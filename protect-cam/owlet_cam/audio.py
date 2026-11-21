"""Audio detections for the Owlet: YAMNet on the camera's microphone.

AudioReader pulls 16 kHz mono PCM from the go2rtc stream with ffmpeg. Every
0.48 s it runs YAMNet (Google's AudioSet classifier, 521 classes) on the last
0.975 s and maps the scores onto Protect's audio types (alrmBabyCry and the
rest). AudioEvents turns those scores into the enter/leave states a UniFi
camera reports in EventSmartAudio.
"""

from __future__ import annotations

import collections
import csv
import logging
import math
import subprocess
import threading
import time
from typing import Callable, Optional

import numpy as np

SAMPLE_RATE = 16000
WINDOW = 15600          # YAMNet's input: 0.975 s
HOP = 7680              # 0.48 s, YAMNet's own frame hop
WARMUP_HOPS = 4         # frames ignored after the stream (re)starts

# AudioSet display names per Protect audio type. A type's score is the highest
# score among its classes. alrmCmonx (CO alarm) has no AudioSet class and its
# beep pattern isn't distinguishable from a smoke alarm here, so it isn't offered.
PROTECT_CLASSES = {
    "alrmBabyCry": ["Baby cry, infant cry", "Crying, sobbing"],
    "alrmSpeak": ["Speech", "Child speech, kid speaking", "Conversation",
                  "Narration, monologue"],
    "alrmBark": ["Bark", "Dog", "Yip", "Bow-wow"],
    "alrmSmoke": ["Smoke detector, smoke alarm", "Fire alarm"],
    "alrmSiren": ["Siren", "Civil defense siren", "Police car (siren)",
                  "Ambulance (siren)", "Fire engine, fire truck (siren)", "Emergency vehicle"],
    "alrmBurglar": ["Car alarm", "Burglar alarm"],
    "alrmCarHorn": ["Vehicle horn, car horn, honking", "Air horn, truck horn"],
    "alrmGlassBreak": ["Shatter", "Glass"],
}

# Per type: (score to count a frame as a hit, hits needed in the last `of` frames
# to start, seconds without hits before the event ends).
DEFAULT_TUNING = {
    "alrmBabyCry": (0.20, 2, 4, 8.0),
    "alrmSpeak": (0.50, 3, 4, 5.0),
    "alrmBark": (0.35, 2, 4, 5.0),
    "alrmSmoke": (0.30, 3, 5, 8.0),
    "alrmSiren": (0.35, 3, 5, 6.0),
    "alrmBurglar": (0.35, 3, 5, 6.0),
    "alrmCarHorn": (0.40, 2, 4, 4.0),
    "alrmGlassBreak": (0.30, 1, 2, 3.0),
}


class Yamnet:
    """The TF-Hub YAMNet TFLite model: 15600 float samples in, 521 scores out."""

    def __init__(self, model_path: str, class_map: str, threads: int = 1) -> None:
        from ai_edge_litert.interpreter import Interpreter

        self.interp = Interpreter(model_path=model_path, num_threads=threads)
        inp = self.interp.get_input_details()[0]
        if list(inp["shape"]) != [WINDOW]:
            self.interp.resize_tensor_input(inp["index"], [WINDOW])
        self.interp.allocate_tensors()
        self.inp = inp["index"]
        self.out = self.interp.get_output_details()[0]["index"]
        with open(class_map) as f:
            self.names = [row["display_name"] for row in csv.DictReader(f)]
        index = {n: i for i, n in enumerate(self.names)}
        self.groups = {t: [index[n] for n in names if n in index]
                       for t, names in PROTECT_CLASSES.items()}

    def scores(self, samples: np.ndarray) -> np.ndarray:
        self.interp.set_tensor(self.inp, samples.astype(np.float32))
        self.interp.invoke()
        return self.interp.get_tensor(self.out).reshape(-1)

    def protect_scores(self, scores: np.ndarray) -> dict[str, float]:
        return {t: float(scores[idx].max()) if idx else 0.0 for t, idx in self.groups.items()}

    def top(self, scores: np.ndarray, n: int = 3) -> str:
        return ", ".join(f"{self.names[i]} {scores[i]:.2f}" for i in scores.argsort()[::-1][:n])


class AudioEvents:
    """Hit counting and lingering per audio type, producing EventSmartAudio states.

    feed() takes one frame's {type: score} and returns the new per-type states
    when they change ({type: "enter" | "moving" | "leave" | "none"}), else None.
    Only types in `enabled` (from ChangeAudioEventsSettings) can become active.
    """

    def __init__(self, tuning: Optional[dict] = None) -> None:
        self.tuning = dict(DEFAULT_TUNING, **(tuning or {}))
        self.enabled: set[str] = set(self.tuning)
        self.hits = {t: collections.deque(maxlen=self.tuning[t][2]) for t in self.tuning}
        self.active: dict[str, float] = {}   # type -> time of its last hit
        self.peak: dict[str, float] = {}

    def feed(self, scores: dict[str, float], now: float) -> Optional[dict[str, str]]:
        started, ended = [], []
        for t, (thr, need, _of, linger) in self.tuning.items():
            hit = t in self.enabled and scores.get(t, 0.0) >= thr
            self.hits[t].append(hit)
            if t in self.active:
                if hit:
                    self.active[t] = now
                    self.peak[t] = max(self.peak.get(t, 0), scores[t])
                elif now - self.active[t] > linger or t not in self.enabled:
                    del self.active[t]
                    ended.append(t)
            elif hit and sum(self.hits[t]) >= need:
                self.active[t] = now
                self.peak[t] = scores[t]
                started.append(t)
        if not started and not ended:
            return None
        states = {t: "none" for t in self.tuning}
        for t in self.active:
            states[t] = "enter" if t in started else "moving"
        for t in ended:
            states[t] = "leave"
        return states

    def clear(self) -> Optional[dict[str, str]]:
        """Ends every active type (on shutdown or disconnect)."""
        if not self.active:
            return None
        states = {t: "none" for t in self.tuning}
        for t in self.active:
            states[t] = "leave"
        self.active.clear()
        return states


class AudioReader:
    """Reads the stream's audio and calls on_frame(scores, dbfs, top, now) every hop.

    inject() mixes a test clip (float samples at 16 kHz) into the live audio, so a
    recorded cry can be checked against the nursery's real background noise.
    """

    def __init__(self, source: str, yamnet: Yamnet,
                 on_frame: Callable[[dict, float, str, float], None],
                 logger: logging.Logger) -> None:
        self.source = source
        self.yamnet = yamnet
        self.on_frame = on_frame
        self.logger = logger
        self._stop = threading.Event()
        self._proc: Optional[subprocess.Popen] = None
        self._inject = np.zeros(0, np.float32)
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="audio", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._proc and self._proc.poll() is None:
            self._proc.kill()

    def inject(self, samples: np.ndarray) -> None:
        with self._lock:
            self._inject = np.concatenate([self._inject, samples.astype(np.float32)])

    def _mix(self, chunk: np.ndarray) -> np.ndarray:
        with self._lock:
            if not len(self._inject):
                return chunk
            n = min(len(chunk), len(self._inject))
            chunk = chunk.copy()
            chunk[:n] += self._inject[:n]
            self._inject = self._inject[n:]
        return np.clip(chunk, -1, 1)

    def _run(self) -> None:
        cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-rtsp_transport", "tcp",
               "-timeout", "10000000", "-i", self.source, "-vn",
               "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "pipe:1"]
        while not self._stop.is_set():
            self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stdin=subprocess.DEVNULL)
            assert self._proc.stdout
            window = np.zeros(WINDOW, np.float32)
            hops = 0
            while not self._stop.is_set():
                raw = self._proc.stdout.read(HOP * 2)
                if len(raw) < HOP * 2:
                    break
                chunk = self._mix(np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0)
                window = np.concatenate([window[HOP:], chunk])
                hops += 1
                # The first windows after a (re)start jump from digital silence to
                # the room, which YAMNet scored as a bark. Skip about 2 s.
                if hops <= WARMUP_HOPS:
                    continue
                try:
                    scores = self.yamnet.scores(window)
                    rms = math.sqrt(float(np.mean(chunk ** 2))) + 1e-9
                    self.on_frame(self.yamnet.protect_scores(scores), 20 * math.log10(rms),
                                  self.yamnet.top(scores), time.time())
                except Exception:
                    self.logger.exception("Audio frame failed")
            if self._proc.poll() is None:
                self._proc.kill()
            self._proc.wait()
            if not self._stop.is_set():
                self.logger.warning("Audio reader exited, restarting in 2s")
                time.sleep(2)
