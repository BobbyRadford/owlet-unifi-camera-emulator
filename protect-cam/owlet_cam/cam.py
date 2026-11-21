"""The Owlet Cam as a native UniFi camera (a UVC G6 Instant) in UniFi Protect.

Built on reolink-unifi-protect's RTSPCam, which does adoption, the video
channels, snapshot uploads and smart-detect events. The Owlet does no analysis
of its own, so this adds it from the go2rtc stream, using the event frames a
real G5/G6 sends (copied from Protect's ds payload log):

  * motion: EventSmartMotion start / pulse / stop inside Protect's motion zones
    (ChangeSmartMotionSettings), with heatmap and snapshots on stop
  * person / vehicle / animal: YOLO boxes into RTSPCam's EventSmartDetect
    pipeline, which honors Protect's smart zones
  * audio: EventSmartAudio for the types Protect enables
    (ChangeAudioEventsSettings), from YAMNet
  * talk-back: Protect's ADTS AAC datagrams, re-encoded at 8 kHz into the
    Owlet bridge's speaker FIFO

Run it as `python -m owlet_cam ... owlet --source rtsp://...`.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
from aiohttp import web

from unifi.cams.rtsp import RTSPCam

from owlet_cam.audio import AudioEvents, AudioReader, Yamnet, SAMPLE_RATE
from owlet_cam.vision import FrameReader, MotionDetector, YoloDetector, save_jpeg

OBJECT_TYPES = ["person", "vehicle", "animal"]
AUDIO_TYPES = ["alrmSmoke", "alrmSiren", "alrmBabyCry", "alrmSpeak", "alrmBurglar",
               "alrmCarHorn", "alrmBark", "alrmGlassBreak"]
# Every audio type Protect knows. EventSmartAudio has to carry all of them:
# Protect only ends an audio event when each one is "leave" or "none".
PROTECT_AUDIO_TYPES = AUDIO_TYPES + ["alrmCmonx"]
# A motion pulse's eventId on a real camera (u64 max: "not an event").
PULSE_EVENT_ID = 18446744073709551615


class OwletCam(RTSPCam):
    @classmethod
    def add_parser(cls, parser) -> None:
        super().add_parser(parser)
        parser.add_argument("--models", default="/app/models", help="Directory with the model files")
        parser.add_argument("--yolo-model", default="yolo11s.onnx")
        parser.add_argument("--yolo-threads", type=int, default=2)
        parser.add_argument("--detect-fps", type=float, default=5.0,
                            help="Frames per second analysed for motion")
        parser.add_argument("--yolo-every", type=int, default=2,
                            help="Run object detection on every Nth analysed frame")
        parser.add_argument("--talk-fifo", default=None,
                            help="The bridge's speaker FIFO (owlet-talk-<name>); enables talk-back")
        parser.add_argument("--talk-rate", default="8000", help="Sample rate the Owlet speaker takes")
        parser.add_argument("--debug-port", type=int, default=0,
                            help="Loopback HTTP port for /state and /inject/audio")
        parser.add_argument("--message-log", default=None,
                            help="Append every message from Protect to this JSONL file")

    def __init__(self, args, logger) -> None:
        super().__init__(args, logger)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._started = False
        self.snapshot_file = Path(self.snapshot_dir, "screen.jpg")
        models = Path(args.models)
        self.motion = MotionDetector()
        self.yolo = YoloDetector(str(models / args.yolo_model), threads=args.yolo_threads,
                                 min_conf=min(0.35, args.ai_min_confidence))
        self.yamnet = Yamnet(str(models / "yamnet.tflite"), str(models / "yamnet_class_map.csv"))
        self.audio_events = AudioEvents()
        self.frames = FrameReader(args.source[0], self.snapshot_file, self._on_frame,
                                  logger, fps=args.detect_fps)
        self.audio = AudioReader(args.source[0], self.yamnet, self._on_audio, logger)
        self._frame_n = 0
        # Motion state (frames thread)
        self.motion_enabled = True
        self.motion_linger_start = 1.0
        self.motion_linger_stop = 3.0
        self.motion_max = 300.0
        self._mo_id = 0
        self._mo_start: Optional[float] = None
        self._mo_trig_since: Optional[float] = None
        self._mo_last_trig = 0.0
        self._mo_last_pulse = 0.0
        self._mo_best = (-1, 0.0)            # (level, wall time) of the best frame
        self._mo_levels: dict = {}
        # Audio state (audio thread)
        self._au_id = 0
        self._au_state: dict = {}
        self._au_last: dict = {}
        self._au_log_ts = 0.0
        # Talk-back
        self._talk_transport = None
        self._talk_queue: asyncio.Queue = asyncio.Queue(maxsize=400)
        self._talk_task: Optional[asyncio.Task] = None
        self._debug_runner = None
        self._fake_motion_until = 0.0
        self._talk_packets = 0

    # ---- identity ------------------------------------------------------------

    async def get_feature_flags(self) -> dict[str, Any]:
        """The hello's `features`, in a real camera's vocabulary (see Protect's
        featureFlags.deserializeFromCamera): `smartDetect` lists object and
        audio classes together, `speaker` turns on talk-back."""
        talk = bool(self.args.talk_fifo)
        return {
            "mic": 1,
            "speaker": talk,
            "aec": [],
            "smartDetect": OBJECT_TYPES + AUDIO_TYPES,
            "motionDetect": ["enhanced"],
            "videoMode": ["default"],
            "videoModeMaxFps": [self.args.hi_fps],
            "videoCodecs": ["h264"],
            "audioCodecs": ["aac"],
            "audioStyle": ["nature"],
            "truedaynight": True,
            "squareEventThumbnail": True,
            "fullHdSnapshot": True,
            "excludeZone": {"maxZones": 16, "rectangleOnly": True},
            # Without a scale-down ladder Protect logs "no max scale down level".
            "maxScaleDownLevel": 2,
            "downScaleResolutions": [[1920, 1080], [1280, 720], [640, 360]],
            "verticalFlip": False,
            "horizontalFlip": False,
            "hasMic": True,
            "hasSpeaker": talk,
            "hasSmartDetect": True,
            "hasMotionZones": True,
        }

    # Snapshots come from FrameReader, which decodes the stream anyway.
    def start_snapshot_stream(self) -> None:
        return

    async def get_snapshot(self) -> Path:
        return self.snapshot_file

    # ---- lifecycle -----------------------------------------------------------

    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()
        if not self._started:
            self._started = True
            self.frames.start()
            self.audio.start()
            if self.args.debug_port:
                await self._start_debug_server()
        await super().run()

    async def close(self) -> None:
        await self._end_motion(time.time())
        states = self.audio_events.clear()
        if states:
            await self._send_audio(states, -60.0)
        await super().close()
        self._session = None

    def _submit(self, coro) -> None:
        """Runs a coroutine on the event loop from a reader thread."""
        if self._loop is None or self._session is None:
            coro.close()
            return
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        fut.add_done_callback(self._log_failure)

    def _log_failure(self, fut) -> None:
        exc = fut.exception()
        if exc:
            self.logger.warning(f"Event send failed: {exc!r}")

    # ---- settings from Protect -------------------------------------------------

    async def process(self, msg: bytes) -> bool:
        m = json.loads(msg)
        fn = m.get("functionName")
        payload = m.get("payload") or {}
        self._log_message(m)
        if fn == "ChangeSmartMotionSettings":
            self.motion_enabled = bool(payload.get("enable", True))
            self.motion_linger_start = payload.get("lingerEventStartMSec", 1000) / 1000
            self.motion_linger_stop = payload.get("lingerEventStopMSec", 3000) / 1000
            self.motion_max = payload.get("eventMaxDurationMSec", 300000) / 1000
            self.motion.set_zones(payload.get("zones"))
            self.logger.info(
                f"Motion: enabled={self.motion_enabled}, zones="
                + ", ".join(f"{z}@{s}" for z, (_, s) in self.motion.zones.items()))
        elif fn == "ChangeAudioEventsSettings":
            enabled = {t for t in AUDIO_TYPES
                       if payload.get("enable" + t[0].upper() + t[1:]) == 1}
            self.audio_events.enabled = enabled
            self.logger.info(f"Audio detections enabled: {sorted(enabled) or 'none'}")
        res = await super().process(msg)
        if fn == "ChangeSmartDetectSettings" and isinstance(payload.get("enableSmartDetect"), list):
            allowed = set(payload["enableSmartDetect"])
            current = self.smart_detect_enabled
            self.smart_detect_enabled = (current & allowed) if current is not None else allowed
            self.logger.info(f"Smart detections enabled: {sorted(self.smart_detect_enabled)}")
        return res

    def _log_message(self, m: dict) -> None:
        path = self.args.message_log
        if not path or m.get("functionName") in ("ubnt_avclient_timeSync", "GetRequest"):
            return
        try:
            if os.path.exists(path) and os.path.getsize(path) > 5_000_000:
                os.replace(path, path + ".1")
            with open(path, "a") as f:
                f.write(json.dumps({"t": time.time(), **m}) + "\n")
        except OSError:
            pass

    # ---- video: motion and objects -------------------------------------------------

    def _on_frame(self, frame: np.ndarray, now: float) -> None:
        if now < self._fake_motion_until:
            # /inject/motion: a block sweeping across the analysed frame only.
            frame = frame.copy()
            x = int((now * 120) % (frame.shape[1] - 120))
            frame[120:240, x:x + 120] = 255
        levels, triggered = self.motion.update(frame)
        self._motion_step(levels, triggered, now)
        self._frame_n += 1
        if self._frame_n % max(1, self.args.yolo_every) == 0:
            boxes = self.yolo.detect(frame)
            self._submit(self._on_ai_boxes(boxes))

    def _motion_step(self, levels: dict, triggered: dict, now: float) -> None:
        self._mo_levels = levels
        hit = self.motion_enabled and any(triggered.values())
        if hit:
            self._mo_last_trig = now
            self._mo_trig_since = self._mo_trig_since or now
        else:
            self._mo_trig_since = None
        top = max(levels.values(), default=0)
        if self._mo_start is None:
            if hit and now - self._mo_trig_since >= self.motion_linger_start:
                self._mo_start = now
                self._mo_id += 1
                self._mo_best = (-1, now)
                self.motion.reset_heat()
                self._submit(self._send_motion("start", self._mo_id, levels, now))
        elif now - self._mo_last_trig > self.motion_linger_stop or now - self._mo_start > self.motion_max:
            self._submit(self._end_motion(now))
        if self._mo_start is not None and top > self._mo_best[0]:
            # Keep the frame with the most movement as the event's snapshot.
            self._mo_best = (top, now)
            try:
                shutil.copyfile(self.snapshot_file, Path(self.snapshot_dir, "motion_best.jpg"))
            except OSError:
                pass
        if (self._mo_start is not None or top > 0) and now - self._mo_last_pulse >= 1.0:
            self._mo_last_pulse = now
            self._submit(self._send_motion("unknown", PULSE_EVENT_ID, levels, now, kind="pulse"))

    def _motion_payload(self, edge: str, event_id: int, now: float, kind: str = "motion") -> dict:
        mono = int(self.get_uptime() * 1000)
        return {
            "clockBestMonotonic": 0, "clockBestWall": 0,
            "clockMonotonic": mono, "clockStream": mono, "clockStreamRate": 1000,
            "clockWall": int(now * 1000),
            "edgeType": edge, "eventId": event_id, "eventType": kind,
            "motionHeatmap": "", "motionHeatmapHeight": 0, "motionHeatmapWidth": 0,
            "motionRawHeatmapNPZ": "", "motionSnapshot": "",
            "motionSnapshotFullFoV": "", "motionSnapshotFullFoVHeight": 0,
            "motionSnapshotFullFoVWidth": 0, "motionSnapshotHeight": 0, "motionSnapshotWidth": 0,
        }

    async def _send_motion(self, edge: str, event_id: int, levels: dict, now: float,
                           kind: str = "motion") -> None:
        payload = self._motion_payload(edge, event_id, now, kind)
        payload["levels"] = {z: int(v) for z, v in levels.items()}
        if edge == "start":
            self.logger.info(f"Motion start #{event_id} {payload['levels']}")
        await self.send(self.gen_event("EventSmartMotion", payload))

    async def _end_motion(self, now: float) -> None:
        start, event_id = self._mo_start, self._mo_id
        if start is None:
            return
        self._mo_start = None
        payload = self._motion_payload("stop", event_id, now)
        best_level, best_ts = self._mo_best
        payload["clockBestWall"] = int(best_ts * 1000)
        payload["clockBestMonotonic"] = int((self.get_uptime() - (now - best_ts)) * 1000)
        payload.update(await asyncio.to_thread(self._motion_files, event_id))
        self.logger.info(f"Motion stop #{event_id} after {now - start:.1f}s (peak {best_level})")
        await self.send(self.gen_event("EventSmartMotion", payload))

    def _motion_files(self, event_id: int) -> dict:
        """Writes the stop frame's heatmap and snapshots and registers them for GetRequest."""
        d = Path(self.snapshot_dir)
        best = d / "motion_best.jpg"
        src = best if best.exists() else self.snapshot_file
        out: dict[str, Any] = {}
        heat = f"heatmap_{event_id:08d}.png"
        (d / heat).write_bytes(self.motion.heatmap_png())
        self._smart_snapshot_files[heat] = d / heat
        out.update(motionHeatmap=heat, motionHeatmapWidth=160, motionHeatmapHeight=90)
        full = f"motionsnap_{event_id:08d}_fullfov.jpg"
        size = save_jpeg(src, d / full, size=(640, 360))
        if size:
            self._smart_snapshot_files[full] = d / full
            out.update(motionSnapshotFullFoV=full, motionSnapshotFullFoVWidth=size[0],
                       motionSnapshotFullFoVHeight=size[1])
        snap = f"motionsnap_{event_id:08d}.jpg"
        size = save_jpeg(src, d / snap, crop_center=self.motion.centroid(), size=(360, 360),
                         square=True)
        if size:
            self._smart_snapshot_files[snap] = d / snap
            out.update(motionSnapshot=snap, motionSnapshotWidth=size[0],
                       motionSnapshotHeight=size[1])
        self.prune_stale_crops()
        return out

    # ---- audio ------------------------------------------------------------------

    def _on_audio(self, scores: dict, dbfs: float, top: str, now: float) -> None:
        self._au_last = {"scores": scores, "dbfs": round(dbfs, 1), "top": top, "t": now}
        if now - self._au_log_ts > 60 or scores.get("alrmBabyCry", 0) > 0.1:
            self._au_log_ts = now
            self.logger.info(f"Audio {dbfs:.0f} dBFS: {top} | cry {scores.get('alrmBabyCry', 0):.2f}")
        states = self.audio_events.feed(scores, now)
        if states:
            self._submit(self._send_audio(states, dbfs))

    async def _send_audio(self, states: dict, dbfs: float) -> None:
        if not self._au_state or all(v in ("none", "leave") for v in self._au_state.values()):
            self._au_id += 1
        self._au_state = states
        mono = int(self.get_uptime() * 1000)
        payload = {
            "clockMonotonic": mono, "clockStream": mono, "clockStreamRate": 1000,
            "clockWall": int(time.time() * 1000),
            "eventId": self._au_id, "eventType": "audio",
            "leveldB": int(round(dbfs)),
            "levels": int(max(0, min(100, (dbfs + 70) * 100 / 70))),
            "loudNoise": "none", "soundLoss": "none",
            **{t: states.get(t, "none") for t in PROTECT_AUDIO_TYPES},
        }
        active = {t: s for t, s in states.items() if s != "none"}
        self.logger.info(f"Audio event #{self._au_id}: {active}")
        await self.send(self.gen_event("EventSmartAudio", payload))

    # ---- talk-back -------------------------------------------------------------

    async def process_talkback_settings(self, m: dict) -> dict:
        payload = m.get("payload") or {}
        if self.args.talk_fifo and self._talk_transport is None:
            port = int(payload.get("bindPort") or 7004)
            cam = self

            class Receiver(asyncio.DatagramProtocol):
                def datagram_received(self, data: bytes, addr) -> None:
                    cam._talk_feed(data)

            self._talk_transport, _ = await asyncio.get_running_loop().create_datagram_endpoint(
                Receiver, local_addr=("0.0.0.0", port))
            self.logger.info(f"Talk-back listening on UDP {port}")
        return self.gen_response("ChangeTalkbackSettings", response_to=m["messageId"],
                                 payload=payload)

    def _talk_feed(self, data: bytes) -> None:
        self._talk_packets += 1
        if self._talk_task is None or self._talk_task.done():
            self._talk_task = asyncio.create_task(self._talk_session())
        try:
            self._talk_queue.put_nowait(data)
        except asyncio.QueueFull:
            pass

    async def _talk_session(self) -> None:
        """One push-to-talk: Protect's ADTS AAC to 8 kHz ADTS AAC in the speaker FIFO."""
        ff = await asyncio.create_subprocess_exec(
            "ffmpeg", "-nostdin", "-loglevel", "error", "-fflags", "nobuffer",
            "-probesize", "32", "-analyzeduration", "0", "-f", "aac", "-i", "pipe:0",
            "-ac", "1", "-ar", str(self.args.talk_rate), "-c:a", "aac", "-b:a", "24k",
            "-f", "adts", "-flush_packets", "1", "pipe:1",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE)
        try:
            fd = os.open(self.args.talk_fifo, os.O_WRONLY | os.O_NONBLOCK)
            os.set_blocking(fd, True)
        except OSError as e:
            self.logger.warning(f"Talk-back: speaker FIFO not available ({e})")
            ff.kill()
            return
        self.logger.info("Talk-back started")
        written = 0

        async def pump() -> None:
            nonlocal written
            assert ff.stdout
            while chunk := await ff.stdout.read(4096):
                await asyncio.to_thread(os.write, fd, chunk)
                written += len(chunk)

        out = asyncio.create_task(pump())
        try:
            while True:
                try:
                    data = await asyncio.wait_for(self._talk_queue.get(), 2.0)
                except asyncio.TimeoutError:
                    break
                assert ff.stdin
                ff.stdin.write(data)
                await ff.stdin.drain()
            assert ff.stdin
            ff.stdin.close()
            await asyncio.wait_for(out, 5)
        except Exception as e:
            self.logger.warning(f"Talk-back failed: {e!r}")
        finally:
            out.cancel()
            if ff.returncode is None:
                ff.kill()
            os.close(fd)
            self.logger.info(f"Talk-back ended ({written} bytes to the speaker)")

    # ---- debug endpoint ---------------------------------------------------------

    async def _start_debug_server(self) -> None:
        app = web.Application(client_max_size=50 * 1024 * 1024)
        app.add_routes([web.get("/state", self._debug_state),
                        web.post("/inject/audio", self._debug_inject),
                        web.post("/inject/motion", self._debug_motion),
                        web.post("/args", self._debug_args)])
        self._debug_runner = web.AppRunner(app)
        await self._debug_runner.setup()
        await web.TCPSite(self._debug_runner, "127.0.0.1", self.args.debug_port).start()
        self.logger.info(f"Debug API on 127.0.0.1:{self.args.debug_port}")

    async def _debug_state(self, request) -> web.Response:
        return web.json_response({
            "connected": self._session is not None,
            "motion": {"enabled": self.motion_enabled, "levels": self._mo_levels,
                       "event": self._mo_id,
                       "active": self._mo_start is not None,
                       "zones": {z: s for z, (_, s) in self.motion.zones.items()}},
            "smart": {"enabled": sorted(self.smart_detect_enabled or []),
                      "zones": list((self.smart_detect_zones or {}).keys()),
                      "active": self._motion_event_ts is not None},
            "talk_packets": self._talk_packets,
            "audio": {"enabled": sorted(self.audio_events.enabled),
                      "active": list(self.audio_events.active), "last": self._au_last},
        })

    async def _debug_motion(self, request) -> web.Response:
        seconds = float(request.query.get("seconds", "6"))
        self._fake_motion_until = time.time() + seconds
        return web.json_response({"motion_seconds": seconds})

    async def _debug_args(self, request) -> web.Response:
        """Sets numeric command-line options at runtime, e.g. {"ai_min_movement": 0}."""
        changes = await request.json()
        for k, v in changes.items():
            if not hasattr(self.args, k) or not isinstance(v, (int, float)):
                return web.json_response({"error": f"bad option {k}"}, status=400)
            setattr(self.args, k, v)
        return web.json_response({k: getattr(self.args, k) for k in changes})

    async def _debug_inject(self, request) -> web.Response:
        """Mixes an uploaded audio file (any format ffmpeg reads) into the live audio."""
        body = await request.read()
        gain = float(request.query.get("gain", "1.0"))
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-nostdin", "-loglevel", "error", "-i", "pipe:0", "-ac", "1",
            "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE)
        pcm, _ = await proc.communicate(body)
        samples = np.frombuffer(pcm, np.float32) * gain
        self.audio.inject(samples)
        return web.json_response({"injected_seconds": round(len(samples) / SAMPLE_RATE, 2)})
