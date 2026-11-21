"""Video analysis for the Owlet: one ffmpeg decode feeds motion detection, object
detection and the camera's snapshots.

FrameReader decodes the go2rtc stream once. It writes a full-resolution JPEG each
second (what Protect gets as the camera's snapshot) and hands 640x360 RGB frames
to a callback at `fps`.

MotionDetector compares each frame with a slowly updated background inside
Protect's motion zones and reports a 0-100 level per zone, like a UniFi camera's
EventSmartMotion pulses. YoloDetector runs a YOLO11 ONNX model and returns
person/vehicle/animal boxes in Protect's 0-1000 [x, y, w, h] space.
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import threading
import time
import zlib
from pathlib import Path
from typing import Callable, Optional

import numpy as np

FRAME_W, FRAME_H = 640, 360
# Motion runs on a 160x90 grid, the size of a real camera's motion heatmap.
GRID_W, GRID_H = 160, 90


class FrameReader:
    """Decodes `source` with ffmpeg until stopped, restarting it when it dies."""

    def __init__(
        self,
        source: str,
        snapshot_path: Path,
        on_frame: Callable[[np.ndarray, float], None],
        logger: logging.Logger,
        fps: float = 5.0,
    ) -> None:
        self.source = source
        self.snapshot_path = snapshot_path
        self.on_frame = on_frame
        self.logger = logger
        self.fps = fps
        self._stop = threading.Event()
        self._proc: Optional[subprocess.Popen] = None
        self._thread = threading.Thread(target=self._run, name="frames", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._proc and self._proc.poll() is None:
            self._proc.kill()

    def _cmd(self) -> list[str]:
        graph = (
            f"[0:v]split=2[a][b];"
            f"[a]fps={self.fps},scale={FRAME_W}:{FRAME_H},format=rgb24[det];"
            f"[b]fps=1[snap]"
        )
        return [
            "ffmpeg", "-nostdin", "-loglevel", "error",
            "-rtsp_transport", "tcp", "-timeout", "10000000",
            "-i", self.source, "-an",
            "-filter_complex", graph,
            "-map", "[det]", "-f", "rawvideo", "pipe:1",
            "-map", "[snap]", "-q:v", "3", "-update", "1", "-y", str(self.snapshot_path),
        ]

    def _run(self) -> None:
        size = FRAME_W * FRAME_H * 3
        while not self._stop.is_set():
            self.logger.info(f"Frame reader: {shlex.join(self._cmd()[:12])} ...")
            self._proc = subprocess.Popen(
                self._cmd(), stdout=subprocess.PIPE, stdin=subprocess.DEVNULL
            )
            assert self._proc.stdout
            while not self._stop.is_set():
                buf = self._proc.stdout.read(size)
                if len(buf) < size:
                    break
                frame = np.frombuffer(buf, np.uint8).reshape(FRAME_H, FRAME_W, 3)
                try:
                    self.on_frame(frame, time.time())
                except Exception:
                    self.logger.exception("Frame handler failed")
            if self._proc.poll() is None:
                self._proc.kill()
            self._proc.wait()
            if not self._stop.is_set():
                self.logger.warning("Frame reader exited, restarting in 2s")
                time.sleep(2)


def zone_mask(coord: list, w: int = GRID_W, h: int = GRID_H) -> np.ndarray:
    """Boolean w x h mask of a Protect zone polygon (coord = [x1, y1, x2, y2, ...] in 0-1000)."""
    pts = [(float(coord[i]) * w / 1000, float(coord[i + 1]) * h / 1000)
           for i in range(0, len(coord) - 1, 2)]
    ys, xs = np.mgrid[0:h, 0:w]
    px, py = xs + 0.5, ys + 0.5
    inside = np.zeros((h, w), bool)
    j = len(pts) - 1
    for i in range(len(pts)):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if yi != yj:
            crosses = ((yi > py) != (yj > py)) & (px < (xj - xi) * (py - yi) / (yj - yi) + xi)
            inside ^= crosses
        j = i
    return inside


def point_in_poly(x: float, y: float, coord: list) -> bool:
    """Whether (x, y) in 0-1000 space is inside a Protect polygon."""
    pts = [(float(coord[i]), float(coord[i + 1])) for i in range(0, len(coord) - 1, 2)]
    inside, j = False, len(pts) - 1
    for i in range(len(pts)):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


class MotionDetector:
    """Background-difference motion per Protect motion zone.

    `zones` is ChangeSmartMotionSettings' {id: {coord, level}} where level is the
    zone's sensitivity (0-100). update() returns {zone id: motion level 0-100}
    and whether each zone is over its trigger, and accumulates the heatmap.
    """

    FULL = {"1": {"coord": [0, 0, 1000, 0, 1000, 1000, 0, 1000], "level": 50}}

    def __init__(self) -> None:
        self.bg: Optional[np.ndarray] = None
        self.heat = np.zeros((GRID_H, GRID_W), np.float32)
        self.last_mask = np.zeros((GRID_H, GRID_W), bool)
        self.set_zones(self.FULL)

    def set_zones(self, zones: Optional[dict]) -> None:
        zones = zones or self.FULL
        self.zones = {
            str(zid): (zone_mask(z["coord"]), int(z.get("level", 50)))
            for zid, z in zones.items()
            if isinstance(z, dict) and len(z.get("coord") or []) >= 6
        }

    @staticmethod
    def gray_grid(frame: np.ndarray) -> np.ndarray:
        """640x360 RGB to a lightly blurred 160x90 luma grid."""
        g = frame.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
        g = g.reshape(GRID_H, 4, GRID_W, 4).mean(axis=(1, 3))
        p = np.pad(g, 1, mode="edge")
        return sum(p[dy:dy + GRID_H, dx:dx + GRID_W] for dy in range(3) for dx in range(3)) / 9

    def update(self, frame: np.ndarray) -> tuple[dict, dict]:
        g = self.gray_grid(frame)
        if self.bg is None:
            self.bg = g
            return {z: 0 for z in self.zones}, {z: False for z in self.zones}
        diff = np.abs(g - self.bg)
        levels, triggered = {}, {}
        # A light switching on or the camera flipping to IR changes most of the
        # picture at once. That isn't motion: take it as the new background.
        if (diff > 12).mean() > 0.6:
            self.bg = g
            self.last_mask[:] = False
            return {z: 0 for z in self.zones}, {z: False for z in self.zones}
        masks = []
        for zid, (zmask, sens) in self.zones.items():
            # Higher sensitivity: smaller per-pixel change and a smaller area count.
            pix_thr = 40 - 0.3 * sens          # 40 at 0, 25 at 50, 10 at 100
            moved = (diff > pix_thr) & zmask
            frac = moved.sum() / max(1, zmask.sum())
            levels[zid] = int(min(100, round(100 * frac ** 0.5)))
            min_frac = 0.002 + (100 - sens) / 100 * 0.02   # 0.2% at 100 .. 2.2% at 0
            triggered[zid] = frac >= min_frac
            masks.append(moved)
        moved_any = np.logical_or.reduce(masks) if masks else np.zeros_like(diff, bool)
        self.last_mask = moved_any
        self.heat += moved_any
        # Follow slow changes (shadows, a sleeping baby shifting) without
        # swallowing movement: changed pixels adapt much slower.
        rate = np.where(moved_any, 0.01, 0.08).astype(np.float32)
        self.bg = self.bg * (1 - rate) + g * rate
        return levels, triggered

    def centroid(self) -> Optional[tuple[float, float]]:
        """Centre of the heat accumulated so far, in 0-1 frame coordinates."""
        total = self.heat.sum()
        if total <= 0:
            return None
        ys, xs = np.mgrid[0:GRID_H, 0:GRID_W]
        return float((xs * self.heat).sum() / total / GRID_W), float((ys * self.heat).sum() / total / GRID_H)

    def heatmap_png(self) -> bytes:
        """The accumulated motion as a 160x90 RGBA PNG (red to yellow, transparent where still)."""
        h = self.heat / self.heat.max() if self.heat.max() > 0 else self.heat
        rgba = np.zeros((GRID_H, GRID_W, 4), np.uint8)
        rgba[..., 0] = 255
        rgba[..., 1] = (h * 220).astype(np.uint8)
        rgba[..., 3] = (np.clip(h * 1.5, 0, 1) * 200).astype(np.uint8)
        return encode_png(rgba)

    def reset_heat(self) -> None:
        self.heat[:] = 0


def encode_png(rgba: np.ndarray) -> bytes:
    h, w, _ = rgba.shape
    raw = b"".join(b"\x00" + rgba[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (len(data).to_bytes(4, "big") + tag + data
                + zlib.crc32(tag + data).to_bytes(4, "big"))

    ihdr = w.to_bytes(4, "big") + h.to_bytes(4, "big") + bytes([8, 6, 0, 0, 0])
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


# COCO class ids per Protect object type.
COCO_TO_PROTECT = {
    0: "person",
    1: "vehicle", 2: "vehicle", 3: "vehicle", 5: "vehicle", 7: "vehicle",
    14: "animal", 15: "animal", 16: "animal", 17: "animal", 18: "animal", 19: "animal",
}


class YoloDetector:
    """YOLO11 (Ultralytics ONNX export, 640x640, output [1, 84, 8400]) on CPU."""

    def __init__(self, model_path: str, threads: int = 2, min_conf: float = 0.35) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        # Worker threads otherwise busy-wait between frames, two cores at 100%.
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name
        self.min_conf = min_conf

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Boxes as [{type, coord: [x, y, w, h] 0-1000, confidence}] for a 640x360 RGB frame."""
        pad = (640 - FRAME_H) // 2
        img = np.full((640, 640, 3), 114, np.uint8)
        img[pad:pad + FRAME_H] = frame
        x = img.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.session.run(None, {self.input: x})[0][0]   # (84, 8400)
        return self.decode(out, pad)

    def decode(self, out: np.ndarray, pad: int) -> list[dict]:
        scores = out[4:]
        cls = scores.argmax(0)
        conf = scores[cls, np.arange(scores.shape[1])]
        keep = (conf >= self.min_conf) & np.isin(cls, list(COCO_TO_PROTECT))
        if not keep.any():
            return []
        cx, cy, w, h = out[0][keep], out[1][keep] - pad, out[2][keep], out[3][keep]
        cls, conf = cls[keep], conf[keep]
        x1 = np.clip(cx - w / 2, 0, FRAME_W)
        y1 = np.clip(cy - h / 2, 0, FRAME_H)
        x2 = np.clip(cx + w / 2, 0, FRAME_W)
        y2 = np.clip(cy + h / 2, 0, FRAME_H)
        boxes = []
        kinds = np.array([COCO_TO_PROTECT[int(c)] for c in cls])
        for kind in set(kinds):
            idx = np.where(kinds == kind)[0]
            for i in nms(x1[idx], y1[idx], x2[idx], y2[idx], conf[idx]):
                j = idx[i]
                boxes.append({
                    "type": str(kind),
                    "coord": [
                        int(x1[j] / FRAME_W * 1000), int(y1[j] / FRAME_H * 1000),
                        int((x2[j] - x1[j]) / FRAME_W * 1000), int((y2[j] - y1[j]) / FRAME_H * 1000),
                    ],
                    "confidence": round(float(conf[j]), 3),
                })
        return boxes


def nms(x1, y1, x2, y2, scores, iou: float = 0.45) -> list[int]:
    order = scores.argsort()[::-1]
    area = (x2 - x1) * (y2 - y1)
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        ratio = inter / (area[i] + area[order[1:]] - inter + 1e-6)
        order = order[1:][ratio < iou]
    return keep


def save_jpeg(src: Path, dst: Path, crop_center: Optional[tuple[float, float]] = None,
              size: Optional[tuple[int, int]] = None, square: bool = False) -> Optional[tuple[int, int]]:
    """Writes a resized or square-cropped copy of the JPEG `src`. Returns its (w, h)."""
    from PIL import Image

    try:
        with Image.open(src) as im:
            im.load()
    except OSError:
        return None
    if square:
        side = min(im.width, im.height)
        cx, cy = crop_center or (0.5, 0.5)
        left = int(min(max(cx * im.width - side / 2, 0), im.width - side))
        top = int(min(max(cy * im.height - side / 2, 0), im.height - side))
        im = im.crop((left, top, left + side, top + side))
    if size:
        im = im.resize(size)
    tmp = dst.with_name(dst.name + ".part")
    im.convert("RGB").save(tmp, "JPEG", quality=85)
    os.replace(tmp, dst)
    return im.width, im.height
