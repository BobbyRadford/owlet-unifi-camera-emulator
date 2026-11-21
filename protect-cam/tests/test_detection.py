import zlib

import numpy as np

from owlet_cam.audio import AudioEvents
from owlet_cam.vision import (FRAME_H, FRAME_W, MotionDetector, YoloDetector, encode_png,
                              point_in_poly, zone_mask)


def test_zone_mask_covers_left_half():
    m = zone_mask([0, 0, 500, 0, 500, 1000, 0, 1000])
    assert m[:, :80].all() and not m[:, 80:].any()


def test_point_in_poly():
    tri = [0, 0, 1000, 0, 0, 1000]
    assert point_in_poly(100, 100, tri)
    assert not point_in_poly(900, 900, tri)


def frame(value=0):
    return np.full((FRAME_H, FRAME_W, 3), value, np.uint8)


def test_motion_only_in_its_zone():
    md = MotionDetector()
    md.set_zones({"1": {"coord": [0, 0, 500, 0, 500, 1000, 0, 1000], "level": 50},
                  "2": {"coord": [500, 0, 1000, 0, 1000, 1000, 500, 1000], "level": 50}})
    md.update(frame(40))
    f = frame(40)
    f[100:200, 50:200] = 200          # a bright block in the left half
    levels, trig = md.update(f)
    assert trig["1"] and not trig["2"]
    assert levels["1"] > 20 and levels["2"] == 0


def test_lighting_change_is_not_motion():
    md = MotionDetector()
    md.update(frame(40))
    levels, trig = md.update(frame(160))
    assert not any(trig.values())


def test_audio_event_enters_and_leaves():
    ev = AudioEvents()
    ev.enabled = {"alrmBabyCry"}
    states = [ev.feed({"alrmBabyCry": 0.6}, t * 0.5) for t in range(2)]
    assert states[0] is None
    assert states[1]["alrmBabyCry"] == "enter"
    assert all(v == "none" for k, v in states[1].items() if k != "alrmBabyCry")
    out = None
    t = 1.0
    while out is None:
        t += 0.5
        out = ev.feed({"alrmBabyCry": 0.0}, t)
    assert out["alrmBabyCry"] == "leave" and t > 8.5   # 8 s after the last hit


def test_disabled_audio_type_never_starts():
    ev = AudioEvents()
    ev.enabled = set()
    assert all(ev.feed({"alrmBabyCry": 0.9}, t) is None for t in range(10))


def test_yolo_decode_maps_and_scales():
    det = YoloDetector.__new__(YoloDetector)
    det.min_conf = 0.35
    out = np.zeros((84, 8400), np.float32)
    pad = (640 - FRAME_H) // 2
    # A person centred at (320, 180) in the frame, 100x200 px, and a cat.
    out[:4, 0] = [320, 180 + pad, 100, 200]
    out[4 + 0, 0] = 0.9
    out[:4, 1] = [100, 100 + pad, 50, 50]
    out[4 + 15, 1] = 0.8
    out[:4, 2] = [100, 100 + pad, 50, 50]
    out[4 + 60, 2] = 0.9               # dining table: not a Protect type
    boxes = sorted(det.decode(out, pad), key=lambda b: b["type"])
    assert [b["type"] for b in boxes] == ["animal", "person"]
    person = boxes[1]
    assert person["coord"] == [421, 222, 156, 555]


def test_png_is_valid():
    png = encode_png(np.zeros((90, 160, 4), np.uint8))
    assert png.startswith(b"\x89PNG")
    idat = png[png.index(b"IDAT") + 4:png.index(b"IEND") - 8]
    assert len(zlib.decompress(idat)) == 90 * (160 * 4 + 1)
