# Owlet Cam as a Native UniFi Protect Camera

`owlet_cam` wraps [reolink-unifi-protect](https://github.com/woutersteegers/reolink-unifi-protect) so an Owlet Cam adopts into UniFi Protect as a `UVC G6 Instant` with local motion detection, smart object detection, smart audio detection (including baby crying), and two-way talk-back.

## How it works

- Motion and vision: `vision.FrameReader` decodes one 1080p frame per second from go2rtc and computes a frame-difference mask against Protect's motion zones. `vision.YoloDetector` runs YOLO11s (ONNX, CPU) on every third analysed frame and passes person, vehicle, and animal boxes to `RTSPCam._on_ai_boxes`. That applies Protect's smart zones, drops boxes that never move (so a sleeping baby does not stay stuck in one endless person event), and sends `EventSmartDetect` with object crops.
- Audio: `audio.AudioReader` runs Google's YAMNet every 0.48 s on the camera's microphone and maps AudioSet classes to Protect's audio types (`alrmBabyCry` comes from "Baby cry, infant cry" and "Crying, sobbing"). `audio.AudioEvents` requires 2 hits in 4 frames to start an event and ends it 8 s after the last hit. Protect enables types per camera with `ChangeAudioEventsSettings`. Every `EventSmartAudio` message lists all nine Protect types because Protect only ends an audio event when every type is `leave` or `none`. CO alarm (`alrmCmonx`) is never detected because AudioSet has no class for it.
- Talk-back: Protect sends ADTS AAC datagrams to UDP 7004. `cam.py` re-encodes them to 8 kHz AAC and writes them to the bridge's speaker FIFO (`run/bridge-tmp/owlet-talk-owlet`).
- Snapshots: camera snapshots come from the same 1 fps ffmpeg decode as the analysis pipeline.

The three video channels are stream-copied without re-encoding, using two ffmpeg bitstream filters (`--ffmpeg-args` in `docker-compose.yml`). The bridge's SDP carries the Owlet's initial 864x480 SPS while the actual stream is 1080p. Protect strips SPS/PPS inside keyframes and decodes using the FLV header, so `extract_extradata` emits a fresh FLV header from each keyframe's SPS/PPS. `setts` spaces the P-frame after each keyframe off the keyframe's timestamp.

## Debug and test endpoints

`owlet-protect` listens on `127.0.0.1:8590` (configurable via `--debug-port`):

- `GET /state` returns current motion levels, active events, and recent audio scores.
- `POST /inject/audio?gain=0.5` mixes a posted audio file into the live audio analyzer (useful for testing baby crying alerts without waking the baby).
- `POST /inject/motion?seconds=6` injects a moving block into analysed frames without affecting recordings.
- `POST /args` with JSON such as `{"ai_min_movement": 0}` adjusts numeric options at runtime.

Every message received from Protect is appended to `protect-cam/data/messages.jsonl`.

## Limits

- Privacy masks, line crossing, loiter zones, face recognition, and license plate detection are not implemented on-camera (face and plate recognition can be handled by an AI Key).
- All three Protect video channels share the same copied 1080p stream with no transcoding.
- Expect roughly half a CPU core on a modern x86_64 host, mostly from YOLO11s and 1 fps H.264 decoding.
