# Owlet UniFi Camera Emulator

Emulates a native UniFi Protect camera (`UVC G6 Instant`) backed by an Owlet Cam. Unlike a generic ONVIF camera, which Protect only reads motion from and cannot talk back to, this gives your Owlet Cam full native UniFi Protect support:

- Motion zones and smart detection zones configured in Protect.
- Local person, vehicle, and animal detections (YOLO11s ONNX on CPU).
- Local audio detections including baby crying, smoke/siren alarms, barking, and speech (Google YAMNet).
- Two-way talk-back from the Protect app to the Owlet Cam speaker.

For AI Key features such as natural-language search, tagging, person re-ID, and speech to text, see the companion [unifi-ai-key-emulator](https://github.com/BobbyRadford/unifi-ai-key-emulator) project.

## How it works

`docker-compose.yml` runs three containers on an x86_64 Linux host:

1. `owlet-bridge` runs [btoth525's Owlet-To-Rtsp bridge](https://github.com/btoth525/Owlet-To-Rtsp) (`owlet-bridge-native`, pinned by digest). It logs into Owlet for your camera's keys, pulls H.264 video and Opus audio over ThroughTek P2P, and serves `rtsp://127.0.0.1:8554/owlet`. Its control panel on port 8088 has no authentication, so it binds to `127.0.0.1` only.
2. `owlet-go2rtc` runs go2rtc 1.9.15 and re-serves the stream on `rtsp://127.0.0.1:8564/owlet`, converting Opus audio to 16 kHz mono AAC (the codec UniFi cameras send) while copying the H.264 video untouched.
3. `owlet-protect` builds `protect-cam/`, which emulates a native `UVC G6 Instant` camera (built on [reolink-unifi-protect](https://github.com/woutersteegers/reolink-unifi-protect)) and runs local motion, vision, audio detection, and talk-back. See [protect-cam/README.md](protect-cam/README.md) for detection internals.

## Setup from scratch

### 1. Extract the ThroughTek libraries

`owlet-bridge` requires three proprietary x86_64 ThroughTek libraries (`libIOTCAPIs.so`, `libAVAPIs.so`, and `libTUTKGlobalAPIs.so`) placed in `libs/x86_64/`. They are not included in this repository.

You can extract them from the legacy Owlet Care Android APK (`com.owletcare.owletcare`, signing cert SHA-1 `2A3BC26DB0B8B0792DBE28E6FFDC2598F9B12B74`, downloadable from APKPure with [apkeep](https://github.com/EFForg/apkeep)). The newer Owlet Dream APK only ships arm64 libraries.

After unzipping the APK's `lib/x86_64/` files into `libs/x86_64/`, add the `libstdc++.so` symlink that those older libraries expect inside the bridge container:

```sh
mkdir -p libs/x86_64
# Copy libIOTCAPIs.so, libAVAPIs.so, libTUTKGlobalAPIs.so into libs/x86_64/
ln -sf /data/data/com.termux/files/usr/lib/libc++_shared.so libs/x86_64/libstdc++.so
```

### 2. Patch the bridge client

The legacy ThroughTek libraries expect a 20-byte connect struct (rather than 160 bytes) and a different `avServStart2` signature for talk-back audio. Run the helper script to extract `tutk_client.py` from the pinned bridge image and apply `patches/tutk_client.diff`:

```sh
./scripts/patch-bridge.sh
```

### 3. Export YOLO11s and generate a camera certificate

Export `yolo11s.onnx` into `protect-cam/models/` (`yamnet.tflite` and `yamnet_class_map.csv` are already in the repository):

```sh
docker run --rm -v "$PWD/protect-cam/models:/out" -w /out python:3.12-slim sh -c \
  "apt-get update -qq && apt-get install -y -qq libxcb1 libgl1 libglib2.0-0 >/dev/null &&
   pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu &&
   pip install -q ultralytics onnx onnxslim && yolo export model=yolo11s.pt format=onnx imgsz=640 opset=17"
```

Generate a self-signed EC client certificate for Protect adoption:

```sh
mkdir -p protect-cam/data
openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 -nodes -days 7300 \
  -subj "/CN=owlet-cam" -keyout protect-cam/data/client.pem -out protect-cam/data/client.pem
```

### 4. Configure `.env` and firewall

Copy `.env.example` to `.env` and fill in your NVR IP, this host's IP, an unused MAC address, and a one-time adoption token:

```sh
cp .env.example .env
```

To fetch a one-time adoption token from your Protect NVR using `tools/unifi.py`:

```sh
UNIFI_HOST=https://<nvr-ip> UNIFI_USER=<username> python3 -I tools/unifi.py login <password-file>
UNIFI_HOST=https://<nvr-ip> python3 -I tools/unifi.py GET /proxy/protect/api/cameras/manage-payload
```

Put `mgmt.token` in `.env` as `OWLET_ADOPTION_TOKEN`. Once the camera is adopted, Protect accepts restarts without needing a fresh token.

If your Protect NVR is on an isolated VLAN, allow UDP port 7004 from the NVR IP to this host's IP so talk-back audio reaches `owlet-protect`.

### 5. Start the stack and configure your Owlet account

```sh
docker compose up -d --build
```

Open the `owlet-bridge` control panel over an SSH tunnel (`ssh -L 8088:127.0.0.1:8088 <host>`, then `http://localhost:8088`) and enter your Owlet email, password, and camera DSN, naming the camera `owlet` so it publishes at `rtsp://127.0.0.1:8554/owlet`.

If your Owlet account normally uses Google Sign-In, set a password through Owlet's password reset flow. To look up your camera's DSN from your Owlet account, run:

```sh
python3 -I tools/find_cameras.py <email-file> <password-file>
```

Once the stream is live, `owlet-protect` connects to Protect and adopts automatically as a `UVC G6 Instant`.

## Checking and testing

- Inspect live motion, object, and audio detection state with `curl http://localhost:8590/state`.
- Test baby crying detection without disturbing anyone by injecting an audio clip into the live audio pipeline: `curl -X POST --data-binary @baby_cry.wav 'http://localhost:8590/inject/audio?gain=0.5'`.
- Test motion detection with `curl -X POST 'http://localhost:8590/inject/motion?seconds=6'`.
- Run the unit tests with:
  ```sh
  docker run --rm --entrypoint python owlet-unifi-camera-emulator -m pytest -q tests
  ```
