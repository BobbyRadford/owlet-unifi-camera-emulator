"""unifi-cam-proxy's command line with the Owlet camera added as the `owlet` source."""

from unifi import main as proxy_main

from owlet_cam.cam import OwletCam

proxy_main.CAMS["owlet"] = OwletCam
proxy_main.main()
