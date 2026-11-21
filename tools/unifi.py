"""Minimal UniFi OS API client.

Logs in once and stores the session cookie in tools/.unifi-session-<host> (mode 600).
Set UNIFI_HOST (for example https://192.168.1.10) and UNIFI_USER in the environment.

Usage:
  UNIFI_HOST=https://192.168.1.10 UNIFI_USER=admin python3 -I tools/unifi.py login <password-file> [mfa-code]
  UNIFI_HOST=https://192.168.1.10 python3 -I tools/unifi.py GET /proxy/protect/api/cameras/manage-payload
"""
import http.cookiejar, json, os, ssl, sys, urllib.error, urllib.request

HOST = os.environ.get("UNIFI_HOST", "").rstrip("/")
if not HOST:
    sys.exit("Set UNIFI_HOST, e.g. UNIFI_HOST=https://192.168.1.10")
USER = os.environ.get("UNIFI_USER", "")
JAR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".unifi-session-" + HOST.split("//", 1)[-1].replace(":", "_"),
)
ctx = ssl._create_unverified_context()
jar = http.cookiejar.LWPCookieJar(JAR)
if os.path.exists(JAR):
    jar.load(ignore_discard=True, ignore_expires=True)
opener = urllib.request.build_opener(
    urllib.request.HTTPCookieProcessor(jar),
    urllib.request.HTTPSHandler(context=ctx),
)


def call(method, path, body=None, headers=None):
    h = {"Content-Type": "application/json", **(headers or {})}
    csrf = os.environ.get("CSRF") or _csrf()
    if csrf:
        h["X-CSRF-Token"] = csrf
    req = urllib.request.Request(
        HOST + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers=h,
        method=method,
    )
    try:
        with opener.open(req, timeout=30) as r:
            out = r.status, r.headers, r.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        out = e.code, e.headers, e.read().decode(errors="replace")
    jar.save(ignore_discard=True, ignore_expires=True)
    os.chmod(JAR, 0o600)
    return out


def _csrf():
    p = JAR + ".csrf"
    return open(p).read().strip() if os.path.exists(p) else ""


def _save_csrf(headers):
    tok = headers.get("x-updated-csrf-token") or headers.get("x-csrf-token")
    if tok:
        p = JAR + ".csrf"
        open(p, "w").write(tok)
        os.chmod(p, 0o600)


if sys.argv[1] == "login":
    if not USER:
        sys.exit("Set UNIFI_USER for login")
    pw = open(sys.argv[2]).read().rstrip("\n")
    body = {"username": USER, "password": pw, "remember": True}
    if len(sys.argv) > 3:
        body["token"] = sys.argv[3]
    st, hd, txt = call("POST", "/api/auth/login", body)
    _save_csrf(hd)
    print(st, txt[:600])
else:
    method, path = sys.argv[1], sys.argv[2]
    body = json.loads(sys.argv[3]) if len(sys.argv) > 3 else None
    st, hd, txt = call(method, path, body)
    _save_csrf(hd)
    print(st)
    print(txt)
