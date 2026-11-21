"""Log into Owlet's Firebase and look for the account's cameras (read-only GETs).
Usage: python3 -I find_cameras.py <email-file> <password-file>"""
import json, sys, urllib.request, urllib.error

KEY = "AIzaSyCsDZ8kWxQuLJAMVnmEhEkayH1TSxKXfGA"
HDR = {"X-Android-Package": "com.owletcare.owletcare",
       "X-Android-Cert": "2A3BC26DB0B8B0792DBE28E6FFDC2598F9B12B74",
       "Content-Type": "application/json", "User-Agent": "OwletCare/Android"}

def call(url, body=None, token=None):
    h = dict(HDR)
    if token:
        h["Authorization"] = token if token.startswith("Bearer") else f"Bearer {token}"
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body else None, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, r.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")
    except Exception as e:
        return 0, str(e)

email = open(sys.argv[1]).read().strip()
pw = open(sys.argv[2]).read().rstrip("\n")
st, txt = call(f"https://www.googleapis.com/identitytoolkit/v3/relyingparty/verifyPassword?key={KEY}",
               {"email": email, "password": pw, "returnSecureToken": True})
d = json.loads(txt)
tok, uid = d["idToken"], d["localId"]
print("login", st)
for url in [
    "https://accounts.owletdata.com/v2/accounts",
    f"https://accounts.owletdata.com/v2/accounts/{uid}",
    f"https://accounts.owletdata.com/v2/accounts/{uid}/devices",
    f"https://accounts.owletdata.com/v1/accounts/{uid}",
    f"https://owletcare-prod.firebaseio.com/users/{uid}.json?auth={tok}&shallow=true",
    f"https://owletcare-prod.firebaseio.com/accounts/{uid}.json?auth={tok}&shallow=true",
]:
    st, txt = call(url, token=None if "firebaseio" in url else tok)
    shown = url.replace(tok, "<token>").replace(uid, "<uid>")
    print(f"--- {st} {shown}\n{txt.replace(uid, '<uid>')[:1500]}")
