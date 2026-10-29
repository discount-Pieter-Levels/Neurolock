"""demo_web/data.js -> Basecamp-sandbox-friendly assets: PNG files in img/ and a
QML JS library nl_data.js (images replaced by file names). Run from this folder."""
import base64, json, os, io
import numpy as np, cv2
src = os.path.expanduser("~/neurolock_run/neurolock/demo_web/data.js")
D = json.loads(open(src).read().split("=", 1)[1].strip().rstrip(";"))
os.makedirs("img", exist_ok=True)
def save(uri, name, invert=False):
    b = base64.b64decode(uri.split(",", 1)[1]); a = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED)
    if invert: a = 255 - a
    cv2.imwrite(f"img/{name}.png", a); return f"img/{name}.png"
for who, ses in (("asha", "s2"), ("bob", "s2")):
    s = D[who][ses]
    D[who][ses] = {k: save(s[k], f"{who}_{k}", invert=(k == "maxcurv")) for k in ("raw", "roi", "enh", "maxcurv")}
D["asha"].pop("s1", None)
D.pop("match", None); D.pop("dist", None)
for row in D["ssvep"]["grid"]:
    for cell in row: cell.pop("oz", None)
open("nl_data.js", "w").write(".pragma library\nvar NL = " + json.dumps(D) + ";\n")
print("ok", os.path.getsize("nl_data.js") // 1024, "KB", sorted(os.listdir("img")))
