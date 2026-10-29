"""Run the Neurolock PoP Basecamp tab (neurolock-ui/Main.qml) outside Basecamp.

  python logos/basecamp/dev_host.py                    # interactive window
  python logos/basecamp/dev_host.py --walkthrough DIR  # scripted demo run, one PNG per step

Basecamp gives QML tabs a `logos` object whose callModule(module, method, args)
reaches core modules. Here `logos` is provided by this file and behaves exactly
like our C++ core module (neurolock-core/src/neurolock_pop_plugin.cpp): same two
methods (state, start), same action allow-list, same 127.0.0.1 forwarding to
logos/pop_service.py, same error strings. Main.qml is loaded unmodified.

The walkthrough drives the tab through its own act() function - the function
each button calls - and screenshots the tab after every step.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from PySide6.QtCore import QMetaObject, QObject, QTimer, QUrl, Q_ARG, Slot
from PySide6.QtGui import QGuiApplication
from PySide6.QtQuick import QQuickView

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("NEUROLOCK_POP_PORT", 47200))
BASE = f"http://127.0.0.1:{PORT}"
ALLOWED = {"enrol", "verify", "consent", "delegate", "revoke", "impostor", "replay"}


def request(verb, path):
    try:
        with urllib.request.urlopen(urllib.request.Request(BASE + path, method=verb, data=b"" if verb == "POST" else None,
                                                           headers={"Content-Type": "application/json"}), timeout=5) as r:
            return r.read().decode()
    except urllib.error.HTTPError as e:
        return e.read().decode() or json.dumps({"error": f"Neurolock service unreachable: {e}"})
    except (TimeoutError, OSError) as e:
        if "timed out" in str(e):
            return '{"error":"Neurolock service did not answer (is pop_service.py running?)"}'
        return json.dumps({"error": f"Neurolock service unreachable: {e}"})


class Logos(QObject):
    """Stand-in for Basecamp's `logos` bridge + the neurolock_pop core module."""

    @Slot(str, str, "QVariantList", result=str)
    def callModule(self, module, method, args):
        if module != "neurolock_pop":
            return json.dumps({"error": f"unknown module {module}"})
        if method == "state":
            return request("GET", "/state")
        if method == "start":
            action, app = (list(args) + ["", ""])[:2]
            if action not in ALLOWED:
                return '{"error":"unknown action"}'
            path = "/" + action + (("?" + urllib.parse.urlencode({"app": app})) if app else "")
            return request("POST", path)
        return json.dumps({"error": f"unknown method {method}"})


# (action or None, caption shown in the walkthrough)
STEPS = [
    (None, "Start: Asha has not enrolled; the chain holds nothing about her"),
    ("enrol", "Enrol at the pseudonym party: 2 fingers' veins + pulse + EEG flicker; dedup against everyone; commitment C written to the ledger"),
    ("verify", "Later, verify: veins unlock her secret S on her own device; pulse and SSVEP prove she is live"),
    ("consent", "Verify + consent: she looks at the CONFIRM flicker; the SSVEP decoder reads her answer from the EEG"),
    ("delegate", "Delegate to an AI agent for 'fieldstation-vote': one active agent per human per app (nullifier)"),
    ("impostor", "Attack 1: Bob presents his own fingers claiming to be Asha -> rejected at the vein layer"),
    ("replay", "Attack 2: a recorded EEG answer is replayed against a fresh flicker challenge -> rejected"),
    ("revoke", "Asha revokes the agent's delegation"),
]


def wait_idle(timeout=600):
    t = time.time()
    while time.time() - t < timeout:
        try:
            jobs = json.loads(request("GET", "/state")).get("jobs", [])
        except ValueError:
            jobs = []
        if not jobs or jobs[-1].get("status") != "running":
            return
        time.sleep(1)


def walkthrough(app, view, out):
    os.makedirs(out, exist_ok=True)
    root = view.rootObject()
    shots = []
    for k, (action, caption) in enumerate(STEPS):
        if action:
            QMetaObject.invokeMethod(root, "act", Q_ARG("QVariant", action))
            # let the job run while the tab keeps polling (its own 1 s timer)
            t = time.time()
            while time.time() - t < 2:
                app.processEvents(); time.sleep(0.05)
            while True:
                app.processEvents()
                jobs = json.loads(request("GET", "/state")).get("jobs", [])
                if not jobs or jobs[-1].get("status") != "running":
                    break
                time.sleep(0.2)
        QMetaObject.invokeMethod(root, "refresh")
        for _ in range(30):
            app.processEvents(); time.sleep(0.05)
        f = os.path.join(out, f"step{k:02d}_{action or 'start'}.png")
        view.grabWindow().save(f)
        shots.append({"file": os.path.basename(f), "action": action, "caption": caption})
        print(f"[{k}] {action or 'start'}: saved {f}", flush=True)
    st = json.loads(request("GET", "/state"))
    json.dump({"steps": shots, "final_state": st}, open(os.path.join(out, "walkthrough.json"), "w"), indent=1)
    app.quit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walkthrough", metavar="DIR")
    a = ap.parse_args()
    app = QGuiApplication(sys.argv)
    view = QQuickView()
    bridge = Logos()
    view.rootContext().setContextProperty("logos", bridge)
    view.setResizeMode(QQuickView.SizeRootObjectToView)
    view.setSource(QUrl.fromLocalFile(os.path.join(HERE, "neurolock-ui", "Main.qml")))
    if view.status() == QQuickView.Error:
        for e in view.errors():
            print(e.toString(), file=sys.stderr)
        sys.exit(1)
    view.resize(900, 820)
    view.setTitle("Neurolock PoP (Basecamp tab, dev host)")
    view.show()
    if a.walkthrough:
        QTimer.singleShot(1500, lambda: walkthrough(app, view, a.walkthrough))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
