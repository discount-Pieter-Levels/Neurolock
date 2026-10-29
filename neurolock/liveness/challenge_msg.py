"""Liveness challenge over Logos Delivery (or a local socket with the same format).

  verifier -> device   {"type":"challenge", challenge_id, nonce, flicker_sequence, seg_s,
                        issued_ts, issued_block, verifier}
  device   -> verifier {"type":"reply", challenge_id, nonce, result:"pass"|"fail", id_pub,
                        ts, sig}            sig = Ed25519(id_sk, canonical reply body)

NO SIGNALS ARE SENT. The device shows the flicker, records EEG + finger video,
runs the SSVEP and pulse checks locally, and answers only pass/fail, signed with
the identity key that the veins unlocked.

The verifier rejects a reply that:
  - names an unknown challenge_id, or one already answered (replay);
  - arrives after the window (MAX_AGE_S seconds / MAX_AGE_BLOCKS blocks);
  - echoes the wrong nonce;
  - is not signed by an id_pub registered on the ledger, or the signature is bad;
  - says "fail".

Transports (same JSON on the wire):
  LogosDelivery   `logosctl call delivery_module send <topic> json:[bytes]` and
                  `logosctl watch delivery_module --event messageReceived --json`
  LocalBus        127.0.0.1 TCP pub/sub (spec fallback when no delivery node is running)
"""
import json
import os
import queue
import secrets
import socket
import subprocess
import threading
import time

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from . import ssvep

TOPIC = "/neurolock/1/liveness/json"      # /<app>/<version>/<subtopic>/<format> (LIP-23)
MAX_AGE_S = 90
MAX_AGE_BLOCKS = 4


def canon(d):
    return json.dumps(d, sort_keys=True, separators=(",", ":")).encode()


# ================================================================ transports
class LocalBus:
    """Tiny topic pub/sub over 127.0.0.1. One broker (start_broker), many clients."""

    @staticmethod
    def start_broker(port=47100):
        clients, lock = [], threading.Lock()
        srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", port)); srv.listen()

        def serve(c):
            for line in c.makefile():
                with lock:
                    for o in clients:
                        if o is not c:
                            try:
                                o.sendall(line.encode())
                            except OSError:
                                pass

        def accept():
            while True:
                c, _ = srv.accept()
                with lock:
                    clients.append(c)
                threading.Thread(target=serve, args=(c,), daemon=True).start()
        threading.Thread(target=accept, daemon=True).start()
        return srv

    def __init__(self, port=47100):
        self.s = socket.create_connection(("127.0.0.1", port))
        self.q = queue.Queue()
        threading.Thread(target=self._rx, daemon=True).start()
        self.name = "local socket (fallback)"

    def _rx(self):
        for line in self.s.makefile():
            m = json.loads(line)
            self.q.put((m["topic"], m["payload"]))

    def send(self, topic, msg):
        self.s.sendall((json.dumps({"topic": topic, "payload": msg}) + "\n").encode())

    def recv(self, topic, timeout):
        end = time.time() + timeout
        while time.time() < end:
            try:
                t, m = self.q.get(timeout=max(0.01, end - time.time()))
            except queue.Empty:
                return None
            if t == topic:
                return m
        return None


class LogosDelivery:
    """Logos Delivery through the logosctl daemon (one daemon per participant,
    `config_dir` keeps them apart). Assumes delivery_module is loaded, createNode'd,
    started and subscribed to TOPIC (see LOGOS_README)."""

    def __init__(self, config_dir, logosctl="logosctl"):
        self.base = [logosctl, "--config-dir", config_dir]
        self.q = queue.Queue()
        self.p = subprocess.Popen(self.base + ["watch", "delivery_module", "--event", "messageReceived", "--json"],
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        threading.Thread(target=self._rx, daemon=True).start()
        self.name = f"Logos Delivery ({config_dir})"

    @staticmethod
    def _payload(data):
        """messageReceived(messageHash, contentTopic, payload, source, timestamp) -> (topic, bytes)."""
        args = data if isinstance(data, list) else [data.get(k) for k in ("messageHash", "contentTopic", "payload")]
        topic, p = args[1], args[2]
        if isinstance(p, list):
            p = bytes(p)
        elif isinstance(p, str):
            import base64
            try:
                p = base64.b64decode(p, validate=True)
            except Exception:
                p = p.encode()
        return topic, p

    def _rx(self):
        for line in self.p.stdout:
            try:
                ev = json.loads(line)
                topic, p = self._payload(ev.get("data"))
                self.q.put((topic, json.loads(p)))
            except Exception:
                continue

    def send(self, topic, msg):
        payload = "json:" + json.dumps(list(canon(msg)))
        subprocess.run(self.base + ["call", "delivery_module", "send", f"str:{topic}", payload],
                       check=True, capture_output=True, timeout=30)

    def recv(self, topic, timeout):
        return LocalBus.recv(self, topic, timeout)


# ================================================================ roles
class Verifier:
    def __init__(self, transport, ledger=None, block_height=None, K=4, seg_s=ssvep.SEG_S):
        self.t, self.ledger, self.block = transport, ledger, block_height
        self.K, self.seg_s = K, seg_s
        self.issued = {}          # challenge_id -> challenge
        self.answered = set()
        self.name = "verifier-" + secrets.token_hex(3)

    def issue(self):
        ch = ssvep.make_challenge(self.K, self.seg_s)
        msg = {"type": "challenge", "challenge_id": secrets.token_hex(8), "nonce": ch["nonce"],
               "flicker_sequence": ch["freqs"], "seg_s": ch["seg_s"], "issued_ts": time.time(),
               "issued_block": self.block() if self.block else None, "verifier": self.name}
        self.issued[msg["challenge_id"]] = msg
        self.t.send(TOPIC, msg)
        return msg

    def check(self, reply, now=None, now_block=None):
        """-> (accepted, reason)."""
        now = time.time() if now is None else now
        cid = reply.get("challenge_id")
        ch = self.issued.get(cid)
        if ch is None:
            return False, "unknown challenge_id"
        if cid in self.answered:
            return False, "challenge_id already used (replay)"
        if now - ch["issued_ts"] > MAX_AGE_S:
            return False, f"late reply ({now - ch['issued_ts']:.0f} s > {MAX_AGE_S} s)"
        if ch["issued_block"] is not None and now_block is not None and now_block - ch["issued_block"] > MAX_AGE_BLOCKS:
            return False, "late reply (blocks)"
        if reply.get("nonce") != ch["nonce"]:
            return False, "wrong nonce"
        if self.ledger is not None and self.ledger.ids.get(reply.get("id_pub")) is None:
            return False, "id_pub is not a registered human"
        body = {k: v for k, v in reply.items() if k != "sig"}
        try:
            Ed25519PublicKey.from_public_bytes(bytes.fromhex(reply["id_pub"])).verify(bytes.fromhex(reply["sig"]), canon(body))
        except Exception:
            return False, "bad signature"
        self.answered.add(cid)
        if reply.get("result") != "pass":
            return False, "device reported FAIL"
        return True, "ok"

    def wait_reply(self, timeout=60):
        return self.t.recv(TOPIC, timeout)


class Device:
    """User side. `run_checks(challenge) -> bool` runs SSVEP + pulse locally on the
    device's own sensors (here: the simulated responders of the pipeline)."""

    def __init__(self, transport, id_sk, id_pub_hex, run_checks):
        self.t, self.sk, self.pub, self.run = transport, id_sk, id_pub_hex, run_checks

    def answer(self, timeout=60):
        while True:
            ch = self.t.recv(TOPIC, timeout)
            if ch is None:
                return None
            if ch.get("type") == "challenge":
                break
        ok = self.run({"freqs": ch["flicker_sequence"], "seg_s": ch["seg_s"], "nonce": ch["nonce"]})
        body = {"type": "reply", "challenge_id": ch["challenge_id"], "nonce": ch["nonce"],
                "result": "pass" if ok else "fail", "id_pub": self.pub, "ts": time.time()}
        reply = {**body, "sig": self.sk.sign(canon(body)).hex()}
        self.t.send(TOPIC, reply)
        return ch, reply
