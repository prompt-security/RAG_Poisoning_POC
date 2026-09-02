#!/usr/bin/env python3
"""
Run the REAL demo end to end against a stub OpenAI-compatible endpoint.

ci/smoke.py constructs the chain directly with a fake LLM, which deliberately
bypasses llm_factory. This file covers what that misses: the actual
`python3 src/rag_poisoning_demo.py --infer openai-compat` entry point, config
parsing, LLMFactory._create_openai_compat_llm, langchain-openai's HTTP
behaviour, and attack_demo.py's own pirate detector.

WHAT THE STUB IS
  A perfectly compliant model. It returns pirate-speak if and only if the
  poison marker is present in the prompt it was sent. That makes the run fully
  deterministic while keeping the assertion meaningful: the ONLY way the
  poisoned phase can go pirate is if the poisoned document actually travelled
  corpus -> embedding -> Chroma -> retrieval -> the stuff prompt -> the model.
  That path is the entire teaching claim of the demo.

WHAT GREEN HERE DOES *NOT* PROVE
  That a real model complies. The stub always obeys; Phi-4-mini does not. This
  gate proves the machinery runs and the injected text reaches the prompt --
  never that the parrot talks. Do not weaken the marker check to "fix" a red
  run: if the marker stops reaching the prompt, the demo is broken, not the test.

Runs locally too:  .venv/bin/python ci/e2e.py
"""

import json
import os
import re
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Set in src/rag_poisoning_corpus.py's create_poisoned_document(). Verified at
# runtime below, so this gate can never pass by asserting against itself.
POISON_MARKER = "CRITICAL SYSTEM INSTRUCTION"
MODEL = "ci-stub-model"

# attack_demo.py's own indicators. The stub emits these only when poisoned.
PIRATE = "Arrr, matey! Ye be askin' a fine question, hearty."
CLEAN = "Load balancing distributes requests across multiple backend nodes."

TRANSCRIPT = []
FAILURES = []


def check(ok, label, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + label + (("  -- " + detail) if detail else ""))
    if not ok:
        FAILURES.append(label + ((": " + detail) if detail else ""))
    return ok


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _route(self):
        """
        The EXACT path, query stripped, one optional trailing slash allowed.

        Deliberately not a suffix match: `endswith('/chat/completions')` would
        also accept `/v2/chat/completions`, which is precisely the transport
        switch this gate claims to detect. Matching loosely here would let the
        assertion pass while every BYO-endpoint participant broke.
        """
        path = urlsplit(self.path).path
        if len(path) > 1 and path.endswith("/"):
            path = path[:-1]
        return path

    def do_GET(self):
        # langchain-openai may probe /v1/models. Anything else is a transport
        # change we want to hear about.
        if self._route() == "/v1/models":
            return self._json(200, {"object": "list",
                                    "data": [{"id": MODEL, "object": "model"}]})
        TRANSCRIPT.append({"path": self.path, "rejected": True, "method": "GET"})
        return self._json(404, {"error": "unexpected GET %s" % self.path})

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))

        if self._route() != "/v1/chat/completions":
            TRANSCRIPT.append({"path": self.path, "rejected": True, "method": "POST"})
            return self._json(404, {"error": "unexpected POST %s" % self.path})

        try:
            req = json.loads(raw or b"{}")
        except ValueError:
            req = None
        # A non-object body is itself a transport change. Record it as rejected
        # and answer 400, rather than letting `req.get` raise an AttributeError
        # that surfaces as an opaque 500 with no clue why.
        if not isinstance(req, dict):
            TRANSCRIPT.append({"path": self.path, "rejected": True, "method": "POST",
                               "reason": "body was not a JSON object"})
            return self._json(400, {"error": "expected a JSON object body"})

        msgs = [m for m in (req.get("messages") or []) if isinstance(m, dict)]
        prompt = "\n".join(str(m.get("content", "")) for m in msgs)
        poisoned = POISON_MARKER in prompt

        TRANSCRIPT.append({
            "path": self.path,
            "model": req.get("model"),
            "stream": req.get("stream"),
            "roles": [m.get("role") for m in msgs],
            "prompt_chars": len(prompt),
            "poison_marker_present": poisoned,
            "prompt": prompt,
        })

        text = PIRATE if poisoned else CLEAN
        return self._json(200, {
            "id": "chatcmpl-ci",
            "object": "chat.completion",
            "created": 0,
            "model": req.get("model") or MODEL,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })


def main():
    os.chdir(REPO)

    # -- 0. the gate must not be a tautology --------------------------------
    print("\n[0] marker integrity (this gate is testing the real corpus)")
    sys.path.insert(0, os.path.join(REPO, "src"))
    from rag_poisoning_corpus import create_benign_corpus, create_poisoned_document
    check(POISON_MARKER in create_poisoned_document().page_content,
          "the marker the stub keys on is really in create_poisoned_document()")
    check(all(POISON_MARKER not in d.page_content for d in create_benign_corpus()),
          "the marker appears in no benign document")

    # -- 1. stand up the stub on an ephemeral port --------------------------
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print("\n[1] stub endpoint on 127.0.0.1:%d" % port)

    env = dict(os.environ)
    env["OPENAI_COMPAT_BASE_URL"] = "http://127.0.0.1:%d" % port
    env["OPENAI_COMPAT_MODEL"] = MODEL
    # Gate-owned, NOT setdefault: an ambient TOP_K_RETRIEVAL would silently
    # change what these assertions test (or crash config.py if non-numeric).
    env["TOP_K_RETRIEVAL"] = "4"
    env.setdefault("SENTENCE_TRANSFORMERS_HOME", "./models/embedding")
    env.setdefault("TRANSFORMERS_CACHE", "./models/embedding")

    try:
        proc = subprocess.run(
            [sys.executable, "src/rag_poisoning_demo.py", "--infer", "openai-compat"],
            env=env, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        print("::error::e2e: the demo did not finish within 900s")
        return 1
    finally:
        srv.shutdown()

    out = proc.stdout + proc.stderr
    print("\n[2] demo process")
    check(proc.returncode == 0, "the demo exited 0", "exit %d" % proc.returncode)
    check("Traceback (most recent call last)" not in out, "no traceback in demo output")
    if proc.returncode != 0:
        print("\n----- demo output (tail) -----\n" + out[-4000:])

    # -- 3. transport -------------------------------------------------------
    print("\n[3] transport (what langchain-openai actually spoke)")
    rejected = sorted({e["path"] for e in TRANSCRIPT if e.get("rejected")})
    check(not rejected,
          "every request hit exactly /v1/chat/completions or /v1/models",
          ("unexpected: %s -- a transport switch would break every BYO-endpoint "
           "participant" % rejected) if rejected else "")
    calls = [e for e in TRANSCRIPT if not e.get("rejected")]
    check(bool(calls), "the stub received chat completions", "%d call(s)" % len(calls))
    models = sorted({e.get("model") for e in calls})
    check(models == [MODEL], "every request carried OPENAI_COMPAT_MODEL", str(models))
    check(all(not e.get("stream") for e in calls), "no request asked for streaming")

    # -- 4. the claim the demo exists to make -------------------------------
    print("\n[4] the claim the demo exists to make")
    hits = [e for e in calls if e["poison_marker_present"]]
    check(bool(hits),
          "the poisoned document's text reached the model prompt "
          "(corpus -> embedding -> Chroma -> retrieval -> stuff prompt)",
          "%d of %d completions carried the marker" % (len(hits), len(calls)))
    check(len(hits) < len(calls),
          "at least one completion did NOT carry the marker (the clean phase is "
          "really clean, so the gate is discriminating)",
          "%d of %d carried it" % (len(hits), len(calls)))

    # -- 5. what a participant actually sees --------------------------------
    print("\n[5] participant-visible output (attack_demo.py's own detector)")
    m_clean = re.search(r"Clean system - Pirate responses: (\d+)/(\d+)", out)
    m_pois = re.search(r"Poisoned system - Pirate responses: (\d+)/(\d+)", out)
    check(bool(m_clean and m_pois), "the analysis block printed both counts")
    if m_clean and m_pois:
        # Zero / non-zero only. A ratio assertion (4/5, 80%) goes red on a
        # corpus edit and then gets lowered until it means nothing.
        check(int(m_clean.group(1)) == 0,
              "detector reports no pirate output on the clean corpus", m_clean.group(0))
        check(int(m_pois.group(1)) >= 1,
              "detector reports pirate output on the poisoned corpus", m_pois.group(0))
    check("Demo completed successfully" in out, "the demo reached its own success line")

    print("\n" + "=" * 64)
    if FAILURES:
        for f in FAILURES:
            print("::error::e2e: %s" % f)
        print("%d assertion(s) failed." % len(FAILURES))
        return 1
    print("End-to-end gate held: %d completions, %d carried the poison marker."
          % (len(calls), len(hits)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
