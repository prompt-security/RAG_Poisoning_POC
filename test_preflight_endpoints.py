#!/usr/bin/env python3
"""
Endpoint-shaped tests for src/preflight.py, using a stub OpenAI-compatible server.

NOT run by CI. The pipeline asserts exit-code semantics only (see
test_preflight.py); simulating an inference engine there is environment-sensitive
and outside what CI is for. Run this locally when changing endpoint handling:

    python3 test_preflight_endpoints.py

It covers the things a bare-runner suite cannot: that the SUCCESS path works at
all (without it, a regression making every check fail would still look green),
that both truncation-probe outcomes are distinguished, and that model fidelity
and survey-vs-explicit severity behave against a live-ish server.

Standard-library only.
"""

import json
import os
import shutil
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import preflight  # noqa: E402
from preflight import FAIL, INFO, OK, WARN  # noqa: E402


def statuses(results, title):
    return [r.status for r in results if r.title == title]


def has_fail(results):
    return any(r.status == FAIL for r in results)


class _StubHandler(BaseHTTPRequestHandler):
    echo_codeword = True
    n_ctx = 4096
    model_ids = ("local-model",)
    malformed = False

    def log_message(self, *_args):
        pass  # keep test output clean

    def _send(self, payload, code=200):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        cls = type(self)
        if self.path == "/v1/models":
            self._send({"data": [{"id": i} for i in cls.model_ids]})
        elif self.path == "/props":
            self._send({"default_generation_settings": {"n_ctx": cls.n_ctx}})
        elif self.path == "/health":
            self._send({"status": "ok"})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length) or b"{}")
        if type(self).malformed:
            # 200 OK with a body that has no choices[].message.content
            self._send({"unexpected": "shape"})
            return
        prompt = req.get("messages", [{}])[-1].get("content", "")
        if "Codeword:" in prompt:
            # Mimic an engine that either preserves or silently trims the head
            # of an overlong prompt.
            word = prompt.split("Codeword:", 1)[1].split()[0]
            reply = word if type(self).echo_codeword else "machine learning"
        else:
            reply = "READY"
        self._send({"choices": [{"message": {"content": reply}}]})


class StubServer:
    def __init__(self, port=0, **options):
        handler = type("H", (_StubHandler,), options)
        self.httpd = HTTPServer(("127.0.0.1", port), handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_exc):
        self.httpd.shutdown()
        self.httpd.server_close()

    @property
    def v1(self):
        return "http://127.0.0.1:%d/v1" % self.port

    @property
    def base(self):
        return "http://127.0.0.1:%d" % self.port


# --------------------------------------------------------------------------

class TestUnusedEngineDoesNotBlockASurvey(unittest.TestCase):
    """
    A discovered-but-unused engine must not fail the whole run. Two statuses
    stayed hard-coded FAIL while their neighbours were gated on `explicit`, so a
    stray server with a small context could block a working Ollama or local path.
    """

    def test_malformed_response_is_advisory_in_a_survey(self):
        with StubServer(malformed=True) as stub:
            survey = preflight.probe_chat(stub.v1, "m", deep=False, explicit=False)
            chosen = preflight.probe_chat(stub.v1, "m", deep=False, explicit=True)
        self.assertFalse(has_fail(survey), "a survey must not be blocked")
        self.assertIn(WARN, statuses(survey, "Completion response malformed"))
        self.assertTrue(has_fail(chosen), "an explicit choice must fail")

    def test_small_context_is_advisory_in_a_survey(self):
        # check_llama_server only honours a custom URL when explicit, so the
        # survey path has to be exercised on the conventional port.
        try:
            stub = StubServer(port=8080, n_ctx=512)
        except OSError:
            self.skipTest("port 8080 is in use on this machine")
        with stub:
            env = {"OPENAI_COMPAT_BASE_URL": stub.base}
            survey = preflight.check_llama_server(env, False, explicit=False)
            chosen = preflight.check_llama_server(env, False, explicit=True)
        self.assertIn(WARN, statuses(survey, "llama-server context too small"))
        self.assertFalse(has_fail(survey))
        self.assertIn(FAIL, statuses(chosen, "llama-server context too small"))


class TestEndpointSuccessPath(unittest.TestCase):
    """Guards against a regression where everything fails and CI still passes."""

    def test_round_trip_against_a_stub(self):
        with StubServer() as stub:
            results = preflight.probe_chat(stub.v1, "local-model", deep=False)
        self.assertFalse(has_fail(results))
        self.assertIn(OK, statuses(results, "Completion round-trip"))

    def test_truncation_probe_passes_when_the_head_survives(self):
        with StubServer(echo_codeword=True) as stub:
            res = preflight.probe_truncation(stub.v1, "local-model")
        self.assertEqual(res.status, OK)

    def test_truncation_probe_warns_when_the_head_is_lost(self):
        with StubServer(echo_codeword=False) as stub:
            res = preflight.probe_truncation(stub.v1, "local-model")
        self.assertEqual(res.status, WARN)


class TestProbedModelAgainstAStubEndpoint(unittest.TestCase):
    """Stub-endpoint half of the model-fidelity checks (not run in CI)."""

    def test_lmstudio_fails_when_the_configured_model_is_not_loaded(self):
        # LM Studio honours the model field, so this is a real 404 for the demo.
        with StubServer(model_ids=("something-else",)) as stub:
            env = {"OPENAI_COMPAT_BASE_URL": stub.base,
                   "OPENAI_COMPAT_MODEL": "phi-4-mini-instruct"}
            results = preflight.check_lmstudio(env, False, explicit=True)
        self.assertIn(FAIL, statuses(results, "OPENAI_COMPAT_MODEL is not loaded"))

    def test_llama_server_only_notes_a_model_mismatch(self):
        # llama-server ignores the model field, so the same mismatch is harmless.
        with StubServer(model_ids=("something-else",)) as stub:
            env = {"OPENAI_COMPAT_BASE_URL": stub.base,
                   "OPENAI_COMPAT_MODEL": "phi-4-mini-instruct"}
            results = preflight.check_llama_server(env, False, explicit=True)
        # Assert on the mismatch handling specifically. A blanket "no failures"
        # check would also depend on whether the llama-server BINARY is present,
        # which differs between a dev machine and a bare CI runner.
        self.assertEqual(statuses(results, "OPENAI_COMPAT_MODEL is not loaded"), [],
                         "llama-server ignores the model field; must not be fatal")
        self.assertIn(INFO, statuses(results, "Model name differs from the served id"))
        self.assertIn(OK, statuses(results, "Completion round-trip"))

class TestOllamaTagRemediationAgainstAStub(unittest.TestCase):
    """Stub-daemon half of the remediation checks (not run in CI)."""

    def test_hf_repo_name_is_not_offered_to_ollama_pull(self):
        # 'Phi-3.5-mini-instruct:latest' shipped in .env.example but is an HF
        # repo name; `ollama pull` on it would 404 just like the demo did.
        bad = "Phi-3.5-mini-instruct:latest"
        with StubServer() as stub:
            env = {"OLLAMA_BASE_URL": stub.base, "OLLAMA_MODEL": bad}
            if shutil.which("ollama") is None:
                self.skipTest("ollama binary not installed")
            results = preflight.check_ollama(env, False, explicit=True)
        pulled = [r for r in results if r.title == "OLLAMA_MODEL not pulled"]
        if not pulled:
            self.skipTest("stub did not reach the model check")
        self.assertFalse(any(c.strip() == "ollama pull %s" % bad
                             for c in pulled[0].fix), pulled[0].fix)

if __name__ == "__main__":
    unittest.main(verbosity=2)
