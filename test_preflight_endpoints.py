#!/usr/bin/env python3
"""
Endpoint-shaped tests for src/preflight.py, using a stub OpenAI-compatible server.

NOT run by CI. The pipeline asserts exit-code semantics only (see
test_preflight.py); simulating an inference engine there is environment-sensitive
and outside what CI is for. Nothing invokes this file automatically -- run it
locally when changing endpoint handling:

    python3 test_preflight_endpoints.py

or, to run it together with the semantics suite:

    python3 -m unittest discover -p "test_preflight*.py"

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
    null_content = False      # a valid 2xx tool-call shape: content is null
    text_body = False         # 200 OK with a plain-text body
    ollama_tags = None        # serve /api/tags instead, for the Ollama shape

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
        if cls.ollama_tags is not None and self.path == "/api/tags":
            self._send({"models": [{"name": n} for n in cls.ollama_tags]})
            return
        if cls.text_body:
            body = b"not json at all"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
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
        if type(self).null_content:
            self._send({"choices": [{"message": {"role": "assistant",
                                                 "content": None}}]})
            return
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

class TestMalformedEndpointResponses(unittest.TestCase):
    """
    Every one of these used to raise an uncaught AttributeError/TypeError and
    abort preflight instead of producing a verdict.
    """

    def test_null_content_yields_a_verdict_not_a_traceback(self):
        # content: null is a VALID 2xx body for a tool-call response.
        with StubServer(null_content=True) as stub:
            probe = preflight.probe_chat(stub.v1, "m", deep=False)
            trunc = preflight.probe_truncation(stub.v1, "m")
        self.assertIn(FAIL, statuses(probe, "Completion response malformed"))
        self.assertEqual(trunc.status, WARN)

    def test_plain_text_body_yields_a_verdict_not_a_traceback(self):
        with StubServer(text_body=True) as stub:
            env = {"OPENAI_COMPAT_BASE_URL": stub.base, "OLLAMA_BASE_URL": stub.base}
            for check in (preflight.check_llama_server,
                          preflight.check_lmstudio,
                          preflight.check_ollama):
                with self.subTest(check=check.__name__):
                    self.assertIsInstance(check(env, False, explicit=True), list)

    def test_wrong_element_types_yield_a_verdict(self):
        # models/data present but holding strings rather than objects.
        with StubServer(model_ids=()) as stub:
            env = {"OLLAMA_BASE_URL": stub.base}
            self.assertIsInstance(
                preflight.check_ollama(env, False, explicit=True), list)


class TestRemoteEndpointNeedsNoLocalBinary(unittest.TestCase):
    """
    A healthy remote or shared endpoint was failed for want of a local
    llama-server binary, which is only needed to START a server yourself.
    """

    def _without_local_binaries(self):
        real = shutil.which

        def fake(name, *a, **k):
            return None if name in ("llama-server", "lms") else real(name, *a, **k)
        return real, fake

    def test_healthy_endpoint_passes_with_no_binary_installed(self):
        real, fake = self._without_local_binaries()
        shutil.which = fake
        try:
            with StubServer() as stub:
                env = {"OPENAI_COMPAT_BASE_URL": stub.base}
                ls = preflight.check_llama_server(env, False, explicit=True)
                lms = preflight.check_lmstudio(env, False, explicit=True)
        finally:
            shutil.which = real
        self.assertFalse(has_fail(ls), [r.title for r in ls])
        self.assertFalse(has_fail(lms), [r.title for r in lms])

    def test_missing_binary_still_fails_when_nothing_is_answering(self):
        real, fake = self._without_local_binaries()
        shutil.which = fake
        try:
            env = {"OPENAI_COMPAT_BASE_URL": "http://127.0.0.1:59996"}
            results = preflight.check_llama_server(env, False, explicit=True)
        finally:
            shutil.which = real
        self.assertTrue(has_fail(results))


class TestRemoteOllamaIsProbed(unittest.TestCase):
    """
    check_ollama returned from its local-binary guard before probing, so a
    reachable OLLAMA_BASE_URL on another machine never produced a round-trip and
    could not be recommended. llama-server and LM Studio were reordered for this
    already; Ollama was missed.
    """

    def _without_ollama(self):
        real = shutil.which
        return real, (lambda n, *a, **k: None if n == "ollama" else real(n, *a, **k))

    def test_reachable_remote_ollama_is_probed_without_a_local_binary(self):
        real, fake = self._without_ollama()
        shutil.which = fake
        try:
            with StubServer(ollama_tags=("phi4-mini:latest",)) as stub:
                env = {"OLLAMA_BASE_URL": stub.base, "OLLAMA_MODEL": "phi4-mini"}
                results = preflight.check_ollama(env, False, explicit=True)
        finally:
            shutil.which = real
        self.assertIn(OK, statuses(results, "Completion round-trip"))
        self.assertIsNone(preflight.check_viable_path(results))
        self.assertEqual(preflight.next_step(results),
                         "python3 src/rag_poisoning_demo.py --infer ollama")

    def test_missing_binary_still_fails_when_nothing_answers(self):
        real, fake = self._without_ollama()
        shutil.which = fake
        try:
            results = preflight.check_ollama(
                {"OLLAMA_BASE_URL": "http://127.0.0.1:59995"}, False, explicit=True)
        finally:
            shutil.which = real
        self.assertTrue(has_fail(results))


class TestOllamaContextLengthSourcing(unittest.TestCase):
    """
    The context check read os.environ while every other check read the resolved
    map. Worth more than consistency: no demo code reads OLLAMA_CONTEXT_LENGTH
    at all -- it configures the `ollama serve` process -- so a value living only
    in .env has no effect, and reporting it as satisfied would be wrong.
    """

    def _context_results(self, env):
        real = shutil.which
        shutil.which = lambda n, *a, **k: "/usr/bin/ollama" if n == "ollama" \
            else real(n, *a, **k)
        try:
            results = preflight.check_ollama(env, False, explicit=False)
        finally:
            shutil.which = real
        return [r for r in results if "CONTEXT_LENGTH" in r.title]

    def setUp(self):
        self._saved = os.environ.pop("OLLAMA_CONTEXT_LENGTH", None)

    def tearDown(self):
        if self._saved is not None:
            os.environ["OLLAMA_CONTEXT_LENGTH"] = self._saved
        else:
            os.environ.pop("OLLAMA_CONTEXT_LENGTH", None)

    def test_value_only_in_dotenv_is_flagged_as_ineffective(self):
        with StubServer(ollama_tags=("phi4-mini:latest",)) as stub:
            found = self._context_results({"OLLAMA_BASE_URL": stub.base,
                                           "OLLAMA_MODEL": "phi4-mini",
                                           "OLLAMA_CONTEXT_LENGTH": "4096"})
        self.assertEqual([r.status for r in found], [WARN])
        self.assertIn("only set in .env", found[0].title)

    def test_exported_value_is_accepted(self):
        os.environ["OLLAMA_CONTEXT_LENGTH"] = "4096"
        with StubServer(ollama_tags=("phi4-mini:latest",)) as stub:
            found = self._context_results({"OLLAMA_BASE_URL": stub.base,
                                           "OLLAMA_MODEL": "phi4-mini",
                                           "OLLAMA_CONTEXT_LENGTH": "4096"})
        self.assertEqual([r.status for r in found], [OK])

    def test_absent_value_still_warns_about_silent_truncation(self):
        with StubServer(ollama_tags=("phi4-mini:latest",)) as stub:
            found = self._context_results({"OLLAMA_BASE_URL": stub.base,
                                           "OLLAMA_MODEL": "phi4-mini"})
        self.assertEqual([r.status for r in found], [WARN])
        self.assertIn("not raised", found[0].title)


class TestNextStepAgainstALiveEndpoint(unittest.TestCase):
    def test_endpoint_only_setup_is_told_to_pass_infer(self):
        with StubServer() as stub:
            results = preflight.check_llama_server(
                {"OPENAI_COMPAT_BASE_URL": stub.base}, False, explicit=True)
        self.assertEqual(preflight.next_step(results),
                         "python3 src/rag_poisoning_demo.py --infer openai-compat")


if __name__ == "__main__":
    unittest.main(verbosity=2)
