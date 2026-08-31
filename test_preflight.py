#!/usr/bin/env python3
"""
Contract tests for src/preflight.py.

These guard the invariants that a human reviewer caught by hand and that are
cheap for CI to check on a bare runner -- no model weights, no inference engine,
no project dependencies:

  * preflight must NOT report success when the selected path cannot run
    (the exit-code bugs: a down endpoint, a missing llama-cpp-python, a cold
    embedding cache were all reported as advisory and exited 0);
  * a survey run must NOT be blocked by an engine you are not using;
  * bad-model detection must match whole tokens, not substrings
    (llama3.2:11b is not a 1B model);
  * .env backups must not be world-readable;
  * --write-env must write the variables config.py actually reads;
  * preflight must import nothing outside the standard library, because one of
    its jobs is to report that the dependencies are missing.

Like preflight itself, this file is standard-library only, so it runs on a bare
interpreter with no venv and no install step.
"""

import json
import os
import shutil
import socket
import sys
import tempfile
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


def dead_port():
    """A port with nothing listening on it."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


# --------------------------------------------------------------------------
# A stub OpenAI-compatible server, so CI can exercise the SUCCESS path too.
# Without this, a regression that made every check fail would still pass CI.
# --------------------------------------------------------------------------

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

class TestBadModelDetection(unittest.TestCase):
    """A separator list containing "" once reduced this to a substring test."""

    NOT_FLAGGED = [
        "llama3.2:11b-vision-instruct",  # contains "1b"
        "qwen2.5:31b",                   # contains "1b"
        "phi4-mini", "phi3.5", "llama3.1:8b", "mistral:7b", "gemma3:27b",
        "phi4-mini:3.8b-q4_K_M",
    ]
    FLAGGED = [
        "llama3.2:1b", "gemma3:1b-it", "qwen2.5:1.5b-instruct", "qwen2.5:0.5b",
        "deepseek-r1:7b", "qwq:32b", "some-thinking-model", "granite-reasoning:8b",
    ]

    def test_real_models_are_not_flagged(self):
        for name in self.NOT_FLAGGED:
            with self.subTest(model=name):
                self.assertIsNone(preflight.assess_model_name(name))

    def test_unsuitable_models_are_flagged(self):
        for name in self.FLAGGED:
            with self.subTest(model=name):
                self.assertIsNotNone(preflight.assess_model_name(name))


class TestExplicitProviderMustFail(unittest.TestCase):
    """
    The core exit-code contract. Asking for a provider that cannot serve the
    demo must be a blocking failure; merely surveying the machine must not be.
    """

    def test_llama_server_down_is_fail_when_explicit(self):
        env = {"OPENAI_COMPAT_BASE_URL": "http://127.0.0.1:%d" % dead_port()}
        self.assertTrue(has_fail(preflight.check_llama_server(env, False, explicit=True)))

    def test_llama_server_down_does_not_block_a_survey(self):
        env = {"OPENAI_COMPAT_BASE_URL": "http://127.0.0.1:%d" % dead_port()}
        self.assertFalse(has_fail(preflight.check_llama_server(env, False, explicit=False)))

    def test_lmstudio_down_is_fail_when_explicit(self):
        env = {"OPENAI_COMPAT_BASE_URL": "http://127.0.0.1:%d" % dead_port()}
        self.assertTrue(has_fail(preflight.check_lmstudio(env, False, explicit=True)))

    def test_ollama_unreachable_is_fail_when_explicit(self):
        env = {"OLLAMA_BASE_URL": "http://127.0.0.1:%d" % dead_port()}
        self.assertTrue(has_fail(preflight.check_ollama(env, False, explicit=True)))

    def test_ollama_unreachable_does_not_block_a_survey(self):
        env = {"OLLAMA_BASE_URL": "http://127.0.0.1:%d" % dead_port()}
        self.assertFalse(has_fail(preflight.check_ollama(env, False, explicit=False)))


class TestFatalPrerequisites(unittest.TestCase):
    def test_cold_embedding_cache_is_fatal(self):
        # config.py forces TRANSFORMERS_OFFLINE=1 and the demo builds embeddings
        # before it reaches any LLM, so this cannot be advisory.
        with tempfile.TemporaryDirectory() as tmp:
            res = preflight.check_embedding_cache({"SENTENCE_TRANSFORMERS_HOME": tmp})
        self.assertEqual(res.status, FAIL)

    def test_missing_gguf_is_fatal(self):
        res = preflight.check_gguf({"LLAMA_MODEL_PATH": "/nonexistent/model.gguf"})
        self.assertEqual(res.status, FAIL)

    def test_non_gguf_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fake.gguf")
            with open(path, "wb") as fh:
                fh.write(b"NOTGGUF" + b"\0" * (600 * 1024 * 1024))
            res = preflight.check_gguf({"LLAMA_MODEL_PATH": path})
        self.assertEqual(res.status, FAIL)

    def test_no_usable_path_is_fatal(self):
        # Nothing importable, nothing downloaded, nothing listening.
        self.assertIsNotNone(preflight.check_viable_path([]))

    def test_a_working_endpoint_counts_as_a_usable_path(self):
        results = [preflight.Result(OK, "Completion round-trip", "0.1s")]
        self.assertIsNone(preflight.check_viable_path(results))

    def test_llama_cpp_missing_is_fatal_only_for_the_local_path(self):
        import importlib.util
        if importlib.util.find_spec("llama_cpp") is not None:
            self.skipTest("llama-cpp-python is installed in this environment")
        self.assertIn(FAIL, statuses(preflight.check_pydeps(local_required=True),
                                     "llama-cpp-python"))
        self.assertIn(WARN, statuses(preflight.check_pydeps(local_required=False),
                                     "llama-cpp-python"))


class TestRemediationIsActionable(unittest.TestCase):
    """A suggested fix that cannot clear the failure it is attached to is a bug."""

    def test_fix_names_the_model_matching_the_configured_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            for key, spec in preflight.UNGATED_MODELS.items():
                with self.subTest(model=key):
                    res = preflight.check_gguf(
                        {"LLAMA_MODEL_PATH": os.path.join(tmp, spec.filename)})
                    self.assertEqual(res.status, FAIL)
                    self.assertTrue(any(key in c for c in res.fix),
                                    "fix %r should mention %r" % (res.fix, key))

    def test_unknown_path_also_tells_you_to_repoint_env(self):
        res = preflight.check_gguf({"LLAMA_MODEL_PATH": "/tmp/custom.gguf"})
        self.assertTrue(any("--write-env" in c for c in res.fix), res.fix)

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


class TestCustomEndpointUrlIsHonoured(unittest.TestCase):
    """
    An explicitly selected provider must be probed at the URL the demo will
    really use. A port-matching heuristic once discarded any custom port and
    silently probed localhost:<conventional> instead -- so a dead endpoint
    "passed" whenever something unrelated was listening on the default port.
    """

    def test_explicit_run_uses_a_custom_port_verbatim(self):
        env = {"OPENAI_COMPAT_BASE_URL": "http://127.0.0.1:59999"}
        self.assertEqual(preflight.compat_base(env, "llama-server", explicit=True),
                         "http://127.0.0.1:59999")

    def test_explicit_run_uses_a_remote_host_verbatim(self):
        env = {"OPENAI_COMPAT_BASE_URL": "http://10.0.0.5:9999"}
        self.assertEqual(preflight.compat_base(env, "lmstudio", explicit=True),
                         "http://10.0.0.5:9999")

    def test_survey_still_discovers_the_conventional_port(self):
        env = {"OPENAI_COMPAT_BASE_URL": "http://127.0.0.1:8080"}
        self.assertEqual(preflight.compat_base(env, "lmstudio", explicit=False),
                         "http://localhost:1234")

    def test_dead_custom_port_is_fail_not_masked_by_the_default_port(self):
        env = {"OPENAI_COMPAT_BASE_URL": "http://127.0.0.1:%d" % dead_port()}
        results = preflight.check_llama_server(env, False, explicit=True)
        self.assertTrue(has_fail(results),
                        "a dead custom endpoint must fail even if :8080 is up")


class TestOnDiskModelIntegrity(unittest.TestCase):
    """An existing file was accepted on the size floor plus magic bytes alone."""

    def _fake_gguf(self, directory, filename, size):
        path = os.path.join(directory, filename)
        with open(path, "wb") as fh:
            fh.write(b"GGUF")
            fh.truncate(size)
        return path

    def test_wrong_exact_size_for_a_known_model_is_rejected(self):
        spec = preflight.UNGATED_MODELS["phi-4-mini"]
        with tempfile.TemporaryDirectory() as tmp:
            # Valid magic, over the size floor, but not the pinned byte count.
            path = self._fake_gguf(tmp, spec.filename, 600 * 1024 * 1024)
            res = preflight.check_gguf({"LLAMA_MODEL_PATH": path})
        self.assertEqual(res.status, FAIL)
        self.assertIn("size", res.title.lower())

    def test_deep_run_reports_that_the_digest_was_checked(self):
        spec = preflight.UNGATED_MODELS["phi-4-mini"]
        path = "./models/llm/%s" % spec.filename
        if not os.path.exists(path) or os.path.getsize(path) != spec.size_bytes:
            self.skipTest("the real %s is not present here" % spec.filename)
        res = preflight.check_gguf({"LLAMA_MODEL_PATH": path}, deep=True)
        self.assertEqual(res.status, OK)
        self.assertIn("sha256 verified", res.detail)


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


class TestEndpointUrlHandling(unittest.TestCase):
    """
    A hand-rolled rsplit(":") for the port raised ValueError on portless and
    path-bearing URLs, and the probe host was hard-coded to 127.0.0.1 so a
    remote endpoint was never actually contacted.
    """

    ODD_URLS = [
        "http://myhost",                 # no port
        "http://host:1234/v1",           # path
        "http://10.0.0.5:1234",          # remote host
        "https://remote.example.com",    # https, no port
    ]

    def test_unusual_urls_do_not_raise(self):
        for url in self.ODD_URLS:
            with self.subTest(url=url):
                env = {"OPENAI_COMPAT_BASE_URL": url}
                preflight.check_lmstudio(env, False, explicit=True)   # must not raise
                preflight.check_llama_server(env, False, explicit=True)

    def test_a_path_bearing_base_url_is_flagged(self):
        # config.py appends /v1 itself, so a path here becomes /v1/v1 and 404s.
        res = preflight.check_base_url_shape(
            {"OPENAI_COMPAT_BASE_URL": "http://host:8080/v1"})
        self.assertIsNotNone(res)
        self.assertEqual(res.status, WARN)

    def test_a_bare_origin_is_not_flagged(self):
        self.assertIsNone(preflight.check_base_url_shape(
            {"OPENAI_COMPAT_BASE_URL": "http://host:8080"}))


class TestCredentialsAreNotPrinted(unittest.TestCase):
    """
    --one-line output is meant to be pasted into a shared thread, so a token in
    the endpoint URL must never reach Result.detail.
    """

    SECRET = "secrettoken"
    URL = "https://user:%s@remote.example.com:1234/v1?api_key=alsosecret#frag" % SECRET

    def test_redact_url_strips_userinfo_query_and_fragment(self):
        shown = preflight.redact_url(self.URL)
        for leak in (self.SECRET, "alsosecret", "#frag"):
            self.assertNotIn(leak, shown)
        self.assertIn("remote.example.com:1234", shown)

    def test_no_check_output_contains_the_secret(self):
        env = {"OPENAI_COMPAT_BASE_URL": self.URL, "OLLAMA_BASE_URL": self.URL}
        results = []
        results += preflight.check_lmstudio(env, False, explicit=True)
        results += preflight.check_llama_server(env, False, explicit=True)
        results += preflight.check_ollama(env, False, explicit=True)
        shape = preflight.check_base_url_shape(env)
        if shape:
            results.append(shape)
        blob = json.dumps([r.as_dict() for r in results])
        self.assertNotIn(self.SECRET, blob)
        self.assertNotIn("alsosecret", blob)


class TestProbedModelMatchesTheDemo(unittest.TestCase):
    """
    The probe used whatever /v1/models listed first, so preflight could validate
    a different model from the one config.py requests.
    """

    def test_configured_model_wins_over_the_first_served_id(self):
        body = {"data": [{"id": "some-other-model"}, {"id": "phi-4-mini-instruct"}]}
        model, served = preflight.resolve_probe_model(
            {"OPENAI_COMPAT_MODEL": "phi-4-mini-instruct"}, body)
        self.assertEqual(model, "phi-4-mini-instruct")
        self.assertEqual(served, ["some-other-model", "phi-4-mini-instruct"])

    def test_first_served_id_is_used_when_nothing_is_configured(self):
        body = {"data": [{"id": "whatever-is-loaded"}]}
        model, _ = preflight.resolve_probe_model({}, body)
        self.assertEqual(model, "whatever-is-loaded")

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


class TestEnvHandling(unittest.TestCase):
    def test_write_env_sets_the_variables_config_py_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, ".env")
            with open(path, "w") as fh:
                fh.write("EXISTING=1\nOPENAI_COMPAT_BASE_URL=http://old:1\n")
            os.chmod(path, 0o600)
            rc = preflight.do_write_env("openai-compat", None, path=path)
            self.assertEqual(rc, 0)
            written = preflight.load_env(path)
            self.assertEqual(written["OPENAI_COMPAT_BASE_URL"], "http://localhost:8080")
            self.assertIn("OPENAI_COMPAT_MODEL", written)
            self.assertEqual(written["EXISTING"], "1")  # unrelated keys survive
            # A backup of a credential-bearing file must not be group/world readable.
            mode = os.stat(path + ".bak").st_mode & 0o777
            self.assertEqual(mode, 0o600, "backup mode is %o" % mode)

    def test_load_env_strips_inline_comments_and_quotes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, ".env")
            with open(path, "w") as fh:
                fh.write('# a comment\nTOP_K=3  # inline\nQUOTED="hello"\nEMPTYLINE=\n\n')
            env = preflight.load_env(path)
        self.assertEqual(env["TOP_K"], "3")
        self.assertEqual(env["QUOTED"], "hello")

    def test_real_environment_overrides_env_even_when_absent_from_it(self):
        key = "OPENAI_COMPAT_BASE_URL"
        old = os.environ.get(key)
        os.environ[key] = "http://from-environ:9999"
        try:
            self.assertEqual(preflight.resolve_env()[key], "http://from-environ:9999")
        finally:
            if old is None:
                del os.environ[key]
            else:
                os.environ[key] = old


class TestStdlibOnly(unittest.TestCase):
    def test_preflight_imports_nothing_third_party(self):
        """
        preflight must report a missing dependency, so it cannot have any.
        """
        stdlib = set(getattr(sys, "stdlib_module_names", ()))
        if not stdlib:
            self.skipTest("sys.stdlib_module_names needs Python 3.10+")
        allowed = stdlib | {"preflight", "sitecustomize", "_distutils_hack"}
        offenders = sorted(
            name.split(".")[0] for name in list(sys.modules)
            if not name.startswith("_")
            and name.split(".")[0] not in allowed
            and "." not in name)
        self.assertEqual(offenders, [], "third-party modules loaded: %s" % offenders)


if __name__ == "__main__":
    unittest.main(verbosity=2)
