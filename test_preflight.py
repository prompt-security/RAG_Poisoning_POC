#!/usr/bin/env python3
"""
Exit-code and semantics tests for src/preflight.py.

SCOPE, deliberately narrow: this file asserts what preflight *concludes* -- which
statuses it assigns, which exit status it returns, what it prints, and how it
parses configuration. It stands up NO fake inference endpoint. The only sockets
it touches are closed local ports, used to represent "nothing is listening".

That keeps the pipeline testing the contract that actually broke in review --
preflight reporting success when the demo cannot run -- rather than simulating an
inference server, which is environment-sensitive and not what CI is for.

Endpoint-shaped tests that need a stub server live in test_preflight_endpoints.py
and are run locally, not by CI.

Standard-library only, like preflight itself, so it runs on a bare interpreter
with no venv and no install step.
"""

import json
import os
import socket
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import preflight  # noqa: E402
from preflight import FAIL, INFO, OK, WARN  # noqa: E402


def statuses(results, title):
    return [r.status for r in results if r.title == title]


def has_fail(results):
    return any(r.status == FAIL for r in results)


def dead_port():
    """A local port with nothing listening on it."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


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


class TestEndpointUrlHandling(unittest.TestCase):
    """
    A hand-rolled rsplit(":") for the port raised ValueError on portless and
    path-bearing URLs, and the probe host was hard-coded to 127.0.0.1 so a
    remote endpoint was never actually contacted.
    """

    def _odd_urls(self):
        port = dead_port()
        return [
            "http://127.0.0.1",                      # no port at all
            "http://127.0.0.1:%d/v1" % port,         # trailing path
            "http://127.0.0.1:%d" % port,            # ordinary host:port
            "https://127.0.0.1",                     # https, no port
            "http://user:tok@127.0.0.1:%d" % port,   # userinfo
        ]

    def test_unusual_urls_do_not_raise(self):
        # A hand-rolled rsplit(":") for the port raised ValueError on the
        # portless and path-bearing forms before this was fixed.
        for url in self._odd_urls():
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
    URL = "https://user:%s@127.0.0.1:1234/v1?api_key=alsosecret#frag" % SECRET

    def test_redact_url_strips_userinfo_query_and_fragment(self):
        shown = preflight.redact_url(self.URL)
        for leak in (self.SECRET, "alsosecret", "#frag"):
            self.assertNotIn(leak, shown)
        self.assertIn("127.0.0.1:1234", shown)

    def test_no_check_output_contains_the_secret(self):
        url = "https://user:%s@127.0.0.1:%d?api_key=alsosecret#frag" % (
            self.SECRET, dead_port())
        env = {"OPENAI_COMPAT_BASE_URL": url, "OLLAMA_BASE_URL": url}
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
        self.assertNotIn("#frag", blob)


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
