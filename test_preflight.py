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
and are run locally, not by CI. To run this file alone:

    python3 test_preflight.py

To run both suites in one go (test_setup.py is excluded deliberately -- it is a
standalone script, not a unittest module, and cannot be imported by the loader):

    python3 -m unittest discover -p "test_preflight*.py"

Standard-library only, like preflight itself, so it runs on a bare interpreter
with no venv and no install step.
"""

import contextlib
import io
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


@contextlib.contextmanager
def patched(**attrs):
    """Temporarily replace attributes of the preflight module."""
    saved = {name: getattr(preflight, name) for name in attrs}
    for name, value in attrs.items():
        setattr(preflight, name, value)
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(preflight, name, value)


def one_line(results, provider=None):
    """(exit status, printed line) of report_one_line."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = preflight.report_one_line(results, {}, provider)
    return rc, buf.getvalue().strip()


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

    def test_missing_gguf_is_fatal_for_the_explicit_local_path(self):
        res = preflight.check_gguf({"LLAMA_MODEL_PATH": "/nonexistent/model.gguf"},
                                   explicit=True)
        self.assertEqual(res.status, FAIL)

    def test_missing_gguf_does_not_block_a_survey(self):
        # Mirrors test_ollama_unreachable_does_not_block_a_survey: someone on
        # llama-server/Ollama has no reason to have a local GGUF at all, so an
        # unfiltered survey must not fail the roster line over it.
        res = preflight.check_gguf({"LLAMA_MODEL_PATH": "/nonexistent/model.gguf"})
        self.assertEqual(res.status, INFO)

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
                        {"LLAMA_MODEL_PATH": os.path.join(tmp, spec.filename)},
                        explicit=True)
                    self.assertEqual(res.status, FAIL)
                    self.assertTrue(any(key in c for c in res.fix),
                                    "fix %r should mention %r" % (res.fix, key))

    def test_unknown_path_also_tells_you_to_repoint_env(self):
        res = preflight.check_gguf({"LLAMA_MODEL_PATH": "/tmp/custom.gguf"})
        self.assertTrue(any("--write-env" in c for c in res.fix), res.fix)

    def test_no_usable_path_points_at_each_engine_check(self):
        # It used to say `--install ollama --run; --download phi-4-mini`
        # whatever the setup -- wrong for a --no-local llama-server user.
        env = {"OPENAI_COMPAT_BASE_URL": "http://10.1.2.3:9000/",
               "OLLAMA_BASE_URL": "http://localhost:11434"}
        res = preflight.check_viable_path([], env)
        self.assertEqual(res.status, FAIL)
        self.assertEqual(res.title, "No runnable inference path")
        for provider in ("llama-server", "lmstudio", "ollama"):
            self.assertTrue(any("--provider %s" % provider in c for c in res.fix), res.fix)
        # The configured URLs are named, in the detail and the fix comments.
        self.assertIn("http://10.1.2.3:9000", res.detail)
        self.assertIn("http://localhost:11434", res.detail)
        joined = " ".join(res.fix)
        self.assertNotIn("--install", joined)
        self.assertNotIn("--download", joined)

    def test_no_usable_path_download_matches_the_configured_gguf(self):
        # Offered only with the local extra, and it must be the file
        # LLAMA_MODEL_PATH names, or check_gguf stays red after following it.
        results = [preflight.Result(OK, "llama-cpp-python", "in-process GGUF path available")]
        res = preflight.check_viable_path(results, {})
        self.assertIn("python3 src/preflight.py --download phi-3.5-mini", res.fix)

    def test_no_usable_path_one_line_is_runnable(self):
        res = preflight.check_viable_path([], {})
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = preflight.report_one_line([res], {})
        line = buf.getvalue().strip()
        self.assertEqual(rc, 1)
        self.assertEqual(line, "PREFLIGHT FAIL: No runnable inference path -- "
                               "python3 src/preflight.py --provider llama-server; "
                               "python3 src/preflight.py --provider lmstudio; "
                               "python3 src/preflight.py --provider ollama")

    def test_no_usable_path_does_not_print_credentials(self):
        env = {"OPENAI_COMPAT_BASE_URL": "https://user:secrettoken@127.0.0.1:1234?api_key=x"}
        res = preflight.check_viable_path([], env)
        self.assertNotIn("secrettoken", json.dumps(res.as_dict()))


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

    def test_padded_ollama_url_is_trimmed_like_config_py(self):
        # config.py trims whitespace and a trailing slash; check_ollama only
        # trimmed the slash, so a quoted or exported "http://host:11434/ "
        # raised InvalidURL out of preflight while the demo ran fine.
        base = "http://127.0.0.1:%d" % dead_port()
        for url in (base + " ", " " + base, base + "/ "):
            with self.subTest(url=url):
                results = preflight.check_ollama({"OLLAMA_BASE_URL": url},
                                                 False, explicit=True)
                self.assertTrue(any(base in r.detail for r in results), results)

    def test_a_path_bearing_base_url_is_flagged(self):
        # config.py appends /v1 itself, so a path here becomes /v1/v1 and 404s.
        res = preflight.check_base_url_shape(
            {"OPENAI_COMPAT_BASE_URL": "http://host:8080/v1"})
        self.assertIsNotNone(res)
        self.assertEqual(res.status, WARN)

    def test_a_bare_origin_is_not_flagged(self):
        self.assertIsNone(preflight.check_base_url_shape(
            {"OPENAI_COMPAT_BASE_URL": "http://host:8080"}))

    def test_a_path_bearing_ollama_base_url_is_flagged(self):
        # llm_factory appends /v1 to OLLAMA_BASE_URL too, and preflight's own
        # probe becomes /v1/api/tags -- which "answers" with a 404, so the old
        # report blamed an unpulled model instead.
        res = preflight.check_base_url_shape(
            {"OLLAMA_BASE_URL": "http://host:11434/v1"}, "OLLAMA_BASE_URL")
        self.assertIsNotNone(res)
        self.assertEqual(res.status, WARN)
        self.assertEqual(res.title, "OLLAMA_BASE_URL is not a bare origin")
        self.assertEqual(res.fix, ["OLLAMA_BASE_URL=http://host:11434"])
        # The wording must be true for Ollama: it names the /api/tags probe,
        # not the OpenAI-compatible /v1/models one.
        self.assertIn("/api/tags", res.detail)

    def test_ollama_check_reads_only_ollama_base_url(self):
        for env in ({}, {"OLLAMA_BASE_URL": ""},
                    {"OLLAMA_BASE_URL": "http://host:11434/"},
                    {"OPENAI_COMPAT_BASE_URL": "http://host:8080/v1"}):
            with self.subTest(env=env):
                self.assertIsNone(
                    preflight.check_base_url_shape(env, "OLLAMA_BASE_URL"))

    def _shape_titles(self, provider):
        env = {"OPENAI_COMPAT_BASE_URL": "http://localhost:9/v1",
               "OLLAMA_BASE_URL": "http://localhost:9/v1"}
        def no_probe(*_args, **_kwargs):
            return []   # touch no socket at all

        with patched(resolve_env=lambda: dict(env), check_ollama=no_probe,
                     check_llama_server=no_probe, check_lmstudio=no_probe):
            results = preflight.run_checks(provider, False)
        return [r.title for r in results if r.title.endswith("not a bare origin")]

    def test_ollama_base_url_is_checked_only_when_ollama_is(self):
        both = ["OPENAI_COMPAT_BASE_URL is not a bare origin",
                "OLLAMA_BASE_URL is not a bare origin"]
        for provider in (None, "ollama"):
            with self.subTest(provider=provider):
                self.assertEqual(self._shape_titles(provider), both)
        # The OPENAI_COMPAT check keeps running for every provider, as before.
        for provider in ("llama-server", "openai-compat", "lmstudio", "local"):
            with self.subTest(provider=provider):
                self.assertEqual(self._shape_titles(provider), both[:1])


class TestOneLineNamesTheBaseUrlShape(unittest.TestCase):
    """
    A base URL with a path (http://localhost:8080/v1) makes the endpoint checks
    fail under names that point elsewhere, while the shape check that explains
    them is only a WARN. --one-line printed just the first FAIL, so the roster
    line said "No runnable inference path" in a survey, or "Completion returned
    HTTP 404 -- see the full report" for one provider, and never named the URL.
    """

    def shape_line(self, var, origin):
        return "PREFLIGHT FAIL: %s is not a bare origin -- %s=%s" % (var, var, origin)

    def test_survey_names_the_path_instead_of_no_runnable_path(self):
        env = {"OPENAI_COMPAT_BASE_URL": "http://localhost:8080/v1"}
        results = [preflight.Result(OK, "Python 3.11.9"),
                   preflight.Result(OK, "Project dependencies"),
                   preflight.check_base_url_shape(env),
                   preflight.check_viable_path([], env)]
        self.assertEqual(one_line(results),
                         (1, self.shape_line("OPENAI_COMPAT_BASE_URL",
                                             "http://localhost:8080")))

    def test_survey_names_an_ollama_path_too(self):
        env = {"OLLAMA_BASE_URL": "http://localhost:11434/v1"}
        results = [preflight.check_base_url_shape(env, "OLLAMA_BASE_URL"),
                   preflight.check_viable_path([], env)]
        self.assertEqual(one_line(results),
                         (1, self.shape_line("OLLAMA_BASE_URL",
                                             "http://localhost:11434")))

    def test_explicit_provider_names_the_variable_it_reads(self):
        base = "http://127.0.0.1:%d" % dead_port()
        compat = {"OPENAI_COMPAT_BASE_URL": base + "/v1"}
        ollama = {"OLLAMA_BASE_URL": base + "/v1"}
        cases = [("llama-server", compat, preflight.check_llama_server),
                 ("openai-compat", compat, preflight.check_llama_server),
                 ("lmstudio", compat, preflight.check_lmstudio),
                 ("ollama", ollama, preflight.check_ollama)]
        for provider, env, check in cases:
            with self.subTest(provider=provider):
                var = preflight.BASE_URL_VARS[provider]
                results = [preflight.check_base_url_shape(env, var)]
                results += check(env, False, explicit=True)
                self.assertTrue(has_fail(results))
                self.assertEqual(one_line(results, provider),
                                 (1, self.shape_line(var, base)))

    def test_explicit_provider_ignores_the_variable_it_does_not_read(self):
        dead = "http://127.0.0.1:%d" % dead_port()
        env = {"OPENAI_COMPAT_BASE_URL": dead + "/v1", "OLLAMA_BASE_URL": dead}
        results = [preflight.check_base_url_shape(env)]
        results += preflight.check_ollama(env, False, explicit=True)
        rc, line = one_line(results, "ollama")
        self.assertEqual(rc, 1)
        self.assertNotIn("OPENAI_COMPAT_BASE_URL", line)
        first = next(r for r in results if r.status == FAIL)
        self.assertTrue(line.startswith("PREFLIGHT FAIL: %s -- " % first.title), line)

        env = {"OPENAI_COMPAT_BASE_URL": dead, "OLLAMA_BASE_URL": dead + "/v1"}
        results = [preflight.check_base_url_shape(env, "OLLAMA_BASE_URL")]
        results += preflight.check_llama_server(env, False, explicit=True)
        rc, line = one_line(results, "llama-server")
        self.assertEqual(rc, 1)
        self.assertNotIn("OLLAMA_BASE_URL", line)

    def test_prerequisite_failures_are_not_masked(self):
        # They come first, a base URL cannot cause them, and each has its own
        # fix. Real check output where it is cheap to produce, so a renamed
        # title breaks this test instead of silently losing precedence.
        env = {"OPENAI_COMPAT_BASE_URL": "http://localhost:8080/v1",
               "OLLAMA_BASE_URL": "http://localhost:11434/v1"}
        tail = [preflight.check_base_url_shape(env),
                preflight.check_base_url_shape(env, "OLLAMA_BASE_URL"),
                preflight.check_viable_path([], env)]
        with tempfile.TemporaryDirectory() as tmp:
            def fake(name, head, size):
                path = os.path.join(tmp, name)
                with open(path, "wb") as fh:
                    fh.write(head)
                    fh.truncate(size)   # sparse: no real 600 MB write
                return {"LLAMA_MODEL_PATH": path}

            known = preflight.UNGATED_MODELS["phi-4-mini"].filename
            prereqs = [
                # check_python cannot be made to fail on a supported interpreter.
                preflight.Result(FAIL, "Python 3.8.18", "", ["uv venv --python=3.11"]),
                preflight.Result(FAIL, "Project dependencies", "",
                                 ["uv sync", "source .venv/bin/activate"]),
                preflight.check_embedding_cache({"SENTENCE_TRANSFORMERS_HOME": tmp}),
                preflight.check_gguf({"LLAMA_MODEL_PATH": "/nonexistent/m.gguf"},
                                     explicit=True),
                preflight.check_gguf(fake("small.gguf", b"GGUF", 1024)),
                preflight.check_gguf(fake(known, b"GGUF", 600 * 1024 * 1024)),
                preflight.check_gguf(fake("notgguf.gguf", b"NOPE", 600 * 1024 * 1024)),
            ]
            prereqs += [r for r in preflight.check_pydeps(local_required=True)
                        if r.status == FAIL]
        for res in prereqs:
            with self.subTest(title=res.title):
                self.assertEqual(res.status, FAIL)
                for provider in (None, "llama-server", "ollama", "local"):
                    rc, line = one_line([res] + tail, provider)
                    self.assertEqual(rc, 1)
                    self.assertTrue(
                        line.startswith("PREFLIGHT FAIL: %s -- " % res.title), line)

    def test_a_pass_is_unchanged_by_a_shape_warning(self):
        env = {"OLLAMA_BASE_URL": "http://localhost:11434/v1"}
        results = [preflight.Result(OK, "Python 3.11.9"),
                   preflight.check_base_url_shape(env, "OLLAMA_BASE_URL"),
                   preflight.Result(OK, "llama-server responding", "http://localhost:8080"),
                   preflight.Result(OK, "Completion round-trip",
                                    "0.1s, model='local-model', said 'READY'",
                                    provider="openai-compat")]
        rc, line = one_line(results)
        self.assertEqual(rc, 0)
        self.assertTrue(line.startswith("PREFLIGHT PASS: "), line)
        self.assertNotIn("BASE_URL", line)


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
        for var in ("OPENAI_COMPAT_BASE_URL", "OLLAMA_BASE_URL"):
            shape = preflight.check_base_url_shape(env, var)
            if shape:
                results.append(shape)
        blob = json.dumps([r.as_dict() for r in results])
        self.assertNotIn(self.SECRET, blob)
        self.assertNotIn("alsosecret", blob)
        self.assertNotIn("#frag", blob)

    def test_one_line_naming_a_base_url_is_redacted(self):
        # The shape line quotes the URL, and this is the line that is pasted.
        url = "https://user:%s@127.0.0.1:%d/v1?api_key=alsosecret#frag" % (
            self.SECRET, dead_port())
        env = {"OPENAI_COMPAT_BASE_URL": url, "OLLAMA_BASE_URL": url}
        for var in ("OPENAI_COMPAT_BASE_URL", "OLLAMA_BASE_URL"):
            with self.subTest(var=var):
                results = [preflight.check_base_url_shape(env, var),
                           preflight.check_viable_path([], env)]
                _rc, line = one_line(results)
                self.assertIn("%s is not a bare origin" % var, line)
                for leak in (self.SECRET, "alsosecret", "frag"):
                    self.assertNotIn(leak, line)


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


class TestMalformedInputDoesNotCrash(unittest.TestCase):
    """
    preflight must always reach a verdict. A traceback is worse than a FAIL: it
    tells the user nothing and, for --one-line, replaces the roster line they
    were told to paste with a stack trace.
    """

    def test_unparseable_port_is_reported_not_raised(self):
        # urlsplit is lazy -- .port is what raises, so it must be inside the guard.
        self.assertEqual(preflight.redact_url("http://host:bad"), "(unparseable URL)")

    def test_unparseable_base_url_is_reported_not_raised(self):
        res = preflight.check_base_url_shape({"OPENAI_COMPAT_BASE_URL": "http://h:bad"})
        self.assertIsNotNone(res)
        self.assertEqual(res.status, WARN)
        res = preflight.check_base_url_shape({"OLLAMA_BASE_URL": "http://h:bad"},
                                             "OLLAMA_BASE_URL")
        self.assertEqual(res.status, WARN)
        self.assertEqual(res.fix, ["OLLAMA_BASE_URL=http://localhost:11434"])

    def test_non_object_config_json_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            snap = os.path.join(tmp, "models--a--b", "snapshot")
            os.makedirs(snap)
            with open(os.path.join(snap, "config.json"), "w") as fh:
                fh.write('["not", "an", "object"]')
            self.assertIsNone(
                preflight.embedding_dim({"SENTENCE_TRANSFORMERS_HOME": tmp}))

    def test_json_objects_tolerates_every_wrong_shape(self):
        for body in ("plain text", None, 42, [], {"data": "str"},
                     {"data": ["a", 1]}, {"data": [{"id": "ok"}]}):
            with self.subTest(body=body):
                out = preflight.json_objects(body, "data")
                self.assertIsInstance(out, list)
        self.assertEqual(preflight.json_objects({"data": [{"id": "ok"}]}, "data"),
                         [{"id": "ok"}])

    def test_completion_text_returns_none_for_a_null_content(self):
        # A tool-call response legitimately carries content: null.
        for body in ({"choices": [{"message": {"content": None}}]},
                     {"choices": [{"message": {}}]},
                     {"choices": []},
                     {"choices": [{"message": {"content": 5}}]},
                     "plain text", None):
            with self.subTest(body=body):
                self.assertIsNone(preflight.completion_text(body))
        self.assertEqual(
            preflight.completion_text({"choices": [{"message": {"content": "hi"}}]}),
            "hi")


class TestEnvValueParsingMatchesDotenv(unittest.TestCase):
    """
    preflight must resolve the SAME endpoint and model the runtime resolves.
    A naive split on "#" cut values that python-dotenv keeps, so a URL fragment
    or a path containing "#" pointed preflight at a different endpoint.

    python-dotenv cannot be imported here (this module runs before the
    dependencies exist), so its rules are matched rather than reused. The
    expectations below were taken from real dotenv_values() output.
    """

    CASES = [
        ("3  # trailing comment", "3", "comment after whitespace is stripped"),
        ("3\t# tab then comment", "3", "tab counts as whitespace"),
        ("http://h:8080#frag", "http://h:8080#frag", "'#' without whitespace is kept"),
        ("/path/with#hash", "/path/with#hash", "'#' inside a path is kept"),
        ('"a # b"', "a # b", "'#' inside double quotes is kept"),
        ("'a # b'", "a # b", "'#' inside single quotes is kept"),
        ("  spaced  ", "spaced", "surrounding whitespace is trimmed"),
        ("", "", "empty value"),
        ("phi4-mini", "phi4-mini", "ordinary value"),
    ]

    def test_values_parse_the_way_dotenv_parses_them(self):
        for raw, expected, why in self.CASES:
            with self.subTest(reason=why):
                self.assertEqual(preflight.parse_env_value(raw), expected)

    def test_unterminated_quote_drops_the_key(self):
        # dotenv rejects the statement and the runtime never sees the key, so
        # preflight must not invent a value it would then go and validate.
        with self.assertRaises(ValueError):
            preflight.parse_env_value('"unterminated')
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, ".env")
            with open(path, "w") as fh:
                fh.write('GOOD=1\nBAD="unterminated\nALSO=2\n')
            env = preflight.load_env(path)
        self.assertEqual(env.get("GOOD"), "1")
        self.assertEqual(env.get("ALSO"), "2")
        self.assertNotIn("BAD", env)

    def test_a_fragment_bearing_endpoint_survives_into_the_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, ".env")
            with open(path, "w") as fh:
                fh.write("OPENAI_COMPAT_BASE_URL=http://host:8080#frag\n")
            self.assertEqual(preflight.load_env(path)["OPENAI_COMPAT_BASE_URL"],
                             "http://host:8080#frag")


class TestTorchIsAMandatoryDependency(unittest.TestCase):
    """
    rag_poisoning_demo.py imports utils unconditionally and utils.py does
    `import torch` at module level, so torch is required by EVERY provider --
    including the endpoint-only ones. find_spec() does not import, so
    sentence-transformers resolving is not evidence that torch is present.
    """

    def test_torch_is_in_the_checked_set(self):
        import importlib.util
        results = preflight.check_pydeps()
        deps = [r for r in results if r.title == "Project dependencies"][0]
        if importlib.util.find_spec("torch") is None:
            self.assertEqual(deps.status, FAIL)
            self.assertIn("torch", deps.detail)
        else:
            self.assertEqual(deps.status, OK)


class TestNextStepMatchesTheVerifiedPath(unittest.TestCase):
    """
    `python3 src/rag_poisoning_demo.py` with no --infer selects provider=None and
    builds LlamaCpp, so recommending it on an endpoint-only machine is advice
    that fails immediately.
    """

    def test_local_path_gets_the_bare_command(self):
        results = [preflight.Result(OK, "llama-cpp-python"),
                   preflight.Result(OK, "Local GGUF")]
        self.assertEqual(preflight.next_step(results),
                         "python3 src/rag_poisoning_demo.py")

    def test_endpoint_only_path_names_the_provider(self):
        results = [preflight.Result(WARN, "llama-cpp-python"),
                   preflight.Result(OK, "Completion round-trip", "0.1s",
                                    provider="openai-compat")]
        self.assertEqual(preflight.next_step(results),
                         "python3 src/rag_poisoning_demo.py --infer openai-compat")

    def test_ollama_path_names_ollama(self):
        results = [preflight.Result(OK, "Completion round-trip", "0.1s",
                                    provider="ollama")]
        self.assertEqual(preflight.next_step(results),
                         "python3 src/rag_poisoning_demo.py --infer ollama")

    def test_no_verified_path_recommends_nothing(self):
        self.assertIsNone(preflight.next_step([preflight.Result(FAIL, "anything")]))


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
            and not name.startswith("test_")   # the harness itself, under discover
            and name.split(".")[0] not in allowed
            and "." not in name)
        self.assertEqual(offenders, [], "third-party modules loaded: %s" % offenders)


if __name__ == "__main__":
    unittest.main(verbosity=2)
