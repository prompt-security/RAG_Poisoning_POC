#!/usr/bin/env python3
"""
Preflight and configuration utility for the RAG Poisoning demo.

Answers three questions before you waste time on a failing demo:

  1. Is an inference engine installed, running, and actually answering?
  2. Is its context window big enough that the retrieved chunks survive?
  3. If something is missing, what exact command fixes it?

IMPORTANT: this module deliberately uses the standard library ONLY. One of its
jobs is to tell you that the project dependencies are not installed yet, so it
must run before (and without) them. Do not add third-party imports here.

Usage:
    python3 src/preflight.py                      # detect and report everything
    python3 src/preflight.py --provider ollama    # check one provider
    python3 src/preflight.py --deep               # add a live context probe
    python3 src/preflight.py --install ollama     # print/run the install commands
    python3 src/preflight.py --download phi-4-mini
    python3 src/preflight.py --write-env ollama --model phi-4-mini
    python3 src/preflight.py --json
"""

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# --------------------------------------------------------------------------
# Demo requirements. These numbers are measured, not guessed -- see the
# workshop's PREFLIGHT-AND-ENDPOINTS notes.
# --------------------------------------------------------------------------

MIN_CTX = 2048          # below this the retrieved chunks get truncated away
RECOMMENDED_CTX = 4096  # what llm_factory.py asks LlamaCpp for
MIN_PARAMS_B = 2.0      # <=1.5B instruct models comply only 40-60% of the time

OK, WARN, FAIL, INFO = "OK", "WARN", "FAIL", "INFO"

# The demo reads OPENAI_COMPAT_BASE_URL (see config.py). These are only the
# conventional listen ports, used to discover an engine that is already up when
# .env has not been pointed at one yet.
DEFAULT_PORTS = {"llama-server": 8080, "lmstudio": 1234}


def compat_base(env: Dict[str, str], engine: str, explicit: bool = False) -> str:
    """
    Resolve the base URL to probe for an OpenAI-compatible engine.

    OPENAI_COMPAT_BASE_URL is the variable the demo actually reads, so when a
    provider is explicitly selected it is honoured VERBATIM -- any host, any
    port. Guessing a conventional port there would check a different endpoint
    from the one the demo will use, which is the whole bug this tool exists to
    catch (a shared endpoint on a non-standard port is a normal setup).

    Only an unfiltered survey falls back to the conventional port, and only when
    the configured URL clearly belongs to a different engine -- otherwise a
    survey would probe the same address twice and miss the other engine.
    """
    port = DEFAULT_PORTS[engine]
    configured = (env.get("OPENAI_COMPAT_BASE_URL") or "").strip().rstrip("/")
    if configured and explicit:
        return configured
    if configured:
        try:
            if urllib.parse.urlsplit(configured).port == port:
                return configured
        except ValueError:
            pass  # malformed port: fall through to the conventional default
    return "http://localhost:%d" % port


@dataclass
class ModelSpec:
    """A GGUF that needs no HuggingFace login and no click-through licence."""
    label: str
    filename: str
    url: str
    size_gb: float
    params_b: float
    licence: str
    ollama_tag: str
    sha256: str
    revision: str
    size_bytes: int = 0
    key: str = ""


# Ungated on purpose: no HF token, no gated-repo TOS acceptance, MIT licence,
# and a US-headquartered publisher -- which matters for some corporate reviews.
UNGATED_MODELS: Dict[str, ModelSpec] = {
    "phi-3.5-mini": ModelSpec(
        label="Phi-3.5-mini-instruct (Microsoft)",
        filename="Phi-3.5-mini-instruct.Q4_K_M.gguf",
        url="https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/"
            "resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf",
        size_gb=2.23,
        params_b=3.8,
        licence="MIT",
        ollama_tag="phi3.5",
        sha256="e4165e3a71af97f1b4820da61079826d8752a2088e313af0c7d346796c38eff5",
        size_bytes=2393232672,
        revision="6d70da17e749a471ccb62ade694486011a75cda3",
        key="phi-3.5-mini",
    ),
    "phi-4-mini": ModelSpec(
        label="Phi-4-mini-instruct (Microsoft)",
        filename="Phi-4-mini-instruct.Q4_K_M.gguf",
        url="https://huggingface.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF/"
            "resolve/main/microsoft_Phi-4-mini-instruct-Q4_K_M.gguf",
        size_gb=2.32,
        params_b=3.8,
        licence="MIT",
        ollama_tag="phi4-mini",
        sha256="01999f17c39cc3074afae5e9c539bc82d45f2dd7faa3917c66cbef76fce8c0c2",
        size_bytes=2491874688,
        revision="7ff82c2aaa4dde30121698a973765f39be5288c0",
        key="phi-4-mini",
    ),
}

# Models that break this demo, and why. Checked as substrings, lowercased.
BAD_MODEL_HINTS = {
    "1.5b": "too small -- 40-60% compliance, high variance between runs",
    "1b": "too small -- 40-60% compliance, high variance between runs",
    "0.5b": "far too small to follow injected instructions reliably",
    "r1": "a reasoning/thinking model -- the visible chain-of-thought breaks the demo",
    "reasoning": "a reasoning/thinking model -- breaks the demo output format",
    "thinking": "a reasoning/thinking model -- breaks the demo output format",
    "qwq": "a reasoning/thinking model -- breaks the demo output format",
}

INSTALL_COMMANDS = {
    "llama-server": {
        "Darwin": ["brew install llama.cpp"],
        "Linux": ["# see https://github.com/ggml-org/llama.cpp -- or:",
                  "brew install llama.cpp"],
        "Windows": ["winget install llama.cpp"],
    },
    "ollama": {
        "Darwin": ["brew install ollama", "ollama serve  # leave running"],
        "Linux": ["curl -fsSL https://ollama.com/install.sh | sh"],
        "Windows": ["winget install Ollama.Ollama"],
    },
    "lmstudio": {
        "Darwin": ["brew install --cask lm-studio"],
        "Linux": ["# download the AppImage from https://lmstudio.ai"],
        "Windows": ["winget install ElementLabs.LMStudio"],
    },
}


@dataclass
class Result:
    status: str
    title: str
    detail: str = ""
    fix: List[str] = field(default_factory=list)
    # Which --infer value this result belongs to, when it proves a usable path.
    # report() needs it to print a next step that actually works.
    provider: Optional[str] = None

    def as_dict(self):
        return {"status": self.status, "title": self.title,
                "detail": self.detail, "fix": self.fix,
                "provider": self.provider}


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def _colour(status: str, text: str) -> str:
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        return text
    codes = {OK: "32", WARN: "33", FAIL: "31", INFO: "36"}
    return "\033[%sm%s\033[0m" % (codes.get(status, "0"), text)


ICONS = {OK: "PASS", WARN: "WARN", FAIL: "FAIL", INFO: "INFO"}


def parse_env_value(raw: str) -> str:
    """
    Parse one .env value the way python-dotenv does.

    Matters because preflight must resolve the SAME endpoint and model the
    runtime will use. Two rules that a naive split on "#" gets wrong:

      * an unquoted "#" only starts a comment when preceded by whitespace, so
        http://host:8080#frag and /path/with#hash keep their "#";
      * a quoted value is taken verbatim, so "a # b" keeps its "#".

    Not reimplemented: escape sequences inside double quotes and variable
    interpolation. Nothing in this project's .env uses either, and importing
    python-dotenv here is impossible -- this module has to run before the
    dependencies exist.
    """
    raw = raw.strip()
    if raw[:1] in ('"', "'"):
        quote = raw[0]
        closing = raw.find(quote, 1)
        if closing == -1:
            # python-dotenv rejects an unterminated quote and drops the key, so
            # the runtime will not have this value either. Signal "no value"
            # rather than inventing one preflight would then go and validate.
            raise ValueError("unterminated quote in .env value")
        return raw[1:closing]
    for i, char in enumerate(raw):
        if char == "#" and i > 0 and raw[i - 1] in " \t":
            return raw[:i].strip()
    return raw


def load_env(path: str = ".env") -> Dict[str, str]:
    """Minimal .env reader -- python-dotenv may not be installed yet."""
    env: Dict[str, str] = {}
    if not os.path.exists(path):
        return env
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            try:
                env[key.strip()] = parse_env_value(value)
            except ValueError:
                continue   # malformed line: dotenv drops it, so do we
    return env


def http_json(url: str, payload: Optional[dict] = None, timeout: float = 10.0):
    """
    GET or POST JSON. Returns (status_code, parsed_body_or_text, error).

    Sends no credentials, by design: every endpoint this demo supports
    (llama-server, Ollama, LM Studio) is unauthenticated, and llm_factory sends
    a literal placeholder key. Nothing here should ever carry a bearer token --
    if an authenticated endpoint is ever in scope, add it deliberately on BOTH
    sides and handle redirects, which strip nothing by default.
    """
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
            try:
                return resp.status, json.loads(raw), None
            except ValueError:
                return resp.status, raw, None
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(raw), None
        except ValueError:
            return exc.code, raw, None
    except (urllib.error.URLError, socket.timeout, OSError) as exc:
        return None, None, str(getattr(exc, "reason", exc))


def redact_url(url: str) -> str:
    """
    A URL safe to print.

    Strips userinfo, query and fragment. --one-line output is meant to be pasted
    into a shared thread, so a token embedded in the endpoint URL must not travel
    with it. The raw value is still used for probing.
    """
    # urlsplit is lazy: .port and .hostname are what actually raise on a
    # malformed authority, so they must be inside the guard too.
    try:
        parts = urllib.parse.urlsplit(url)
        netloc = parts.hostname or ""
        if parts.port:
            netloc += ":%d" % parts.port
        username = parts.username
    except ValueError:
        return "(unparseable URL)"
    if username:
        netloc = "***@" + netloc
    shown = urllib.parse.urlunsplit((parts.scheme, netloc, parts.path, "", ""))
    if parts.query or parts.fragment:
        shown += " (query redacted)"
    return shown or "(empty URL)"


def json_objects(body, key: str) -> List[dict]:
    """
    The mapping elements of body[key], defensively.

    An endpoint can answer 200 with plain text or an unexpected shape; every
    caller here wants "the list of objects, or nothing".
    """
    if not isinstance(body, dict):
        return []
    items = body.get(key)
    if not isinstance(items, list):
        return []
    return [i for i in items if isinstance(i, dict)]


def resolve_probe_model(env: Dict[str, str], body, fallback: str = "local-model"):
    """
    Which model will the demo actually ask for, and what does the server serve?

    config.py sends OPENAI_COMPAT_MODEL, so that is the name to validate. Probing
    whatever /v1/models happens to list first can pass while the demo 404s on a
    model that is not loaded.
    """
    served = [m["id"] for m in json_objects(body, "data")
              if isinstance(m.get("id"), str) and m["id"]]
    configured = (env.get("OPENAI_COMPAT_MODEL") or "").strip()
    if configured:
        return configured, served
    return (served[0] if served else fallback), served


def check_base_url_shape(env: Dict[str, str]) -> Optional[Result]:
    """
    config.py appends /v1 itself, so a base URL that already carries a path
    becomes /v1/v1 and 404s. This is a documented participant trip-hazard.
    """
    raw = (env.get("OPENAI_COMPAT_BASE_URL") or "").strip()
    if not raw:
        return None
    try:
        parts = urllib.parse.urlsplit(raw)
        parts.port          # lazy: this is the attribute that validates the port
        path = parts.path.rstrip("/")
    except ValueError:
        return Result(
            WARN, "OPENAI_COMPAT_BASE_URL is unparseable",
            "%r is not a usable URL, so the endpoint can never be reached."
            % redact_url(raw),
            ["OPENAI_COMPAT_BASE_URL=http://localhost:8080"])
    if path:
        return Result(
            WARN, "OPENAI_COMPAT_BASE_URL is not a bare origin",
            "%r has a path. config.py appends /v1 itself, so this becomes "
            "%s/v1 and 404s. Use scheme://host:port only."
            % (redact_url(raw), redact_url(raw)),
            ["OPENAI_COMPAT_BASE_URL=%s" % redact_url(
                urllib.parse.urlunsplit(urllib.parse.urlsplit(raw)[:2] + ("", "", "")))])
    return None


def install_fix(engine: str) -> List[str]:
    return INSTALL_COMMANDS.get(engine, {}).get(platform.system(), [])


# Model names are delimited by -, :, _, / and whitespace. Dots are kept INSIDE
# tokens so "1.5b" and "llama3.2" survive splitting.
_NAME_TOKENS = re.compile(r"[^a-z0-9.]+")


def assess_model_name(name: str) -> Optional[str]:
    """
    Return a warning string if this model is known to break the demo.

    Matches whole tokens, not substrings. A previous version also tried an empty
    separator, which collapsed the test to `hint in name` and flagged real models
    such as llama3.2:11b as "too small" because "11b" contains "1b".
    """
    tokens = set(t for t in _NAME_TOKENS.split(name.lower()) if t)
    for hint, why in BAD_MODEL_HINTS.items():
        if hint in tokens:
            return "%r looks like it is %s" % (name, why)
    return None


# --------------------------------------------------------------------------
# Environment checks
# --------------------------------------------------------------------------

def check_python() -> Result:
    major, minor = sys.version_info[:2]
    ver = "%d.%d.%d" % sys.version_info[:3]
    if (major, minor) < (3, 9):
        return Result(FAIL, "Python %s" % ver,
                      "The demo needs Python 3.9+ (setup.sh pins 3.11).",
                      ["uv venv --python=3.11"])
    if (major, minor) >= (3, 13):
        return Result(WARN, "Python %s" % ver,
                      "Newer than the pinned 3.11; the pinned langchain/chromadb "
                      "versions are not tested here.",
                      ["uv venv --python=3.11", "source .venv/bin/activate"])
    return Result(OK, "Python %s" % ver)


def check_pydeps(local_required: bool = False) -> List[Result]:
    """Import-check the project dependencies without importing them for real."""
    import importlib.util
    # torch is not optional: rag_poisoning_demo.py imports utils unconditionally
    # and utils.py does `import torch` at module level, so every provider needs
    # it -- including the endpoint-only paths. find_spec() does not import, so
    # sentence-transformers resolving is not evidence that torch is installed.
    wanted = [
        ("langchain", "langchain"),
        ("langchain_community", "langchain-community"),
        ("langchain_openai", "langchain-openai"),
        ("chromadb", "chromadb"),
        ("sentence_transformers", "sentence-transformers"),
        ("torch", "torch"),
        ("dotenv", "python-dotenv"),
    ]
    results = []
    missing = []
    for module, dist in wanted:
        if importlib.util.find_spec(module) is None:
            missing.append(dist)
    if missing:
        results.append(Result(
            FAIL, "Project dependencies",
            "Not importable: %s" % ", ".join(missing),
            ["source .venv/bin/activate", "uv pip install -r requirements.txt"]))
    else:
        results.append(Result(OK, "Project dependencies",
                              "langchain, chromadb, sentence-transformers present"))

    # llama-cpp-python is only needed for the in-process local path -- but if
    # that IS the selected path, its absence is fatal, not advisory.
    if importlib.util.find_spec("llama_cpp") is None:
        results.append(Result(
            FAIL if local_required else WARN, "llama-cpp-python",
            "Required for the selected in-process GGUF path (--infer cpu/darwin)."
            if local_required else
            "Absent, so --infer cpu/darwin cannot run. Only needed for the "
            "in-process GGUF path; endpoint providers do not use it.",
            ["uv pip install llama-cpp-python"]))
    else:
        results.append(Result(OK, "llama-cpp-python", "in-process GGUF path available"))
    return results


def check_embedding_cache(env: Dict[str, str]) -> Result:
    home = env.get("SENTENCE_TRANSFORMERS_HOME", "./models/embedding")
    model = env.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    parts = model.split("/")
    folder = "models--%s--%s" % (parts[0], parts[1]) if len(parts) > 1 else ""
    path = os.path.join(home, folder)
    if folder and os.path.isdir(path) and os.listdir(path):
        return Result(OK, "Embedding model cached", model)
    return Result(
        FAIL, "Embedding model not cached",
        "%s is not in %s. config.py forces TRANSFORMERS_OFFLINE=1 and the demo "
        "builds embeddings before it ever reaches the LLM, so this fails the run "
        "outright -- with a misleading offline error even on working internet." % (model, home),
        ["./setup.sh --no-local  # pre-downloads the embedding model"])


def embedding_dim(env: Dict[str, str]) -> Optional[int]:
    """Read the embedding width from the cached HF snapshot, without torch."""
    home = env.get("SENTENCE_TRANSFORMERS_HOME", "./models/embedding")
    for root, _dirs, files in os.walk(home):
        if "config.json" in files:
            try:
                with open(os.path.join(root, "config.json"), encoding="utf-8") as fh:
                    blob = json.load(fh)
            except (ValueError, OSError):
                continue
            # A cache entry can hold any JSON, not necessarily an object.
            if isinstance(blob, dict) and isinstance(blob.get("hidden_size"), int):
                return blob["hidden_size"]
    return None


def model_for_path(path: str) -> Optional[ModelSpec]:
    """Which known model does this LLAMA_MODEL_PATH refer to, if any?"""
    base = os.path.basename(path)
    for spec in UNGATED_MODELS.values():
        if spec.filename == base:
            return spec
    return None


def _download_fix(path: str) -> List[str]:
    """
    Remediation for a missing GGUF.

    Downloading a model whose filename does not match LLAMA_MODEL_PATH would
    leave this check failing, so name the matching model where we can and
    otherwise repoint .env as well.
    """
    spec = model_for_path(path)
    if spec is not None:
        return ["python3 src/preflight.py --download %s" % spec.key]
    return ["python3 src/preflight.py --download phi-4-mini",
            "python3 src/preflight.py --write-env local --model phi-4-mini"]


def check_gguf(env: Dict[str, str], deep: bool = False, explicit: bool = False) -> Result:
    path = env.get("LLAMA_MODEL_PATH", "./models/llm/Phi-3.5-mini-instruct.Q4_K_M.gguf")
    if not os.path.exists(path):
        # Mirrors check_ollama/check_llama_server/check_lmstudio: in an
        # unfiltered survey (explicit=False), a path nobody is trying to use
        # not existing is informational, not a blocking problem -- someone on
        # llama-server/Ollama has no reason to have a local GGUF at all. Only
        # an explicitly selected --provider local/llamacpp should FAIL here.
        return Result(FAIL if explicit else INFO, "Local GGUF missing", path,
                      _download_fix(path))
    size_gb = os.path.getsize(path) / (1024 ** 3)
    if size_gb < 0.5:
        return Result(FAIL, "Local GGUF looks truncated",
                      "%s is only %.2f GB -- a failed or partial download." % (path, size_gb),
                      ["rm %s" % path] + _download_fix(path))
    # For a model we ship a digest for, the exact byte length is free to check
    # and catches a substituted or resumed-wrong file that passes the size floor.
    spec = model_for_path(path)
    actual_bytes = os.path.getsize(path)
    if spec is not None and spec.size_bytes and actual_bytes != spec.size_bytes:
        return Result(FAIL, "Local GGUF has unexpected size",
                      "%s is %d bytes; %s is %d. Wrong or partial file."
                      % (path, actual_bytes, spec.label, spec.size_bytes),
                      ["rm %s" % path] + _download_fix(path))

    with open(path, "rb") as fh:
        magic = fh.read(4)
    if magic != b"GGUF":
        return Result(FAIL, "Local model is not a GGUF file",
                      "%s does not start with the GGUF magic bytes. The demo needs a "
                      "GGUF quantisation, not the raw HF weights." % path,
                      ["rm %s" % path] + _download_fix(path))
    # Full integrity verification is opt-in: hashing ~2.3 GB costs ~10 s, too
    # slow to run on every preflight, but worth offering for a file already on
    # disk that --download never validated.
    if deep and spec is not None:
        digest = sha256_file(path)
        if digest != spec.sha256:
            return Result(FAIL, "Local GGUF failed digest check",
                          "%s does not match the pinned sha256 for %s.\n"
                          "         expected %s\n         actual   %s"
                          % (path, spec.label, spec.sha256, digest),
                          ["rm %s" % path] + _download_fix(path))
        return Result(OK, "Local GGUF", "%s (%.2f GB), sha256 verified"
                      % (path, size_gb))
    detail = "%s (%.2f GB)" % (path, size_gb)
    if spec is not None and not deep:
        detail += " -- exact size ok; add --deep to verify sha256"
    return Result(OK, "Local GGUF", detail)


# --------------------------------------------------------------------------
# Endpoint checks
# --------------------------------------------------------------------------

def check_ollama(env: Dict[str, str], deep: bool,
                 explicit: bool = False) -> List[Result]:
    results: List[Result] = []
    base = env.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    # Mirror config.py's default, or preflight would pass while the demo pulls a
    # model nobody checked.
    configured = (env.get("OLLAMA_MODEL") or "").strip()
    model = configured or "phi4-mini"

    # Probe FIRST, matching llama-server and LM Studio: OLLAMA_BASE_URL may point
    # at another machine, where a local binary is beside the point.
    status, body, err = http_json(base + "/api/tags", timeout=4.0)
    if status is None:
        if shutil.which("ollama") is None:
            results.append(Result(
                FAIL if explicit else INFO, "ollama not installed",
                "Nothing is answering at %s and there is no local binary to "
                "start." % redact_url(base),
                install_fix("ollama")))
            return results
        results.append(Result(OK, "ollama installed", shutil.which("ollama")))
        results.append(Result(
            FAIL if explicit else INFO, "ollama daemon not reachable",
            "%s -- %s" % (redact_url(base), err),
            ["ollama serve  # leave this running in another terminal"]))
        return results

    tags = [m["name"] for m in json_objects(body, "models")
            if isinstance(m.get("name"), str) and m["name"]]
    results.append(Result(OK, "ollama daemon up",
                          "%d model(s) pulled" % len(tags)))

    if not configured:
        results.append(Result(
            INFO, "OLLAMA_MODEL not set in .env",
            "config.py falls back to %r, so that is what the demo would use "
            "and what is checked below." % model,
            ["python3 src/preflight.py --write-env ollama --model phi-4-mini"]))

    if not any(t == model or t.startswith(model + ":") for t in tags):
        # An HF repo name (uppercase, or a '/') is not a valid Ollama tag, so
        # telling the user to pull it verbatim would just fail again.
        looks_like_hf_repo = "/" in model or model != model.lower()
        if looks_like_hf_repo:
            fix = ["# %r is a HuggingFace repo name, not an Ollama tag" % model,
                   "python3 src/preflight.py --write-env ollama --model phi-4-mini",
                   "ollama pull phi4-mini"]
            detail = ("%r cannot be pulled -- Ollama tags are lowercase library "
                      "names. Pulled: %s" % (model, ", ".join(tags) or "(none)"))
        else:
            fix = ["ollama pull %s" % model]
            detail = "%r is not in: %s" % (model, ", ".join(tags) or "(none)")
        results.append(Result(FAIL if explicit else WARN,
                              "OLLAMA_MODEL not pulled", detail, fix))
        model_pulled = False
    else:
        model_pulled = True
        results.append(Result(OK, "OLLAMA_MODEL pulled", model))
        bad = assess_model_name(model)
        if bad:
            results.append(Result(WARN, "Model unsuitable for the demo", bad))

    # Context: ollama truncates SILENTLY, which is the single nastiest failure
    # mode here -- the demo just quietly stops working.
    ctx_env = env.get("OLLAMA_CONTEXT_LENGTH")
    ctx_ok = bool(ctx_env) and ctx_env.isdigit() and int(ctx_env) >= MIN_CTX
    if ctx_ok and "OLLAMA_CONTEXT_LENGTH" not in os.environ:
        # No demo code reads this variable -- it configures the `ollama serve`
        # process, so a value that only lives in .env has no effect at all.
        results.append(Result(
            WARN, "OLLAMA_CONTEXT_LENGTH is only set in .env",
            "%s is set to %s in .env, but nothing in the demo reads it -- it "
            "configures the `ollama serve` process. Export it in the shell that "
            "starts the daemon, or the prompt is still truncated silently."
            % ("OLLAMA_CONTEXT_LENGTH", ctx_env),
            ["export OLLAMA_CONTEXT_LENGTH=%s" % ctx_env,
             "export OLLAMA_KEEP_ALIVE=60m",
             "# then restart: ollama serve"]))
    elif ctx_ok:
        results.append(Result(OK, "OLLAMA_CONTEXT_LENGTH", ctx_env))
    else:
        results.append(Result(
            WARN, "OLLAMA_CONTEXT_LENGTH not raised",
            "Ollama truncates the prompt SILENTLY when the context is too small, "
            "so the poisoned chunk can vanish with no error at all. Want >= %d." % RECOMMENDED_CTX,
            ["export OLLAMA_CONTEXT_LENGTH=%d" % RECOMMENDED_CTX,
             "export OLLAMA_KEEP_ALIVE=60m",
             "# then restart: ollama serve"]))

    # Probing a model that is not pulled would just surface a second, noisier
    # copy of the same 404 we already reported above.
    if model_pulled:
        results.extend(probe_chat(base + "/v1", model, deep, explicit,
                                  provider="ollama"))
    return results


def check_llama_server(env: Dict[str, str], deep: bool,
                       explicit: bool = False) -> List[Result]:
    results: List[Result] = []
    base = compat_base(env, "llama-server", explicit)
    # Probe FIRST. A reachable endpoint is sufficient on its own -- it may be a
    # shared or remote server, in which case a local binary is irrelevant.
    status, body, err = http_json(base + "/v1/models", timeout=4.0)
    if status is None:
        if shutil.which("llama-server") is None:
            results.append(Result(
                FAIL if explicit else WARN, "llama-server not installed",
                "Nothing is answering at %s and there is no local binary to "
                "start. It is the most forgiving option for a live room: it "
                "ignores the model field and fails LOUDLY on context overflow."
                % redact_url(base),
                install_fix("llama-server")))
            return results
        results.append(Result(OK, "llama-server installed",
                              shutil.which("llama-server")))
        results.append(Result(
            FAIL if explicit else INFO, "llama-server not running",
            "%s -- %s" % (redact_url(base), err),
            # One line, no continuation: --one-line joins these with "; ", and a
            # trailing backslash would make the pasted command invalid.
            ["llama-server -hf bartowski/microsoft_Phi-4-mini-instruct-GGUF:Q4_K_M "
             "-c %d -np 1 -cb --host 127.0.0.1 --port 8080 -a local-model --jinja"
             % RECOMMENDED_CTX,
             "# -c is DIVIDED by -np, so -c must be >= %d * np" % MIN_CTX]))
        return results

    results.append(Result(OK, "llama-server responding", redact_url(base)))

    # /props is authoritative for the real context size.
    status, props, err = http_json(base + "/props", timeout=4.0)
    n_ctx = None
    if isinstance(props, dict):
        settings = props.get("default_generation_settings")
        if isinstance(settings, dict):
            n_ctx = settings.get("n_ctx")
        if not isinstance(n_ctx, int):
            n_ctx = props.get("n_ctx")
    if isinstance(n_ctx, int):
        if n_ctx < MIN_CTX:
            results.append(Result(
                FAIL if explicit else WARN, "llama-server context too small",
                "n_ctx=%d, need >= %d. Remember -c is DIVIDED by -np." % (n_ctx, MIN_CTX),
                ["# restart with: -c %d -np 1" % RECOMMENDED_CTX]))
        elif n_ctx < RECOMMENDED_CTX:
            results.append(Result(WARN, "llama-server context tight",
                                  "n_ctx=%d, prefer %d" % (n_ctx, RECOMMENDED_CTX)))
        else:
            results.append(Result(OK, "llama-server context", "n_ctx=%d" % n_ctx))

    model, served = resolve_probe_model(env, body)
    if served and model not in served:
        results.append(Result(
            INFO, "Model name differs from the served id",
            "Requesting %r while the server lists %s. llama-server ignores the "
            "model field, so this is harmless here." % (model, ", ".join(served))))
    results.extend(probe_chat(base + "/v1", model, deep, explicit,
                              provider="openai-compat"))
    return results


def check_lmstudio(env: Dict[str, str], deep: bool,
                   explicit: bool = False) -> List[Result]:
    results: List[Result] = []
    base = compat_base(env, "lmstudio", explicit)
    # Probe first: a reachable endpoint needs no local CLI (it may be remote).
    status, body, err = http_json(base + "/v1/models", timeout=4.0)
    if status is None:
        if shutil.which("lms") is None:
            results.append(Result(FAIL if explicit else INFO,
                                  "LM Studio CLI not installed",
                                  "Nothing is answering at %s and there is no "
                                  "local CLI to start it." % redact_url(base),
                                  install_fix("lmstudio")))
            return results
        results.append(Result(OK, "LM Studio CLI installed", shutil.which("lms")))
        results.append(Result(
            FAIL if explicit else INFO, "LM Studio server not running",
            "%s -- %s. Its server is OFF by default and models JIT-load with a "
            "25s+ stall." % (redact_url(base), err),
            ["lms server start --port 1234",
             "lms load phi-4-mini-instruct --context-length %d --parallel 1 --ttl 3600" % RECOMMENDED_CTX]))
        return results

    results.append(Result(OK, "LM Studio server up", redact_url(base)))

    # Unlike llama-server, LM Studio honours the model field, so requesting a
    # model it has not loaded is a real 404 for the demo.
    model, served = resolve_probe_model(env, body)
    if served and model not in served:
        results.append(Result(
            FAIL if explicit else WARN, "OPENAI_COMPAT_MODEL is not loaded",
            "The demo will request %r, but LM Studio serves: %s. LM Studio "
            "honours the model field, so this 404s." % (model, ", ".join(served)),
            ["lms load %s --context-length %d --parallel 1 --ttl 3600"
             % (model, RECOMMENDED_CTX),
             "# or set OPENAI_COMPAT_MODEL to one of the ids above"]))
        return results

    results.extend(probe_chat(base + "/v1", model, deep, explicit,
                              provider="openai-compat"))
    return results


def completion_text(body) -> Optional[str]:
    """
    The assistant text from an OpenAI-shaped completion, or None.

    content is legitimately null for a tool-call response, and the whole body may
    not be an object at all, so every hop is checked rather than assumed.
    """
    choices = json_objects(body, "choices")
    if not choices:
        return None
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    return content if isinstance(content, str) else None


def probe_chat(v1_base: str, model: str, deep: bool,
               explicit: bool = True,
               provider: Optional[str] = None) -> List[Result]:
    """Round-trip a real completion, then optionally probe for silent truncation."""
    import time
    results: List[Result] = []
    started = time.time()
    status, body, err = http_json(
        v1_base + "/chat/completions",
        {"model": model, "temperature": 0,
         "max_tokens": 8,
         "messages": [{"role": "user", "content": "Reply with exactly: READY"}]},
        timeout=90.0)
    elapsed = time.time() - started

    if status is None:
        return [Result(FAIL if explicit else WARN,
                       "Completion round-trip failed", str(err))]
    if status >= 400:
        snippet = json.dumps(body)[:300] if not isinstance(body, str) else body[:300]
        return [Result(FAIL if explicit else WARN,
                       "Completion returned HTTP %d" % status, snippet)]

    text = completion_text(body)
    if text is None:
        return [Result(FAIL if explicit else WARN, "Completion response malformed",
                       "No string content in choices[0].message: %s"
                       % json.dumps(body)[:240])]
    text = text.strip()

    results.append(Result(OK, "Completion round-trip",
                          "%.1fs, model=%r, said %r" % (elapsed, model, text[:40]),
                          provider=provider))
    if elapsed > 20:
        results.append(Result(WARN, "Endpoint is slow",
                              "%.1fs for 8 tokens. A 25-slide live demo will drag." % elapsed))

    if deep:
        results.append(probe_truncation(v1_base, model))
    return results


def probe_truncation(v1_base: str, model: str) -> Result:
    """
    Detect SILENT prompt truncation.

    Puts a codeword at the very START of a deliberately long prompt and asks for
    it back. Engines that trim the front of an overflowing prompt lose the
    codeword and answer confidently without it -- which is exactly how this demo
    dies quietly. A loud HTTP 400 is the GOOD outcome.
    """
    codeword = "PARROT7"
    filler = ("The retrieval corpus contains routine documentation text. " * 120)
    prompt = ("Codeword: %s\n\n%s\n\nWhat was the codeword at the very top? "
              "Answer with the codeword only." % (codeword, filler))
    status, body, err = http_json(
        v1_base + "/chat/completions",
        {"model": model, "temperature": 0, "max_tokens": 16,
         "messages": [{"role": "user", "content": prompt}]},
        timeout=120.0)

    if status is None:
        return Result(WARN, "Truncation probe inconclusive", str(err))
    if status >= 400:
        snippet = json.dumps(body)[:200] if not isinstance(body, str) else body[:200]
        return Result(OK, "Context overflow fails LOUDLY (good)",
                      "HTTP %d: %s -- you will see the problem instead of "
                      "debugging a silently broken demo." % (status, snippet))
    text = completion_text(body)
    if text is None:
        return Result(WARN, "Truncation probe inconclusive",
                      "no string content in the response")

    if codeword in text.upper():
        return Result(OK, "No silent truncation",
                      "The head of a ~1.5k-token prompt survived intact.")
    return Result(WARN, "Possible SILENT truncation",
                  "The codeword at the top of a long prompt did not come back "
                  "(got %r). Either the context is too small and the engine is "
                  "trimming quietly, or this model is too weak to follow the "
                  "instruction. Raise the context and re-run." % text.strip()[:40])


# --------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------

def sha256_file(path: str, chunk: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def do_download(key: str, dest_dir: str = "./models/llm") -> int:
    spec = UNGATED_MODELS.get(key)
    if spec is None:
        print("Unknown model %r. Choose from: %s"
              % (key, ", ".join(UNGATED_MODELS)))
        return 2
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, spec.filename)
    if os.path.exists(dest) and os.path.getsize(dest) > 512 * 1024 * 1024:
        print("Already present: %s (%.2f GB) -- verifying sha256 ..."
              % (dest, os.path.getsize(dest) / (1024 ** 3)))
        if sha256_file(dest) == spec.sha256:
            print("sha256 verified.")
            return 0
        print("Digest does NOT match the pinned value; re-downloading.")
        os.remove(dest)
    # Pin the immutable commit rather than the mutable /main ref, so the bytes
    # cannot change under us between releases.
    url = spec.url.replace("/resolve/main/", "/resolve/%s/" % spec.revision)
    print("Downloading %s" % spec.label)
    print("  licence : %s (no HuggingFace login, no gated-repo TOS)" % spec.licence)
    print("  size    : ~%.2f GB" % spec.size_gb)
    print("  from    : %s" % url)
    print("  to      : %s" % dest)
    print("  sha256  : %s" % spec.sha256)
    if shutil.which("curl") is None:
        print("curl not found; download manually with the URL above.")
        return 1
    # -C - resumes a partial file, so a dropped download is cheap to retry.
    rc = subprocess.call(["curl", "-L", "--fail", "--retry", "3",
                          "-C", "-", "-o", dest, url])
    if rc != 0:
        print("Download failed (curl exit %d). Re-run to resume." % rc)
        return rc

    print("Verifying sha256 ...")
    actual = sha256_file(dest)
    if actual != spec.sha256:
        print("DIGEST MISMATCH -- refusing to keep this file.")
        print("  expected: %s" % spec.sha256)
        print("  actual  : %s" % actual)
        print("This file is parsed by a native library, so it is not left on disk.")
        try:
            os.remove(dest)
        except OSError:
            print("Could not remove %s -- delete it manually." % dest)
        return 1

    print("Done: %s (%.2f GB), sha256 verified."
          % (dest, os.path.getsize(dest) / (1024 ** 3)))
    # A model whose filename does not match LLAMA_MODEL_PATH would still leave
    # the GGUF check failing, so say what to do next.
    print("If .env does not already point here, run:")
    print("    python3 src/preflight.py --write-env local --model %s" % spec.key)
    return 0


def do_install(engine: str, run: bool) -> int:
    cmds = install_fix(engine)
    if not cmds:
        print("No install recipe for %r on %s." % (engine, platform.system()))
        return 2
    print("Install commands for %s on %s:\n" % (engine, platform.system()))
    for cmd in cmds:
        print("    %s" % cmd)
    if not run:
        print("\nRe-run with --run to execute the non-comment lines.")
        return 0
    for cmd in cmds:
        if cmd.strip().startswith("#"):
            continue
        print("\n$ %s" % cmd)
        rc = subprocess.call(cmd, shell=True)
        if rc != 0:
            print("Command failed (exit %d); stopping." % rc)
            return rc
    return 0


def do_write_env(provider: str, model_key: Optional[str], path: str = ".env") -> int:
    """Point .env at a provider/model. Backs the file up first."""
    updates: Dict[str, str] = {}
    if provider == "ollama":
        spec = UNGATED_MODELS.get(model_key or "phi-4-mini")
        if spec is None:
            print("Unknown model %r." % model_key)
            return 2
        updates["OLLAMA_BASE_URL"] = "http://localhost:11434"
        updates["OLLAMA_MODEL"] = spec.ollama_tag
    elif provider in ("local", "llamacpp"):
        spec = UNGATED_MODELS.get(model_key or "phi-4-mini")
        if spec is None:
            print("Unknown model %r." % model_key)
            return 2
        updates["LLAMA_MODEL_PATH"] = "./models/llm/%s" % spec.filename
    elif provider in ("llama-server", "lmstudio", "openai-compat"):
        # config.py reads OPENAI_COMPAT_BASE_URL / OPENAI_COMPAT_MODEL. Writing
        # anything else leaves the demo pointed at its own default.
        port = DEFAULT_PORTS.get(provider, 8080)
        updates["OPENAI_COMPAT_BASE_URL"] = "http://localhost:%d" % port
        # llama-server ignores the model field; LM Studio needs the loaded id.
        updates["OPENAI_COMPAT_MODEL"] = (
            "local-model" if provider != "lmstudio" else "phi-4-mini-instruct")
    else:
        print("Unknown provider %r." % provider)
        return 2

    lines: List[str] = []
    if os.path.exists(path):
        # .env can hold credentials. copyfile() would create the backup at the
        # default umask (commonly 0644), so tighten it to owner-only and create
        # it exclusively rather than inheriting whatever was there before.
        backup = path + ".bak"
        with open(path, "rb") as src:
            payload = src.read()
        fd = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)
        os.chmod(backup, 0o600)
        with open(path, "r", encoding="utf-8") as fh:
            lines = fh.read().splitlines()

    seen = set()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in updates:
            lines[idx] = "%s=%s" % (key, updates[key])
            seen.add(key)
    for key, value in updates.items():
        if key not in seen:
            lines.append("%s=%s" % (key, value))

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines).rstrip("\n") + "\n")

    print("Updated %s (backup at %s.bak):" % (path, path))
    for key, value in updates.items():
        print("    %s=%s" % (key, value))
    return 0


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def next_step(results: List[Result]) -> Optional[str]:
    """
    The demo invocation that the verified path actually supports.

    `python3 src/rag_poisoning_demo.py` with no --infer selects provider=None and
    builds LlamaCpp, so recommending it on an endpoint-only machine is wrong.
    """
    def ok(title):
        return any(r.title == title and r.status == OK for r in results)

    if ok("llama-cpp-python") and ok("Local GGUF"):
        return "python3 src/rag_poisoning_demo.py"
    for res in results:
        if res.title == "Completion round-trip" and res.status == OK and res.provider:
            return "python3 src/rag_poisoning_demo.py --infer %s" % res.provider
    return None


def check_viable_path(results: List[Result]) -> Optional[Result]:
    """
    Generalises the false-success findings: a survey run must not claim success
    unless at least ONE inference path is actually usable end to end. Individual
    engines being down is fine -- having no working path at all is not.
    """
    def ok(title: str) -> bool:
        return any(r.title == title and r.status == OK for r in results)

    local_ok = ok("llama-cpp-python") and ok("Local GGUF")
    endpoint_ok = ok("Completion round-trip")
    if local_ok or endpoint_ok:
        return None
    return Result(
        FAIL, "No runnable inference path",
        "Neither the in-process GGUF path (llama-cpp-python + a local GGUF) nor "
        "any OpenAI-compatible endpoint is usable, so the demo cannot run "
        "whichever --infer you pick.",
        ["python3 src/preflight.py --install ollama --run",
         "python3 src/preflight.py --download phi-4-mini",
         "# or start one: llama-server -hf bartowski/microsoft_Phi-4-mini-instruct-GGUF:Q4_K_M -c 4096 -np 1 --port 8080"])


def resolve_env() -> Dict[str, str]:
    """
    .env as the base, with the real environment winning (python-dotenv order).

    Deliberately does NOT read .keys. That file holds the DeepSeek API key, which
    belongs to a provider preflight does not check, and reading credentials here
    served only a bearer-token path that no supported endpoint needs.
    """
    env = dict(load_env())
    # Overlay the whole environment: the previous version only replaced keys that
    # already existed in .env, so `OPENAI_COMPAT_BASE_URL=... python preflight.py`
    # was silently ignored when .env did not mention it.
    env.update(os.environ)
    return env


def run_checks(provider: Optional[str], deep: bool) -> List[Result]:
    env = resolve_env()
    local = provider in ("local", "llamacpp")

    results = [check_python()]
    results.extend(check_pydeps(local_required=local))
    results.append(check_embedding_cache(env))

    if provider in (None, "local", "llamacpp"):
        results.append(check_gguf(env, deep, explicit=provider in ("local", "llamacpp")))
    shape = check_base_url_shape(env)
    if shape is not None:
        results.append(shape)
    if provider in (None, "ollama"):
        results.extend(check_ollama(env, deep, explicit=provider == "ollama"))
    if provider in (None, "llama-server", "openai-compat"):
        results.extend(check_llama_server(
            env, deep, explicit=provider in ("llama-server", "openai-compat")))
    if provider in (None, "lmstudio"):
        results.extend(check_lmstudio(env, deep, explicit=provider == "lmstudio"))

    if provider is None:
        viable = check_viable_path(results)
        if viable is not None:
            results.append(viable)
    return results


def report(results: List[Result]) -> int:
    print("\nRAG Poisoning demo -- preflight")
    print("=" * 64)
    for res in results:
        print("  [%s] %s" % (_colour(res.status, ICONS[res.status]), res.title))
        if res.detail:
            print("         %s" % res.detail)
        for cmd in res.fix:
            print("         > %s" % cmd)

    fails = sum(1 for r in results if r.status == FAIL)
    warns = sum(1 for r in results if r.status == WARN)
    print("=" * 64)
    if fails:
        print("%d blocking problem(s), %d warning(s)." % (fails, warns))
        print("Fix the FAIL lines above, then re-run.")
        return 1
    command = next_step(results)
    if warns:
        print("No blocking problems, %d warning(s) -- the demo should run." % warns)
        if command:
            print("Run: %s" % command)
        return 0
    if command:
        print("All checks passed. Run: %s" % command)
    else:
        print("All checks passed.")
    return 0


def report_one_line(results: List[Result], env: Dict[str, str]) -> int:
    """
    Emit a single pasteable line for the workshop pre-flight roster.

    Participants paste this into the shared thread 24h ahead so the instructor
    can see the real BYO success rate before the room fills.
    """
    first_fail = next((r for r in results if r.status == FAIL), None)
    if first_fail is not None:
        cmds = [c.split("#", 1)[0].strip() for c in first_fail.fix]
        fix = ("; ".join(c for c in cmds if c)
               or "see the full report: python3 src/preflight.py")
        print("PREFLIGHT FAIL: %s -- %s" % (first_fail.title, fix))
        return 1

    parts = ["python %d.%d.%d" % sys.version_info[:3]]
    parts.append("deps ok")
    dim = embedding_dim(env)
    parts.append("embeddings cached (dim %s)" % dim if dim else "embeddings cached")

    endpoint = next((r for r in results
                     if r.title in ("llama-server responding", "ollama daemon up",
                                    "LM Studio server up")), None)
    fired = next((r for r in results if r.title == "Completion round-trip"), None)
    if endpoint is not None:
        parts.append("endpoint %s reachable" % (endpoint.detail or "ok"))
    gguf = next((r for r in results if r.title == "Local GGUF"), None)
    if endpoint is None and gguf is not None and gguf.status == OK:
        parts.append("local GGUF ok")
    if fired is not None:
        said = fired.detail.split("said ")[-1] if "said " in fired.detail else "ok"
        parts.append("model fired: %s" % said)

    command = next_step(results)
    if command:
        parts.append("run: %s" % command.replace("python3 src/rag_poisoning_demo.py",
                                                 "demo").strip())
    print("PREFLIGHT PASS: " + " | ".join(parts))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preflight and configure the RAG Poisoning demo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Models (all ungated, MIT, no HF login):\n" + "\n".join(
            "  %-14s %-38s ~%.2f GB  ollama: %s"
            % (k, s.label, s.size_gb, s.ollama_tag)
            for k, s in UNGATED_MODELS.items()))
    parser.add_argument("--provider",
                        choices=["local", "llamacpp", "ollama", "llama-server",
                                 "openai-compat", "lmstudio"],
                        help="check only this provider (default: all). "
                             "openai-compat checks OPENAI_COMPAT_BASE_URL, which is "
                             "what --infer openai-compat actually reads.")
    parser.add_argument("--deep", action="store_true",
                        help="add a live probe for silent prompt truncation")
    parser.add_argument("--json", action="store_true",
                        help="emit machine-readable results")
    parser.add_argument("--one-line", action="store_true",
                        help="emit a single pasteable PASS/FAIL line for the "
                             "workshop pre-flight roster")
    parser.add_argument("--download", metavar="MODEL",
                        help="download an ungated GGUF (%s)" % ", ".join(UNGATED_MODELS))
    parser.add_argument("--install", metavar="ENGINE",
                        choices=["llama-server", "ollama", "lmstudio"],
                        help="show the install commands for an inference engine")
    parser.add_argument("--run", action="store_true",
                        help="with --install, actually execute the commands")
    parser.add_argument("--write-env", metavar="PROVIDER",
                        choices=["local", "llamacpp", "ollama", "llama-server",
                                 "openai-compat", "lmstudio"],
                        help="point .env at a provider (backs up the old file 0600)")
    parser.add_argument("--model", metavar="MODEL",
                        help="model key for --write-env (%s)" % ", ".join(UNGATED_MODELS))
    args = parser.parse_args()

    if args.download:
        return do_download(args.download)
    if args.install:
        return do_install(args.install, args.run)
    if args.write_env:
        return do_write_env(args.write_env, args.model)

    results = run_checks(args.provider, args.deep)
    if args.one_line:
        return report_one_line(results, resolve_env())
    if args.json:
        payload = {"results": [r.as_dict() for r in results],
                   "fails": sum(1 for r in results if r.status == FAIL),
                   "warns": sum(1 for r in results if r.status == WARN)}
        print(json.dumps(payload, indent=2))
        return 1 if payload["fails"] else 0
    return report(results)


if __name__ == "__main__":
    sys.exit(main())
