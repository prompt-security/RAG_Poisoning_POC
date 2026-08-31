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
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import urllib.error
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

    def as_dict(self):
        return {"status": self.status, "title": self.title,
                "detail": self.detail, "fix": self.fix}


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def _colour(status: str, text: str) -> str:
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        return text
    codes = {OK: "32", WARN: "33", FAIL: "31", INFO: "36"}
    return "\033[%sm%s\033[0m" % (codes.get(status, "0"), text)


ICONS = {OK: "PASS", WARN: "WARN", FAIL: "FAIL", INFO: "INFO"}


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
            # strip inline comments, then quotes
            value = value.split("#", 1)[0].strip().strip('"').strip("'")
            env[key.strip()] = value
    return env


def http_json(url: str, payload: Optional[dict] = None, timeout: float = 10.0):
    """GET or POST JSON. Returns (status_code, parsed_body_or_text, error)."""
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


def port_open(host: str, port: int, timeout: float = 1.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def install_fix(engine: str) -> List[str]:
    return INSTALL_COMMANDS.get(engine, {}).get(platform.system(), [])


def assess_model_name(name: str) -> Optional[str]:
    """Return a warning string if this model is known to break the demo."""
    low = name.lower()
    for hint, why in BAD_MODEL_HINTS.items():
        # match on token boundaries so "31b" doesn't trip the "1b" rule
        for sep in ("-", ":", "_", ".", " ", ""):
            if (sep + hint) in low or low.startswith(hint):
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


def check_pydeps() -> List[Result]:
    """Import-check the project dependencies without importing them for real."""
    import importlib.util
    wanted = [
        ("langchain", "langchain"),
        ("langchain_community", "langchain-community"),
        ("langchain_openai", "langchain-openai"),
        ("chromadb", "chromadb"),
        ("sentence_transformers", "sentence-transformers"),
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

    # llama-cpp-python is only needed for the in-process local path.
    if importlib.util.find_spec("llama_cpp") is None:
        results.append(Result(
            WARN, "llama-cpp-python",
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
        WARN, "Embedding model not cached", 
        "%s is not in %s. config.py forces TRANSFORMERS_OFFLINE=1, so a cold "
        "cache raises a misleading offline error even with working internet." % (model, home),
        ["./setup.sh --no-local  # pre-downloads the embedding model"])


def embedding_dim(env: Dict[str, str]) -> Optional[int]:
    """Read the embedding width from the cached HF snapshot, without torch."""
    home = env.get("SENTENCE_TRANSFORMERS_HOME", "./models/embedding")
    for root, _dirs, files in os.walk(home):
        if "config.json" in files:
            try:
                with open(os.path.join(root, "config.json"), encoding="utf-8") as fh:
                    dim = json.load(fh).get("hidden_size")
                if isinstance(dim, int):
                    return dim
            except (ValueError, OSError):
                continue
    return None


def check_gguf(env: Dict[str, str]) -> Result:
    path = env.get("LLAMA_MODEL_PATH", "./models/llm/Phi-3.5-mini-instruct.Q4_K_M.gguf")
    if not os.path.exists(path):
        return Result(FAIL, "Local GGUF missing", path,
                      ["python3 src/preflight.py --download phi-4-mini"])
    size_gb = os.path.getsize(path) / (1024 ** 3)
    if size_gb < 0.5:
        return Result(FAIL, "Local GGUF looks truncated",
                      "%s is only %.2f GB -- a failed or partial download." % (path, size_gb),
                      ["rm %s" % path,
                       "python3 src/preflight.py --download phi-4-mini"])
    with open(path, "rb") as fh:
        magic = fh.read(4)
    if magic != b"GGUF":
        return Result(FAIL, "Local model is not a GGUF file",
                      "%s does not start with the GGUF magic bytes. The demo needs a "
                      "GGUF quantisation, not the raw HF weights." % path,
                      ["rm %s" % path,
                       "python3 src/preflight.py --download phi-4-mini"])
    return Result(OK, "Local GGUF", "%s (%.2f GB)" % (path, size_gb))


# --------------------------------------------------------------------------
# Endpoint checks
# --------------------------------------------------------------------------

def check_ollama(env: Dict[str, str], deep: bool) -> List[Result]:
    results: List[Result] = []
    base = env.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = env.get("OLLAMA_MODEL", "")

    if shutil.which("ollama") is None:
        results.append(Result(FAIL, "ollama not installed", "",
                              install_fix("ollama")))
        return results
    results.append(Result(OK, "ollama installed", shutil.which("ollama")))

    status, body, err = http_json(base + "/api/tags", timeout=4.0)
    if status is None:
        results.append(Result(
            FAIL, "ollama daemon not reachable",
            "%s -- %s" % (base, err),
            ["ollama serve  # leave this running in another terminal"]))
        return results

    tags = [m.get("name", "") for m in (body or {}).get("models", [])]
    results.append(Result(OK, "ollama daemon up",
                          "%d model(s) pulled" % len(tags)))

    if not model:
        results.append(Result(WARN, "OLLAMA_MODEL not set in .env", "",
                              ["python3 src/preflight.py --write-env ollama --model phi-4-mini"]))
    elif not any(t == model or t.startswith(model + ":") for t in tags):
        results.append(Result(
            FAIL, "OLLAMA_MODEL not pulled",
            "%r is not in: %s" % (model, ", ".join(tags) or "(none)"),
            ["ollama pull %s" % model]))
    else:
        results.append(Result(OK, "OLLAMA_MODEL pulled", model))
        bad = assess_model_name(model)
        if bad:
            results.append(Result(WARN, "Model unsuitable for the demo", bad))

    # Context: ollama truncates SILENTLY, which is the single nastiest failure
    # mode here -- the demo just quietly stops working.
    ctx_env = os.environ.get("OLLAMA_CONTEXT_LENGTH")
    if ctx_env and ctx_env.isdigit() and int(ctx_env) >= MIN_CTX:
        results.append(Result(OK, "OLLAMA_CONTEXT_LENGTH", ctx_env))
    else:
        results.append(Result(
            WARN, "OLLAMA_CONTEXT_LENGTH not raised",
            "Ollama truncates the prompt SILENTLY when the context is too small, "
            "so the poisoned chunk can vanish with no error at all. Want >= %d." % RECOMMENDED_CTX,
            ["export OLLAMA_CONTEXT_LENGTH=%d" % RECOMMENDED_CTX,
             "export OLLAMA_KEEP_ALIVE=60m",
             "# then restart: ollama serve"]))

    if model:
        results.extend(probe_chat(base + "/v1", model, deep))
    return results


def check_llama_server(env: Dict[str, str], deep: bool) -> List[Result]:
    results: List[Result] = []
    base = env.get("LLAMA_SERVER_BASE_URL", "http://localhost:8080").rstrip("/")

    if shutil.which("llama-server") is None:
        results.append(Result(WARN, "llama-server not installed",
                              "The most forgiving option for a live room: it ignores "
                              "the model field and fails LOUDLY on context overflow.",
                              install_fix("llama-server")))
    else:
        results.append(Result(OK, "llama-server installed", shutil.which("llama-server")))

    status, body, err = http_json(base + "/v1/models", timeout=4.0)
    if status is None:
        spec = UNGATED_MODELS["phi-4-mini"]
        results.append(Result(
            INFO, "llama-server not running", "%s -- %s" % (base, err),
            ["llama-server -hf bartowski/microsoft_Phi-4-mini-instruct-GGUF:Q4_K_M \\",
             "    -c %d -np 1 -cb --host 127.0.0.1 --port 8080 -a local-model --jinja" % RECOMMENDED_CTX,
             "# -c is DIVIDED by -np, so -c must be >= %d * np" % MIN_CTX]))
        return results

    results.append(Result(OK, "llama-server responding", base))

    # /props is authoritative for the real context size.
    status, props, err = http_json(base + "/props", timeout=4.0)
    n_ctx = None
    if isinstance(props, dict):
        n_ctx = (props.get("default_generation_settings", {}) or {}).get("n_ctx")
        if n_ctx is None:
            n_ctx = props.get("n_ctx")
    if isinstance(n_ctx, int):
        if n_ctx < MIN_CTX:
            results.append(Result(
                FAIL, "llama-server context too small",
                "n_ctx=%d, need >= %d. Remember -c is DIVIDED by -np." % (n_ctx, MIN_CTX),
                ["# restart with: -c %d -np 1" % RECOMMENDED_CTX]))
        elif n_ctx < RECOMMENDED_CTX:
            results.append(Result(WARN, "llama-server context tight",
                                  "n_ctx=%d, prefer %d" % (n_ctx, RECOMMENDED_CTX)))
        else:
            results.append(Result(OK, "llama-server context", "n_ctx=%d" % n_ctx))

    served = "local-model"
    if isinstance(body, dict) and body.get("data"):
        served = body["data"][0].get("id", served)
    results.extend(probe_chat(base + "/v1", served, deep))
    return results


def check_lmstudio(env: Dict[str, str], deep: bool) -> List[Result]:
    results: List[Result] = []
    base = env.get("LMSTUDIO_BASE_URL", "http://localhost:1234").rstrip("/")
    if shutil.which("lms") is None:
        results.append(Result(INFO, "LM Studio CLI not installed", "",
                              install_fix("lmstudio")))
    else:
        results.append(Result(OK, "LM Studio CLI installed", shutil.which("lms")))

    if not port_open("127.0.0.1", int(base.rsplit(":", 1)[-1])):
        results.append(Result(
            INFO, "LM Studio server not running",
            "Its server is OFF by default and models JIT-load with a 25s+ stall.",
            ["lms server start --port 1234",
             "lms load phi-4-mini-instruct --context-length %d --parallel 1 --ttl 3600" % RECOMMENDED_CTX]))
        return results

    results.append(Result(OK, "LM Studio server up", base))
    status, body, err = http_json(base + "/v1/models", timeout=4.0)
    served = None
    if isinstance(body, dict) and body.get("data"):
        served = body["data"][0].get("id")
    if served:
        results.extend(probe_chat(base + "/v1", served, deep))
    return results


def probe_chat(v1_base: str, model: str, deep: bool) -> List[Result]:
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
        return [Result(FAIL, "Completion round-trip failed", str(err))]
    if status >= 400:
        snippet = json.dumps(body)[:300] if not isinstance(body, str) else body[:300]
        return [Result(FAIL, "Completion returned HTTP %d" % status, snippet)]

    text = ""
    try:
        text = body["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError):
        return [Result(FAIL, "Completion response malformed",
                       json.dumps(body)[:300])]

    results.append(Result(OK, "Completion round-trip",
                          "%.1fs, model=%r, said %r" % (elapsed, model, text[:40])))
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
    try:
        text = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return Result(WARN, "Truncation probe inconclusive", "malformed response")

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

def do_download(key: str, dest_dir: str = "./models/llm") -> int:
    spec = UNGATED_MODELS.get(key)
    if spec is None:
        print("Unknown model %r. Choose from: %s"
              % (key, ", ".join(UNGATED_MODELS)))
        return 2
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, spec.filename)
    if os.path.exists(dest) and os.path.getsize(dest) > 512 * 1024 * 1024:
        print("Already present: %s (%.2f GB)"
              % (dest, os.path.getsize(dest) / (1024 ** 3)))
        return 0
    print("Downloading %s" % spec.label)
    print("  licence : %s (no HuggingFace login, no gated-repo TOS)" % spec.licence)
    print("  size    : ~%.2f GB" % spec.size_gb)
    print("  from    : %s" % spec.url)
    print("  to      : %s" % dest)
    if shutil.which("curl") is None:
        print("curl not found; download manually with the URL above.")
        return 1
    # -C - resumes a partial file, so a dropped download is cheap to retry.
    rc = subprocess.call(["curl", "-L", "--fail", "--retry", "3",
                          "-C", "-", "-o", dest, spec.url])
    if rc != 0:
        print("Download failed (curl exit %d). Re-run to resume." % rc)
        return rc
    print("Done: %s (%.2f GB)" % (dest, os.path.getsize(dest) / (1024 ** 3)))
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
    elif provider == "llama-server":
        updates["LLAMA_SERVER_BASE_URL"] = "http://localhost:8080"
    else:
        print("Unknown provider %r." % provider)
        return 2

    lines: List[str] = []
    if os.path.exists(path):
        shutil.copyfile(path, path + ".bak")
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

def run_checks(provider: Optional[str], deep: bool) -> List[Result]:
    env = dict(load_env())
    # real environment wins over .env, same as python-dotenv's default
    for key in list(env):
        if key in os.environ:
            env[key] = os.environ[key]

    results = [check_python()]
    results.extend(check_pydeps())
    results.append(check_embedding_cache(env))

    if provider in (None, "local", "llamacpp"):
        results.append(check_gguf(env))
    if provider in (None, "ollama"):
        results.extend(check_ollama(env, deep))
    if provider in (None, "llama-server"):
        results.extend(check_llama_server(env, deep))
    if provider in (None, "lmstudio"):
        results.extend(check_lmstudio(env, deep))
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
    if warns:
        print("No blocking problems, %d warning(s) -- the demo should run." % warns)
        return 0
    print("All checks passed. Run: python3 src/rag_poisoning_demo.py")
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
                        choices=["local", "llamacpp", "ollama", "llama-server", "lmstudio"],
                        help="check only this provider (default: all)")
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
                        choices=["local", "llamacpp", "ollama", "llama-server"],
                        help="point .env at a provider (backs up the old file)")
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
        env = dict(load_env())
        for key in list(env):
            if key in os.environ:
                env[key] = os.environ[key]
        return report_one_line(results, env)
    if args.json:
        payload = {"results": [r.as_dict() for r in results],
                   "fails": sum(1 for r in results if r.status == FAIL),
                   "warns": sum(1 for r in results if r.status == WARN)}
        print(json.dumps(payload, indent=2))
        return 1 if payload["fails"] else 0
    return report(results)


if __name__ == "__main__":
    sys.exit(main())
