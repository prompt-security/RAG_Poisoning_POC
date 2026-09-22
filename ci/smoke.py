#!/usr/bin/env python3
"""
Construct the demo's REAL chain and query it. This is the gate.

Why this file exists rather than an import sweep: every third-party import that
matters in this repo lives inside a METHOD BODY.

    src/rag_system.py:43   from langchain_community.vectorstores import Chroma
    src/rag_system.py:77   from langchain.chains import RetrievalQA      <- regression #3
    src/rag_system.py:99   from rag_poisoning_corpus import ...          <- Document
    src/utils.py:39        import chromadb                               <- regression #2

So `import rag_system`, `python -m compileall` and `python3 test_setup.py` all
succeed while the demo is completely unrunnable. RAGSystem.__init__ executes the
first two; setup_vector_database() executes the third plus add_documents() and
the deprecated Chroma.persist(); query() proves .invoke() still returns
{"result", "source_documents"} and that result is still str-compatible, which
attack_demo.py's re.search depends on.

Runs locally too:  .venv/bin/python ci/smoke.py --embeddings fake

  --embeddings fake : DeterministicFakeEmbedding + a canned chat model. No
                      network, no weights, ~5 s. top_k is left unset so Chroma's
                      default returns the whole corpus -- this phase is about
                      plumbing, and a fake-embedding ranking assertion would be
                      arbitrary.
  --embeddings real : the real MiniLM the demo uses, TOP_K_RETRIEVAL=3, against
                      the cache setup.sh primed. Asserts the MECHANISM (the
                      poisoned document's source is present in the poisoned
                      phase and absent in the clean one), never a score -- a
                      count assertion goes red on a corpus edit and then gets
                      lowered until it means nothing.
"""

import argparse
import os
import re
import shutil
import sys
import warnings

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

POISON_SOURCE = "distributed_systems_advanced.md"
# The demo's own first query. The poisoned document is the distributed-systems
# one, so with real MiniLM it ranks top for this by a wide margin -- not a
# knife-edge assertion.
QUERY = "How do distributed systems handle load balancing?"

FAILURES = []


def check(ok, label, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + label + (("  -- " + detail) if detail else ""))
    if not ok:
        FAILURES.append(label + ((": " + detail) if detail else ""))
    return ok


class Cfg:
    """The only attributes RAGSystem / create_embeddings read."""

    def __init__(self, db_path, model, top_k):
        self.vector_db_path = db_path
        self.embedding_model = model
        self.top_k_retrieval = top_k


def fake_llm():
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage
    return GenericFakeChatModel(messages=iter([AIMessage(content="Arrr, matey!")] * 64))


def build_embeddings(mode, cfg):
    if mode == "fake":
        from langchain_core.embeddings import DeterministicFakeEmbedding
        return DeterministicFakeEmbedding(size=384), "fake"

    import torch
    from utils import get_device, create_embeddings
    mps = getattr(torch.backends, "mps", None)
    print("  torch %s | mps built=%s available=%s"
          % (torch.__version__,
             mps and mps.is_built(), mps and mps.is_available()))
    device = get_device()
    try:
        emb = create_embeddings(cfg, device)
        emb.embed_query("probe")
    except Exception as exc:
        # src/utils.py:get_device() picks 'mps' from platform.machine() alone; it
        # never calls torch.backends.mps.is_available(). Record the divergence
        # loudly rather than hiding it behind a forced-CPU env var.
        print("::warning::create_embeddings failed on device %r (%s). "
              "src/utils.py:get_device() selects 'mps' from platform.machine() "
              "alone and never checks torch.backends.mps.is_available(). "
              "Falling back to cpu." % (device, exc))
        device = "cpu"
        emb = create_embeddings(cfg, device)
    return emb, device


def sources(response):
    return [d.metadata.get("source", "unknown") for d in response.get("source_documents", [])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", choices=["fake", "real"], default="fake")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    os.chdir(REPO)

    print("\n[1] the pin regression #2 came from")
    import numpy
    check(numpy.__version__.startswith("1."), "numpy is on the 1.x line",
          "numpy %s installed; chromadb 0.4.24 imports np.float_, removed in "
          "NumPy 2.0" % numpy.__version__)
    if numpy.__version__.startswith("1."):
        import chromadb          # the import that actually crashed
        print("  PASS  chromadb %s imports for real  -- not find_spec()"
              % chromadb.__version__)

    print("\n[2] the one lazy import no other step reaches")
    try:
        # llm_factory.py. Importing the class does not import llama_cpp, which
        # only loads inside validate_environment -- so this is safe without the
        # `local` extra installed.
        from langchain_community.llms import LlamaCpp  # noqa: F401
        print("  PASS  langchain_community.llms.LlamaCpp resolves")
    except Exception as exc:
        check(False, "langchain_community.llms.LlamaCpp resolves", repr(exc))

    print("\n[3] constructing the chain (%s embeddings)" % args.embeddings)
    db = os.path.join("data", "ci_smoke_%s" % args.embeddings)
    shutil.rmtree(db, ignore_errors=True)
    os.makedirs(db, exist_ok=True)

    top_k = 3 if args.embeddings == "real" else None
    cfg = Cfg(db, os.environ.get("EMBEDDING_MODEL",
                                 "sentence-transformers/all-MiniLM-L6-v2"), top_k)
    emb, device = build_embeddings(args.embeddings, cfg)
    if args.embeddings == "real":
        check(len(emb.embed_query("probe")) == 384,
              "MiniLM-L6-v2 is 384-dimensional (device=%s)" % device)

    from rag_system import RAGSystem     # imports fine even when the chain is dead
    rs = RAGSystem(cfg, emb, fake_llm(), collection_name="ci_smoke_%s" % args.embeddings)
    print("  PASS  RAGSystem.__init__ completed "
          "-- this is what regression #3 broke (src/rag_system.py:77)")

    qa = rs.qa_chain
    check(type(qa).__name__ == "RetrievalQA", "the chain is a RetrievalQA", type(qa).__name__)
    # The artifact names its own vulnerability: it is a "stuff" chain, so every
    # retrieved chunk lands in one prompt verbatim. A silent switch to
    # map_reduce/refine would still construct and would quietly stop
    # demonstrating the attack.
    check(type(qa.combine_documents_chain).__name__ == "StuffDocumentsChain",
          "chain_type is still 'stuff'", type(qa.combine_documents_chain).__name__)
    check(qa.return_source_documents is True,
          "return_source_documents is on (attack_demo.py reads it)")

    print("\n[4] poisoned phase")
    # setup_vector_database rebuilds self.vectorstore but leaves the chain's
    # retriever bound to the deleted collection. attack_demo.py calls
    # refresh_chain() in between; mirroring it pins that undocumented ordering.
    rs.setup_vector_database(include_poison=True)
    rs.refresh_chain()
    out = rs.query(QUERY)
    check(set(out) >= {"query", "result", "source_documents"},
          ".invoke() still returns query/result/source_documents", str(sorted(out)))
    check(isinstance(out["result"], str),
          "response['result'] is still str-compatible "
          "(attack_demo.py runs re.search on it)", type(out["result"]).__name__)
    if isinstance(out["result"], str):
        re.search(r"\barrr\b", out["result"], re.IGNORECASE)   # must not raise
    src = sources(out)
    check(bool(src), "the retriever returned documents")
    check(POISON_SOURCE in src,
          "the poisoned document reached the chain's prompt", ", ".join(src))
    if top_k:
        check(len(src) == top_k, "TOP_K_RETRIEVAL=%d honoured" % top_k, str(len(src)))

    print("\n[5] clean phase")
    rs.setup_vector_database(include_poison=False)
    rs.refresh_chain()
    clean = sources(rs.query(QUERY))
    check(POISON_SOURCE not in clean,
          "the poisoned document is absent from the clean corpus", ", ".join(clean))

    print("\n" + "=" * 64)
    if FAILURES:
        for f in FAILURES:
            print("::error::smoke: %s" % f)
        print("%d assertion(s) failed." % len(FAILURES))
        return 1
    print("Chain constructs, ingests, persists, retrieves and answers (%s embeddings)."
          % args.embeddings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
