# Verifiable Decomposition — Step 1 Teaching Demo

A minimal, fully offline teaching demo showing **why decomposing an AI pipeline to the verifiability boundary matters** — and what it looks like in practice using LangGraph.

## The Mistake (Step 1)

The classic failure is a **single do-everything loop**: one prompt retrieves docs, answers a question, and returns a raw string. When it's wrong, you cannot tell which part went wrong — retrieval? the model? context formatting? You get one opaque output and no handle on the failure.

## The Fix: Decompose to the Verifiability Boundary

Break the pipeline into bounded steps, but **only as far as it improves your ability to check each step**. Not smaller for its own sake — smaller until each step can be tested in isolation and its output can be inspected before it is passed to the next step.

> "Decompose to the verifiability boundary" means: if you can't write a unit test for a step and observe its output independently, it isn't decomposed enough.

## What this demo shows

Two pipelines, same inputs, side by side:

| | Monolith | Decomposed Graph |
|---|---|---|
| Architecture | One `retrieve → generate → return` call | `StateGraph` with 6 nodes |
| Trap questions | Returns confident-but-**wrong** answer | Escalates — refuses to guess |
| When it fails | Opaque — no idea which step | Names the **exact node** that caught the problem |
| Testable per-step? | ❌ No | ✅ Yes |

## Node / Edge Architecture

```
START
  └─► validate_question
        ├─(invalid)─► escalate ─► END
        └─(valid)──► retrieve
                       ├─(weak retrieval)──► escalate ─► END
                       └─(strong)──────────► answer
                                               └─► validate_grounding
                                                     ├─(not grounded)─► escalate ─► END
                                                     └─(grounded)────► format_output ─► END
```

Each node does **one verifiable thing** and appends a human-readable line to `state["trace"]`. The `escalate` node calls `interrupt()` to pause for a human decision; the in-memory checkpointer persists state so `Command(resume=...)` picks up exactly where it stopped.

## Run commands

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.compare      # CLI side-by-side table (no UI)
streamlit run app.py       # Side-by-side Streamlit UI (screen-recordable)
pytest -q                  # All tests — must all pass
```

## Acceptance checklist

- [x] Everything runs with **no API key** set (StubModel is the only model)
- [x] `pytest -q` passes (72 tests)
- [x] At least one test exercises a **single graph node in isolation** (`test_graph_nodes.py`)
- [x] One test drives the graph to an `interrupt` and resumes via `Command(resume=...)` (`test_graph_resume.py::test_graph_interrupt_and_resume`)
- [x] On trap questions, the **monolith returns ≥1 confident-but-ungrounded answer** (`test_evaluation.py::test_monolith_returns_at_least_one_confident_ungrounded_answer`)
- [x] On trap questions, the **graph returns 0** — it escalates instead (`test_evaluation.py::test_graph_returns_zero_confident_ungrounded_answers`)
- [x] `compare.py` output clearly shows the graph names the deciding node; the monolith cannot
- [x] The Streamlit UI shows both pipelines side by side with the per-node trace
- [x] Results are deterministic across repeated runs

## Project structure

```
src/
  corpus.py          # Fixed in-memory documents (AcmePay FAQ, 10 docs)
  retrieval.py       # TF-IDF cosine similarity + retrieval-strength signal (stdlib only)
  stub_model.py      # Deterministic StubModel — no network, no API key
  monolith.py        # The single do-everything pipeline (intentionally wrong)
  graph_pipeline.py  # The decomposed LangGraph StateGraph
  evaluation.py      # Eval set (5 normal + 3 trap) + scoring
  compare.py         # CLI: run both, print table
app.py               # Streamlit side-by-side UI
tests/
  test_retrieval.py
  test_stub_model.py
  test_monolith.py
  test_graph_nodes.py   # Per-node isolation tests
  test_graph_resume.py  # Full graph + interrupt/resume
  test_evaluation.py    # Trap-question delta assertion
```

## Exact installed versions

- `langgraph==1.2.5`
- `streamlit==1.58.0`
- `pytest==9.1.0`
- Python 3.12.3

## Learn more

- [LangGraph docs](https://langchain-ai.github.io/langgraph/)
- [LangGraph repo](https://github.com/langchain-ai/langgraph)
- [Streamlit docs](https://docs.streamlit.io/)

> **Note on observability:** Each node logs one INFO line (node name + decision) via stdlib `logging`. For production, [LangSmith](https://smith.langchain.com/) or [Langfuse](https://langfuse.com/) provide full trace replay — that's Step 8 scope, out of scope here.
