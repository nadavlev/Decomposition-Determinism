# Build Progress

| Step | Module(s) | Status |
|------|-----------|--------|
| 1 | Scaffold (layout, requirements, stubs) | ✅ done |
| 2 | corpus.py + retrieval.py + tests | ✅ done |
| 3 | stub_model.py + tests | ✅ done |
| 4 | monolith.py + tests | ✅ done |
| 5 | graph_pipeline.py nodes + state + per-node tests | ✅ done |
| 6 | Conditional edges + checkpointer + resume test | ✅ done |
| 7 | evaluation.py + tests | ✅ done |
| 8 | compare.py CLI | ✅ done |
| 9 | app.py Streamlit UI | ✅ done |
| 10 | README.md + final PROGRESS update | ✅ done |

## All acceptance criteria met

- No API key required (StubModel only)
- `pytest -q` → 72 tests pass
- Per-node isolation tests in `test_graph_nodes.py`
- Interrupt/resume test in `test_graph_resume.py::test_graph_interrupt_and_resume`
- Trap-question delta: monolith confident_ungrounded=3, graph=0
- `compare.py` names the deciding node for every question
- Streamlit UI shows side-by-side with per-node trace expander
- Results are fully deterministic
