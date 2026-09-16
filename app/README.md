# Chatbot Learning Museum — single `app/` folder

> For learning / portfolio only. **Not expected to run or execute.**
> This folder losslessly represents the work across `app_rag` → `app_kshitij` → `app_home`.

## Evolution in one glance

| Phase | Folder (archived into `app/`) | What it is |
|---|---|---|
| 1. RAG CLI | `app_rag/backend/` → `app/backend/legacy/rag_*` | SQL (`flogbook.fault_bookv3`) → clean → Chroma → local GGUF (`DeepSeek-R1-Distill-Llama-8B-Q2_K`) CLI loop. No frontend, no server. |
| 2. Monolith web app | `app_kshitij/` → `app/backend/legacy/kshitij_app_monolith.py` + `app/frontend/legacy/` | Single-file Flask + SSE + Phi-3.1-mini + `all-MiniLM` + Chroma `accelerator_papers`, glassmorphism UI with vendored `marked`. |
| 3. Modular hybrid | `app_home/` → `app/backend/*.py` + `app/frontend/` (canonical) | Intent-routed (BERT+rules, 5 intents) RAG over fault logs + domain docs, `llama-cpp` streaming, `/frontend_log`, dark/light theme, offline-first. |

## Layout

```
app/
  README.md                # this file
  requirements.txt         # reconstructed union (never existed per-app)
  .env.example             # redacted secrets (HF tokens, proxy, SQL)
  backend/
    app.py                 # canonical server (ex-app_home, paths relativized)
    config.py              # MERGED config (relative paths + CHROMA_DB_DIR fix + legacy aliases)
    intent_classifier.py   # BERT + keyword hybrid, 5 intents
    rag_setup.py           # generic txt/pdf/json → bge-en → Chroma
    rag_db.py              # MSSQL fault_bookv3 → clean → Document → Chroma
    legacy/
      kshitij_app_monolith.py  # verbatim ex-app_kshitij/backend/app.py
      rag_cli_main.py          # verbatim ex-app_rag/backend/main.py
      rag_preprocess.py        # verbatim ex-app_rag/backend/preprocess.py
      rag_pipeline.py          # verbatim ex-app_rag/backend/rag_pipeline.py
      rag_minimal_config.py    # verbatim ex-app_rag/backend/config.py (1 line)
  frontend/
    templates/index.html   # canonical 72L shell (ex-app_home)
    static/css/style.css   # single copy (was byte-identical x4)
    static/js/script.js    # canonical SSE client (ex-app_home, 262L)
    static/js/marked.min.js# single vendored copy (offline intranet)
    static/icons/send.svg  # single copy
    legacy/
      index_full_273L.html     # ex-app_kshitij active SPA
      index_checkpoint_274L.html # .ipynb checkpoint era (inline script)
      style_legacy_568L.css    # diverged text-button-era stylesheet
  docs/
    lab-notes.md           # full root Readme.md (14 sections), secrets redacted
    concepts.md            # what was done + concepts learned per phase
    provenance.md          # old path → new path + SHA256 + dedup/drop log
```

## Known non-runnable issues (kept intentionally as learning record)

- `rag_setup.py` referenced `config.CHROMA_DB_DIR` which never existed in `app_home` — now defined in merged `config.py`.
- Absolute paths (`C:\Users\Administrator\...`, `C:\Users\offic\Documents\...`) relativized in canonical files only; legacy files keep originals verbatim.
- `app_rag/rag_pipeline.py:80` calls `config.setup_logging()` which its `config.py` lacks → `AttributeError` (preserved).
- `langchain_chroma` vs `langchain_community.vectorstores.Chroma` import split between `main.py`/`rag_pipeline.py` (preserved).
- Frontend expects `clean_response`/`CLEAN_STREAM_STARTED`, backend sends `status:DONE` (preserved mismatch).
- No `requirements.txt` existed; models/`chroma_db`/`rag_data`/logs were git-ignored and absent.

## Browse order for reviewers

1. `docs/concepts.md` — 5-minute story of what was learned.
2. `docs/lab-notes.md` — full 14-section lab notes.
3. `backend/legacy/` → `backend/` — CLI → monolith → modular progression.
4. `frontend/legacy/` → `frontend/` — UI evolution.
