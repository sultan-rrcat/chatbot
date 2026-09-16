# Concepts learned (what this repo represents)

## Phase 1 — `app_rag`: SQL → RAG fundamentals (CLI, no UI)
- Facility logbook ontology: `fault_bookv3` (fault_id/time/duration, system/device, description, persons_involved, action_taken, faulty_system, human_error, fda_entry, logged_by, log_time, first_observation, beam_affected).
- ETL with `pandas + SQLAlchemy + pyodbc` (Trusted_Connection, ODBC Driver 17); two SQL identities (`DESKTOP-FG7N2DC\SQLEXPRESS/flogbook` → `localhost\SQLEXPRESS/CONTROLS`).
- Cleaning: duration regex (`X mins Y secs` → `X.X min`), `pd.to_datetime(errors='coerce')`, `fillna('N/A')`, `safe_strip` against `UnicodeDecodeError`.
- Dual path discovered: `preprocess.py` dumps `fault_bookv3_cleaned.txt` (`col: value` + `----` separator) while `rag_pipeline.py` ingests the live `df` directly (TXT unused).
- Row→chunk templating (14 fields) + metadata (`fault_id/time/system`) → `HuggingFaceEmbeddings(all-MiniLM-L6-v2, local_files_only)` → `Chroma.from_documents(persist_directory=new_rag_data/chroma_faultbook_index)`.
- Grounded generation: top-`k=4`, low temperature (`0.1`), `top_p=0.9`, `n_ctx=8192` on `DeepSeek-R1-Distill-Llama-8B-Q2_K.gguf` via `llama-cpp-python`.
- Failure modes kept: `config.setup_logging()` missing, import side-effect (`from preprocess import df` fetches DB on import), `langchain_chroma` vs `langchain_community` split, relative `persist_directory` CWD dependence.

## Phase 2 — `app_kshitij`: monolith web app (one `app.py`, 326L)
- Recursive ingestion (`os.walk`, `.txt/.pdf/.json`, skip others); custom JSON shaping (`System/Description/Severity/Cause/Solution`); TXT/PDF whitespace-normalize + sliding window (`500/50`) with `ids="{file}_chunk_{i}"`.
- Startup wipe (`shutil.rmtree(CHROMA_DB_PATH)`) + `get_or_create_collection("accelerator_papers")`, ingest-if-empty.
- Prompt: `<|system|> accelerator-physics specialist … <|end|>` + optional `Previous conversation` (`deque(maxlen=5)`) + `Context: <3 docs>` + `Question:` + `<|assistant|>`; `LlamaCpp` streaming (`n_ctx=4096, max_tokens=2048, T=0.7, top_p=0.95`).
- SSE (`text/event-stream`, `data: {"chunk"/"status":"DONE"/"clean_response"}`), `503` uninitialized / `400` empty guards, `AbortController` + `\n\n` buffering on the client.
- Frontend: single SPA, `marked v15.0.12` vendored (offline intranet, no CDN), `localStorage` + `prefers-color-scheme` theme, auto-resize textarea, `Enter(!Shift)` send, spinner, modal errors, welcome bubble; glassmorphism (`:root` dark + `[data-theme=light]`, `Inter`, `send.svg`).
- Ops: dual logging (`app_chatbot.log` + stdout), hard-coded `Administrator` paths (non-portable), unused imports (`torch, DirectoryLoader, Thread…`), Jupyter `.ipynb_checkpoints` editing traces, `CHROMA_TELEMETRY=false`, SSH `-L 5000`.

## Phase 3 — `app_home`: modular hybrid chatbot (canonical `app/`)
- Intent taxonomy + hybrid classifier (`intent_classifier.py`): keyword rules (realtime/analytical/fault/domain) gated with BERT (`BertTokenizer/BertForSequenceClassification`, 5 labels; `conf>0.7` → classifier, agree → rule, disagree+`conf>0.5` → classifier, else `Uncertain`).
- Routing in `app.py`: `INTENT1/2` stubbed (`under development`), `INTENT3_FAULTINFO` → `FAULT_INFO_COLLECTION`, `INTENT4_DOMAININFO` → `ACC_PY_DOCS` + `sources[]`, else general; `ChatPromptTemplate + dedent`; token streaming + terminal `sources` + `DONE`; `/frontend_log` telemetry.
- Generic RAG (`rag_setup.py`): `TextLoader/PyPDFLoader/JSONLoader(jq=".")`, `RecursiveCharacterTextSplitter(500/100)`, `HuggingFaceEmbeddings(bge-en, local_files_only)`, `chromadb.PersistentClient` + batched `Chroma` ingest (`MAX_CHROMA_BATCH_SIZE=5000`), collection wipe, `k=3` retrieve with `unique_sources`, telemetry disabled.
- Domain ETL (`rag_db.py`): `FaultbookIngestor` (fetch → `faultbook_data.csv` → clean → `row_to_chunk` 15 fields → ingest; SQL fetch currently commented to CSV path).
- Offline-first: local GGUFs (Phi-3-mini default, `n_ctx=8096, n_gpu_layers=1000`; DeepSeek + Hermes alternates), local embedder, local `marked.min.js`.
- Preserved debt: `CHROMA_DB_DIR` undefined (fixed in merged `config.py`), absolute `Documents\chatbot` paths (relativized in canonical only), unused `llama_cpp` import, frontend/backend `clean_response` contract mismatch, `deque(5)` context loss, no math rendering / persistence / auth (see lab-notes future scope).

## Cross-cutting lab notes (root `Readme.md` → `docs/lab-notes.md`)
GPT-2 setup; CUDA `cu118/cu113` pinning; NLTK WordNet manual corpus fix; quantized (4-bit) vs FP16 trade-offs; accelerator-physics Q&A samples + FermiLab IOTA/MI/Mu2e + Courant-Snyder; `lm-eval` (MMLU/GSM8K, `hellaswag` blocked-network note); full MMLU task list; memory strategies table (CPU/FP16/quant/`device_map`/DeepSpeed+WSL2); RHEL/conda/proxy/Jupyter/SSH/`llama.cpp`/`xformers`/`libaio`/`nohup`/zip/Chroma-telemetry; HF model download catalog; RAG features/limits/future scope; hybrid SQL+RAG control-chatbot vision + `fault_bookv3` 18-col schema + Ollama/Windows-proxy/Python2.7/Moba notes; venv-vs-conda `PATH` fixes; git daily workflow.
