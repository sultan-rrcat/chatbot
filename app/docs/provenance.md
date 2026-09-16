# Provenance — old path → new path (SHA256, bytes)

History preserved with `git mv` (rename detection: `git status` shows `R`).
Deduplicated files keep ONE physical copy; all source paths listed.
Dropped files are generated artifacts only. Verified 2026-09-16.

## Canonical backend (ex-`app_home`, latest modular phase)

| Original | New | SHA256 (prefix) | Size |
|---|---|---|---|
| `app_home/backend/app.py` | `app/backend/app.py` (+ relativized Flask folders) | `1B9CC7A3` | 6250 B |
| `app_home/backend/config.py` | `app/backend/config.py` (MERGED: relative paths + `CHROMA_DB_DIR` + legacy aliases) | `2D6CF33A` | 2446 B → merged |
| `app_home/backend/intent_classifier.py` | `app/backend/intent_classifier.py` (verbatim) | `1D840B73` | 4719 B |
| `app_home/backend/rag_setup.py` | `app/backend/rag_setup.py` (verbatim) | `EB7314DA` | 14383 B |
| `app_home/backend/rag_db.py` | `app/backend/rag_db.py` (verbatim) | `2DD529A4` | 9979 B |

## Legacy backend (verbatim, light-fix free)

| Original | New | SHA256 (prefix) | Size |
|---|---|---|---|
| `app_kshitij/backend/app.py` | `app/backend/legacy/kshitij_app_monolith.py` | `60FFC1D6` | 13953 B |
| `app_rag/backend/main.py` | `app/backend/legacy/rag_cli_main.py` | `9E9CD85F` | 2047 B |
| `app_rag/backend/preprocess.py` | `app/backend/legacy/rag_preprocess.py` | `F3151668` | 2596 B |
| `app_rag/backend/rag_pipeline.py` | `app/backend/legacy/rag_pipeline.py` | `A0DEA206` | 2736 B |
| `app_rag/backend/config.py` | `app/backend/legacy/rag_minimal_config.py` | `3380A596` | 88 B |

## Canonical frontend

| Original | New | SHA256 (prefix) | Size |
|---|---|---|---|
| `app_home/frontend/templates/index.html` | `app/frontend/templates/index.html` (verbatim) | `577BA30F` | 3167 B |
| `app_home/frontend/static/js/script.js` | `app/frontend/static/js/script.js` (verbatim) | `19AE82B0` | 9579 B |
| `app_home/frontend/static/js/marked.min.js` | `app/frontend/static/js/marked.min.js` (single copy; identical to kshitij's) | `742EE84B` | 39972 B |
| `app_home/frontend/static/icons/send.svg` | `app/frontend/static/icons/send.svg` (single copy; identical to kshitij's) | `8A1D7C19` | 232 B |
| `app_home/frontend/static/css/style.css` | `app/frontend/static/css/style.css` (single copy; identical x4, see below) | `824CEBF9` | 18480 B |

## Legacy frontend (diverged variants kept)

| Original | New | SHA256 (prefix) | Size |
|---|---|---|---|
| `app_kshitij/frontend/templates/index.html` | `app/frontend/legacy/index_full_273L.html` (verbatim) | `AA5058A4` | 11865 B |
| `app_home/frontend/templates/.ipynb_checkpoints/index-checkpoint.html` | `app/frontend/legacy/index_checkpoint_274L.html` (verbatim; byte-identical to kshitij's checkpoint, which was deduplicated) | `D999D9A6` | 12025 B |
| `app_kshitij/frontend/templates/styles/style.css` (+ byte-identical `.ipynb_checkpoints` twin; git pairs either — same bytes) | `app/frontend/legacy/style_legacy_568L.css` (verbatim; byte-identical to its own checkpoint, which was deduplicated) | `E8E16CE5` | 13870 B |

## Deduplicated (one copy kept, sources logged — user-approved)

- `marked.min.js` (`742EE84B`): `app_home/.../marked.min.js` + `app_kshitij/.../marked.min.js` → one copy.
- `send.svg` (`8A1D7C19`): `app_home/.../send.svg` + `app_kshitij/.../send.svg` → one copy.
- `style.css` active (`824CEBF9`): `app_home/.../style.css` + `app_home/.../style-checkpoint.css` + `app_kshitij/.../style.css` + `app_kshitij/.../style-checkpoint.css` → one copy.
- `index-checkpoint.html` (`D999D9A6`): `app_home/.../index-checkpoint.html` + `app_kshitij/.../index-checkpoint.html` → `index_checkpoint_274L.html`.
- `style legacy` (`E8E16CE5`): `app_kshitij/.../styles/style.css` + `app_kshitij/.../styles/.ipynb_checkpoints/style-checkpoint.css` → `style_legacy_568L.css`.

## Dropped (generated only, not learning content)

- `app_home/backend/__pycache__/config.cpython-310.pyc` (`F8B9E5EB`, 2076 B)
- `app_home/backend/__pycache__/intent_classifier.cpython-310.pyc` (`AFC7E55C`, 3422 B)
- `app_home/backend/__pycache__/rag_setup.cpython-310.pyc` (`5F619B07`, 7703 B)

## Docs / root

- `Readme.md` (643 lines, root) → `app/docs/lab-notes.md` (full content, 4 secrets redacted → `app/.env.example`); root `Readme.md` becomes a pointer to `app/`.
- `.gitignore` (21 lines, root) → updated in place for `app/` paths (models, chroma_db, rag_data, logs, fonts, `__pycache__`, `.ipynb_checkpoints`).
- New (no original): `app/README.md`, `app/requirements.txt` (reconstructed), `app/.env.example`, `app/docs/concepts.md`, this file.

## Reconciliation

- Tracked before: 30 files (`git ls-files`).
- After: 13 canonical+legacy backend/frontend files via `git mv` (18 `R` entries include rename-detection pairing) + 10 deduplicated/dropped via `git rm` + ~7 new museum files.
- No unique logic/UI lost: every distinct hash prefix above is present exactly once in `app/` (except intentional single-copy dedups logged here).
