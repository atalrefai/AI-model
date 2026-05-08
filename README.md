# Phi-3-Mini Local AI Assistant — Trainable RAG Workspace

> Bilingual document — English first, then Arabic (الإنجليزية أولاً ثم العربية).

---

## 🇬🇧 English

### 1. Project Overview

This workspace is a **fully local** AI assistant built around the
**Phi-3-mini-4k-instruct (Q4 quantized GGUF)** model running through
[`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python). It is not
just a chat front-end — it is a complete **Retrieval-Augmented Generation
(RAG)** workspace with three independent training pipelines:

1. **ADTC training** — upload domain PDFs / DOCX / images / spreadsheets,
   convert them into structured "trained knowledge" with executive
   summaries, semantic tags and chunked retrieval.
2. **Dataset training** — same idea but tuned for tabular / structured
   files (CSV, XLSX, JSON).
3. **Database training** — connect to a live SQL database (MySQL / SQLite
   / PostgreSQL via SQLAlchemy), profile schema + sample rows, and turn
   the result into searchable knowledge documents.

The single Python file [web2.py](web2.py) implements the entire stack:
Flask server, HTML/JS UI, retrieval, reasoning, audit log, evaluation,
feedback and learning backlog.

The system has been built across **7 phases** (Foundation → Continuous
Learning); this README is the canonical guide to all of them.

### 2. Repository Layout

```
AI Model Phi-3-Mini/
├── web2.py                              # ⭐ Main app (server + UI + RAG)
├── web.py                               # Older / minimal variant
├── run_llama_cli.py                     # Local CLI runner for the GGUF model
├── Phi-3-mini-4k-instruct-q4.gguf       # Quantized model weights (Q4)
├── llama_cpp_python-0.3.22-...whl       # Pre-built wheel (Windows)
│
├── chat_memory/                         # Per-chat JSON files (history)
│   └── <chat_id>.json
│
├── training_memory/                     # All persistent trained knowledge
│   ├── adtc_datasets.json               # ADTC datasets registry
│   ├── db_links.json                    # DB connections registry
│   ├── knowledge_docs.json              # Indexed knowledge corpus
│   ├── audit_log.jsonl                  # Phase 1 — audit trail
│   ├── feedback_log.jsonl               # Phase 7 — user feedback
│   ├── eval_cases.json                  # Phase 6 — evaluation cases
│   ├── learning_backlog.jsonl           # Phase 7 — weak-answer backlog
│   ├── adtc_files/<dataset_id>/         # Uploaded ADTC source files
│   └── dataset_files/<dataset_id>/      # Uploaded dataset source files
│
├── _embedded.js / _served.html / _served.js   # Render snapshots
├── _rendered.html                       # Last test-client render (debug)
└── README.md                            # ← this file
```

### 3. Architecture (high level)

```
┌─────────────────────────────────────────────────────────────────┐
│                       Browser UI (single page)                  │
│  Chat • Monitoring • Settings • Training • Feedback (👍/👎)     │
└──────────────────────────────┬──────────────────────────────────┘
                               │  fetch (withBase)
┌──────────────────────────────▼──────────────────────────────────┐
│                    Flask app (web2.py)                          │
│                                                                 │
│  ┌─────────────┐   ┌───────────────┐   ┌─────────────────────┐  │
│  │  Chat API   │   │  Training API │   │  Eval / Feedback    │  │
│  └──────┬──────┘   └───────┬───────┘   └──────────┬──────────┘  │
│         │                  │                      │             │
│  ┌──────▼──────┐    ┌──────▼──────┐       ┌───────▼──────┐      │
│  │  Retrieval  │    │  Ingestion  │       │ Audit log    │      │
│  │  (TF-IDF +  │    │  (PDF/DOCX/ │       │ Backlog      │      │
│  │   semantic) │    │  XLSX/CSV/  │       │ Quality      │      │
│  │             │    │  Images/DB) │       │ Confidence   │      │
│  └──────┬──────┘    └──────┬──────┘       └──────────────┘      │
│         │                  │                                    │
│  ┌──────▼──────────────────▼──────┐                             │
│  │   Reasoning bundle             │                             │
│  │   (evidence, conflicts,        │                             │
│  │    missing-evidence flag)      │                             │
│  └──────────────┬─────────────────┘                             │
│                 │                                               │
│  ┌──────────────▼──────────────────┐                            │
│  │   model_reply()                 │                            │
│  │   • System prompt with rules    │                            │
│  │   • Knowledge context injection │                            │
│  │   • Hedging detector            │                            │
│  │   • Synthesis fallback          │                            │
│  └──────────────┬──────────────────┘                            │
│                 │                                               │
│  ┌──────────────▼──────────────────┐                            │
│  │  llama_cpp.Llama (Phi-3-mini)   │                            │
│  └─────────────────────────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
```

### 4. End-to-end Request Flow (Chat)

1. **Browser** sends `POST /api/chat` with
   `{chat_id, message, knowledge_mode, response_mode}`.
2. **`chat_api`** (web2.py) generates a `request_id`, opens an audit
   event `chat.request`, loads the chat memory from `chat_memory/`.
3. **Knowledge retrieval** — `_retrieve_knowledge_with_trace()`:
   - Builds a TF-IDF + token-overlap score over `knowledge_docs.json`.
   - Filters by `knowledge_mode` (`all`, `adtc_only`, `dataset_only`,
     `database_only`).
   - Returns top-K docs **plus** a retrieval trace (which terms matched,
     which docs were considered).
4. **Reasoning bundle** — `_build_reasoning_bundle()`:
   - `_extract_evidence_lines()` picks the strongest snippet lines.
   - `_detect_reasoning_conflicts()` flags contradictory dates / hours
     / numbers between snippets.
   - Adds `missing_evidence` if no good lines were found.
5. **Generation** — `model_reply()`:
   - Builds the system prompt with response-style rules, ADTC-style
     rules (when applicable), and the reasoning pipeline.
   - Streams the user history + `TRAINED KNOWLEDGE` block into Phi-3.
   - Detects hedging ("As an AI…", "consult the…") and triggers
     `_synthesize_from_snippets()` to produce a clean rewritten answer.
6. **Quality + confidence** — `_response_quality_metrics()` and
   `_confidence_from_quality()` produce a 0-100 score using
   knowledge-overlap, length, evidence-presence and conflict count.
7. **Persist** — chat history is appended, `chat.reply` is audited,
   weak answers are pushed to `learning_backlog.jsonl`.
8. **Response** — JSON: `{request_id, chat_id, reply, quality,
   confidence, sources, retrieval_trace, reasoning}`.

### 5. Source-code Walkthrough (web2.py)

> Line numbers below are approximate; use `Ctrl+G` in VS Code to jump.

| Section | Purpose |
| --- | --- |
| **Globals & paths** (~50-90) | `MODEL_PATH`, `TRAINING_STORE_DIR`, `ADTC_DATASETS_PATH`, `KNOWLEDGE_DOCS_PATH`, `AUDIT_LOG_PATH`, `FEEDBACK_LOG_PATH`, `EVAL_CASES_PATH`, `LEARNING_BACKLOG_PATH`, `SCHEMA_VERSION = 1`. |
| **Llama init** | Loads the Phi-3 GGUF via `llama_cpp.Llama(...)` with `n_ctx=4096`, configurable threads. |
| **JSON helpers** | `_safe_load_json`, `json_dump` — atomic, lock-protected reads/writes. |
| **Normalizers** | `_normalize_chat`, `_normalize_dataset`, `_normalize_adtc_dataset`, `_normalize_knowledge_doc`, `_normalize_db_link` — guarantee a canonical shape regardless of historical schema drift. |
| **Audit & quality** (Phase 1) | `_append_audit_event`, `_response_quality_metrics`. |
| **Ingestion** | `_extract_pdf_text` (lazy `pypdf`), `_extract_docx_text`, `_extract_xlsx_text`, `_extract_csv_text`, `_extract_image_text` (OCR optional), `_chunk_text_semantic` (Phase 2). |
| **Knowledge layer** (Phase 3) | `_retrieve_knowledge_with_trace`, `_build_dataset_knowledge_links`, `/api/knowledge/retrieve`. |
| **Reasoning** (Phase 4) | `_extract_evidence_lines`, `_detect_reasoning_conflicts`, `_build_reasoning_bundle`, `_confidence_from_quality`, `_reply_has_evidence_marker`, `_append_reasoning_evidence`. |
| **Grounded extractors** | `_answer_from_knowledge` — deterministic answers for "how many rows", communication channels, closure dates, locations, etc. |
| **Synthesis fallback** | `_synthesize_from_snippets` — second LLM pass with a strict "no copy-paste, rewrite cleanly" prompt. |
| **Generation** | `model_reply(message, history, knowledge_docs, force_synthesis, reasoning_bundle, response_mode)`. |
| **HTML templates** | `INDEX_HTML` (chat), `MONITORING_HTML`, `TRAINING_INDEX_HTML`, `ADTC_RUN_HTML`, `DATASET_RUN_HTML`. |
| **Routes** | See section 7. |
| **Eval & feedback** (Phases 6-7) | `_load_eval_cases`, `_save_eval_cases`, `_append_learning_item`, `_capture_learning_backlog`. |

### 6. Data Schemas

#### `knowledge_docs.json`
```json
{
  "schema_version": 1,
  "docs": [
    {
      "id": "doc_xxx",
      "title": "Dataset:HP - Dental.pdf (part 1/2)",
      "content": "…full chunked text…",
      "source": "adtc | dataset | database | manual",
      "tags": ["medical", "schedule"],
      "meta": { "dataset_id": "...", "page": 1, "row_count": 42 },
      "created_at": 1715000000
    }
  ]
}
```

#### `audit_log.jsonl` (one JSON object per line)
```json
{"ts": 1715000000, "event": "chat.reply", "data": {"request_id": "...", "quality_score": 86}}
```

#### `feedback_log.jsonl`
```json
{"ts": 1715000010, "request_id": "...", "chat_id": "...", "rating": 5, "category": "positive", "note": ""}
```

#### `learning_backlog.jsonl`
```json
{"ts": 1715000020, "kind": "chat_improvement_needed", "data": {"question": "...", "confidence": 38, "missing_evidence": true}}
```

### 7. HTTP API Reference

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/` | Chat UI (single-page) |
| GET | `/monitoring` | Live monitoring dashboard |
| GET | `/training` | Training console |
| GET | `/training/run/<link_id>` | DB ingestion runner |
| GET | `/training/dataset/run/<dataset_id>` | Dataset training runner |
| GET | `/training/adtc/run/<dataset_id>` | ADTC training runner |
| GET / POST | `/api/config` | Read / update generation config (`max_tokens`, `temperature`, `top_p`, `stop`) |
| POST | `/api/chat` | Main chat endpoint (returns reply + quality + sources) |
| GET / POST / DELETE | `/api/chats[/<id>]` | List, create, fetch, clear, delete chats |
| POST | `/api/training/docs` | Add a manual knowledge doc |
| POST | `/api/training/dataset/upload` | Upload a dataset file |
| GET / DELETE | `/api/training/datasets[/<id>]` | Manage dataset registry |
| POST | `/api/training/datasets/<id>/start` | Start dataset training |
| GET | `/api/training/datasets/<id>/progress` | Poll progress |
| POST | `/api/training/adtc/upload` | Upload an ADTC file |
| GET / DELETE | `/api/training/adtc/datasets[/<id>]` | Manage ADTC registry |
| POST | `/api/training/adtc/datasets/<id>/start` | Start ADTC training |
| GET | `/api/training/adtc/datasets/<id>/progress` | Poll progress |
| POST | `/api/training/db/test` / `save` | Test / register DB connection |
| GET / DELETE | `/api/training/db/links[/<id>]` | Manage DB links |
| POST | `/api/training/db/links/<id>/start` | Start DB ingestion |
| GET | `/api/knowledge/retrieve?q=...&limit=5` | Inspect retrieval (Phase 3) |
| GET / POST | `/api/eval/cases` | Manage eval cases (Phase 6) |
| POST | `/api/eval/run` | Run eval suite, return pass-rate |
| POST | `/api/chat/feedback` | Submit thumbs feedback |
| GET | `/api/learning/backlog` | Read weak-answer backlog |
| GET | `/api/learning/retrain/suggest` | Heuristic retrain suggestions |
| GET | `/api/ops/daily_summary` | Operational rollup |
| GET | `/api/monitoring` | Live monitor JSON |
| GET | `/api/docs[/json]` | Self-documentation |

### 8. How the Model "Learns" (Training Pipeline)

The Phi-3 weights are **frozen**; "learning" here means building a
high-quality, retrievable, well-structured knowledge corpus that the
model is grounded on at inference time. The pipeline:

1. **Upload** — file is saved under `training_memory/<kind>_files/<dataset_id>/`.
2. **Extract** — text is pulled out:
   - PDF → `pypdf` (lazy import).
   - DOCX → `python-docx`.
   - XLSX → `openpyxl`.
   - CSV → built-in `csv`.
   - Images → optional OCR (Tesseract if available).
   - DB → SQLAlchemy schema + sample rows + column profiles.
3. **Semantic chunking** — `_chunk_text_semantic()` (Phase 2) splits by
   headings, bullets and natural paragraph boundaries, keeping a small
   overlap for context.
4. **Tagging** — heuristic semantic tags (medical, schedule, contacts,
   pricing, etc.) are attached to each chunk.
5. **Executive summary** — for each ADTC file an "Executive summary"
   doc is produced with key facts, training confidence, coverage score,
   pages-with-text and section hints.
6. **Indexing** — chunks are normalized and appended to
   `knowledge_docs.json` with `schema_version: 1`.
7. **Quality diagnostics** — extraction quality (chars/page,
   pages-with-text, table coverage) is recorded so the UI can flag
   weak files.
8. **Audit** — every step writes a `train.*` event to
   `audit_log.jsonl`.

### 9. How the Model "Understands" Inputs

At chat time the model receives a carefully composed prompt:

1. **System block** — response style (professional/practical/clear),
   ADTC style rules, time context, mandatory rules.
2. **Reasoning pipeline block** — explicit 5-step instruction:
   understand → select evidence → resolve conflicts → answer →
   acknowledge missing facts.
3. **Evidence block** — the strongest evidence lines from the
   reasoning bundle.
4. **Trained knowledge block** — top-K retrieved chunks (max 5,
   1800 chars each) labelled with their title.
5. **Conversation history** — last user turns only (in `adtc_only`
   mode) or full history.
6. **User message**.

Two safety nets prevent low-quality output:
- **Hedging detector** — if the reply contains "as an AI", "consult
  the database administrator", etc., a second pass via
  `_synthesize_from_snippets()` produces a clean rewrite.
- **Refusal detector** — `_looks_like_refusal()` catches generic
  refusals and routes the request to the synthesis fallback.

### 10. Algorithms in Detail

- **Retrieval scoring**: token overlap (`_tokenize`) + title boost +
  source-mode filter. Lightweight, no external embedding service.
- **Conflict detection**: regex-based date/hour/number extraction
  across snippets; mismatches are reported to the model so it can
  resolve or surface them.
- **Confidence**: weighted blend of quality score, evidence count,
  conflict count and knowledge-hit count → 0-100 with a short
  human-readable explanation.
- **Quality score**: penalizes very short answers, missing prefixes,
  echoing only a file title; rewards multiple matched evidence lines.
- **Eval runner** (`/api/eval/run`): replays cases via Flask's
  `test_client`, checks expected substrings + minimum confidence,
  saves a JSON report.
- **Learning backlog**: any reply with `missing_evidence`,
  `conflict_count > 0` or `confidence < 50` is queued for human
  review and possible retraining.

### 11. Local Setup & Run

#### Requirements
- Windows 10/11 (the included `.whl` is Windows-built; on Linux/macOS,
  install `llama-cpp-python` via pip).
- Python 3.10 – 3.12.
- ~4 GB RAM free for the Q4 model.

#### Install
```powershell
# 1) (Optional) create a venv
python -m venv .venv
.venv\Scripts\activate

# 2) Install llama-cpp-python (Windows wheel included)
pip install .\llama_cpp_python-0.3.22-py3-none-win_amd64.whl

# 3) Install runtime dependencies
pip install flask pypdf python-docx openpyxl sqlalchemy psutil
# OCR (optional):
pip install pytesseract pillow
```

#### Run
```powershell
python .\web2.py
# → http://127.0.0.1:5000
```

#### First-time checklist
1. Open `http://127.0.0.1:5000`.
2. Click **Training** → upload a PDF under **ADTC** → **Start**.
3. Watch the run page until "completed".
4. Back in chat, set **Knowledge mode = ADTC Only** and ask a
   question grounded in your file.
5. Use 👍 / 👎 to feed the learning backlog.

#### Backup / Restore
```powershell
.\scripts\backup_memory.ps1   # snapshots training_memory/ + chat_memory/
.\scripts\restore_memory.ps1  # restores latest snapshot
```

#### Run evaluation
```powershell
curl -X POST http://127.0.0.1:5000/api/eval/run
```

### 12. Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Buttons dead in browser | JS syntax error in template | Check `_rendered.html` + Node `vm.Script` lint (see `validate_js.ps1` flow used in this repo) |
| `pypdf` import error | Library missing | `pip install pypdf` |
| Chat returns "from trained ADTC data:" + raw text | Ingestion produced a weak file | Re-run ADTC training; check Executive Summary coverage score |
| Hedging answers ("contact the clinic") | Synthesis fallback didn't trigger | Check `_looks_like_refusal` markers; ensure `knowledge_mode = adtc_only` |
| MySQL/MariaDB stats 500 (`ONLY_FULL_GROUP_BY`) | Missing GROUP BY columns | Already handled in helpers; otherwise add the missing columns |

---

## 🇸🇦 العربية

### 1. نظرة عامة على المشروع

هذا المشروع عبارة عن **مساعد ذكاء اصطناعي يعمل محلياً بالكامل** يعتمد على
نموذج **Phi-3-mini-4k-instruct (مكمَّم Q4 بصيغة GGUF)** عبر مكتبة
`llama-cpp-python`. ليس مجرد واجهة محادثة، بل هو منظومة **توليد معزَّز
بالاسترجاع (RAG)** متكاملة، مع ثلاث منظومات تدريب مستقلة:

1. **تدريب ADTC**: رفع ملفات PDF / DOCX / صور / جداول، وتحويلها إلى
   "معرفة مدرَّبة" منظَّمة مع ملخصات تنفيذية ووسوم دلالية وتقطيع للاسترجاع.
2. **تدريب الـ Datasets**: نفس الفكرة لكن مُحسَّن للملفات الجدولية
   (CSV / XLSX / JSON).
3. **تدريب قواعد البيانات**: الاتصال بقاعدة بيانات حقيقية (MySQL /
   SQLite / PostgreSQL عبر SQLAlchemy)، وتحليل المخطط والصفوف، وتحويل
   الناتج إلى مستندات معرفة قابلة للبحث.

ملف بايثون واحد — [web2.py](web2.py) — يحتوي المنظومة كاملة: خادم Flask،
وواجهة HTML/JS، والاسترجاع، والاستدلال، وسجل التدقيق، والتقييم،
والتغذية الراجعة، وقائمة التعلم المتأخرة.

تم بناء النظام عبر **7 مراحل** (الأساسات → التعلم المستمر)، وهذا الملف
هو الدليل الرسمي لها جميعاً.

### 2. هيكل المستودع

```
AI Model Phi-3-Mini/
├── web2.py                              # ⭐ التطبيق الرئيسي (الخادم + الواجهة + RAG)
├── web.py                               # نسخة قديمة/مبسَّطة
├── run_llama_cli.py                     # مشغّل CLI محلي للنموذج
├── Phi-3-mini-4k-instruct-q4.gguf       # أوزان النموذج (Q4)
├── llama_cpp_python-0.3.22-…whl         # عجلة جاهزة لويندوز
│
├── chat_memory/                         # ملفات JSON للمحادثات
│   └── <chat_id>.json
│
├── training_memory/                     # كل المعرفة المدرَّبة الدائمة
│   ├── adtc_datasets.json
│   ├── db_links.json
│   ├── knowledge_docs.json
│   ├── audit_log.jsonl
│   ├── feedback_log.jsonl
│   ├── eval_cases.json
│   ├── learning_backlog.jsonl
│   ├── adtc_files/<dataset_id>/
│   └── dataset_files/<dataset_id>/
│
└── README.md                            # ← هذا الملف
```

### 3. البنية المعمارية (نظرة عُليا)

نفس المخطط الموجود في القسم الإنجليزي 3: المتصفح → Flask → استرجاع
→ بناء حزمة استدلال → `model_reply()` → نموذج Phi-3 → الرد +
تدقيق + ثقة + مصادر.

### 4. تدفق طلب المحادثة من البداية إلى النهاية

1. **المتصفح** يرسل `POST /api/chat` ومعه
   `{chat_id, message, knowledge_mode, response_mode}`.
2. **`chat_api`** يولّد `request_id`، يفتح حدث تدقيق `chat.request`،
   ويحمّل ذاكرة المحادثة من `chat_memory/`.
3. **استرجاع المعرفة** عبر `_retrieve_knowledge_with_trace()`:
   - يحسب درجة TF-IDF + تطابق الرموز فوق `knowledge_docs.json`.
   - يطبّق فلتر `knowledge_mode` (`all` / `adtc_only` /
     `dataset_only` / `database_only`).
   - يعيد أعلى K مستند مع **أثر استرجاع** يبيّن أي المصطلحات تطابقت
     وأي المستندات نُظر فيها.
4. **حزمة الاستدلال** عبر `_build_reasoning_bundle()`:
   - `_extract_evidence_lines()` يلتقط أقوى أسطر الأدلة.
   - `_detect_reasoning_conflicts()` يكشف التعارضات في التواريخ /
     الساعات / الأرقام بين المقاطع.
   - يضع علامة `missing_evidence` عند عدم وجود أسطر كافية.
5. **التوليد** عبر `model_reply()`:
   - يبني موجِّه نظام يحتوي قواعد الأسلوب وقواعد ADTC والسياق الزمني.
   - يحقن سجل المستخدم + كتلة "TRAINED KNOWLEDGE" في النموذج.
   - عند اكتشاف صياغة تهرّبية ("As an AI..." إلخ)، يُشغّل
     `_synthesize_from_snippets()` لإعادة كتابة الرد بشكل نظيف.
6. **الجودة والثقة** عبر `_response_quality_metrics()` و
   `_confidence_from_quality()` (0-100) مع شرح مختصر.
7. **الحفظ** — تحديث المحادثة، تدقيق `chat.reply`، ودفع الإجابات
   الضعيفة إلى `learning_backlog.jsonl`.
8. **الرد** بصيغة JSON تشمل: `request_id, chat_id, reply, quality,
   confidence, sources, retrieval_trace, reasoning`.

### 5. شرح الكود (web2.py)

| القسم | الوظيفة |
| --- | --- |
| **الثوابت والمسارات** (~50-90) | `MODEL_PATH`, `TRAINING_STORE_DIR`, `ADTC_DATASETS_PATH`, `KNOWLEDGE_DOCS_PATH`, `AUDIT_LOG_PATH`, `FEEDBACK_LOG_PATH`, `EVAL_CASES_PATH`, `LEARNING_BACKLOG_PATH`, `SCHEMA_VERSION = 1`. |
| **تهيئة Llama** | تحميل ملف GGUF عبر `llama_cpp.Llama(...)` بـ `n_ctx=4096`. |
| **مساعدات JSON** | `_safe_load_json`, `json_dump` — قراءة/كتابة ذرّية محمية بقفل. |
| **المعايرات (Normalizers)** | تضمن شكلاً قانونياً موحّداً للمحادثات والـ datasets ومستندات المعرفة وروابط القواعد. |
| **التدقيق والجودة** (المرحلة 1) | `_append_audit_event`, `_response_quality_metrics`. |
| **الاستخراج** | `_extract_pdf_text` (استيراد كسول لـ `pypdf`)، `_extract_docx_text`، `_extract_xlsx_text`، `_extract_csv_text`، `_extract_image_text` (OCR اختياري)، `_chunk_text_semantic` (المرحلة 2). |
| **طبقة المعرفة** (المرحلة 3) | `_retrieve_knowledge_with_trace`, `_build_dataset_knowledge_links`, `/api/knowledge/retrieve`. |
| **الاستدلال** (المرحلة 4) | `_extract_evidence_lines`, `_detect_reasoning_conflicts`, `_build_reasoning_bundle`, `_confidence_from_quality`. |
| **مستخرِجات مؤسَّسة على الدليل** | `_answer_from_knowledge` لإجابات حتمية لأسئلة "كم عدد الصفوف"، قنوات التواصل، الإغلاق، المواقع. |
| **بديل الصياغة** | `_synthesize_from_snippets` — تمرير ثانٍ بصياغة صارمة "أعد الكتابة، لا تنسخ". |
| **التوليد** | `model_reply(message, history, knowledge_docs, force_synthesis, reasoning_bundle, response_mode)`. |
| **قوالب HTML** | `INDEX_HTML`, `MONITORING_HTML`, `TRAINING_INDEX_HTML`, `ADTC_RUN_HTML`, `DATASET_RUN_HTML`. |
| **التقييم والتغذية الراجعة** (5-7) | `_load_eval_cases`, `_save_eval_cases`, `_append_learning_item`, `_capture_learning_backlog`. |

### 6. مخططات البيانات

نفس المخططات في القسم الإنجليزي 6 لكل من `knowledge_docs.json` و
`audit_log.jsonl` و `feedback_log.jsonl` و `learning_backlog.jsonl`.

### 7. مرجع واجهة الـ HTTP API

نفس الجدول في القسم الإنجليزي 7. أهم النقاط:
- `POST /api/chat` نقطة الدخول الرئيسية.
- `POST /api/training/{adtc|dataset|db}/...` لتدريب أنواع المصادر.
- `GET /api/knowledge/retrieve?q=...` لفحص جودة الاسترجاع.
- `POST /api/eval/run` لتشغيل حزمة التقييم.
- `POST /api/chat/feedback` لإرسال 👍 / 👎.
- `GET /api/learning/backlog` لقراءة قائمة الإجابات الضعيفة.
- `GET /api/ops/daily_summary` للتجميع التشغيلي اليومي.

### 8. كيف "يتعلّم" النموذج (منظومة التدريب)

أوزان Phi-3 **مجمّدة**؛ "التعلّم" هنا يعني بناء مصدر معرفة عالي
الجودة قابل للاسترجاع، يُسنَد إليه النموذج وقت الاستدلال:

1. **الرفع** — يُحفظ الملف تحت
   `training_memory/<kind>_files/<dataset_id>/`.
2. **الاستخراج** — استخراج النص:
   - PDF عبر `pypdf` (استيراد كسول).
   - DOCX عبر `python-docx`.
   - XLSX عبر `openpyxl`.
   - CSV عبر `csv` المدمج.
   - الصور عبر OCR اختياري.
   - قواعد البيانات: مخطط + صفوف عيّنة + تحليل أعمدة عبر SQLAlchemy.
3. **التقطيع الدلالي** — `_chunk_text_semantic()` يقطع النص بحسب
   العناوين والقوائم والفقرات الطبيعية مع تداخل صغير للحفاظ على السياق.
4. **الوسوم** — وسوم دلالية (طبية / جدولة / تواصل / تسعير...) لكل
   مقطع.
5. **الملخّص التنفيذي** — لكل ملف ADTC يُنشَأ مستند "Executive summary"
   يحوي حقائق رئيسية، ثقة التدريب، درجة التغطية، صفحات نص،
   وتلميحات الأقسام.
6. **الفهرسة** — تُلحَق المقاطع المعيَّرة في `knowledge_docs.json`
   مع `schema_version: 1`.
7. **تشخيص الجودة** — تخزين مؤشرات جودة الاستخراج (حروف/صفحة،
   صفحات بنص، تغطية الجداول) لتمييز الملفات الضعيفة في الواجهة.
8. **التدقيق** — كل خطوة تكتب حدث `train.*` في `audit_log.jsonl`.

### 9. كيف "يفهم" النموذج المدخلات

عند المحادثة يستلم النموذج موجِّه نظام مُحكم البناء:

1. **كتلة النظام** — قواعد الأسلوب (احترافي/عملي/واضح)، قواعد
   ADTC، السياق الزمني، قواعد إلزامية.
2. **كتلة منظومة الاستدلال** — تعليمات صريحة من 5 خطوات:
   فهم السؤال → اختيار الأدلة → حلّ التعارضات → الإجابة →
   الإقرار بنقص الأدلة عند الحاجة.
3. **كتلة الأدلة** — أقوى أسطر الأدلة من حزمة الاستدلال.
4. **كتلة المعرفة المدرَّبة** — أعلى K مقطع (حتى 5، بحد 1800 حرف
   لكل مقطع) مع عنوان كل مقطع.
5. **سجل المحادثة** — أحدث رسائل المستخدم فقط (في وضع `adtc_only`)
   أو السجل الكامل.
6. **رسالة المستخدم**.

شبكتا أمان لمنع الإجابات الضعيفة:
- **كاشف التهرّب**: في حال احتوى الرد على "As an AI..." أو
  "consult the database administrator" يُنفَّذ تمرير ثانٍ عبر
  `_synthesize_from_snippets()` لإنتاج إعادة كتابة نظيفة.
- **كاشف الرفض**: `_looks_like_refusal()` يلتقط الرفض العام
  ويوجّه الطلب للبديل التركيبي.

### 10. الخوارزميات بالتفصيل

- **درجة الاسترجاع**: تطابق رموز (`_tokenize`) + تعزيز للعناوين +
  فلتر حسب نوع المصدر. خفيف ولا يعتمد على خدمة embedding خارجية.
- **كشف التعارض**: استخراج تواريخ/ساعات/أرقام بقواعد regex عبر المقاطع
  ورفع التعارضات إلى النموذج لحلّها أو الإقرار بها.
- **الثقة**: مزيج موزون بين درجة الجودة، عدد الأدلة، عدد التعارضات،
  وعدد ضربات المعرفة → 0-100 مع شرح بشري قصير.
- **درجة الجودة**: تعاقب الإجابات القصيرة جداً وفقدان البادئة وتكرار
  عنوان ملف فقط، وتُكافئ تعدد أسطر الأدلة المتطابقة.
- **مشغّل التقييم** (`/api/eval/run`): يعيد تشغيل الحالات عبر
  `test_client` ويتحقق من السلاسل المتوقعة وحدّ الثقة، ويحفظ تقريراً.
- **قائمة التعلّم المتأخر**: أي إجابة بـ `missing_evidence` أو
  `conflict_count > 0` أو `confidence < 50` تُدرَج للمراجعة وإعادة
  التدريب لاحقاً.

### 11. الإعداد والتشغيل المحلي

#### المتطلبات
- ويندوز 10/11 (الـ wheel المرفقة لويندوز؛ على لينكس/ماك ثبّت
  `llama-cpp-python` عبر pip).
- بايثون 3.10 – 3.12.
- ~4 جيجابايت RAM متاحة لنموذج Q4.

#### التثبيت
```powershell
python -m venv .venv
.venv\Scripts\activate

pip install .\llama_cpp_python-0.3.22-py3-none-win_amd64.whl
pip install flask pypdf python-docx openpyxl sqlalchemy psutil
# اختياري للـ OCR:
pip install pytesseract pillow
```

#### التشغيل
```powershell
python .\web2.py
# → http://127.0.0.1:5000
```

#### قائمة الفحص الأولى
1. افتح `http://127.0.0.1:5000`.
2. **Training** ← ارفع PDF داخل **ADTC** ← **Start**.
3. تابع صفحة التشغيل حتى "completed".
4. ارجع للمحادثة، اختر **Knowledge mode = ADTC Only**، واسأل سؤالاً
   مرتبطاً بالملف.
5. استخدم 👍 / 👎 لتغذية قائمة التعلم المتأخرة.

#### نسخ احتياطي / استعادة
```powershell
.\scripts\backup_memory.ps1
.\scripts\restore_memory.ps1
```

#### تشغيل التقييم
```powershell
curl -X POST http://127.0.0.1:5000/api/eval/run
```

### 12. حلّ المشكلات الشائعة

| العَرَض | السبب الأرجح | الحل |
| --- | --- | --- |
| الأزرار لا تعمل في المتصفح | خطأ صياغي JS في القالب | تحقّق من `_rendered.html` + فحص `vm.Script` في Node (راجع تدفق `validate_js.ps1`) |
| خطأ استيراد `pypdf` | المكتبة غير مثبَّتة | `pip install pypdf` |
| الإجابة تعرض اسم الملف فقط أو نص خام طويل | استخراج ضعيف | أعد تشغيل تدريب ADTC وراجع درجة التغطية في الملخص التنفيذي |
| إجابات تهرّبية ("راجع العيادة") | لم يُفعَّل البديل التركيبي | تحقّق من علامات `_looks_like_refusal` وتأكد من `knowledge_mode = adtc_only` |
| خطأ MySQL `ONLY_FULL_GROUP_BY` | أعمدة غير مجمّعة في الاستعلام | أضف الأعمدة المفقودة لـ GROUP BY |

---

> **رخصة:** الاستخدام الداخلي فقط ضمن مساحة العمل الحالية. أوزان نموذج
> Phi-3 خاضعة لرخصة Microsoft الخاصة بها.
