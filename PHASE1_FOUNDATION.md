# Phase 1 - Foundation and Governance

Status: Completed

## Phase Goal
Build a stable foundation before advanced ADTC reasoning work, including scope, data structure, quality baseline, observability, and backup policy.

## Tasks
1. Define product scope and boundaries.
2. Unify data model and add schema versioning.
3. Define response quality baseline and measurable KPIs.
4. Define operation logging policy (train/update/delete/fail).
5. Define backup and restore policy.

---

## Task 1 - Product Scope and Boundaries
Status: Completed

### In-Scope (must support)
- Local chat with Phi-3 model.
- Knowledge-augmented answers from trained memory.
- ADTC file training (PDF, DOCX, XLSX, TXT, CSV, JSON, JSONL).
- Dataset and ADTC lifecycle operations: upload, train, view progress, delete.
- Monitoring page for request and system metrics.
- Grounded answering from trained content.

### Out-of-Scope (for now)
- Internet live retrieval.
- Direct model weight fine-tuning.
- Multi-user authentication and role-based access control.
- Distributed training orchestration.
- Cloud object storage.

### Scope Rules
- All answers must prioritize trained memory first.
- If evidence is missing, the assistant should state the limitation briefly.
- Deletion must remove both file storage and linked knowledge records.

---

## Task 2 - Data Model Unification
Status: Completed

Planned implementation:
- Add `schema_version` to persisted entities (chat, dataset, adtc_dataset, knowledge_doc, db_link).
- Add migration-safe normalizers that auto-fill missing fields.
- Standardize timestamps (`created_at`, `updated_at`, `saved_at`, `last_trained_at`) and id fields.
- Add one helper to validate and normalize records before write.

Implementation notes:
- Implemented in web2.py.
- Added global `SCHEMA_VERSION = 1`.
- Added normalizers for chat and knowledge docs (`_normalize_chat`, `_normalize_knowledge_doc`).
- Extended existing normalizers to include `schema_version`, `created_at`, `updated_at`.
- Ensured `_load_*` functions normalize records on read.
- Ensured `_persist_*` functions normalize and refresh `updated_at` on write.

---

## Task 3 - Response Quality Baseline
Status: Completed

Implementation notes:
- Added `_response_quality_metrics(query, reply, knowledge_docs)` in web2.py.
- Added quality metadata to `/api/chat` responses under `quality`.
- Metrics include: `score`, `char_count`, `sentence_count`, `bullet_count`, `grounded_prefix`, `knowledge_hits`.

## Task 4 - Operation Logging Policy
Status: Completed

Implementation notes:
- Added JSONL audit logger (`_append_audit_event`) writing to `training_memory/audit_log.jsonl`.
- Added audit events for chat requests/replies/errors.
- Added audit events for dataset upload/delete/start/finish.
- Added audit events for ADTC upload/delete/start/finish.

## Task 5 - Backup and Restore Policy
Status: Completed

Implementation notes:
- Added backup script: `scripts/backup_memory.ps1`.
- Added restore script: `scripts/restore_memory.ps1`.
- Backup covers both `chat_memory` and `training_memory`.
