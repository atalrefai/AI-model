# Phase 4 - Reasoning and Decision Layer

Status: Completed

## Phase Goal
Enable real multi-step reasoning over trained knowledge with evidence grounding, conflict awareness, and transparent reasoning output.

## Tasks
1. Build evidence extraction for query-aligned reasoning.
2. Add conflict detection over retrieved evidence.
3. Build reasoning bundle object for each chat request.
4. Enforce reasoning pipeline instructions in model generation.
5. Return reasoning metadata and log conflicts for observability.

---

## Task 1 - Evidence Extraction
Status: Completed

Implemented in web2.py:
- Added `_extract_evidence_lines(query, docs)`.
- Scores evidence lines by token overlap + numeric/schedule cues.

## Task 2 - Conflict Detection
Status: Completed

Implemented in web2.py:
- Added `_detect_reasoning_conflicts(evidence)`.
- Detects open/closed schedule conflicts and numeric key conflicts.

## Task 3 - Reasoning Bundle
Status: Completed

Implemented in web2.py:
- Added `_build_reasoning_bundle(query, docs)`.
- Includes evidence count, conflict count, and missing-evidence indicator.

## Task 4 - Generation Guardrails
Status: Completed

Implemented in web2.py:
- Extended `model_reply(..., reasoning_bundle=...)`.
- Injects mandatory reasoning pipeline block into system prompt.
- Appends compact evidence block when model reply lacks explicit evidence markers.

## Task 5 - API + Audit Integration
Status: Completed

Implemented in web2.py:
- `/api/chat` now computes and returns `reasoning` metadata.
- `chat.reply` audit event now includes `reasoning_conflicts`.
- ADTC-only and normal knowledge paths both use reasoning bundle.
