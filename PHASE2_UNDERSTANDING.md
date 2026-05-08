# Phase 2 - Understanding and Extraction Layer

Status: Completed

## Phase Goal
Improve document understanding quality by introducing semantic chunking, richer metadata tags, extraction diagnostics, and per-file executive summaries.

## Tasks
1. Semantic chunking instead of raw fixed-size splitting.
2. Add semantic tags and section labels to chunks.
3. Add extraction quality diagnostics (duplicates/noise/sparsity).
4. Generate executive summary doc per trained ADTC file.
5. Expose richer diagnostics in ADTC learned details/results.

---

## Task 1 - Semantic Chunking
Status: Completed

Implemented in web2.py:
- Added `_chunk_text_semantic`.
- Preserves paragraph blocks and boundaries better than plain fixed-size split.

## Task 2 - Semantic Tags and Section Labels
Status: Completed

Implemented in web2.py:
- Added `_extract_semantic_tags`.
- Added `_infer_chunk_section_label`.
- Stored `section` and `semantic_tags` in chunk metadata during ingestion.

## Task 3 - Extraction Quality Diagnostics
Status: Completed

Implemented in web2.py:
- Added `_build_extraction_quality_report`.
- Tracks duplicate ratio, noise ratio, sparse structure, and low-text flags.

## Task 4 - ADTC Executive Summary
Status: Completed

Implemented in web2.py:
- Added per-file knowledge document `ADTC Executive Summary: <file>`.
- Includes key facts, topic, confidence, and section hints.

## Task 5 - Richer ADTC Diagnostics
Status: Completed

Implemented in web2.py:
- Added extraction quality fields into `learned_details` and `results`.
- Added extraction diagnostics into ADTC understanding document text.
