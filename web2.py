import os
import re
import ast
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone, timedelta
from copy import deepcopy
from collections import deque, Counter
from pathlib import Path

try:
  from zoneinfo import ZoneInfo
except Exception:
  ZoneInfo = None

from flask import Flask, jsonify, render_template_string, request
from llama_cpp import Llama

try:
  import psutil
  _PSUTIL_AVAILABLE = True
  _PROC = psutil.Process(os.getpid())
  # Prime cpu_percent so subsequent calls return a non-zero delta.
  try:
    _PROC.cpu_percent(interval=None)
    psutil.cpu_percent(interval=None)
  except Exception:
    pass
except Exception:
  psutil = None
  _PSUTIL_AVAILABLE = False
  _PROC = None

try:
  from sqlalchemy import create_engine, inspect, text
  from sqlalchemy.exc import SQLAlchemyError
  SQLALCHEMY_AVAILABLE = True
except Exception:
  SQLALCHEMY_AVAILABLE = False
  create_engine = None
  inspect = None
  text = None
  SQLAlchemyError = Exception

app = Flask(__name__)

MODEL_PATH = "Phi-3-mini-4k-instruct-q4.gguf"
MODEL_CTX = 4096
MODEL_THREADS = max(1, os.cpu_count() or 4)
CHAT_STORE_DIR = Path(__file__).resolve().parent / "chat_memory"
CHAT_STORE_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_STORE_DIR = Path(__file__).resolve().parent / "training_memory"
TRAINING_STORE_DIR.mkdir(parents=True, exist_ok=True)
KNOWLEDGE_DOCS_PATH = TRAINING_STORE_DIR / "knowledge_docs.json"
DB_LINKS_PATH = TRAINING_STORE_DIR / "db_links.json"
DATASETS_PATH = TRAINING_STORE_DIR / "datasets.json"
DATASET_FILES_DIR = TRAINING_STORE_DIR / "dataset_files"
DATASET_FILES_DIR.mkdir(parents=True, exist_ok=True)
ADTC_DATASETS_PATH = TRAINING_STORE_DIR / "adtc_datasets.json"
ADTC_FILES_DIR = TRAINING_STORE_DIR / "adtc_files"
ADTC_FILES_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_LOG_PATH = TRAINING_STORE_DIR / "audit_log.jsonl"
FEEDBACK_LOG_PATH = TRAINING_STORE_DIR / "feedback_log.jsonl"
LEARNING_BACKLOG_PATH = TRAINING_STORE_DIR / "learning_backlog.jsonl"
EVAL_CASES_PATH = TRAINING_STORE_DIR / "eval_cases.json"

try:
  PdfReader = __import__("pypdf").PdfReader
  _PDF_AVAILABLE = True
except Exception:
  PdfReader = None
  _PDF_AVAILABLE = False

try:
  Document = __import__("docx").Document
  _DOCX_AVAILABLE = True
except Exception:
  Document = None
  _DOCX_AVAILABLE = False

try:
  load_workbook = __import__("openpyxl").load_workbook
  _XLSX_AVAILABLE = True
except Exception:
  load_workbook = None
  _XLSX_AVAILABLE = False

STOPWORDS = {
  "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "i", "in", "is", "it",
  "of", "on", "or", "that", "the", "this", "to", "we", "what", "when", "where", "which", "who", "why", "with",
  "about", "according", "trained", "snippet", "snippets", "database", "db", "table", "tables",
}

RUNTIME_CONFIG = {
    "max_tokens": None,
    "temperature": 0.7,
    "top_p": 0.95,
    "stop": ["<|end|>", "<|user|>", "<|assistant|>"],
}

SCHEMA_VERSION = 1

_config_lock = threading.Lock()
_chat_lock = threading.Lock()
_monitor_lock = threading.Lock()
_training_lock = threading.Lock()
_audit_lock = threading.Lock()

MONITOR_STATE = {
  "current": {
    "active": False,
    "request_id": None,
    "chat_id": None,
    "stage": "idle",
    "started_at": None,
  },
  "recent": deque(maxlen=120),
  "totals": {
    "requests": 0,
    "errors": 0,
    "total_ms": 0.0,
    "infer_ms": 0.0,
    "tokens_out": 0,
  },
}

llm = Llama(
    model_path=MODEL_PATH,
  n_ctx=MODEL_CTX,
  n_threads=MODEL_THREADS,
    verbose=False,
)


def _now_ts() -> int:
    return int(time.time())


def _live_time_context() -> dict:
    now_local = datetime.now().astimezone()
    now_utc = datetime.now(timezone.utc)

    weekday_en = now_local.strftime("%A")
    weekday_ar_map = {
      "Monday": "الاثنين",
      "Tuesday": "الثلاثاء",
      "Wednesday": "الأربعاء",
      "Thursday": "الخميس",
      "Friday": "الجمعة",
      "Saturday": "السبت",
      "Sunday": "الأحد",
    }
    weekday_ar = weekday_ar_map.get(weekday_en, weekday_en)

    world_times = []
    city_zones = [
      ("Riyadh", "Asia/Riyadh"),
      ("Dubai", "Asia/Dubai"),
      ("London", "Europe/London"),
      ("New York", "America/New_York"),
      ("Tokyo", "Asia/Tokyo"),
    ]
    if ZoneInfo is not None:
      for city, tz_name in city_zones:
        try:
          dt = datetime.now(ZoneInfo(tz_name))
          world_times.append({
            "city": city,
            "tz": tz_name,
            "iso": dt.isoformat(timespec="seconds"),
            "date": dt.strftime("%Y-%m-%d"),
            "time": dt.strftime("%H:%M:%S"),
            "weekday_en": dt.strftime("%A"),
          })
        except Exception:
          continue

    return {
      "local": {
        "iso": now_local.isoformat(timespec="seconds"),
        "date": now_local.strftime("%Y-%m-%d"),
        "time": now_local.strftime("%H:%M:%S"),
        "weekday_en": weekday_en,
        "weekday_ar": weekday_ar,
        "tz": str(now_local.tzinfo) if now_local.tzinfo else "local",
      },
      "utc": {
        "iso": now_utc.isoformat(timespec="seconds"),
        "date": now_utc.strftime("%Y-%m-%d"),
        "time": now_utc.strftime("%H:%M:%S"),
        "weekday_en": now_utc.strftime("%A"),
        "tz": "UTC",
      },
      "world": world_times,
    }


def _is_time_question(text: str) -> bool:
    q = (text or "").strip().lower()
    if not q:
      return False
    ar_keys = [
      "ما هو اليوم", "ما اليوم", "اليوم ايه", "اليوم اي", "التاريخ", "كم الساعة", "الوقت الآن", "الوقت الان", "الساعة الآن", "الساعة الان",
      "غد", "غدا", "غدًا", "بكرة", "بكرا", "بعد غد", "امس", "أمس",
    ]
    en_keys = [
      "what day", "what is today", "today date", "current date", "current time", "what time", "date today",
      "tomorrow", "yesterday", "day after tomorrow",
    ]
    return any(k in q for k in ar_keys) or any(k in q for k in en_keys)


def _build_time_answer(query: str) -> str:
    ctx = _live_time_context()
    local = ctx["local"]
    utc = ctx["utc"]
    is_ar = any("\u0600" <= ch <= "\u06FF" for ch in (query or ""))
    q_low = (query or "").lower()

    weekday_ar_map = {
      "Monday": "الاثنين",
      "Tuesday": "الثلاثاء",
      "Wednesday": "الأربعاء",
      "Thursday": "الخميس",
      "Friday": "الجمعة",
      "Saturday": "السبت",
      "Sunday": "الأحد",
    }

    offset_days = 0
    if "day after tomorrow" in q_low or "بعد غد" in q_low:
      offset_days = 2
    elif "tomorrow" in q_low or any(k in q_low for k in ["غد", "غدا", "غدًا", "بكرة", "بكرا"]):
      offset_days = 1
    elif "yesterday" in q_low or any(k in q_low for k in ["امس", "أمس"]):
      offset_days = -1

    now_local_dt = datetime.now().astimezone()
    target_dt = now_local_dt + timedelta(days=offset_days)
    target_weekday_en = target_dt.strftime("%A")
    target_weekday_ar = weekday_ar_map.get(target_weekday_en, target_weekday_en)
    target_date = target_dt.strftime("%Y-%m-%d")

    if offset_days != 0:
      if is_ar:
        if offset_days == 1:
          label = "غدًا"
        elif offset_days == 2:
          label = "بعد غد"
        else:
          label = "أمس"
        return "\n".join([
          "من الوقت الفعلي الحالي:",
          f"- {label}: {target_weekday_ar}",
          f"- التاريخ: {target_date}",
          f"- الوقت المحلي الآن: {local['time']} ({local['tz']})",
        ])

      label_en = "Tomorrow" if offset_days == 1 else ("Day after tomorrow" if offset_days == 2 else "Yesterday")
      return "\n".join([
        "From live current time:",
        f"- {label_en}: {target_weekday_en}",
        f"- Date: {target_date}",
        f"- Local time now: {local['time']} ({local['tz']})",
      ])

    if is_ar:
      lines = [
        "من الوقت الفعلي الحالي:",
        f"- اليوم: {local['weekday_ar']}",
        f"- التاريخ المحلي: {local['date']}",
        f"- الوقت المحلي: {local['time']} ({local['tz']})",
        f"- UTC: {utc['date']} {utc['time']} ({utc['tz']})",
      ]
      if ctx["world"]:
        lines.append("- أوقات عالمية:")
        for w in ctx["world"][:5]:
          lines.append(f"  - {w['city']}: {w['date']} {w['time']} ({w['tz']})")
      return "\n".join(lines)

    lines = [
      "From live current time:",
      f"- Today: {local['weekday_en']}",
      f"- Local date: {local['date']}",
      f"- Local time: {local['time']} ({local['tz']})",
      f"- UTC: {utc['date']} {utc['time']} ({utc['tz']})",
    ]
    if ctx["world"]:
      lines.append("- World clocks:")
      for w in ctx["world"][:5]:
        lines.append(f"  - {w['city']}: {w['date']} {w['time']} ({w['tz']})")
    return "\n".join(lines)


def _default_title() -> str:
    return "New chat"


def _chat_path(chat_id: str) -> Path:
    if not chat_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for ch in chat_id):
        raise ValueError("invalid chat id")
    return CHAT_STORE_DIR / f"{chat_id}.json"


def _new_chat_id() -> str:
    return uuid.uuid4().hex[:12]


def _normalize_chat(chat: dict) -> dict:
  now = _now_ts()
  if not chat.get("id"):
    chat["id"] = _new_chat_id()
  chat.setdefault("schema_version", SCHEMA_VERSION)
  chat.setdefault("title", _default_title())
  chat.setdefault("messages", [])
  chat.setdefault("created_at", now)
  chat.setdefault("updated_at", chat.get("created_at", now))
  return chat


def _derive_title(messages: list[dict]) -> str:
    for m in messages:
        if m.get("role") == "user":
            text = str(m.get("content", "")).replace("\n", " ").strip()
            if not text:
                continue
            return (text[:36] + "...") if len(text) > 36 else text
    return _default_title()


def _read_chat(chat_id: str):
  path = _chat_path(chat_id)
  if not path.exists():
    return None
  try:
    data = json_load(path)
    if isinstance(data, dict):
      return _normalize_chat(data)
    return None
  except Exception:
    return None


def _write_chat(chat: dict):
    path = _chat_path(chat["id"])
    json_dump(path, chat)


def _delete_chat(chat_id: str):
    path = _chat_path(chat_id)
    if path.exists():
        path.unlink()


def _chat_summary(chat: dict) -> dict:
    return {
        "id": chat["id"],
        "title": chat.get("title", _default_title()),
        "updated_at": chat.get("updated_at", 0),
        "message_count": len(chat.get("messages", [])),
    }


def _list_chats() -> list[dict]:
    chats = []
    for path in CHAT_STORE_DIR.glob("*.json"):
        try:
            chat = json_load(path)
            if not isinstance(chat, dict) or "id" not in chat:
                continue
            chats.append(_chat_summary(chat))
        except Exception:
            continue
    chats.sort(key=lambda x: x.get("updated_at", 0), reverse=True)
    return chats


def _create_chat(title: str | None = None) -> dict:
    now = _now_ts()
    chat = {
        "id": _new_chat_id(),
    "schema_version": SCHEMA_VERSION,
        "title": title or _default_title(),
        "messages": [],
        "created_at": now,
        "updated_at": now,
    }
    _write_chat(chat)
    return chat


def json_load(path: Path):
    return __import__("json").loads(path.read_text(encoding="utf-8"))


def json_dump(path: Path, data: dict):
    path.write_text(__import__("json").dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")


def _append_audit_event(event_type: str, payload: dict | None = None):
  event = {
    "ts": _now_ts(),
    "event": str(event_type or "unknown"),
    "payload": payload or {},
  }
  line = __import__("json").dumps(event, ensure_ascii=True)
  with _audit_lock:
    with AUDIT_LOG_PATH.open("a", encoding="utf-8") as f:
      f.write(line + "\n")


def _append_jsonl(path: Path, obj: dict):
  line = __import__("json").dumps(obj, ensure_ascii=True)
  with _audit_lock:
    with path.open("a", encoding="utf-8") as f:
      f.write(line + "\n")


def _read_jsonl(path: Path, max_lines: int = 500) -> list[dict]:
  if not path.exists():
    return []
  try:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
  except Exception:
    return []
  out = []
  for ln in lines[-max_lines:]:
    ln = ln.strip()
    if not ln:
      continue
    try:
      obj = __import__("json").loads(ln)
    except Exception:
      continue
    if isinstance(obj, dict):
      out.append(obj)
  return out


def _load_eval_cases() -> list[dict]:
  data = _safe_load_json(EVAL_CASES_PATH, {"cases": []})
  cases = data.get("cases", []) if isinstance(data, dict) else []
  if not isinstance(cases, list):
    return []
  out = []
  for i, c in enumerate(cases):
    if not isinstance(c, dict):
      continue
    q = str(c.get("question") or "").strip()
    if not q:
      continue
    out.append({
      "id": str(c.get("id") or f"case_{i+1}"),
      "question": q,
      "expect_keywords": [str(x).strip().lower() for x in (c.get("expect_keywords") or []) if str(x).strip()],
      "knowledge_mode": str(c.get("knowledge_mode") or "all").strip().lower() or "all",
    })
  return out


def _save_eval_cases(cases: list[dict]):
  json_dump(EVAL_CASES_PATH, {"cases": cases})


def _build_reply_confidence(quality: dict, reasoning: dict, knowledge_docs: list[dict]) -> dict:
  qscore = int(quality.get("score") or 0)
  evidence_count = int(reasoning.get("evidence_count") or 0)
  conflict_count = int(reasoning.get("conflict_count") or 0)
  knowledge_hits = len(knowledge_docs or [])
  score = qscore
  score += min(15, evidence_count * 3)
  score += min(10, knowledge_hits * 2)
  score -= min(20, conflict_count * 8)
  if reasoning.get("missing_evidence"):
    score -= 15
  score = max(0, min(100, score))
  explanation = []
  explanation.append(f"Quality baseline: {qscore}/100")
  explanation.append(f"Evidence lines used: {evidence_count}")
  explanation.append(f"Knowledge hits: {knowledge_hits}")
  if conflict_count:
    explanation.append(f"Conflict signals detected: {conflict_count}")
  return {
    "score": score,
    "explanation": "; ".join(explanation),
  }


def _build_sources_payload(knowledge_docs: list[dict], limit: int = 5) -> list[dict]:
  out = []
  for d in (knowledge_docs or [])[:limit]:
    meta = d.get("meta") if isinstance(d.get("meta"), dict) else {}
    out.append({
      "id": d.get("id"),
      "title": d.get("title"),
      "source": d.get("source"),
      "dataset_id": meta.get("dataset_id"),
      "file": meta.get("file") or meta.get("filename"),
      "kind": meta.get("kind"),
    })
  return out


def _append_learning_item(kind: str, payload: dict):
  item = {
    "ts": _now_ts(),
    "kind": str(kind or "general"),
    "payload": payload or {},
  }
  _append_jsonl(LEARNING_BACKLOG_PATH, item)


def _response_quality_metrics(query: str, reply: str, knowledge_docs: list[dict] | None = None) -> dict:
  text = (reply or "").strip()
  lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
  bullet_lines = [ln for ln in lines if ln.startswith("-") or ln.startswith("*")]
  sentence_count = max(1, len(re.findall(r"[.!?\u061F]+", text))) if text else 0
  char_count = len(text)
  has_grounded_prefix = text.lower().startswith("from trained") or text.startswith("من ")
  mentions_limitation = any(k in text.lower() for k in ("i couldn't find", "not found", "missing", "لم أجد", "غير متوفر"))
  # Baseline score for practical, structured, grounded responses.
  score = 50
  if has_grounded_prefix:
    score += 15
  if bullet_lines:
    score += 10
  if 180 <= char_count <= 1400:
    score += 10
  if sentence_count >= 2:
    score += 5
  if mentions_limitation:
    score += 5
  if knowledge_docs:
    score += min(10, len(knowledge_docs) * 2)
  score = max(0, min(100, score))
  return {
    "score": score,
    "char_count": char_count,
    "sentence_count": sentence_count,
    "bullet_count": len(bullet_lines),
    "grounded_prefix": has_grounded_prefix,
    "knowledge_hits": len(knowledge_docs or []),
  }


def _safe_load_json(path: Path, default):
  if not path.exists():
    return deepcopy(default)
  try:
    return json_load(path)
  except Exception:
    return deepcopy(default)


def _load_knowledge_docs() -> list[dict]:
  data = _safe_load_json(KNOWLEDGE_DOCS_PATH, {"docs": []})
  docs = data.get("docs", []) if isinstance(data, dict) else []
  if not isinstance(docs, list):
    return []
  out = []
  for d in docs:
    if isinstance(d, dict):
      out.append(_normalize_knowledge_doc(d))
  return out


def _save_knowledge_docs(docs: list[dict]):
  json_dump(KNOWLEDGE_DOCS_PATH, {"docs": docs})


def _load_db_links() -> list[dict]:
  data = _safe_load_json(DB_LINKS_PATH, {"links": []})
  links = data.get("links", []) if isinstance(data, dict) else []
  if not isinstance(links, list):
    return []
  out = []
  for link in links:
    if isinstance(link, dict):
      out.append(_normalize_link(link))
  return out


def _save_db_links(links: list[dict]):
  json_dump(DB_LINKS_PATH, {"links": links})


def _load_datasets() -> list[dict]:
  data = _safe_load_json(DATASETS_PATH, {"datasets": []})
  items = data.get("datasets", []) if isinstance(data, dict) else []
  if not isinstance(items, list):
    return []
  out = []
  for ds in items:
    if isinstance(ds, dict):
      out.append(_normalize_dataset(ds))
  return out


def _save_datasets(items: list[dict]):
  json_dump(DATASETS_PATH, {"datasets": items})


def _new_dataset_id() -> str:
  return uuid.uuid4().hex[:10]


def _normalize_dataset(ds: dict) -> dict:
  now = _now_ts()
  if not ds.get("id"):
    ds["id"] = _new_dataset_id()
  ds.setdefault("schema_version", SCHEMA_VERSION)
  ds.setdefault("name", "Dataset")
  ds.setdefault("files", [])
  ds.setdefault("status", "saved")
  ds.setdefault("saved_at", now)
  ds.setdefault("created_at", ds.get("saved_at", now))
  ds.setdefault("updated_at", ds.get("created_at", now))
  ds.setdefault("last_trained_at", None)
  ds.setdefault("last_error", None)
  ds.setdefault("last_added_docs", 0)
  ds.setdefault("last_results", [])
  ds.setdefault("doc_ids", [])
  return ds


def _public_dataset(ds: dict) -> dict:
  return {
    "id": ds.get("id"),
    "name": ds.get("name"),
    "files": [
      {"name": f.get("name"), "size_bytes": f.get("size_bytes", 0), "kind": f.get("kind", "")}
      for f in ds.get("files", [])
    ],
    "file_count": len(ds.get("files", [])),
    "total_size": sum(int(f.get("size_bytes", 0) or 0) for f in ds.get("files", [])),
    "status": ds.get("status", "saved"),
    "saved_at": ds.get("saved_at"),
    "last_trained_at": ds.get("last_trained_at"),
    "last_added_docs": ds.get("last_added_docs", 0),
    "last_error": ds.get("last_error"),
    "last_results": ds.get("last_results", []),
  }


def _find_dataset(dataset_id: str) -> dict | None:
  with _training_lock:
    items = _load_datasets()
    for ds in items:
      if ds.get("id") == dataset_id:
        _normalize_dataset(ds)
        return ds
  return None


def _persist_dataset(updated: dict):
  with _training_lock:
    updated = _normalize_dataset(updated)
    updated["updated_at"] = _now_ts()
    items = _load_datasets()
    found = False
    for i, ds in enumerate(items):
      _normalize_dataset(ds)
      if ds.get("id") == updated.get("id"):
        items[i] = updated
        found = True
        break
    if not found:
      items.append(updated)
    _save_datasets(items)


def _delete_dataset_storage(dataset_id: str):
  ds_dir = DATASET_FILES_DIR / dataset_id
  if ds_dir.exists():
    try:
      for child in ds_dir.iterdir():
        try:
          child.unlink()
        except Exception:
          pass
      ds_dir.rmdir()
    except Exception:
      pass


def _load_adtc_datasets() -> list[dict]:
  data = _safe_load_json(ADTC_DATASETS_PATH, {"datasets": []})
  items = data.get("datasets", []) if isinstance(data, dict) else []
  if not isinstance(items, list):
    return []
  out = []
  for ds in items:
    if isinstance(ds, dict):
      out.append(_normalize_adtc_dataset(ds))
  return out


def _save_adtc_datasets(items: list[dict]):
  json_dump(ADTC_DATASETS_PATH, {"datasets": items})


def _new_adtc_id() -> str:
  return "adtc_" + uuid.uuid4().hex[:10]


def _normalize_adtc_dataset(ds: dict) -> dict:
  now = _now_ts()
  if not ds.get("id"):
    ds["id"] = _new_adtc_id()
  ds.setdefault("schema_version", SCHEMA_VERSION)
  ds.setdefault("name", "ADTC Dataset")
  ds.setdefault("topic", "general")
  ds.setdefault("files", [])
  ds.setdefault("status", "saved")
  ds.setdefault("saved_at", now)
  ds.setdefault("created_at", ds.get("saved_at", now))
  ds.setdefault("updated_at", ds.get("created_at", now))
  ds.setdefault("last_trained_at", None)
  ds.setdefault("last_error", None)
  ds.setdefault("last_added_docs", 0)
  ds.setdefault("last_results", [])
  ds.setdefault("doc_ids", [])
  ds.setdefault("relationships", [])
  ds.setdefault("insights", [])
  ds.setdefault("learned_details", [])
  ds.setdefault("integration_report", {})
  return ds


def _public_adtc_dataset(ds: dict) -> dict:
  return {
    "id": ds.get("id"),
    "name": ds.get("name"),
    "topic": ds.get("topic", "general"),
    "files": [
      {
        "name": f.get("name"),
        "size_bytes": f.get("size_bytes", 0),
        "kind": f.get("kind", ""),
      }
      for f in ds.get("files", [])
    ],
    "file_count": len(ds.get("files", [])),
    "total_size": sum(int(f.get("size_bytes", 0) or 0) for f in ds.get("files", [])),
    "status": ds.get("status", "saved"),
    "saved_at": ds.get("saved_at"),
    "last_trained_at": ds.get("last_trained_at"),
    "last_added_docs": ds.get("last_added_docs", 0),
    "last_error": ds.get("last_error"),
    "last_results": ds.get("last_results", []),
    "relationships": ds.get("relationships", []),
    "insights": ds.get("insights", []),
    "learned_details": ds.get("learned_details", []),
    "integration_report": ds.get("integration_report", {}),
  }


def _find_adtc_dataset(dataset_id: str) -> dict | None:
  with _training_lock:
    items = _load_adtc_datasets()
    for ds in items:
      if ds.get("id") == dataset_id:
        _normalize_adtc_dataset(ds)
        return ds
  return None


def _persist_adtc_dataset(updated: dict):
  with _training_lock:
    updated = _normalize_adtc_dataset(updated)
    updated["updated_at"] = _now_ts()
    items = _load_adtc_datasets()
    found = False
    for i, ds in enumerate(items):
      _normalize_adtc_dataset(ds)
      if ds.get("id") == updated.get("id"):
        items[i] = updated
        found = True
        break
    if not found:
      items.append(updated)
    _save_adtc_datasets(items)


def _delete_adtc_storage(dataset_id: str):
  ds_dir = ADTC_FILES_DIR / dataset_id
  if ds_dir.exists():
    try:
      shutil.rmtree(ds_dir, ignore_errors=True)
    except Exception:
      pass


def _delete_adtc_related_knowledge(dataset_id: str, doc_ids: list[str] | None = None) -> int:
  wanted_doc_ids = {str(x) for x in (doc_ids or []) if x}
  removed = 0
  with _training_lock:
    docs = _load_knowledge_docs()
    kept = []
    for d in docs:
      src = str(d.get("source", "")).strip().lower()
      meta = d.get("meta") if isinstance(d.get("meta"), dict) else {}
      meta_dataset_id = str(meta.get("dataset_id") or "").strip()
      doc_id = str(d.get("id") or "").strip()
      should_remove = (
        src == "adtc"
        and (
          meta_dataset_id == dataset_id
          or (doc_id and doc_id in wanted_doc_ids)
        )
      )
      if should_remove:
        removed += 1
      else:
        kept.append(d)
    if removed:
      _save_knowledge_docs(kept)
  return removed


def _norm_key(text_value: str) -> str:
  return re.sub(r"[^a-z0-9\u0600-\u06FF]+", "", (text_value or "").strip().lower())


def _analyze_relationships(file_profiles: list[dict]) -> list[dict]:
  rels = []
  for i in range(len(file_profiles)):
    for j in range(i + 1, len(file_profiles)):
      a = file_profiles[i]
      b = file_profiles[j]
      a_cols = set(a.get("schema_keys", []))
      b_cols = set(b.get("schema_keys", []))
      shared = sorted(a_cols.intersection(b_cols))
      if len(shared) >= 2:
        rels.append({
          "from": a.get("file"),
          "to": b.get("file"),
          "shared_keys": shared[:12],
          "score": len(shared),
        })
  rels.sort(key=lambda x: x.get("score", 0), reverse=True)
  return rels[:25]


def _extract_pdf_payload(file_path: Path) -> dict:
  global PdfReader, _PDF_AVAILABLE
  if not _PDF_AVAILABLE or PdfReader is None:
    try:
      PdfReader = __import__("pypdf").PdfReader
      _PDF_AVAILABLE = True
    except Exception:
      raise ValueError("PDF support requires package 'pypdf' (pip install pypdf)")
  reader = PdfReader(str(file_path))
  pages = []
  page_char_counts = []
  page_samples = []
  for idx, p in enumerate(reader.pages, start=1):
    try:
      txt = (p.extract_text() or "").strip()
      pages.append(txt)
      page_char_counts.append(len(txt))
      sample = re.sub(r"\s+", " ", txt)[:220]
      page_samples.append({"page": idx, "chars": len(txt), "sample": sample})
    except Exception:
      pages.append("")
      page_char_counts.append(0)
      page_samples.append({"page": idx, "chars": 0, "sample": ""})
  merged = "\n\n".join(p for p in pages if p)
  pages_total = len(pages)
  pages_with_text = sum(1 for c in page_char_counts if c > 0)
  low_text_pages = sum(1 for c in page_char_counts if c < 40)
  avg_chars = int(sum(page_char_counts) / pages_total) if pages_total else 0
  return {
    "kind": "pdf",
    "text": merged,
    "schema_keys": [],
    "units": len(pages),
    "pages_total": pages_total,
    "pages_with_text": pages_with_text,
    "pages_low_text": low_text_pages,
    "avg_chars_per_page": avg_chars,
    "page_char_counts": page_char_counts[:80],
    "page_samples": page_samples[:20],
  }


def _extract_contact_signals(text_value: str) -> dict:
  text_value = text_value or ""
  urls = []
  emails = []
  phones = []
  for u in re.findall(r"https?://[^\s)]+", text_value, flags=re.IGNORECASE):
    if u not in urls:
      urls.append(u)
  for e in re.findall(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", text_value, flags=re.IGNORECASE):
    if e not in emails:
      emails.append(e)
  for p in re.findall(r"\+?\d[\d\s\-()]{6,}\d", text_value):
    cleaned = re.sub(r"\s+", " ", p).strip()
    if cleaned not in phones:
      phones.append(cleaned)
  return {
    "urls": urls[:12],
    "emails": emails[:12],
    "phones": phones[:12],
  }


def _extract_key_facts(text_value: str, limit: int = 8) -> list[str]:
  if not text_value:
    return []
  candidates = []
  lines = [ln.strip() for ln in re.split(r"[\n\r]+", text_value) if ln and ln.strip()]
  boost_keywords = {
    "contact", "phone", "email", "website", "hotline", "hours", "location", "address", "service",
    "ramadan", "clinic", "hospital", "department", "guide", "help", "support",
    "التواصل", "الهاتف", "البريد", "الموقع", "العنوان", "الخدمات", "المستشفى", "العيادة",
  }

  for ln in lines:
    if len(ln) < 12:
      continue
    if len(ln) > 260:
      continue
    ll = ln.lower()
    score = 0
    if any(k in ll for k in boost_keywords):
      score += 3
    if re.search(r"https?://|@|\+?\d", ln):
      score += 3
    if re.search(r"\d", ln):
      score += 1
    if ":" in ln:
      score += 1
    weird_ratio = len(re.findall(r"[^\w\s\u0600-\u06FF:/@.()\-]", ln)) / max(1, len(ln))
    if weird_ratio > 0.25:
      score -= 2
    if score > 0:
      candidates.append((score, ln))

  candidates.sort(key=lambda x: x[0], reverse=True)
  out = []
  seen = set()
  for _, ln in candidates:
    nk = _norm_key(ln)
    if not nk or nk in seen:
      continue
    seen.add(nk)
    out.append(ln)
    if len(out) >= limit:
      break
  return out


def _build_understanding_report(raw_text: str, kind: str, payload_meta: dict | None = None) -> dict:
  raw_text = raw_text or ""
  payload_meta = payload_meta or {}
  tokens = _tokenize(raw_text)
  freq = Counter(tokens)
  token_count = len(tokens)
  unique_terms_count = len(freq)
  lexical_density = round(unique_terms_count / max(1, token_count), 3)
  digit_count = len(re.findall(r"\d", raw_text))
  digit_density = round(digit_count / max(1, len(raw_text)), 4)
  top_terms = [w for w, _ in freq.most_common(12)]
  key_facts = _extract_key_facts(raw_text, limit=8)
  contacts = _extract_contact_signals(raw_text)

  unclear = []
  if len(raw_text.strip()) < 120:
    unclear.append("Very little extractable text was found.")
  if kind == "pdf":
    low_pages = int(payload_meta.get("pages_low_text") or 0)
    pages_total = int(payload_meta.get("pages_total") or 0)
    if pages_total > 0 and low_pages >= max(1, pages_total // 3):
      unclear.append(f"{low_pages} page(s) had very low text extraction; file may contain scanned/image-only pages.")
  if not key_facts:
    unclear.append("Could not confidently extract high-value factual statements.")
  if not contacts.get("urls") and not contacts.get("emails") and not contacts.get("phones"):
    unclear.append("No explicit contact channels (URL/email/phone) were detected in extracted text.")

  summary_bits = []
  if key_facts:
    summary_bits.append(f"Extracted {len(key_facts)} key fact line(s).")
  if contacts.get("urls"):
    summary_bits.append(f"Detected {len(contacts['urls'])} website/link reference(s).")
  if contacts.get("emails"):
    summary_bits.append(f"Detected {len(contacts['emails'])} email reference(s).")
  if contacts.get("phones"):
    summary_bits.append(f"Detected {len(contacts['phones'])} phone/hotline reference(s).")
  if kind == "pdf" and payload_meta.get("pages_total") is not None:
    summary_bits.append(
      f"PDF pages: {payload_meta.get('pages_with_text', 0)}/{payload_meta.get('pages_total', 0)} with text; avg chars/page {payload_meta.get('avg_chars_per_page', 0)}."
    )

  return {
    "top_terms": top_terms,
    "key_facts": key_facts,
    "contact_signals": contacts,
    "unclear_points": unclear,
    "understanding_summary": " ".join(summary_bits) if summary_bits else "Basic text extraction completed.",
    "quality_score": max(0, min(100, 70 + len(key_facts) * 3 + len(contacts.get("urls", [])) * 2 - len(unclear) * 10)),
    "signal_metrics": {
      "char_count": len(raw_text),
      "token_count": token_count,
      "unique_terms_count": unique_terms_count,
      "lexical_density": lexical_density,
      "digit_density": digit_density,
    },
  }


def _extract_section_titles(raw_text: str, limit: int = 12) -> list[str]:
  out = []
  seen = set()
  for ln in (raw_text or "").splitlines():
    line = re.sub(r"\s+", " ", ln).strip(" -•\t")
    if len(line) < 4 or len(line) > 90:
      continue
    if not re.search(r"[A-Za-z\u0600-\u06FF]", line):
      continue
    # Heading-like candidates: short lines with title-like structure and low punctuation.
    punct = len(re.findall(r"[,:;]", line))
    if len(line.split()) <= 8 and punct <= 1:
      nk = _norm_key(line)
      if nk and nk not in seen:
        seen.add(nk)
        out.append(line)
    if len(out) >= limit:
      break
  return out


def _build_topic_coverage(raw_text: str, topic: str = "") -> dict:
  ll = (raw_text or "").lower()
  buckets = {
    "location": ["location", "address", "gate", "entrance", "abu dhabi", "uae", "شارع", "العنوان", "الموقع", "بوابة"],
    "contacts": ["phone", "email", "website", "hotline", "contact", "الهاتف", "البريد", "التواصل", "الموقع الإلكتروني"],
    "services": ["service", "clinic", "department", "surgery", "treatment", "خدمة", "عيادة", "قسم", "علاج"],
    "schedule": ["hour", "timing", "open", "appointment", "موعد", "ساعات", "دوام", "مفتوح"],
    "insurance": ["insurance", "payer", "coverage", "تامين", "تأمين"],
  }
  coverage = {}
  for k, keys in buckets.items():
    cnt = 0
    for kw in keys:
      cnt += ll.count(kw.lower())
    coverage[k] = cnt
  coverage["topic"] = topic or "general"
  coverage["coverage_score"] = min(100, sum(1 for v in coverage.values() if isinstance(v, int) and v > 0) * 20)
  return coverage


def _build_qa_probes(raw_text: str, topic: str = "") -> list[dict]:
  lines = [re.sub(r"\s+", " ", x).strip() for x in (raw_text or "").splitlines() if x and x.strip()]
  probes = [
    {
      "id": "location",
      "question": "Where is the hospital/clinic located?",
      "keywords": ["location", "address", "gate", "entrance", "abu dhabi", "uae", "العنوان", "الموقع", "بوابة"],
    },
    {
      "id": "contacts",
      "question": "How can users contact this provider?",
      "keywords": ["phone", "email", "website", "hotline", "contact", "الهاتف", "البريد", "التواصل"],
    },
    {
      "id": "services",
      "question": "What services or clinics are available?",
      "keywords": ["service", "clinic", "department", "surgery", "treatment", "خدمة", "عيادة", "قسم"],
    },
    {
      "id": "schedule",
      "question": "What are opening hours/appointment details?",
      "keywords": ["hour", "timing", "appointment", "open", "دوام", "ساعات", "موعد"],
    },
  ]
  if topic and "dental" in topic.lower():
    probes.append({
      "id": "dental_scope",
      "question": "What dental specialties are covered?",
      "keywords": ["dental", "orthodont", "prosthodont", "implant", "root canal", "أسنان", "تقويم", "زراعة"],
    })

  out = []
  for p in probes:
    evidence = []
    for ln in lines:
      ll = ln.lower()
      if any(k in ll for k in p["keywords"]):
        evidence.append(ln)
      if len(evidence) >= 2:
        break
    out.append({
      "id": p["id"],
      "question": p["question"],
      "answerable": bool(evidence),
      "confidence": 90 if len(evidence) >= 2 else (65 if len(evidence) == 1 else 25),
      "evidence": evidence,
    })
  return out


def _score_training_confidence(understand: dict, coverage: dict, qa_probes: list[dict]) -> int:
  quality = int(understand.get("quality_score") or 0)
  cov = int(coverage.get("coverage_score") or 0)
  qa_conf = 0
  if qa_probes:
    qa_conf = int(round(sum(int(p.get("confidence") or 0) for p in qa_probes) / len(qa_probes)))
  # Weighted confidence to represent how reliable this file is for question-answering.
  return max(0, min(100, int(round(quality * 0.5 + cov * 0.2 + qa_conf * 0.3))))


def _extract_docx_payload(file_path: Path) -> dict:
  if not _DOCX_AVAILABLE:
    raise ValueError("Word support requires package 'python-docx' (pip install python-docx)")
  doc = Document(str(file_path))
  paragraphs = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
  table_lines = []
  schema_keys = set()
  for t in doc.tables:
    rows = []
    for r in t.rows:
      cells = [c.text.strip() for c in r.cells]
      rows.append(cells)
    if rows:
      header = rows[0]
      for col in header:
        nk = _norm_key(col)
        if nk:
          schema_keys.add(nk)
      table_lines.append("Columns: " + ", ".join(header))
      for row in rows[1 : 1 + DATASET_ROW_CHUNK_SIZE]:
        table_lines.append("- " + ", ".join(row))
  text_blob = "\n".join(paragraphs + table_lines)
  return {
    "kind": "docx",
    "text": text_blob,
    "schema_keys": sorted(schema_keys),
    "units": len(paragraphs) + len(table_lines),
  }


def _extract_xlsx_payload(file_path: Path) -> dict:
  if not _XLSX_AVAILABLE:
    raise ValueError("Excel support requires package 'openpyxl' (pip install openpyxl)")
  wb = load_workbook(str(file_path), read_only=True, data_only=True)
  lines = []
  schema_keys = set()
  sheet_count = 0
  total_rows = 0
  for sheet in wb.worksheets:
    sheet_count += 1
    iterator = sheet.iter_rows(values_only=True)
    try:
      header_raw = next(iterator)
    except StopIteration:
      continue
    header = [str(v).strip() if v is not None else "" for v in header_raw]
    norm_header = []
    for idx, col in enumerate(header):
      cname = col or f"col_{idx+1}"
      norm_header.append(cname)
      nk = _norm_key(cname)
      if nk:
        schema_keys.add(nk)
    lines.append(f"Sheet: {sheet.title}")
    lines.append("Columns: " + ", ".join(norm_header))
    count = 0
    for row in iterator:
      vals = ["" if v is None else str(v) for v in row]
      pairs = []
      for i, val in enumerate(vals):
        if i < len(norm_header):
          pairs.append(f"{norm_header[i]}={val}")
      lines.append("- " + ", ".join(pairs))
      count += 1
      if count >= DATASET_ROW_CHUNK_SIZE:
        break
    total_rows += count
  try:
    wb.close()
  except Exception:
    pass
  return {
    "kind": "xlsx",
    "text": "\n".join(lines),
    "schema_keys": sorted(schema_keys),
    "units": max(sheet_count, total_rows),
  }


def _tokenize(text_value: str) -> list[str]:
  text_value = (text_value or "").lower()
  tokens = []
  for w in re.split(r"[^\w\u0600-\u06FF]+", text_value):
    if not w:
      continue
    if len(w) > 3 and w.endswith("s"):
      w = w[:-1]
    if len(w) < 2 or w in STOPWORDS:
      continue
    tokens.append(w)
  return tokens


def _doc_semantic_token_set(doc: dict) -> set[str]:
  title = str(doc.get("title", ""))
  content = str(doc.get("content", ""))
  meta = doc.get("meta") if isinstance(doc.get("meta"), dict) else {}
  semantic_tags = meta.get("semantic_tags", []) if isinstance(meta, dict) else []
  section = str(meta.get("section", "")) if isinstance(meta, dict) else ""
  tag_text = " ".join(str(t) for t in semantic_tags if t)
  token_pool = f"{title}\n{content}\n{section}\n{tag_text}"
  return set(_tokenize(token_pool))


def _retrieve_knowledge_with_trace(
  query: str,
  limit: int = 5,
  source_filter: str | None = None,
) -> tuple[list[dict], list[dict]]:
  q_text = (query or "").strip()
  q_tokens = set(_tokenize(q_text))
  if not q_tokens:
    return [], []

  with _training_lock:
    docs = _load_knowledge_docs()

  if source_filter:
    wanted = str(source_filter).strip().lower()
    docs = [d for d in docs if str(d.get("source", "")).strip().lower() == wanted]

  scored = []
  for doc in docs:
    tokens = _doc_semantic_token_set(doc)
    if not tokens:
      continue
    overlap_tokens = q_tokens.intersection(tokens)
    overlap = len(overlap_tokens)
    if overlap == 0:
      continue

    title = str(doc.get("title", ""))
    title_low = title.lower()
    q_low = q_text.lower()
    meta = doc.get("meta") if isinstance(doc.get("meta"), dict) else {}
    semantic_tags = [str(t).lower() for t in (meta.get("semantic_tags", []) if isinstance(meta, dict) else []) if t]

    lexical_score = overlap / max(1, len(q_tokens))
    title_boost = 0.25 if q_low and q_low in title_low else 0.0
    tag_overlap = len(set(semantic_tags).intersection(q_tokens))
    tag_boost = min(0.25, tag_overlap * 0.08)
    summary_boost = 0.1 if str(meta.get("summary_kind", "")) == "executive" else 0.0
    created_at = int(doc.get("created_at") or 0)
    age_days = max(0.0, (time.time() - created_at) / 86400.0) if created_at else 99999.0
    recency_boost = 0.08 if age_days <= 7 else (0.04 if age_days <= 30 else 0.0)

    score = lexical_score + title_boost + tag_boost + summary_boost + recency_boost
    reasons = [
      f"lexical_overlap={overlap}/{len(q_tokens)}",
      f"tag_overlap={tag_overlap}",
    ]
    if title_boost > 0:
      reasons.append("title_match")
    if summary_boost > 0:
      reasons.append("executive_summary_boost")
    if recency_boost > 0:
      reasons.append("recent_doc_boost")

    scored.append((score, doc, overlap_tokens, reasons))

  scored.sort(key=lambda x: x[0], reverse=True)
  top_docs = [item[1] for item in scored[:limit]]
  trace = []
  for score, doc, overlap_tokens, reasons in scored[: max(12, limit)]:
    trace.append({
      "doc_id": doc.get("id"),
      "title": doc.get("title"),
      "source": doc.get("source"),
      "score": round(float(score), 4),
      "matched_terms": sorted(list(overlap_tokens))[:12],
      "reasons": reasons,
    })
  return top_docs, trace


def _retrieve_knowledge(query: str, limit: int = 5, source_filter: str | None = None) -> list[dict]:
  docs, _ = _retrieve_knowledge_with_trace(query, limit=limit, source_filter=source_filter)
  return docs


def _build_dataset_knowledge_links(dataset_id: str, source: str = "adtc", limit: int = 24) -> list[dict]:
  if not dataset_id:
    return []
  with _training_lock:
    all_docs = _load_knowledge_docs()
  docs = []
  for d in all_docs:
    if str(d.get("source", "")).strip().lower() != str(source).strip().lower():
      continue
    meta = d.get("meta") if isinstance(d.get("meta"), dict) else {}
    if str(meta.get("dataset_id") or "").strip() != str(dataset_id).strip():
      continue
    docs.append(d)

  edges = []
  for i in range(len(docs)):
    for j in range(i + 1, len(docs)):
      a = docs[i]
      b = docs[j]
      am = a.get("meta") if isinstance(a.get("meta"), dict) else {}
      bm = b.get("meta") if isinstance(b.get("meta"), dict) else {}
      a_tags = {str(t).lower() for t in (am.get("semantic_tags", []) if isinstance(am, dict) else []) if t}
      b_tags = {str(t).lower() for t in (bm.get("semantic_tags", []) if isinstance(bm, dict) else []) if t}
      shared_tags = sorted(list(a_tags.intersection(b_tags)))
      a_tokens = _doc_semantic_token_set(a)
      b_tokens = _doc_semantic_token_set(b)
      shared_terms = len(a_tokens.intersection(b_tokens))
      if not shared_tags and shared_terms < 6:
        continue
      score = len(shared_tags) * 3 + min(10, shared_terms)
      edges.append({
        "from_id": a.get("id"),
        "to_id": b.get("id"),
        "from_title": a.get("title"),
        "to_title": b.get("title"),
        "shared_tags": shared_tags[:8],
        "shared_term_count": shared_terms,
        "score": score,
      })
  edges.sort(key=lambda x: x.get("score", 0), reverse=True)
  return edges[:limit]


def _append_knowledge_doc(title: str, content: str, source: str, meta: dict | None = None) -> dict:
  now = _now_ts()
  doc = {
    "id": uuid.uuid4().hex[:12],
    "schema_version": SCHEMA_VERSION,
    "title": title.strip() or "Untitled",
    "content": content.strip(),
    "source": source,
    "meta": meta or {},
    "created_at": now,
    "updated_at": now,
  }
  with _training_lock:
    docs = _load_knowledge_docs()
    docs.append(doc)
    _save_knowledge_docs(docs)
  return doc


def _normalize_knowledge_doc(doc: dict) -> dict:
  now = _now_ts()
  if not doc.get("id"):
    doc["id"] = uuid.uuid4().hex[:12]
  doc.setdefault("schema_version", SCHEMA_VERSION)
  doc.setdefault("title", "Untitled")
  doc.setdefault("content", "")
  doc.setdefault("source", "manual")
  doc.setdefault("meta", {})
  doc.setdefault("created_at", now)
  doc.setdefault("updated_at", doc.get("created_at", now))
  return doc


# =====================================================================
# Database Training: presets, URL builders, persistence, background jobs
# =====================================================================

from urllib.parse import quote_plus

DB_PRESETS = {
  "sqlite": {
    "label": "SQLite",
    "driver_pip": None,
    "fields": [
      {"key": "database", "label": "Database file path", "placeholder": "C:\\path\\to\\file.db", "required": True},
    ],
  },
  "mysql": {
    "label": "MySQL",
    "driver_pip": "pymysql",
    "fields": [
      {"key": "host", "label": "Host", "required": True, "default": "localhost"},
      {"key": "port", "label": "Port", "required": True, "default": "3306"},
      {"key": "user", "label": "Username", "required": True},
      {"key": "password", "label": "Password", "required": False, "secret": True},
      {"key": "database", "label": "Database name", "required": True},
    ],
  },
  "mariadb": {
    "label": "MariaDB",
    "driver_pip": "pymysql",
    "fields": [
      {"key": "host", "label": "Host", "required": True, "default": "localhost"},
      {"key": "port", "label": "Port", "required": True, "default": "3306"},
      {"key": "user", "label": "Username", "required": True},
      {"key": "password", "label": "Password", "required": False, "secret": True},
      {"key": "database", "label": "Database name", "required": True},
    ],
  },
  "postgresql": {
    "label": "PostgreSQL",
    "driver_pip": "psycopg2-binary",
    "fields": [
      {"key": "host", "label": "Host", "required": True, "default": "localhost"},
      {"key": "port", "label": "Port", "required": True, "default": "5432"},
      {"key": "user", "label": "Username", "required": True},
      {"key": "password", "label": "Password", "required": False, "secret": True},
      {"key": "database", "label": "Database name", "required": True},
    ],
  },
  "mssql": {
    "label": "Microsoft SQL Server",
    "driver_pip": "pyodbc",
    "fields": [
      {"key": "host", "label": "Host", "required": True, "default": "localhost"},
      {"key": "port", "label": "Port", "required": True, "default": "1433"},
      {"key": "user", "label": "Username", "required": True},
      {"key": "password", "label": "Password", "required": False, "secret": True},
      {"key": "database", "label": "Database name", "required": True},
      {"key": "odbc_driver", "label": "ODBC Driver", "required": True, "default": "ODBC Driver 17 for SQL Server"},
    ],
  },
  "oracle": {
    "label": "Oracle",
    "driver_pip": "cx_Oracle",
    "fields": [
      {"key": "host", "label": "Host", "required": True, "default": "localhost"},
      {"key": "port", "label": "Port", "required": True, "default": "1521"},
      {"key": "user", "label": "Username", "required": True},
      {"key": "password", "label": "Password", "required": False, "secret": True},
      {"key": "database", "label": "Service Name / SID", "required": True},
    ],
  },
}


def _build_url_from_preset(db_type: str, fields: dict) -> str:
  preset = DB_PRESETS.get(db_type)
  if not preset:
    raise ValueError(f"Unsupported database type: {db_type}")
  fields = fields or {}

  def get(k: str, default: str = "") -> str:
    v = fields.get(k)
    return ("" if v is None else str(v)).strip() or default

  if db_type == "sqlite":
    path = get("database")
    if not path:
      raise ValueError("Database file path is required")
    return f"sqlite:///{path}"

  user = quote_plus(get("user"))
  password_raw = get("password")
  password = quote_plus(password_raw)
  host = get("host", "localhost")
  port = get("port")
  database = get("database")
  auth = user + (":" + password if password_raw else "")
  netloc = f"{auth}@{host}" + (f":{port}" if port else "")

  if db_type == "mysql":
    return f"mysql+pymysql://{netloc}/{database}"
  if db_type == "mariadb":
    return f"mariadb+pymysql://{netloc}/{database}"
  if db_type == "postgresql":
    return f"postgresql+psycopg2://{netloc}/{database}"
  if db_type == "mssql":
    driver = quote_plus(get("odbc_driver", "ODBC Driver 17 for SQL Server"))
    return f"mssql+pyodbc://{netloc}/{database}?driver={driver}"
  if db_type == "oracle":
    return f"oracle+cx_oracle://{netloc}/?service_name={database}"

  raise ValueError(f"Unsupported database type: {db_type}")


def _mask_url(url: str) -> str:
  if not url:
    return ""
  return re.sub(r"://([^:@/]+):([^@/]+)@", r"://\1:***@", url)


def _new_link_id() -> str:
  return uuid.uuid4().hex[:10]


def _normalize_link(link: dict) -> dict:
  now = _now_ts()
  if "id" not in link or not link["id"]:
    link["id"] = _new_link_id()
  link.setdefault("schema_version", SCHEMA_VERSION)
  link.setdefault("type", "custom")
  link.setdefault("fields", {})
  link.setdefault("tables", "")
  link.setdefault("sample_rows", 5)
  link.setdefault("status", "saved")
  link.setdefault("saved_at", now)
  link.setdefault("created_at", link.get("saved_at", now))
  link.setdefault("updated_at", link.get("created_at", now))
  link.setdefault("last_trained_at", None)
  link.setdefault("last_error", None)
  link.setdefault("last_added_docs", 0)
  link.setdefault("last_table_count", 0)
  return link


def _public_link(link: dict) -> dict:
  fields = link.get("fields") or {}
  return {
    "id": link.get("id"),
    "name": link.get("name"),
    "type": link.get("type"),
    "fields": {k: ("***" if k == "password" and v else v) for k, v in fields.items()},
    "url_masked": _mask_url(link.get("url", "")),
    "tables": link.get("tables", ""),
    "sample_rows": link.get("sample_rows", 5),
    "status": link.get("status", "saved"),
    "saved_at": link.get("saved_at"),
    "last_used": link.get("last_used"),
    "last_trained_at": link.get("last_trained_at"),
    "last_error": link.get("last_error"),
    "last_added_docs": link.get("last_added_docs", 0),
    "last_table_count": link.get("last_table_count", 0),
  }


def _find_link(link_id: str) -> dict | None:
  links = _load_db_links()
  for link in links:
    _normalize_link(link)
    if link.get("id") == link_id:
      return link
  return None


def _persist_link(updated: dict):
  with _training_lock:
    updated = _normalize_link(updated)
    updated["updated_at"] = _now_ts()
    links = _load_db_links()
    out = []
    found = False
    for link in links:
      _normalize_link(link)
      if link.get("id") == updated.get("id"):
        out.append(updated)
        found = True
      else:
        out.append(link)
    if not found:
      out.append(updated)
    _save_db_links(out)


def _delete_link(link_id: str) -> bool:
  with _training_lock:
    links = _load_db_links()
    new_links = []
    removed = False
    for link in links:
      _normalize_link(link)
      if link.get("id") == link_id:
        removed = True
        continue
      new_links.append(link)
    if removed:
      _save_db_links(new_links)
  return removed


# In-memory training jobs registry: link_id -> progress dict
TRAINING_JOBS: dict = {}
_jobs_lock = threading.Lock()


def _new_job_state(link_id: str, name: str) -> dict:
  return {
    "link_id": link_id,
    "name": name,
    "status": "starting",
    "stage": "starting",
    "current_table": None,
    "tables_total": 0,
    "tables_done": 0,
    "added_docs": 0,
    "rows_ingested": 0,
    "rows_profiled": 0,
    "started_at": time.time(),
    "finished_at": None,
    "elapsed_ms": 0.0,
    "eta_ms": None,
    "percent": 0,
    "logs": [],
    "error": None,
    "stop_requested": False,
    "learnings": [],            # per-table detailed learning records
    "analysis": None,           # cross-table inferred relations + summary
    "model_integration": None,  # how it was added to the assistant
    "knowledge_base_path": str(KNOWLEDGE_DOCS_PATH),
  }


def _job_log(job: dict, message: str):
  job["logs"].append({"t": time.time(), "msg": message})
  if len(job["logs"]) > 200:
    job["logs"] = job["logs"][-200:]


def _job_update(job: dict, **kwargs):
  with _jobs_lock:
    job.update(kwargs)
    job["elapsed_ms"] = (time.time() - job["started_at"]) * 1000.0
    total = job.get("tables_total") or 0
    done = job.get("tables_done") or 0
    if total > 0:
      job["percent"] = int(min(100, max(0, (done / total) * 100)))
      if 0 < done < total:
        per = job["elapsed_ms"] / done
        job["eta_ms"] = per * (total - done)
      elif done >= total:
        job["eta_ms"] = 0
    else:
      if job["status"] in ("starting", "connecting", "listing"):
        job["percent"] = max(job.get("percent", 0), 5)


# ---------------------------------------------------------------------------
# Deep-analysis helpers used by the training job
# ---------------------------------------------------------------------------

_PATTERN_REGEX = {
  "email": re.compile(r"^[\w.+-]+@[\w-]+\.[\w.-]+$"),
  "url": re.compile(r"^https?://", re.I),
  "phone": re.compile(r"^[+()\d][\d\s().+-]{6,}$"),
  "ipv4": re.compile(r"^\d{1,3}(\.\d{1,3}){3}$"),
  "uuid": re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I),
  "iso_date": re.compile(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?)?$"),
  "money": re.compile(r"^\$?-?\d+([.,]\d+)?$"),
  "json_blob": re.compile(r"^\s*[\[{].*[\]}]\s*$", re.S),
}


def _detect_value_pattern(value) -> str | None:
  if value is None:
    return None
  s = str(value).strip()
  if not s:
    return None
  for name, rgx in _PATTERN_REGEX.items():
    try:
      if rgx.match(s):
        return name
    except Exception:
      continue
  return None


def _profile_column(col_name: str, col_type: str, values: list) -> dict:
  total = len(values)
  nulls = sum(1 for v in values if v is None or (isinstance(v, str) and v == ""))
  non_null = [v for v in values if v is not None and not (isinstance(v, str) and v == "")]
  distinct: dict = {}
  numeric_vals: list[float] = []
  str_lens: list[int] = []
  pattern_counts: dict[str, int] = {}
  for v in non_null:
    key = v if isinstance(v, (int, float, bool)) else str(v)[:120]
    distinct[key] = distinct.get(key, 0) + 1
    if isinstance(v, (int, float)) and not isinstance(v, bool):
      numeric_vals.append(float(v))
    elif isinstance(v, str):
      str_lens.append(len(v))
      try:
        numeric_vals.append(float(v))
      except Exception:
        pass
    p = _detect_value_pattern(v)
    if p:
      pattern_counts[p] = pattern_counts.get(p, 0) + 1
  top = sorted(distinct.items(), key=lambda kv: kv[1], reverse=True)[:5]
  inferred_kind = "unknown"
  lc = (col_type or "").lower()
  if any(t in lc for t in ("int", "decimal", "numeric", "float", "double", "real")):
    inferred_kind = "numeric"
  elif any(t in lc for t in ("date", "time")):
    inferred_kind = "datetime"
  elif any(t in lc for t in ("bool", "bit")):
    inferred_kind = "boolean"
  elif any(t in lc for t in ("json", "blob", "text", "char")):
    inferred_kind = "text"
  if pattern_counts:
    dominant = max(pattern_counts.items(), key=lambda kv: kv[1])
    if dominant[1] >= max(1, int(0.6 * len(non_null))):
      inferred_kind = dominant[0]
  numeric_summary = None
  if numeric_vals:
    numeric_summary = {
      "min": min(numeric_vals),
      "max": max(numeric_vals),
      "avg": sum(numeric_vals) / len(numeric_vals),
      "count": len(numeric_vals),
    }
  text_summary = None
  if str_lens:
    text_summary = {
      "avg_len": sum(str_lens) / len(str_lens),
      "min_len": min(str_lens),
      "max_len": max(str_lens),
    }
  return {
    "column": col_name,
    "declared_type": col_type,
    "inferred_kind": inferred_kind,
    "rows_seen": total,
    "null_count": nulls,
    "null_pct": (nulls / total * 100.0) if total else 0.0,
    "distinct_count": len(distinct),
    "uniqueness_pct": (len(distinct) / max(1, len(non_null))) * 100.0,
    "top_values": [{"value": str(k)[:80], "count": c} for k, c in top],
    "patterns_detected": pattern_counts,
    "numeric_summary": numeric_summary,
    "text_summary": text_summary,
  }


def _infer_table_role(tname: str, col_records: list[dict], profiles: list[dict]) -> str:
  name = (tname or "").lower()
  cols_lower = [(c.get("name") or "").lower() for c in col_records]
  joined = " ".join(cols_lower)
  id_like = sum(1 for c in cols_lower if c.endswith("_id") or c == "id" or c.endswith("id"))
  total_cols = len(cols_lower) or 1
  if id_like >= 2 and id_like / total_cols >= 0.5 and total_cols <= 6:
    return "junction/mapping table — primarily links other entities together"
  hints = {
    "user/identity": ("user", "account", "member", "person", "employee", "customer"),
    "authentication/security": ("password", "token", "session", "auth"),
    "product/catalog": ("product", "item", "sku", "catalog", "asset"),
    "order/transaction": ("order", "invoice", "transaction", "payment", "sale", "cart"),
    "log/audit": ("log", "audit", "event", "history", "activity"),
    "configuration/settings": ("config", "setting", "preference", "option"),
    "permission/role": ("role", "permission", "grant", "access", "privilege"),
    "messaging/notification": ("message", "notif", "comment", "chat", "email"),
    "geo/location": ("country", "city", "region", "location", "address"),
    "content/document": ("post", "article", "document", "doc", "page", "note"),
  }
  scores: dict[str, int] = {}
  for role, words in hints.items():
    for w in words:
      if w in name:
        scores[role] = scores.get(role, 0) + 3
      if w in joined:
        scores[role] = scores.get(role, 0) + 1
  if scores:
    return max(scores.items(), key=lambda kv: kv[1])[0]
  if any(p.get("inferred_kind") in ("email", "phone") for p in profiles):
    return "contact/people-related"
  return "generic data entity"


def _infer_relationships(table_info: list[dict]) -> dict:
  """Infer logical relationships across tables (independent of declared FKs)."""
  by_table = {t["table"]: t for t in table_info}
  inferred: list[dict] = []
  junction_tables: list[dict] = []

  for ti in table_info:
    tname = ti["table"]
    cols = [(c.get("name") or "") for c in ti.get("columns", [])]
    cols_lower = [c.lower() for c in cols]
    pks = [p.lower() for p in (ti.get("primary_key") or [])]

    id_like_cols = [c for c in cols if c.lower().endswith("_id") or (c.lower().endswith("id") and c.lower() not in pks and c.lower() != "id")]
    if len(cols) and len(id_like_cols) >= 2 and len(id_like_cols) / max(1, len(cols)) >= 0.5 and len(cols) <= 7:
      junction_tables.append({
        "table": tname,
        "links": id_like_cols,
        "explanation": f"Table '{tname}' looks like a junction/mapping table connecting {', '.join(id_like_cols)}.",
      })

    for col in cols:
      cl = col.lower()
      if cl in ("id",) or cl in pks:
        continue
      candidate_targets = []
      if cl.endswith("_id"):
        stem = cl[:-3]
        candidate_targets.append(stem)
        candidate_targets.append(stem + "s")
      if cl.endswith("id") and len(cl) > 2:
        stem = cl[:-2]
        candidate_targets.append(stem)
        candidate_targets.append(stem + "s")
      candidate_targets.append(cl)
      for cand in candidate_targets:
        for other in by_table:
          if other == tname:
            continue
          if other.lower() == cand:
            inferred.append({
              "from_table": tname,
              "from_column": col,
              "to_table": other,
              "to_column": (by_table[other].get("primary_key") or ["id"])[0],
              "confidence": "high" if cl.endswith("_id") else "medium",
              "evidence": f"Column name '{col}' matches table name '{other}'.",
            })
            break
        else:
          continue
        break
  return {"inferred_relations": inferred, "junction_tables": junction_tables}


def _build_knowledge_summary(link: dict, table_info: list[dict], analysis: dict) -> str:
  lines: list[str] = []
  lines.append(f"Knowledge structure summary for database '{link.get('name')}' ({link.get('type')}):")
  lines.append(f"- Tables analysed: {len(table_info)}")
  total_rows = sum((ti.get('row_count_in_source') or 0) for ti in table_info)
  lines.append(f"- Total rows seen across tables: {total_rows}")
  for ti in table_info:
    cols = [c.get("name") for c in ti.get("columns", [])]
    lines.append(
      f"  • {ti['table']} — role: {ti.get('semantic_role')}; "
      f"{len(cols)} columns; rows in source: {ti.get('row_count_in_source')}; "
      f"deeply analysed: {ti.get('rows_profiled')} rows"
    )
  inferred = analysis.get("inferred_relations") or []
  if inferred:
    lines.append(f"- Inferred logical relationships: {len(inferred)}")
    for rel in inferred[:20]:
      lines.append(
        f"  • {rel['from_table']}.{rel['from_column']} → {rel['to_table']}.{rel['to_column']} "
        f"({rel['confidence']} confidence: {rel['evidence']})"
      )
  jt = analysis.get("junction_tables") or []
  if jt:
    lines.append(f"- Junction/mapping tables: {len(jt)}")
    for j in jt:
      lines.append(f"  • {j['table']} links {', '.join(j['links'])}")
  return "\n".join(lines)


def _run_training_job(link_id: str):
  link = _find_link(link_id)
  if not link:
    return

  job = _new_job_state(link_id, link.get("name", "Training"))
  with _jobs_lock:
    TRAINING_JOBS[link_id] = job

  link["status"] = "running"
  link["last_error"] = None
  _persist_link(link)

  try:
    if not SQLALCHEMY_AVAILABLE:
      raise RuntimeError("SQLAlchemy is not installed. Run: pip install sqlalchemy")

    url = link.get("url", "")
    if not url:
      raise RuntimeError("Connection has no URL. Re-save the connection.")

    _job_update(job, status="connecting", stage="connecting")
    _job_log(job, f"Connecting to {link.get('type', '?')} database...")
    engine = create_engine(url)
    with engine.connect() as conn:
      conn.execute(text("SELECT 1"))
    _job_log(job, "Connection successful.")

    if job.get("stop_requested"):
      raise RuntimeError("Training stopped by user before listing tables.")

    _job_update(job, status="listing", stage="listing-tables")
    _job_log(job, "Listing tables from database...")
    inspector = inspect(engine)
    table_names = inspector.get_table_names()

    selected = [t.strip() for t in str(link.get("tables", "")).split(",") if t.strip()]
    if selected:
      table_names = [t for t in table_names if t in selected]
    sample_rows = max(1, min(50, int(link.get("sample_rows") or 5)))

    _job_update(job, tables_total=len(table_names))
    _job_log(job, f"Found {len(table_names)} table(s). Sample rows per table: {sample_rows}")

    if not table_names:
      _job_update(job, status="completed", stage="done", finished_at=time.time(), percent=100)
      _job_log(job, "No tables to ingest. Marking as completed.")
      link["status"] = "completed"
      link["last_trained_at"] = _now_ts()
      link["last_added_docs"] = 0
      link["last_table_count"] = 0
      _persist_link(link)
      return

    added_docs = 0
    rows_total = 0
    rows_profiled_total = 0
    table_info_list: list[dict] = []
    deep_cap = max(50, min(2000, int(link.get("deep_rows") or 500)))
    with engine.connect() as conn:
      for tname in table_names:
        if job.get("stop_requested"):
          raise RuntimeError(f"Training stopped by user before table '{tname}'.")
        _job_update(job, status="ingesting", stage="ingesting-table", current_table=tname)
        _job_log(job, f"Ingesting table: {tname}")

        columns = inspector.get_columns(tname)
        col_records = []
        for c in columns:
          col_records.append({
            "name": c.get("name"),
            "type": str(c.get("type")) if c.get("type") is not None else "?",
            "nullable": bool(c.get("nullable", True)),
            "primary_key": bool(c.get("primary_key", False)),
          })
        col_names = [c["name"] for c in col_records if c.get("name")]
        try:
          pk_info = inspector.get_pk_constraint(tname) or {}
          pk_cols = list(pk_info.get("constrained_columns") or [])
        except Exception:
          pk_cols = []
        try:
          fk_info = inspector.get_foreign_keys(tname) or []
          fk_summary = [
            {
              "columns": list(fk.get("constrained_columns") or []),
              "referred_table": fk.get("referred_table"),
              "referred_columns": list(fk.get("referred_columns") or []),
            } for fk in fk_info
          ]
        except Exception:
          fk_summary = []

        # Total row count (best-effort)
        try:
          total_rows = conn.execute(text(f"SELECT COUNT(*) FROM {tname}")).scalar() or 0
          total_rows = int(total_rows)
        except Exception:
          total_rows = None
        _job_log(job, f"  • Schema: {len(col_names)} columns, "
                      f"{len(pk_cols)} PK, {len(fk_summary)} FK, "
                      f"rows in source: {total_rows if total_rows is not None else 'unknown'}")

        schema_text = (
          f"Table: {tname}\n"
          f"Columns: {', '.join(col_names)}\n"
          f"Primary key: {', '.join(pk_cols) if pk_cols else '(none)'}\n"
          f"Foreign keys: {len(fk_summary)}\n"
          f"Total rows in source: {total_rows if total_rows is not None else 'unknown'}"
        )

        sample_rows_data: list[dict] = []
        deep_rows_data: list[dict] = []
        try:
          deep_limit = min(deep_cap, total_rows) if total_rows else deep_cap
          rows = conn.execute(text(f"SELECT * FROM {tname} LIMIT {deep_limit}"))
          for r in rows.fetchall():
            try:
              row_dict = {k: (v if isinstance(v, (str, int, float, bool, type(None))) else str(v))
                           for k, v in dict(r._mapping).items()}
            except Exception:
              row_dict = {"_raw": str(r)}
            deep_rows_data.append(row_dict)
          sample_rows_data = deep_rows_data[:sample_rows]
          rows_total += len(sample_rows_data)
          rows_profiled_total += len(deep_rows_data)
          sample_text = "\n".join(str(r) for r in sample_rows_data) if sample_rows_data else "(empty table)"
          _job_log(job, f"  • Deep-read {len(deep_rows_data)} row(s) (cap {deep_limit}) for analysis; kept {len(sample_rows_data)} as memorised samples.")
        except Exception as exc:
          sample_text = f"(Could not fetch sample rows: {exc})"
          _job_log(job, f"  • Could not sample rows: {exc}")

        _job_update(job, status="profiling", stage="profiling-table")
        column_profiles: list[dict] = []
        for c in col_records:
          cname = c.get("name")
          if not cname:
            continue
          values = [row.get(cname) for row in deep_rows_data]
          column_profiles.append(_profile_column(cname, c.get("type") or "?", values))
        _job_log(job, f"  • Profiled {len(column_profiles)} column(s): null %, distinct count, top values, patterns.")

        patterns_overall: dict[str, int] = {}
        for prof in column_profiles:
          for pname, cnt in (prof.get("patterns_detected") or {}).items():
            patterns_overall[pname] = patterns_overall.get(pname, 0) + cnt
        semantic_role = _infer_table_role(tname, col_records, column_profiles)
        _job_log(job, f"  • Inferred role for '{tname}': {semantic_role}.")

        # Build a rich human-readable analysis section for the doc text
        analysis_lines = [
          f"Inferred semantic role: {semantic_role}",
          f"Patterns detected across rows: {patterns_overall or '(none)'}",
          "Per-column profile:",
        ]
        for prof in column_profiles:
          ns = prof.get("numeric_summary")
          ts = prof.get("text_summary")
          tops = ", ".join(f"{tv['value']}×{tv['count']}" for tv in prof.get("top_values") or [])
          extra = []
          if ns:
            extra.append(f"min={ns['min']}, max={ns['max']}, avg={ns['avg']:.2f}")
          if ts:
            extra.append(f"avg_len={ts['avg_len']:.1f}")
          if prof.get("patterns_detected"):
            extra.append(f"patterns={prof['patterns_detected']}")
          analysis_lines.append(
            f"  - {prof['column']} ({prof['declared_type']} → {prof['inferred_kind']}): "
            f"distinct={prof['distinct_count']}, null%={prof['null_pct']:.1f}, "
            f"unique%={prof['uniqueness_pct']:.1f}; top={tops or '-'}"
            + (f"; {'; '.join(extra)}" if extra else "")
          )

        combined = (
          schema_text
          + "\n\nDeep analysis:\n" + "\n".join(analysis_lines)
          + "\nSample rows:\n" + sample_text
        )
        doc = _append_knowledge_doc(
          title=f"DB:{link.get('name')}:{tname}",
          content=combined,
          source="database",
          meta={"db_name": link.get("name"), "db_type": link.get("type"),
                "table": tname, "link_id": link_id,
                "columns": col_records, "primary_key": pk_cols,
                "foreign_keys": fk_summary, "row_count": total_rows,
                "rows_profiled": len(deep_rows_data),
                "semantic_role": semantic_role,
                "patterns": patterns_overall,
                "column_profiles": column_profiles},
        )
        added_docs += 1

        # Build a human-readable "what was understood" summary for this table
        understood = (
          f"Table '{tname}' is interpreted as a {semantic_role}. "
          f"It has {len(col_names)} column(s) and {total_rows if total_rows is not None else 'an unknown number of'} row(s) in the source. "
          f"The assistant deeply analysed {len(deep_rows_data)} row(s): computed null %, distinct counts, "
          f"top values and value patterns ({patterns_overall or 'none detected'}) per column, "
          f"and stored {len(sample_rows_data)} representative sample row(s) as factual context."
        )

        learning_record = {
          "table": tname,
          "columns": col_records,
          "primary_key": pk_cols,
          "foreign_keys": fk_summary,
          "row_count_in_source": total_rows,
          "rows_sampled": len(sample_rows_data),
          "rows_profiled": len(deep_rows_data),
          "semantic_role": semantic_role,
          "patterns_detected": patterns_overall,
          "column_profiles": column_profiles,
          "sample_preview": sample_rows_data[:3],
          "doc_id": doc.get("id"),
          "doc_title": doc.get("title"),
          "doc_chars": len(combined),
          "stored_in": str(KNOWLEDGE_DOCS_PATH),
          "understood": understood,
          "ingested_at": time.time(),
        }
        with _jobs_lock:
          job.setdefault("learnings", []).append(learning_record)

        table_info_list.append({
          "table": tname,
          "columns": col_records,
          "primary_key": pk_cols,
          "foreign_keys": fk_summary,
          "row_count_in_source": total_rows,
          "rows_profiled": len(deep_rows_data),
          "semantic_role": semantic_role,
        })

        _job_update(job,
                    tables_done=job["tables_done"] + 1,
                    added_docs=added_docs,
                    rows_ingested=rows_total,
                    rows_profiled=rows_profiled_total)

    # Cross-table relationship inference
    _job_update(job, status="analyzing", stage="analyzing-relations", current_table=None)
    _job_log(job, "Inferring logical relationships across tables...")
    analysis = _infer_relationships(table_info_list)
    _job_log(job,
             f"  • Found {len(analysis['inferred_relations'])} inferred relation(s) and "
             f"{len(analysis['junction_tables'])} junction table(s).")

    # Semantic summary doc
    _job_update(job, status="summarizing", stage="semantic-summary")
    summary_text = _build_knowledge_summary(link, table_info_list, analysis)
    summary_doc = _append_knowledge_doc(
      title=f"DB:{link.get('name')}:__summary__",
      content=summary_text,
      source="database",
      meta={"db_name": link.get("name"), "db_type": link.get("type"),
            "link_id": link_id, "kind": "knowledge_summary",
            "tables": [ti["table"] for ti in table_info_list],
            "inferred_relations": analysis["inferred_relations"],
            "junction_tables": analysis["junction_tables"]},
    )
    added_docs += 1
    _job_log(job, f"  • Wrote knowledge structure summary doc ({len(summary_text)} chars).")
    with _jobs_lock:
      job["analysis"] = {
        "inferred_relations": analysis["inferred_relations"],
        "junction_tables": analysis["junction_tables"],
        "knowledge_summary": summary_text,
        "summary_doc_id": summary_doc.get("id"),
      }

    _job_update(job, status="saving", stage="saving", current_table=None)
    _job_log(job, "Persisting training results to knowledge memory...")

    link["status"] = "completed"
    link["last_trained_at"] = _now_ts()
    link["last_added_docs"] = added_docs
    link["last_table_count"] = len(table_names)
    link["last_error"] = None
    _persist_link(link)

    integration = {
      "method": "Retrieval-Augmented Generation (RAG) with deep schema + content profiling",
      "model_weights_changed": False,
      "model_file": MODEL_PATH if "MODEL_PATH" in globals() else None,
      "knowledge_base_path": str(KNOWLEDGE_DOCS_PATH),
      "docs_added_this_run": added_docs,
      "total_docs_in_kb": len(_load_knowledge_docs()),
      "rows_profiled_this_run": rows_profiled_total,
      "tables_analysed": len(table_info_list),
      "inferred_relations_count": len(analysis["inferred_relations"]),
      "junction_tables_count": len(analysis["junction_tables"]),
      "depth_level": (
        "deep — schema + per-column profiling (null %, distinct, top values, "
        "pattern detection) + cross-table relationship inference + semantic role "
        "tagging + a global knowledge-structure summary"
      ),
      "how_it_is_used": (
        "Each chat turn the assistant retrieves the most relevant of these "
        "rich documents (per-table profiles plus the global summary) and grounds "
        "its answer on them. The base Phi-3 model file is NOT modified; the "
        "knowledge memory is what grows and gets richer with every training run."
      ),
      "what_it_understood": (
        f"From '{link.get('name')}' ({link.get('type')}) the assistant now understands "
        f"{len(table_info_list)} table(s), their semantic roles "
        f"({', '.join(sorted({ti.get('semantic_role','?') for ti in table_info_list}))}), "
        f"per-column data shape, dominant value patterns, and "
        f"{len(analysis['inferred_relations'])} logical relationship(s) it inferred "
        f"between tables (independent of declared foreign keys)."
      ),
    }
    with _jobs_lock:
      job["model_integration"] = integration

    _job_update(job, status="completed", stage="done", finished_at=time.time(), percent=100, eta_ms=0)
    _job_log(job, f"Training completed. Tables: {len(table_names)}, Docs added: {added_docs}, Rows sampled: {rows_total}")
    _job_log(job, "Note: knowledge was added to the assistant's retrieval memory (RAG). The Phi-3 model weights were NOT modified.")
  except Exception as exc:
    err = str(exc)
    is_stop = err.startswith("Training stopped by user")
    final_status = "stopped" if is_stop else "failed"
    _job_update(job, status=final_status, stage=final_status, finished_at=time.time(), error=err)
    _job_log(job, ("Stopped: " if is_stop else "Failed: ") + err)
    link["status"] = final_status
    link["last_error"] = None if is_stop else err
    _persist_link(link)


def _extract_row_dicts(content: str) -> list[dict]:
  rows = []
  for line in (content or "").splitlines():
    raw = line.strip()
    if not raw.startswith("{") or not raw.endswith("}"):
      continue
    try:
      parsed = ast.literal_eval(raw)
    except Exception:
      continue
    if isinstance(parsed, dict):
      rows.append(parsed)
  return rows


def _answer_from_knowledge(query: str, docs: list[dict]) -> str | None:
  q_tokens = set(_tokenize(query))
  if not q_tokens or not docs:
    return None

  db_docs = [d for d in docs if str(d.get("source", "")).lower() == "database"]

  # Fast-path: "how many rows" / "كم عدد الصفوف" — answer directly from doc meta/text.
  q_low = (query or "").lower()
  count_keywords_en = ("how many", "count", "total rows", "number of rows", "rows in", "row count")
  count_keywords_ar = ("كم عدد", "عدد الصفوف", "كم صف", "كم سجل", "عدد السجلات")
  asks_count = (
    any(k in q_low for k in count_keywords_en)
    or any(k in query for k in count_keywords_ar)
    or bool(q_tokens.intersection({"how", "many", "count", "total", "number"}) and q_tokens.intersection({"row", "rows", "record", "records"}))
  )
  if asks_count:
    best = db_docs[0]
    title = str(best.get("title", ""))
    content = str(best.get("content", ""))
    meta = best.get("meta") or {}

    row_count = None
    deep_count = None

    if isinstance(meta, dict):
      for k in ("row_count", "rows", "total_rows"):
        if meta.get(k) is not None:
          try:
            row_count = int(meta.get(k))
            break
          except Exception:
            pass
      for k in ("rows_profiled", "deep_rows", "rows_deeply_analysed"):
        if meta.get(k) is not None:
          try:
            deep_count = int(meta.get(k))
            break
          except Exception:
            pass

    if row_count is None:
      m = re.search(r"total rows in source\s*:\s*([0-9,]+)", content, flags=re.IGNORECASE)
      if m:
        try:
          row_count = int(m.group(1).replace(",", ""))
        except Exception:
          pass
    if deep_count is None:
      m = re.search(r"deeply analysed\s*:?\s*([0-9,]+)\s*rows?", content, flags=re.IGNORECASE)
      if not m:
        m = re.search(r"analysed\s*([0-9,]+)\s*rows", content, flags=re.IGNORECASE)
      if m:
        try:
          deep_count = int(m.group(1).replace(",", ""))
        except Exception:
          pass

    if row_count is not None or deep_count is not None:
      is_arabic = any("\u0600" <= ch <= "\u06FF" for ch in query)
      parts = []
      if is_arabic:
        head = "من بيانات قاعدة البيانات المدربة:\n"
        if title:
          parts.append(f"المصدر: {title}")
        if row_count is not None:
          parts.append(f"إجمالي الصفوف في المصدر: {row_count}")
        if deep_count is not None:
          parts.append(f"الصفوف التي تم تحليلها بعمق: {deep_count}")
      else:
        head = "From trained database data:\n"
        if title:
          parts.append(f"Source: {title}")
        if row_count is not None:
          parts.append(f"Total rows in source: {row_count}")
        if deep_count is not None:
          parts.append(f"Rows deeply analysed: {deep_count}")
      return head + "\n".join(f"- {p}" for p in parts)

  row_dicts = []
  for d in db_docs:
    row_dicts.extend(_extract_row_dicts(str(d.get("content", ""))))
  if row_dicts:
    asks_list = bool(q_tokens.intersection({"list", "show", "all", "product", "name", "price", "item"}))
    if asks_list:
      lines = []
      for row in row_dicts[:20]:
        keys = {str(k).lower(): k for k in row.keys()}
        name_key = keys.get("name") or keys.get("product") or keys.get("title")
        price_key = keys.get("price") or keys.get("cost")
        if name_key and price_key:
          lines.append(f"- {row.get(name_key)}: {row.get(price_key)}")
        else:
          preview = ", ".join(f"{k}={v}" for k, v in row.items())
          lines.append(f"- {preview}")
      if lines:
        return "From trained database data:\n" + "\n".join(lines)

  # Generic grounded extraction for ADTC/dataset/manual docs.
  source_counts = {}
  for d in docs:
    s = str(d.get("source", "")).lower() or "unknown"
    source_counts[s] = source_counts.get(s, 0) + 1
  dominant_source = max(source_counts, key=source_counts.get) if source_counts else "unknown"

  matched_lines = []
  used_titles = []
  for d in docs[:6]:
    title = str(d.get("title", "Knowledge"))
    title_low = title.lower()
    # Skip meta/summary docs from raw line extraction so we never echo only a file-title.
    if any(k in title_low for k in ("executive summary", "understanding", "adtc summary", "summary for", "schema overview")):
      continue
    content = str(d.get("content", ""))
    if title not in used_titles:
      used_titles.append(title)

    for ln in content.splitlines():
      line = ln.strip()
      if len(line) < 3:
        continue
      # Drop lines that are themselves the document title or a "filename.pdf" only.
      if line.lower() == title_low or re.fullmatch(r"[\w\s\-\.]+\.(pdf|docx?|xlsx?|csv|txt)", line, flags=re.IGNORECASE):
        continue
      ltok = set(_tokenize(line))
      overlap = len(q_tokens.intersection(ltok))
      if overlap >= 2 or (overlap >= 1 and len(q_tokens) <= 3):
        matched_lines.append((overlap, title, line))

  matched_lines.sort(key=lambda x: x[0], reverse=True)
  top = matched_lines[:10]

  is_ar = any("\u0600" <= ch <= "\u06FF" for ch in query)
  if dominant_source == "adtc":
    head = "من بيانات ADTC المدربة:\n" if is_ar else "From trained ADTC data:\n"
  elif dominant_source == "dataset":
    head = "من بيانات الملفات المدربة:\n" if is_ar else "From trained dataset files:\n"
  elif dominant_source == "database":
    head = "من بيانات قاعدة البيانات المدربة:\n" if is_ar else "From trained database data:\n"
  else:
    head = "من البيانات المدربة:\n" if is_ar else "From trained knowledge data:\n"

  # ADTC-specific synthesis for communication channel questions.
  asks_channels = (
    any(k in q_low for k in ("communication channel", "communication channels", "contact", "reach", "hotline", "phone", "email", "website", "social"))
    or any(k in query for k in ("قنوات التواصل", "وسائل التواصل", "طرق التواصل", "التواصل", "البريد", "الهاتف", "الموقع"))
  )
  adtc_docs = [d for d in docs if str(d.get("source", "")).lower() == "adtc"]

  # ADTC-specific synthesis for closure/holiday schedule questions.
  asks_closure = (
    any(k in q_low for k in (
      "closed", "close", "closure", "will be closed", "holiday", "eid", "operating hours", "operation hours", "opening hours"
    ))
    or any(k in query for k in ("مغلق", "اغلاق", "إغلاق", "العيد", "الدوام", "ساعات العمل", "مواعيد العمل"))
  )
  if asks_closure and adtc_docs:
    all_lines = []
    for d in adtc_docs:
      all_lines.extend([ln.strip() for ln in str(d.get("content", "")).splitlines() if ln.strip()])

    date_range = None
    closure_line = None
    hours_line = None
    date_patterns = [
      r"\b\d{1,2}(?:st|nd|rd|th)?\s*[-–]\s*\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4}\b",
      r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s*[-–]\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
    ]
    for line in all_lines:
      ll = line.lower()
      if closure_line is None and any(k in ll for k in ("closed", "closure", "holiday", "eid", "مغلق", "إغلاق", "العيد")):
        closure_line = line
      if hours_line is None and any(k in ll for k in ("operating hours", "operation hours", "opening hours", "clinic and laboratory", "ساعات العمل", "الدوام")):
        hours_line = line
      if date_range is None:
        for pat in date_patterns:
          m = re.search(pat, line, flags=re.IGNORECASE)
          if m:
            date_range = m.group(0)
            break
      if closure_line and date_range and hours_line:
        break

    if closure_line or date_range or hours_line:
      if is_ar:
        parts = ["استنادًا إلى بيانات ADTC المدربة:"]
        if date_range:
          parts.append(f"- عيادات Healthpoint مغلقة خلال الفترة: {date_range}.")
        elif closure_line:
          parts.append(f"- الخلاصة: {closure_line}")
        if hours_line:
          parts.append(f"- ملاحظة ساعات التشغيل: {hours_line}")
        return head + "\n".join(parts)

      parts = ["Based on trained ADTC files:"]
      if date_range:
        parts.append(f"- Healthpoint clinics are closed during: {date_range}.")
      elif closure_line:
        parts.append(f"- Summary: {closure_line}")
      if hours_line:
        parts.append(f"- Operating-hours note: {hours_line}")
      return head + "\n".join(parts)

  if asks_channels and adtc_docs:
    all_lines = []
    for d in adtc_docs:
      all_lines.extend([ln.strip() for ln in str(d.get("content", "")).splitlines() if ln.strip()])

    urls = []
    emails = []
    phones = []
    social_tags = set()

    for line in all_lines:
      for u in re.findall(r"https?://[^\s)]+", line, flags=re.IGNORECASE):
        if u not in urls:
          urls.append(u)
      for e in re.findall(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", line, flags=re.IGNORECASE):
        if e not in emails:
          emails.append(e)
      for p in re.findall(r"\+?\d[\d\s\-()]{6,}\d", line):
        cleaned = re.sub(r"\s+", " ", p).strip()
        if cleaned not in phones:
          phones.append(cleaned)

      ll = line.lower()
      if any(k in ll for k in ("facebook", "instagram", "linkedin", "x.com", "twitter", "youtube", "tiktok", "whatsapp", "telegram")):
        if "facebook" in ll: social_tags.add("Facebook")
        if "instagram" in ll: social_tags.add("Instagram")
        if "linkedin" in ll: social_tags.add("LinkedIn")
        if "x.com" in ll or "twitter" in ll: social_tags.add("X/Twitter")
        if "youtube" in ll: social_tags.add("YouTube")
        if "tiktok" in ll: social_tags.add("TikTok")
        if "whatsapp" in ll: social_tags.add("WhatsApp")
        if "telegram" in ll: social_tags.add("Telegram")

    if urls or emails or phones or social_tags:
      if is_ar:
        parts = ["حسب ما تعلّمته من ملفات ADTC، قنوات تواصل HP هي:"]
        parts.append("- الموقع/الروابط الرسمية: " + (" | ".join(urls[:4]) if urls else "غير مذكور بوضوح"))
        parts.append("- البريد الإلكتروني: " + (" | ".join(emails[:4]) if emails else "غير مذكور بوضوح"))
        parts.append("- الهاتف/الخط الساخن: " + (" | ".join(phones[:4]) if phones else "غير مذكور بوضوح"))
        parts.append("- قنوات التواصل الاجتماعي: " + (" | ".join(sorted(social_tags)) if social_tags else "غير مذكور بوضوح"))
        return head + "\n".join(parts)
      parts = ["Based on what I learned from ADTC files, HP communication channels are:"]
      parts.append("- Official website/links: " + (" | ".join(urls[:4]) if urls else "Not explicitly stated"))
      parts.append("- Email: " + (" | ".join(emails[:4]) if emails else "Not explicitly stated"))
      parts.append("- Phone/Hotline: " + (" | ".join(phones[:4]) if phones else "Not explicitly stated"))
      parts.append("- Social channels: " + (" | ".join(sorted(social_tags)) if social_tags else "Not explicitly stated"))
      return head + "\n".join(parts)

    if is_ar:
      return head + "بحثت داخل بيانات ADTC المدربة ولم أجد ذكرًا مباشرًا وواضحًا لقنوات التواصل (هاتف/بريد/روابط تواصل اجتماعي) في المقاطع المسترجعة حاليًا."
    return head + "I searched the trained ADTC content but did not find an explicit communication-channels entry (phone/email/social) in the currently retrieved snippets."

  # Synthesis for location/address questions (hospital/clinic location).
  asks_location = (
    any(k in q_low for k in ("location", "address", "where is", "where are", "located", "directions", "how to reach", "find us"))
    or any(k in query for k in ("الموقع", "العنوان", "أين", "اين", "مكان", "كيف اصل", "كيف أصل"))
  )
  grounded_docs = [d for d in docs if str(d.get("source", "")).lower() in ("adtc", "dataset", "manual")]
  if asks_location and grounded_docs:
    # Collect every non-trivial line from grounded docs, dedupe, drop pure heading-like lines.
    raw_lines = []
    seen = set()
    q_entity_tokens = [
      t for t in _tokenize(query)
      if t not in {
        "where", "what", "are", "is", "the", "a", "an", "location", "address", "located",
        "clinic", "clinics", "hospital", "how", "reach", "find", "us",
        "اين", "أين", "الموقع", "العنوان", "مكان", "عيادة", "عيادات", "مستشفى", "ما", "هو", "هي"
      }
    ]
    for d in grounded_docs:
      title = str(d.get("title", "")).lower()
      # Skip understanding/summary docs for raw extraction (they are meta).
      if "understanding" in title or "adtc summary" in title:
        continue
      for ln in str(d.get("content", "")).splitlines():
        line = re.sub(r"\s+", " ", ln).strip(" -•\t")
        if len(line) < 8:
          continue
        ll = line.lower()
        # drop pure heading repetitions like "Hospital Location" / "Clinic Locations"
        if len(line.split()) <= 4 and re.fullmatch(r"[A-Za-z\s\(\)/\-]+", line) and any(w in ll for w in ("location", "locations", "clinic", "hospital", "ifhas")):
          continue
        # Skip common non-location noise.
        if any(k in ll for k in ("flower shop", "wards", "general surgery", "occupational medicine", "tel", "fax", "call center", "media, iframe", "embed and object tags", "same ent")):
          continue
        if ll.startswith("http://") or ll.startswith("https://"):
          continue
        if line.lower() in seen:
          continue
        seen.add(line.lower())
        raw_lines.append(line)

    address_keywords = (
      "street", "st.", "road", "rd", "building", "tower", "floor", "block", "po box", "p.o.", "zone",
      "city", "abu dhabi", "dubai", "sharjah", "ajman", "uae", "u.a.e", "united arab emirates",
      "al ", "mubadala", "healthpoint", "khalifa", "zayed", "hamdan", "corniche", "island",
      "near ", "next to", "opposite", "behind", "beside", "located", "located at", "located in",
      "شارع", "طريق", "مبنى", "برج", "طابق", "منطقة", "أبوظبي", "ابوظبي", "دبي", "الإمارات", "الامارات",
    )
    def _line_score(line: str) -> int:
      ll = line.lower()
      score = 0
      has_address_marker = any(k in ll for k in address_keywords) or re.search(r"\bP\.?\s*O\.?\s*Box\b", line, re.IGNORECASE)
      has_direction_marker = bool(re.search(r"\b(gate|entrance|between|near|next to|opposite|behind)\b", ll))
      if has_address_marker:
        score += 6
      if has_direction_marker:
        score += 3
      # Discard non-location lines early.
      if not has_address_marker and not has_direction_marker:
        return 0
      if any(tok and tok in ll for tok in q_entity_tokens):
        score += 5
      if "ifhas" in ll and any(tok in query.lower() for tok in ("ifhas", "clinic", "clinics")):
        score += 4
      if len(line) > 40:
        score += 1
      # Penalize obvious heading-like lines.
      if len(line.split()) <= 4:
        score -= 3
      if not re.search(r"[A-Za-z]", line):
        score -= 1
      return score

    scored = []
    for line in raw_lines:
      sc = _line_score(line)
      if sc >= 6:
        scored.append((sc, line))

    scored.sort(key=lambda x: x[0], reverse=True)
    address_lines = [line for _, line in scored]

    # Also pick lines containing landmark / clinic name + content (longer informative lines).
    informative_lines = [l for l in raw_lines if len(l) >= 25 and l not in address_lines and not l.lower().startswith(("http://", "https://"))]

    chosen = []
    for l in address_lines[:6]:
      if l not in chosen:
        chosen.append(l)
    if len(chosen) < 4 and not address_lines:
      for l in informative_lines:
        if l not in chosen:
          chosen.append(l)
        if len(chosen) >= 6:
          break

    if chosen:
      if is_ar:
        body = ["استنادًا إلى ما تعلّمته من ملفات ADTC، المواقع الأقرب للدقة هي:"]
        for l in chosen[:4]:
          body.append(f"- {l}")
        return head + "\n".join(body)
      body = ["Based on what I learned from ADTC files, the most likely clinic/hospital locations are:"]
      for l in chosen[:4]:
        body.append(f"- {l}")
      return head + "\n".join(body)

    if is_ar:
      return head + "لم أجد عنوانًا صريحًا (شارع/مبنى/مدينة) داخل المقاطع المسترجعة حاليًا. حاول إعادة التدريب أو طرح السؤال باسم الفرع/العيادة."
    return head + "I couldn't find an explicit address (street/building/city) inside the currently retrieved snippets. Try re-running ADTC training, or ask by branch/clinic name."

  if top:
    # Human-style summary: dedupe lines, drop headings, drop file-name prefixes.
    seen_lines = set()
    cleaned = []
    for _, _title, line in top:
      norm = re.sub(r"\s+", " ", line).strip(" -•\t")
      key = norm.lower()
      if key in seen_lines or len(norm) < 6:
        continue
      # Skip lines that look like a pure heading already echoed by the question.
      if len(norm.split()) <= 3 and key in q_low:
        continue
      seen_lines.add(key)
      cleaned.append(norm)
      if len(cleaned) >= 6:
        break
    if cleaned:
      if is_ar:
        body = ["حسب ما تعلّمته من البيانات المدربة:"]
      else:
        body = ["Based on what I learned from the trained data:"]
      for l in cleaned:
        body.append(f"- {l}")
      return head + "\n".join(body)

  # If no high-overlap lines are found, still avoid hallucination and report limitation from trained data only.
  if is_ar:
    return head + "لم أجد سطرًا مطابقًا بشكل كافٍ داخل البيانات المدربة لهذا السؤال. حاول إعادة الصياغة بكلمات موجودة في الملفات مثل أسماء الأقسام أو المصطلحات الأساسية."
  return head + "I couldn't find a high-confidence matching line in the trained data for this exact query. Please rephrase using terms that appear in the files (section names, entities, key phrases)."


def _estimate_tokens(text: str) -> int:
  text = text or ""
  # Approximation for quick monitoring when tokenizer usage is unavailable.
  return max(1, len(text) // 4)


def _looks_like_refusal(text_value: str) -> bool:
  low = (text_value or "").strip().lower()
  if not low:
    return False
  markers = (
    "i cannot respond",
    "i can't respond",
    "i cannot provide",
    "i can't provide",
    "i cannot answer",
    "i can't answer",
    "i do not have access",
    "i don't have access",
    "as an ai",
    "contact healthpoint",
    "visit their official website",
    "consult",
    "لا استطيع",
    "لا أستطيع",
    "لا يمكنني",
  )
  return any(m in low for m in markers)


def _extract_evidence_lines(query: str, docs: list[dict], max_items: int = 8) -> list[dict]:
  q_tokens = set(_tokenize(query or ""))
  if not q_tokens:
    return []

  candidates = []
  for d in docs or []:
    title = str(d.get("title", "Knowledge"))
    doc_id = str(d.get("id", ""))
    for raw in str(d.get("content", "")).splitlines():
      line = re.sub(r"\s+", " ", raw).strip(" -•\t")
      if len(line) < 8:
        continue
      l_tokens = set(_tokenize(line))
      overlap = len(q_tokens.intersection(l_tokens))
      if overlap == 0:
        continue
      score = overlap
      if re.search(r"\d", line):
        score += 1
      if any(k in line.lower() for k in ("open", "close", "closed", "hour", "موعد", "دوام", "مغلق", "العيد")):
        score += 1
      candidates.append({
        "score": score,
        "line": line,
        "title": title,
        "doc_id": doc_id,
      })

  candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
  out = []
  seen = set()
  for c in candidates:
    k = (c.get("line", "").lower(), c.get("title", "").lower())
    if k in seen:
      continue
    seen.add(k)
    out.append(c)
    if len(out) >= max_items:
      break
  return out


def _detect_reasoning_conflicts(evidence: list[dict]) -> list[str]:
  lines = [str(e.get("line", "")) for e in evidence or []]
  conflicts = []

  has_open = any(re.search(r"\b(open|opening)\b|مفتوح", ln, flags=re.IGNORECASE) for ln in lines)
  has_closed = any(re.search(r"\b(closed|closure|close)\b|مغلق|إغلاق|اغلاق", ln, flags=re.IGNORECASE) for ln in lines)
  if has_open and has_closed:
    conflicts.append("schedule_open_closed_conflict")

  numeric_by_key = {}
  for ln in lines:
    m = re.match(r"^\s*([A-Za-z\u0600-\u06FF][^:]{2,40})\s*:\s*([0-9][0-9,./-]*)", ln)
    if not m:
      continue
    key = _norm_key(m.group(1))[:24]
    val = m.group(2).strip()
    if not key:
      continue
    numeric_by_key.setdefault(key, set()).add(val)

  for k, vals in numeric_by_key.items():
    if len(vals) > 1:
      conflicts.append(f"numeric_conflict:{k}")

  return conflicts


def _build_reasoning_bundle(query: str, docs: list[dict] | None = None) -> dict:
  docs = docs or []
  evidence = _extract_evidence_lines(query, docs, max_items=8)
  conflicts = _detect_reasoning_conflicts(evidence)
  missing = len(evidence) == 0
  return {
    "query": query,
    "evidence": evidence,
    "evidence_count": len(evidence),
    "conflicts": conflicts,
    "conflict_count": len(conflicts),
    "has_evidence": not missing,
    "missing_evidence": missing,
  }


def _reply_has_evidence_marker(reply: str, evidence: list[dict]) -> bool:
  low = (reply or "").lower()
  if "evidence" in low or "الدليل" in low:
    return True
  for ev in evidence[:3]:
    line = str(ev.get("line", "")).lower()
    if len(line) >= 20 and line[:20] in low:
      return True
  return False


def _append_reasoning_evidence(reply: str, bundle: dict, is_ar: bool) -> str:
  evidence = list(bundle.get("evidence") or [])
  conflicts = list(bundle.get("conflicts") or [])
  if not evidence:
    return reply

  parts = [reply.rstrip()]
  if conflicts:
    if is_ar:
      parts.append("\nملاحظة تحقق: توجد إشارات تعارض بين بعض المقاطع المسترجعة، وتم تقديم الإجابة بصيغة حذرة.")
    else:
      parts.append("\nValidation note: there are conflicting retrieved snippets, so the answer is presented conservatively.")

  if is_ar:
    parts.append("\nأدلة مختصرة من البيانات المدربة:")
  else:
    parts.append("\nEvidence from trained data:")
  for ev in evidence[:3]:
    parts.append(f"- [{ev.get('title')}] {ev.get('line')}")
  return "\n".join(parts)


def _monitor_start(request_id: str, chat_id: str):
  with _monitor_lock:
    MONITOR_STATE["current"] = {
      "active": True,
      "request_id": request_id,
      "chat_id": chat_id,
      "stage": "loading-memory",
      "started_at": time.time(),
    }


def _monitor_stage(stage: str):
  with _monitor_lock:
    MONITOR_STATE["current"]["stage"] = stage


def _monitor_finish(entry: dict, is_error: bool = False):
  with _monitor_lock:
    MONITOR_STATE["current"] = {
      "active": False,
      "request_id": None,
      "chat_id": None,
      "stage": "idle",
      "started_at": None,
    }
    MONITOR_STATE["recent"].appendleft(entry)
    MONITOR_STATE["totals"]["requests"] += 1
    if is_error:
      MONITOR_STATE["totals"]["errors"] += 1
    MONITOR_STATE["totals"]["total_ms"] += float(entry.get("total_ms", 0.0))
    MONITOR_STATE["totals"]["infer_ms"] += float(entry.get("infer_ms", 0.0))
    MONITOR_STATE["totals"]["tokens_out"] += int(entry.get("output_tokens", 0))


def _monitor_snapshot() -> dict:
  with _monitor_lock:
    current = dict(MONITOR_STATE["current"])
    recent = list(MONITOR_STATE["recent"])
    totals = dict(MONITOR_STATE["totals"])

  req_count = max(1, totals["requests"])
  success_count = max(0, totals["requests"] - totals["errors"])
  avg_total_ms = totals["total_ms"] / req_count
  avg_infer_ms = totals["infer_ms"] / req_count
  avg_tps = (totals["tokens_out"] / (totals["infer_ms"] / 1000.0)) if totals["infer_ms"] > 0 else 0.0

  elapsed_ms = 0.0
  if current.get("active") and current.get("started_at"):
    elapsed_ms = (time.time() - current["started_at"]) * 1000.0

  # Annotate recent entries with the dominant stage so users can see WHY a
  # request was slow (model inference vs. memory IO vs. knowledge retrieval).
  STAGE_LABELS = {
    "load_ms": "memory load",
    "knowledge_ms": "knowledge retrieval",
    "infer_ms": "model inference",
    "save_ms": "memory save",
  }
  enriched_recent = []
  for r in recent:
    stages = {k: float(r.get(k) or 0.0) for k in STAGE_LABELS.keys()}
    if any(v > 0 for v in stages.values()):
      bottleneck_key = max(stages, key=stages.get)
      bottleneck = {
        "stage": STAGE_LABELS[bottleneck_key],
        "ms": round(stages[bottleneck_key], 2),
        "share_pct": round(
          (stages[bottleneck_key] / max(1.0, float(r.get("total_ms") or 0.0))) * 100.0, 1
        ),
      }
    else:
      bottleneck = {"stage": "-", "ms": 0.0, "share_pct": 0.0}
    item = dict(r)
    item["bottleneck"] = bottleneck
    enriched_recent.append(item)

  # System resource snapshot (CPU + memory). Falls back gracefully if psutil
  # is unavailable on the host.
  system = {"available": False}
  if _PSUTIL_AVAILABLE:
    try:
      vm = psutil.virtual_memory()
      sys_cpu = psutil.cpu_percent(interval=None)
      proc_cpu = _PROC.cpu_percent(interval=None)
      try:
        proc_cpu_normalized = proc_cpu / max(1, psutil.cpu_count(logical=True) or 1)
      except Exception:
        proc_cpu_normalized = proc_cpu
      proc_mem = _PROC.memory_info()
      try:
        num_threads = _PROC.num_threads()
      except Exception:
        num_threads = 0
      system = {
        "available": True,
        "cpu": {
          "system_percent": round(sys_cpu, 1),
          "process_percent": round(proc_cpu, 1),
          "process_percent_normalized": round(proc_cpu_normalized, 1),
          "logical_cores": psutil.cpu_count(logical=True) or 0,
          "physical_cores": psutil.cpu_count(logical=False) or 0,
        },
        "memory": {
          "system_total_mb": round(vm.total / (1024 * 1024), 1),
          "system_used_mb": round((vm.total - vm.available) / (1024 * 1024), 1),
          "system_available_mb": round(vm.available / (1024 * 1024), 1),
          "system_percent": round(vm.percent, 1),
          "process_rss_mb": round(proc_mem.rss / (1024 * 1024), 1),
          "process_vms_mb": round(proc_mem.vms / (1024 * 1024), 1),
        },
        "process": {
          "pid": _PROC.pid,
          "threads": num_threads,
        },
      }
    except Exception as exc:
      system = {"available": False, "error": str(exc)}

  return {
    "model": {
      "path": MODEL_PATH,
      "ctx": MODEL_CTX,
      "threads": MODEL_THREADS,
    },
    "current": {
      **current,
      "elapsed_ms": round(elapsed_ms, 2),
    },
    "summary": {
      "total_requests": totals["requests"],
      "total_errors": totals["errors"],
      "success_rate": round((success_count / req_count) * 100.0, 2),
      "avg_total_ms": round(avg_total_ms, 2),
      "avg_infer_ms": round(avg_infer_ms, 2),
      "avg_tokens_per_sec": round(avg_tps, 2),
    },
    "system": system,
    "recent": enriched_recent,
    "server_time": int(time.time()),
  }


INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Phi-3 Mini Chat</title>
  <style>
    :root {
      --bg: #1f1f1f;
      --panel: #171717;
      --border: #2f2f2f;
      --muted: #a1a1a1;
      --text: #ececec;
      --accent: #10a37f;
      --hover: #262626;
      --active: #2d2d2d;
      --bubble-user: #2f2f2f;
      --bubble-bot: transparent;
      --input: #2a2a2a;
    }

    body.light {
      --bg: #ffffff;
      --panel: #f5f5f6;
      --border: #e2e2e2;
      --muted: #666;
      --text: #202020;
      --accent: #0f9b78;
      --hover: #ececec;
      --active: #e5e5e5;
      --bubble-user: #efefef;
      --bubble-bot: transparent;
      --input: #f8f8f8;
    }

    * { box-sizing: border-box; }
    html, body { margin: 0; height: 100%; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: "Segoe UI", "Inter", system-ui, sans-serif;
    }

    .app {
      height: 100vh;
      display: grid;
      grid-template-columns: 280px 1fr;
      grid-template-rows: 1fr;
    }

    .sidebar {
      border-right: 1px solid var(--border);
      background: var(--panel);
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 10px;
      min-height: 0;
    }

    .btn {
      width: 100%;
      border: 1px solid var(--border);
      background: transparent;
      color: var(--text);
      border-radius: 12px;
      padding: 10px 12px;
      text-align: left;
      cursor: pointer;
      font-size: 14px;
    }
    .btn:hover { background: var(--hover); }

    .chat-list {
      overflow: auto;
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 6px;
      min-height: 0;
      flex: 1;
    }

    .chat-item-row {
      display: grid;
      grid-template-columns: 1fr 34px;
      gap: 6px;
      margin-bottom: 4px;
    }

    .chat-item {
      border: 0;
      background: transparent;
      color: var(--text);
      text-align: left;
      border-radius: 10px;
      padding: 10px;
      cursor: pointer;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .chat-item:hover { background: var(--hover); }
    .chat-item.active { background: var(--active); }

    .chat-delete {
      border: 0;
      background: transparent;
      color: var(--muted);
      border-radius: 10px;
      cursor: pointer;
      font-size: 14px;
    }
    .chat-delete:hover {
      background: var(--hover);
      color: #ff7b7b;
    }

    .main {
      display: grid;
      grid-template-rows: 56px 1fr auto;
      min-height: 0;
    }

    .topbar {
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 16px;
      font-weight: 600;
    }

    .top-actions {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .icon-btn {
      width: 36px;
      height: 36px;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: transparent;
      color: var(--text);
      cursor: pointer;
      font-size: 16px;
    }
    .icon-btn:hover { background: var(--hover); }

    .messages {
      overflow: auto;
      padding: 18px 20px;
    }

    .msg {
      margin-bottom: 14px;
      display: flex;
    }

    .bubble {
      max-width: min(820px, 100%);
      padding: 12px 14px;
      border-radius: 14px;
      line-height: 1.6;
      white-space: pre-wrap;
      word-wrap: break-word;
    }

    .msg.user { justify-content: flex-end; }
    .msg.user .bubble { background: var(--bubble-user); }
    .msg.bot { justify-content: flex-start; }
    .msg.bot .bubble { background: var(--bubble-bot); }

    .composer-wrap {
      border-top: 1px solid var(--border);
      padding: 12px;
    }

    .composer {
      border: 1px solid var(--border);
      background: var(--input);
      border-radius: 18px;
      display: flex;
      align-items: flex-end;
      gap: 8px;
      padding: 8px;
    }

    #messageInput {
      flex: 1;
      min-height: 44px;
      max-height: 180px;
      resize: none;
      border: 0;
      outline: none;
      background: transparent;
      color: var(--text);
      font-size: 15px;
      padding: 8px;
      font-family: inherit;
    }

    .send-btn {
      width: 42px;
      height: 42px;
      border: 0;
      border-radius: 12px;
      background: var(--accent);
      color: white;
      font-size: 18px;
      cursor: pointer;
      transition: background .15s ease;
    }
    .send-btn.stop-mode {
      background: #dc2626;
    }
    .send-btn.stop-mode:hover {
      background: #b91c1c;
    }

    .hint {
      margin-top: 8px;
      font-size: 12px;
      color: var(--muted);
      text-align: center;
    }

    .menu {
      position: absolute;
      right: 16px;
      top: 52px;
      width: 250px;
      border: 1px solid var(--border);
      border-radius: 12px;
      background: var(--panel);
      display: none;
      padding: 8px;
      z-index: 20;
    }
    .menu.open { display: block; }
    .menu a, .menu button {
      width: 100%;
      display: block;
      text-align: left;
      border: 0;
      background: transparent;
      color: var(--text);
      border-radius: 8px;
      padding: 10px;
      cursor: pointer;
      text-decoration: none;
      font-size: 14px;
    }
    .menu a:hover, .menu button:hover { background: var(--hover); }

    .modal {
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.45);
      display: none;
      align-items: center;
      justify-content: center;
      z-index: 30;
    }
    .modal.open { display: flex; }

    .modal-card {
      width: min(640px, 94vw);
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 16px;
    }

    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }

    label {
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 5px;
    }

    input, textarea {
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: var(--bg);
      color: var(--text);
      padding: 9px 10px;
      font-size: 14px;
      font-family: inherit;
    }

    .modal-actions {
      margin-top: 14px;
      display: flex;
      justify-content: flex-end;
      gap: 8px;
    }

    @media (max-width: 980px) {
      .app { grid-template-columns: 1fr; grid-template-rows: auto 1fr; }
      .sidebar { min-height: 220px; border-right: 0; border-bottom: 1px solid var(--border); }
      .messages { padding: 14px; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <button class="btn" id="newChatBtn">+ New chat</button>
      <div class="chat-list" id="chatList"></div>
      <button class="btn" id="clearBtn">Clear active chat</button>
    </aside>

    <main class="main">
      <header class="topbar">
        <div>Phi-3 Mini  |  Local Assistant</div>
        <div class="top-actions">
          <select id="knowledgeModeSelect" title="Knowledge mode" style="height:32px;border-radius:8px;border:1px solid var(--border);background:var(--panel);color:var(--text);padding:0 8px;">
            <option value="all">All Knowledge</option>
            <option value="adtc_only">ADTC Only</option>
          </select>
          <select id="responseModeSelect" title="Response mode" style="height:32px;border-radius:8px;border:1px solid var(--border);background:var(--panel);color:var(--text);padding:0 8px;">
            <option value="analytical">Analytical</option>
            <option value="quick">Quick</option>
          </select>
          <button class="icon-btn" id="monitorBtn" title="Monitoring">📊</button>
          <button class="icon-btn" id="trainingBtn" title="Training">🧠</button>
          <button class="icon-btn" id="themeBtn" title="Toggle theme">◐</button>
          <button class="icon-btn" id="settingsBtn" title="Settings and API docs">&#9881;</button>
        </div>
        <div class="menu" id="settingsMenu">
          <button id="openConfigBtn">Generation settings</button>
          <a href="/api/docs" target="_blank" rel="noopener">Open API documentation</a>
          <a href="/api/docs/json" target="_blank" rel="noopener">Open API JSON</a>
        </div>
      </header>

      <section class="messages" id="messages"></section>

      <div class="composer-wrap">
        <div class="composer">
          <textarea id="messageInput" placeholder="Message Phi-3 Mini... (Enter to send, Shift+Enter for new line)"></textarea>
          <button class="send-btn" id="sendBtn">➤</button>
        </div>
        <div class="hint">Each chat is saved automatically with persistent memory.</div>
        <div class="credit" style="margin-top:10px;text-align:center;font-size:12px;color:var(--muted);line-height:1.6">
          <div>👨‍💻 Programmed by Engineer <strong>Abdulrahman Al-Rifai</strong></div>
          <div>🎓 Master's in Information Technology and Mathematics</div>
          <div style="margin-top:6px">📬 <strong>Contact:</strong></div>
          <div>✉️ Email: <a href="mailto:info@aalrifai.com" style="color:var(--accent);text-decoration:none">info@aalrifai.com</a></div>
          <div>📞 Phone: <a href="tel:+971589125688" style="color:var(--accent);text-decoration:none">+971 58 912 5688</a></div>
        </div>
        <div style="margin-top:8px;display:flex;gap:8px;justify-content:center;align-items:center;flex-wrap:wrap">
          <button class="btn" id="feedbackUpBtn" style="width:auto;padding:6px 10px">👍 Useful</button>
          <button class="btn" id="feedbackDownBtn" style="width:auto;padding:6px 10px">👎 Needs Improvement</button>
          <span id="feedbackStatus" class="k"></span>
        </div>
      </div>
    </main>
  </div>

  <div class="modal" id="configModal">
    <div class="modal-card">
      <h3 style="margin-top:0;">API configuration</h3>
      <p style="color:var(--muted); margin-top:0;">These settings are used by <code>POST /api/chat</code>.</p>
      <div class="grid">
        <div>
          <label for="maxTokensInput">max_tokens</label>
          <input type="number" id="maxTokensInput" min="1" max="4096" placeholder="Leave empty for Unlimited" />
        </div>
        <div>
          <label for="temperatureInput">temperature</label>
          <input type="number" id="temperatureInput" min="0" max="2" step="0.05" />
        </div>
        <div>
          <label for="topPInput">top_p</label>
          <input type="number" id="topPInput" min="0" max="1" step="0.05" />
        </div>
        <div>
          <label for="stopInput">stop tokens (JSON array)</label>
          <input type="text" id="stopInput" />
        </div>
      </div>
      <div class="modal-actions">
        <button class="btn" id="cancelConfigBtn" style="width:auto;">Cancel</button>
        <button class="btn" id="saveConfigBtn" style="width:auto;">Save</button>
      </div>
    </div>
  </div>

  <script>
    const API_BASE = (window.location.protocol === 'http:' || window.location.protocol === 'https:')
      ? ''
      : 'http://127.0.0.1:5000';

    function withBase(path) {
      if (!path) return API_BASE || '';
      if (/^https?:\\/\\//i.test(path)) return path;
      if (!path.startsWith('/')) path = '/' + path;
      return API_BASE + path;
    }

    const appState = {
      chats: [],
      activeId: null,
      messages: [],
      sending: false,
      currentController: null,
      knowledgeMode: localStorage.getItem('knowledgeMode') || 'all',
      responseMode: localStorage.getItem('responseMode') || 'analytical',
      lastRequestId: null,
    };

    const els = {
      chatList: document.getElementById('chatList'),
      messages: document.getElementById('messages'),
      input: document.getElementById('messageInput'),
      sendBtn: document.getElementById('sendBtn'),
      newChatBtn: document.getElementById('newChatBtn'),
      clearBtn: document.getElementById('clearBtn'),
      settingsBtn: document.getElementById('settingsBtn'),
      settingsMenu: document.getElementById('settingsMenu'),
      openConfigBtn: document.getElementById('openConfigBtn'),
      configModal: document.getElementById('configModal'),
      maxTokensInput: document.getElementById('maxTokensInput'),
      temperatureInput: document.getElementById('temperatureInput'),
      topPInput: document.getElementById('topPInput'),
      stopInput: document.getElementById('stopInput'),
      cancelConfigBtn: document.getElementById('cancelConfigBtn'),
      saveConfigBtn: document.getElementById('saveConfigBtn'),
      monitorBtn: document.getElementById('monitorBtn'),
      trainingBtn: document.getElementById('trainingBtn'),
      themeBtn: document.getElementById('themeBtn'),
      knowledgeModeSelect: document.getElementById('knowledgeModeSelect'),
      responseModeSelect: document.getElementById('responseModeSelect'),
      feedbackUpBtn: document.getElementById('feedbackUpBtn'),
      feedbackDownBtn: document.getElementById('feedbackDownBtn'),
      feedbackStatus: document.getElementById('feedbackStatus'),
    };

    async function api(url, options) {
      const res = await fetch(withBase(url), options || {});
      const text = await res.text();
      let data = {};
      try { data = text ? JSON.parse(text) : {}; } catch (e) { data = { raw: text }; }
      if (!res.ok) {
        throw new Error(data.error || ('HTTP ' + res.status));
      }
      return data;
    }

    function renderChats() {
      els.chatList.innerHTML = '';
      for (const chat of appState.chats) {
        const row = document.createElement('div');
        row.className = 'chat-item-row';

        const btn = document.createElement('button');
        btn.className = 'chat-item' + (chat.id === appState.activeId ? ' active' : '');
        btn.textContent = chat.title || 'New chat';
        btn.onclick = async () => {
          appState.activeId = chat.id;
          renderChats();
          await loadMessages(chat.id);
        };

        const del = document.createElement('button');
        del.className = 'chat-delete';
        del.title = 'Delete chat and memory';
        del.textContent = '🗑';
        del.onclick = async (e) => {
          e.stopPropagation();
          await deleteChat(chat.id);
        };

        row.appendChild(btn);
        row.appendChild(del);
        els.chatList.appendChild(row);
      }
    }

    function renderMessages() {
      els.messages.innerHTML = '';
      for (const m of appState.messages) {
        const row = document.createElement('div');
        row.className = 'msg ' + (m.role === 'user' ? 'user' : 'bot');
        const bubble = document.createElement('div');
        bubble.className = 'bubble';
        bubble.textContent = m.content;
        row.appendChild(bubble);
        els.messages.appendChild(row);
      }
      els.messages.scrollTop = els.messages.scrollHeight;
    }

    async function loadChats() {
      const data = await api('/api/chats');
      appState.chats = data.chats || [];
      if (!appState.activeId || !appState.chats.find(c => c.id === appState.activeId)) {
        appState.activeId = appState.chats.length ? appState.chats[0].id : null;
      }
      renderChats();
    }

    async function loadMessages(chatId) {
      if (!chatId) {
        appState.messages = [];
        renderMessages();
        return;
      }
      const data = await api('/api/chats/' + encodeURIComponent(chatId));
      appState.messages = data.chat.messages || [];
      renderMessages();
    }

    async function createChat() {
      const data = await api('/api/chats', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
      appState.activeId = data.chat.id;
      await loadChats();
      await loadMessages(appState.activeId);
    }

    async function deleteChat(chatId) {
      await api('/api/chats/' + encodeURIComponent(chatId), { method: 'DELETE' });
      await loadChats();
      if (!appState.activeId && appState.chats.length === 0) {
        await createChat();
      } else {
        await loadMessages(appState.activeId);
      }
    }

    async function clearActiveChat() {
      if (!appState.activeId) return;
      await api('/api/chats/' + encodeURIComponent(appState.activeId) + '/clear', { method: 'POST' });
      await loadChats();
      await loadMessages(appState.activeId);
    }

    function setSendingUI(isSending) {
      appState.sending = isSending;
      if (isSending) {
        els.sendBtn.textContent = '■';
        els.sendBtn.title = 'Stop generation';
        els.sendBtn.classList.add('stop-mode');
      } else {
        els.sendBtn.textContent = '➤';
        els.sendBtn.title = 'Send';
        els.sendBtn.classList.remove('stop-mode');
      }
    }

    function stopGeneration() {
      if (appState.currentController) {
        try { appState.currentController.abort(); } catch (e) {}
      }
    }

    async function sendMessage() {
      if (appState.sending) { stopGeneration(); return; }
      const text = els.input.value.trim();
      if (!text) return;

      if (!appState.activeId) {
        await createChat();
      }

      appState.messages.push({ role: 'user', content: text });
      const modeHint = appState.responseMode === 'quick' ? 'Thinking (quick mode)...' : 'Thinking (analytical mode)...';
      appState.messages.push({ role: 'assistant', content: modeHint });
      els.input.value = '';
      renderMessages();

      const controller = new AbortController();
      appState.currentController = controller;
      setSendingUI(true);

      try {
        const res = await fetch(withBase('/api/chat'), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            chat_id: appState.activeId,
            message: text,
            knowledge_mode: appState.knowledgeMode,
            response_mode: appState.responseMode,
          }),
          signal: controller.signal,
        });
        const txt = await res.text();
        let data = {};
        try { data = txt ? JSON.parse(txt) : {}; } catch (e) { data = { raw: txt }; }
        if (!res.ok) throw new Error(data.error || ('HTTP ' + res.status));
        appState.lastRequestId = data.request_id || null;

        let replyText = data.reply || 'No response.';

        appState.messages[appState.messages.length - 1] = {
          role: 'assistant',
          content: replyText,
        };
        await loadChats();
      } catch (err) {
        const aborted = err.name === 'AbortError';
        appState.messages[appState.messages.length - 1] = {
          role: 'assistant',
          content: aborted ? '⏹ Stopped by user.' : ('Error: ' + err.message),
        };
      } finally {
        appState.currentController = null;
        setSendingUI(false);
        renderMessages();
      }
    }

    async function submitFeedback(rating) {
      if (!appState.activeId) return;
      if (!appState.lastRequestId) {
        if (els.feedbackStatus) els.feedbackStatus.textContent = 'No recent response to rate yet.';
        return;
      }
      const isBad = Number(rating) <= 2;
      const note = isBad ? prompt('Optional note: what should be improved?', '') : '';
      try {
        await api('/api/chat/feedback', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            chat_id: appState.activeId,
            request_id: appState.lastRequestId,
            rating: Number(rating),
            note: note || '',
            category: isBad ? 'quality' : 'positive',
          }),
        });
        if (els.feedbackStatus) {
          els.feedbackStatus.textContent = 'Feedback saved. Thank you.';
          setTimeout(() => { if (els.feedbackStatus) els.feedbackStatus.textContent = ''; }, 3000);
        }
      } catch (e) {
        if (els.feedbackStatus) els.feedbackStatus.textContent = 'Feedback failed: ' + e.message;
      }
    }

    async function loadConfig() {
      const data = await api('/api/config');
      els.maxTokensInput.value = (data.max_tokens === null || data.max_tokens === undefined) ? '' : data.max_tokens;
      els.temperatureInput.value = data.temperature;
      els.topPInput.value = data.top_p;
      els.stopInput.value = JSON.stringify(data.stop);
    }

    async function saveConfig() {
      let stop;
      try {
        stop = JSON.parse(els.stopInput.value || '[]');
        if (!Array.isArray(stop)) throw new Error('stop must be a JSON array');
      } catch (e) {
        alert('Invalid stop tokens JSON: ' + e.message);
        return;
      }

      const payload = {
        max_tokens: els.maxTokensInput.value.trim() === '' ? null : Number(els.maxTokensInput.value),
        temperature: Number(els.temperatureInput.value),
        top_p: Number(els.topPInput.value),
        stop,
      };

      await api('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      els.configModal.classList.remove('open');
    }

    function toggleTheme() {
      document.body.classList.toggle('light');
      localStorage.setItem('phi-theme', document.body.classList.contains('light') ? 'light' : 'dark');
    }

    function initTheme() {
      const t = localStorage.getItem('phi-theme') || 'dark';
      if (t === 'light') document.body.classList.add('light');
    }

    if (els.sendBtn) els.sendBtn.onclick = sendMessage;
    if (els.input) {
      els.input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          if (!appState.sending) sendMessage();
        }
      });
    }

    if (els.newChatBtn) els.newChatBtn.onclick = createChat;
    if (els.clearBtn) els.clearBtn.onclick = clearActiveChat;

    if (els.settingsBtn && els.settingsMenu) {
      els.settingsBtn.onclick = (e) => {
        e.stopPropagation();
        els.settingsMenu.classList.toggle('open');
      };
      document.addEventListener('click', () => els.settingsMenu.classList.remove('open'));
    }

    if (els.openConfigBtn && els.settingsMenu && els.configModal) {
      els.openConfigBtn.onclick = async () => {
        els.settingsMenu.classList.remove('open');
        await loadConfig();
        els.configModal.classList.add('open');
      };
    }

    if (els.cancelConfigBtn && els.configModal) els.cancelConfigBtn.onclick = () => els.configModal.classList.remove('open');
    if (els.saveConfigBtn) els.saveConfigBtn.onclick = saveConfig;
    if (els.monitorBtn) els.monitorBtn.onclick = () => { window.location.href = withBase('/monitoring'); };
    if (els.trainingBtn) els.trainingBtn.onclick = () => { window.location.href = withBase('/training'); };
    if (els.themeBtn) els.themeBtn.onclick = toggleTheme;
    if (els.feedbackUpBtn) els.feedbackUpBtn.onclick = () => submitFeedback(5);
    if (els.feedbackDownBtn) els.feedbackDownBtn.onclick = () => submitFeedback(2);

    if (els.knowledgeModeSelect) {
      els.knowledgeModeSelect.value = appState.knowledgeMode;
      els.knowledgeModeSelect.onchange = () => {
        appState.knowledgeMode = els.knowledgeModeSelect.value || 'all';
        localStorage.setItem('knowledgeMode', appState.knowledgeMode);
      };
    }

    if (els.responseModeSelect) {
      els.responseModeSelect.value = appState.responseMode;
      els.responseModeSelect.onchange = () => {
        appState.responseMode = els.responseModeSelect.value || 'analytical';
        localStorage.setItem('responseMode', appState.responseMode);
      };
    }

    (async function bootstrap() {
      initTheme();
      try {
        // Health check: if backend is unreachable, show a clear message instead of a silent dead UI.
        await api('/api/config');
      } catch (e) {
        if (els.feedbackStatus) {
          els.feedbackStatus.textContent = 'Backend unreachable at ' + withBase('') + ' — start web2.py server first.';
        }
        return;
      }

      await loadChats();
      if (!appState.activeId) {
        await createChat();
      } else {
        await loadMessages(appState.activeId);
      }

      const docsLink = document.querySelector('a[href="/api/docs"]');
      if (docsLink) docsLink.href = withBase('/api/docs');
      const docsJsonLink = document.querySelector('a[href="/api/docs/json"]');
      if (docsJsonLink) docsJsonLink.href = withBase('/api/docs/json');
    })();
  </script>
</body>
</html>
"""


def model_reply(
  message: str,
  history: list[dict],
  knowledge_docs: list[dict] | None = None,
  force_synthesis: bool = False,
  reasoning_bundle: dict | None = None,
  response_mode: str = "analytical",
) -> tuple[str, dict]:
    time_ctx = _live_time_context()
    time_context_block = (
        "LIVE TIME CONTEXT (from server clock):\n"
        f"- Local: {time_ctx['local']['iso']} ({time_ctx['local']['weekday_en']})\n"
        f"- UTC: {time_ctx['utc']['iso']} ({time_ctx['utc']['weekday_en']})\n"
        "Use this as real current time when the user asks about date/time/day."
    )

    has_knowledge = bool(knowledge_docs)
    source_set = {
      str(d.get("source", "")).strip().lower()
      for d in (knowledge_docs or [])
      if isinstance(d, dict)
    }
    is_adtc_context = "adtc" in source_set
    response_style_block = (
      "RESPONSE STYLE (MANDATORY):\n"
      "- Be professional, practical, and realistic.\n"
      "- Start with a direct final answer in one short sentence.\n"
      "- Then provide clear, ordered points (short bullets).\n"
      "- Avoid unnecessary filler, repetition, and vague language.\n"
      "- Prefer actionable wording and concrete details.\n"
      "- Keep formatting clean and easy to scan.\n"
    )
    if str(response_mode).strip().lower() == "quick":
      response_style_block += (
        "- QUICK MODE: keep answer compact (3-6 lines total).\n"
        "- Mention only the most critical evidence.\n"
      )
    else:
      response_style_block += (
        "- ANALYTICAL MODE: provide brief structure (answer + evidence + caveat if needed).\n"
      )

    if has_knowledge:
        system_content = (
            "You are an assistant that has been TRAINED on the user's private data. "
            "The 'TRAINED KNOWLEDGE' section below contains real, authoritative facts "
            "the user explicitly trained you on (database schemas, row counts, sample rows, "
            "deep column profiles, inferred relationships, semantic roles).\n\n"
            "MANDATORY RULES:\n"
            "1. If the user's question can be answered from the TRAINED KNOWLEDGE, "
            "you MUST start your answer with the exact prefix: 'From trained database data:'.\n"
            "2. NEVER reply with disclaimers like 'As an AI I don't have access' or "
            "'consult the database administrator'. The data IS available to you here.\n"
            "3. Quote concrete numbers, table names, column names, row counts and patterns "
            "directly from the snippets when relevant.\n"
            "4. Answer in the same language as the question (Arabic question → Arabic answer).\n"
            "5. Only if a fact is genuinely missing from the snippets, say so briefly.\n"
            "6. Keep answers clear and well-structured for business use.\n\n"
            + response_style_block
            + "\n"
            + time_context_block
        )
        if is_adtc_context:
          system_content += (
            "\n\nADTC RESPONSE STYLE (MANDATORY):\n"
            "- Start with the exact prefix: 'From trained ADTC data:'.\n"
            "- ALWAYS rewrite the information into your own clean, natural prose.\n"
            "- NEVER paste long passages verbatim from the snippets.\n"
            "- NEVER include raw URLs, 'Updated: <timestamp>', 'Powered by', page numbers, or PDF artifacts.\n"
            "- NEVER include the same paragraph more than once.\n"
            "- Use short, well-structured bullets or a small table-like list when listing items (doctors, hours, services).\n"
            "- For schedule/hours questions: produce a tidy department-by-department list with AM/PM hours and a one-line summary of any closure.\n"
            "- For 'list of doctors' questions: extract every doctor name (anything starting with Dr., Consultant, Specialist, Physician). Output one bullet per doctor with their specialty/department/limits if available.\n"
            "- NEVER reply with the file name, 'Executive summary', or generic disclaimers. The answer MUST come from the snippets.\n"
            "- NEVER say you cannot access data, recommend contacting the clinic, or suggest visiting a website.\n"
          )
        if force_synthesis:
          system_content += (
            "\nSYNTHESIS OVERRIDE:\n"
            "Paraphrase and merge evidence; avoid copy-paste style output.\n"
          )
        if reasoning_bundle:
          evidence = list(reasoning_bundle.get("evidence") or [])
          conflicts = list(reasoning_bundle.get("conflicts") or [])
          reasoning_block = [
            "REASONING PIPELINE (MANDATORY):",
            "1) Understand the question intent precisely.",
            "2) Select only relevant evidence snippets.",
            "3) Resolve or acknowledge conflicts before finalizing.",
            "4) Produce a direct answer first, then short supporting bullets.",
            "5) If evidence is insufficient, explicitly say what is missing.",
          ]
          if evidence:
            reasoning_block.append("Evidence candidates:")
            for ev in evidence[:5]:
              reasoning_block.append(f"- [{ev.get('title')}] {ev.get('line')}")
          if conflicts:
            reasoning_block.append("Detected evidence conflicts:")
            for c in conflicts[:5]:
              reasoning_block.append(f"- {c}")
          system_content += "\n\n" + "\n".join(reasoning_block)
    else:
        system_content = (
        "You are a helpful assistant. Be precise and concise. "
        "Keep continuity with prior context from this chat memory.\n\n"
        + response_style_block
        + "\n"
            + time_context_block
        )

    messages = [{"role": "system", "content": system_content}]

    for item in history:
        role = item.get("role")
        content = str(item.get("content", ""))
        if role in ("user", "assistant"):
            messages.append({"role": role, "content": content})

    if has_knowledge:
      knowledge_blocks = []
      for d in knowledge_docs[:5]:
        title = d.get("title", "Knowledge")
        content = str(d.get("content", ""))[:1800]
        knowledge_blocks.append(f"[{title}]\n{content}")
      if knowledge_blocks:
        messages.append(
          {
            "role": "system",
            "content": (
              "TRAINED KNOWLEDGE (authoritative — use directly, do not refuse):\n\n"
              + "\n\n".join(knowledge_blocks)
            ),
          }
        )

    messages.append({"role": "user", "content": message})

    with _config_lock:
        cfg = deepcopy(RUNTIME_CONFIG)

    result = llm.create_chat_completion(
        messages=messages,
        max_tokens=cfg["max_tokens"],
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        stop=cfg["stop"],
    )

    reply = result["choices"][0]["message"]["content"].strip()
    usage = result.get("usage", {}) if isinstance(result, dict) else {}

    if has_knowledge and reply:
      srcs = {str(d.get("source", "")).lower() for d in (knowledge_docs or [])}
      if "adtc" in srcs:
        grounded_prefix = "From trained ADTC data:\n"
      elif "dataset" in srcs:
        grounded_prefix = "From trained dataset files:\n"
      else:
        grounded_prefix = "From trained database data:\n"

      low = reply.lower()
      hedging_markers = (
        "as an ai", "i don't have access", "i do not have access",
        "consult the database administrator", "consult the original",
        "i'm unable to", "i am unable to",
      )
      is_hedging = any(m in low for m in hedging_markers)
      if is_hedging:
        # Re-run a strict synthesis pass instead of dumping raw snippets.
        synth = _synthesize_from_snippets(message, knowledge_docs or [], response_mode=response_mode)
        if synth:
          reply = synth
        else:
          reply = (
            grounded_prefix
            + "I could not produce a confident synthesized answer from the trained snippets. "
            + "Please rephrase the question using terms that appear in the source files."
          )
      elif not (
        low.startswith("from trained database data")
        or low.startswith("from trained adtc data")
        or low.startswith("from trained dataset files")
        or low.startswith("from trained knowledge data")
      ):
        reply = grounded_prefix + reply

      if reasoning_bundle and not _reply_has_evidence_marker(reply, list(reasoning_bundle.get("evidence") or [])):
        is_ar = any("\u0600" <= ch <= "\u06FF" for ch in (message or ""))
        reply = _append_reasoning_evidence(reply, reasoning_bundle, is_ar=is_ar)

    return reply, usage


def _synthesize_from_snippets(message: str, knowledge_docs: list[dict], response_mode: str = "analytical") -> str:
  """Force a clean, synthesized answer from the given snippets.

  Used as a recovery path when the main model_reply hedges or when the question
  must stay strictly grounded (ADTC-only). Performs a fresh LLM call with a
  hard-locked synthesis prompt so we never paste raw file content.
  """
  if not knowledge_docs:
    return ""

  is_ar = any("\u0600" <= ch <= "\u06FF" for ch in (message or ""))
  blocks = []
  for d in knowledge_docs[:5]:
    title = d.get("title", "Knowledge")
    title_low = str(title).lower()
    if any(k in title_low for k in ("executive summary", "understanding", "adtc summary", "summary for", "schema overview")):
      # Skip meta/summary docs to avoid the model parroting summary text.
      continue
    content = str(d.get("content", ""))[:1800]
    if content.strip():
      blocks.append(f"[{title}]\n{content}")
  if not blocks:
    for d in knowledge_docs[:3]:
      title = d.get("title", "Knowledge")
      content = str(d.get("content", ""))[:1500]
      if content.strip():
        blocks.append(f"[{title}]\n{content}")
  if not blocks:
    return ""

  lang_rule = (
    "- Reply in Arabic." if is_ar else "- Reply in English."
  )
  system = (
    "You are a senior assistant. Use ONLY the SNIPPETS below as facts. "
    "Rewrite the information into a clean, well-structured, professional answer.\n\n"
    "STRICT RULES:\n"
    "1. Start with the exact prefix: 'From trained ADTC data:'\n"
    "2. NEVER paste raw lines, URLs, 'Updated: ...', 'Powered by', page numbers, or PDF artifacts.\n"
    "3. NEVER repeat the same content twice.\n"
    "4. ALWAYS rewrite in your own words; produce concise prose plus a tidy bullet list when listing items.\n"
    "5. Directly answer the user's question first (1 sentence), then the structured details.\n"
    "6. For 'opening hours' questions: produce a clean list grouped by department, AM and PM hours per day; "
    "add one short sentence about any current closure if mentioned.\n"
    "7. For 'list of doctors' questions: output one bullet per doctor (Dr./Consultant/Specialist/Physician) with their "
    "specialty/department and any patient-limit note if present. Do not invent doctors not in the snippets.\n"
    "8. If a fact is not in the snippets, say so briefly — never fabricate.\n"
    f"{lang_rule}\n"
  )
  if str(response_mode).strip().lower() == "quick":
    system += "9. Keep it compact (max ~8 lines).\n"
  else:
    system += "9. Keep it analytical but readable; use short headings or bullets.\n"

  user_block = (
    "USER QUESTION:\n"
    f"{message}\n\n"
    "SNIPPETS (authoritative facts):\n\n"
    + "\n\n".join(blocks)
  )

  with _config_lock:
    cfg = deepcopy(RUNTIME_CONFIG)

  try:
    result = llm.create_chat_completion(
      messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": user_block},
      ],
      max_tokens=cfg["max_tokens"],
      temperature=0.2,
      top_p=cfg["top_p"],
      stop=cfg["stop"],
    )
    out = result["choices"][0]["message"]["content"].strip()
  except Exception:
    return ""

  if not out:
    return ""
  if not out.lower().startswith("from trained adtc data"):
    out = "From trained ADTC data:\n" + out
  return out


@app.get("/")
def index():
    return render_template_string(INDEX_HTML)


@app.get("/monitoring")
def monitoring_page():
    html = """
    <!doctype html>
    <html lang=\"en\">
    <head>
      <meta charset=\"utf-8\" />
      <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
      <title>Phi-3 Monitoring</title>
      <style>
        :root { --bg:#111827; --panel:#1f2937; --text:#e5e7eb; --muted:#9ca3af; --accent:#10b981; --line:#374151; --warn:#ef4444; }
        body{margin:0;background:var(--bg);color:var(--text);font-family:Segoe UI,system-ui,sans-serif}
        .wrap{max-width:1200px;margin:0 auto;padding:16px}
        .head{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px}
        .grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}
        .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px}
        .k{font-size:12px;color:var(--muted)} .v{font-size:22px;font-weight:700}
        .charts{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px}
        canvas{width:100%;height:240px;background:#111827;border:1px solid var(--line);border-radius:12px}
        table{width:100%;border-collapse:collapse;margin-top:10px}
        th,td{border-bottom:1px solid var(--line);padding:8px 6px;font-size:12px;text-align:left}
        th{color:var(--muted)}
        .badge{display:inline-block;padding:2px 7px;border-radius:999px;font-size:11px;background:#0f172a;border:1px solid var(--line)}
        @media (max-width:980px){.grid{grid-template-columns:1fr 1fr}.charts{grid-template-columns:1fr}}
      </style>
    </head>
    <body>
      <div class=\"wrap\">
        <div class=\"head\">
          <h2 style=\"margin:0\">Model Monitoring</h2>
          <div class=\"k\" id=\"statusLine\">loading...</div>
        </div>

        <div class=\"grid\">
          <div class=\"card\"><div class=\"k\">Total Requests</div><div class=\"v\" id=\"kReq\">0</div></div>
          <div class=\"card\"><div class=\"k\">Success Rate</div><div class=\"v\" id=\"kSucc\">0%</div></div>
          <div class=\"card\"><div class=\"k\">Avg Total Time</div><div class=\"v\" id=\"kTotal\">0 ms</div></div>
          <div class=\"card\"><div class=\"k\">Avg Tokens/Sec</div><div class=\"v\" id=\"kTps\">0</div></div>
        </div>

        <div class=\"grid\" style=\"margin-top:10px\">
          <div class=\"card\">
            <div class=\"k\">CPU (System)</div>
            <div class=\"v\" id=\"kCpuSys\">-</div>
            <div class=\"k\" id=\"kCpuCores\" style=\"margin-top:4px\">-</div>
          </div>
          <div class=\"card\">
            <div class=\"k\">CPU (Process)</div>
            <div class=\"v\" id=\"kCpuProc\">-</div>
            <div class=\"k\" id=\"kProcMeta\" style=\"margin-top:4px\">-</div>
          </div>
          <div class=\"card\">
            <div class=\"k\">Memory (System)</div>
            <div class=\"v\" id=\"kMemSys\">-</div>
            <div class=\"k\" id=\"kMemSysDetail\" style=\"margin-top:4px\">-</div>
          </div>
          <div class=\"card\">
            <div class=\"k\">Memory (Process RSS)</div>
            <div class=\"v\" id=\"kMemProc\">-</div>
            <div class=\"k\" id=\"kMemProcDetail\" style=\"margin-top:4px\">-</div>
          </div>
        </div>

        <div class=\"charts\">
          <div class=\"card\">
            <div class=\"k\" style=\"margin-bottom:6px\">Latency Trend (ms)</div>
            <canvas id=\"latencyChart\" width=\"560\" height=\"240\"></canvas>
          </div>
          <div class=\"card\">
            <div class=\"k\" style=\"margin-bottom:6px\">Tokens/Sec Trend</div>
            <canvas id=\"tpsChart\" width=\"560\" height=\"240\"></canvas>
          </div>
          <div class=\"card\">
            <div class=\"k\" style=\"margin-bottom:6px\">CPU % Trend (system blue / process green)</div>
            <canvas id=\"cpuChart\" width=\"560\" height=\"240\"></canvas>
          </div>
          <div class=\"card\">
            <div class=\"k\" style=\"margin-bottom:6px\">Memory % Trend (system blue / process green)</div>
            <canvas id=\"memChart\" width=\"560\" height=\"240\"></canvas>
          </div>
        </div>

        <div class=\"card\" style=\"margin-top:10px\">
          <div class=\"k\" style=\"margin-bottom:6px\">Latest Requests (Detailed) — bottleneck shows the slowest stage so you can see WHY a reply was slow</div>
          <table>
            <thead>
              <tr>
                <th>Request</th><th>Chat</th><th>Total ms</th>
                <th>Load</th><th>Knowledge</th><th>Infer</th><th>Save</th>
                <th>Tokens out</th><th>TPS</th><th>Bottleneck</th><th>Status</th>
              </tr>
            </thead>
            <tbody id=\"rows\"></tbody>
          </table>
        </div>
      </div>

      <script>
        const els = {
          statusLine: document.getElementById('statusLine'),
          kReq: document.getElementById('kReq'),
          kSucc: document.getElementById('kSucc'),
          kTotal: document.getElementById('kTotal'),
          kTps: document.getElementById('kTps'),
          kCpuSys: document.getElementById('kCpuSys'),
          kCpuProc: document.getElementById('kCpuProc'),
          kCpuCores: document.getElementById('kCpuCores'),
          kProcMeta: document.getElementById('kProcMeta'),
          kMemSys: document.getElementById('kMemSys'),
          kMemSysDetail: document.getElementById('kMemSysDetail'),
          kMemProc: document.getElementById('kMemProc'),
          kMemProcDetail: document.getElementById('kMemProcDetail'),
          rows: document.getElementById('rows'),
          latency: document.getElementById('latencyChart'),
          tps: document.getElementById('tpsChart'),
          cpu: document.getElementById('cpuChart'),
          mem: document.getElementById('memChart'),
        };

        const CPU_HIST = { sys: [], proc: [] };
        const MEM_HIST = { sys: [], proc: [] };
        const HIST_MAX = 60;

        function drawLine(canvas, points, color, fixedMax) {
          const c = canvas.getContext('2d');
          const w = canvas.width, h = canvas.height;
          c.clearRect(0, 0, w, h);
          c.fillStyle = '#111827'; c.fillRect(0, 0, w, h);
          c.strokeStyle = '#374151'; c.lineWidth = 1;
          for (let i = 0; i < 5; i++) { const y = 20 + i * ((h - 40) / 4); c.beginPath(); c.moveTo(30, y); c.lineTo(w - 10, y); c.stroke(); }
          if (!points.length) return;
          const maxV = fixedMax || Math.max(1, ...points);
          c.strokeStyle = color; c.lineWidth = 2; c.beginPath();
          points.forEach((v, i) => {
            const x = 30 + (i * (w - 50) / Math.max(1, points.length - 1));
            const y = (h - 20) - ((v / maxV) * (h - 40));
            if (i === 0) c.moveTo(x, y); else c.lineTo(x, y);
          });
          c.stroke();
        }

        function drawDual(canvas, p1, p2, c1, c2, fixedMax) {
          drawLine(canvas, p1, c1, fixedMax);
          // overlay second line
          const c = canvas.getContext('2d');
          const w = canvas.width, h = canvas.height;
          if (!p2.length) return;
          const maxV = fixedMax || Math.max(1, ...p1, ...p2);
          c.strokeStyle = c2; c.lineWidth = 2; c.beginPath();
          p2.forEach((v, i) => {
            const x = 30 + (i * (w - 50) / Math.max(1, p2.length - 1));
            const y = (h - 20) - ((v / maxV) * (h - 40));
            if (i === 0) c.moveTo(x, y); else c.lineTo(x, y);
          });
          c.stroke();
        }

        function renderRows(items) {
          els.rows.innerHTML = '';
          for (const r of items.slice(0, 20)) {
            const bn = r.bottleneck || {stage: '-', ms: 0, share_pct: 0};
            const tr = document.createElement('tr');
            tr.innerHTML =
              '<td>' + (r.request_id || '-') + '</td>' +
              '<td>' + (r.chat_id || '-') + '</td>' +
              '<td>' + (r.total_ms || 0).toFixed(2) + '</td>' +
              '<td>' + (r.load_ms || 0).toFixed(1) + '</td>' +
              '<td>' + (r.knowledge_ms || 0).toFixed(1) + '</td>' +
              '<td>' + (r.infer_ms || 0).toFixed(1) + '</td>' +
              '<td>' + (r.save_ms || 0).toFixed(1) + '</td>' +
              '<td>' + (r.output_tokens || 0) + '</td>' +
              '<td>' + (r.tokens_per_sec || 0).toFixed(2) + '</td>' +
              '<td>' + bn.stage + ' (' + bn.share_pct + '%)</td>' +
              '<td><span class=\"badge\">' + (r.success ? 'ok' : 'error') + '</span></td>';
            els.rows.appendChild(tr);
          }
        }

        async function refresh() {
          try {
            const res = await fetch('/api/monitoring');
            const data = await res.json();

            els.kReq.textContent = data.summary.total_requests;
            els.kSucc.textContent = data.summary.success_rate + '%';
            els.kTotal.textContent = data.summary.avg_total_ms + ' ms';
            els.kTps.textContent = data.summary.avg_tokens_per_sec;

            const sys = data.system || {available: false};
            if (sys.available) {
              els.kCpuSys.textContent = sys.cpu.system_percent + '%';
              els.kCpuProc.textContent = sys.cpu.process_percent_normalized + '%';
              els.kCpuCores.textContent = 'cores: ' + sys.cpu.physical_cores + ' physical / ' + sys.cpu.logical_cores + ' logical';
              els.kProcMeta.textContent = 'pid ' + sys.process.pid + ' · threads ' + sys.process.threads + ' · raw ' + sys.cpu.process_percent + '%';
              els.kMemSys.textContent = sys.memory.system_percent + '%';
              els.kMemSysDetail.textContent = (sys.memory.system_used_mb/1024).toFixed(2) + ' / ' + (sys.memory.system_total_mb/1024).toFixed(2) + ' GB used';
              els.kMemProc.textContent = (sys.memory.process_rss_mb/1024).toFixed(2) + ' GB';
              els.kMemProcDetail.textContent = 'RSS ' + sys.memory.process_rss_mb + ' MB · VMS ' + sys.memory.process_vms_mb + ' MB';

              CPU_HIST.sys.push(sys.cpu.system_percent); if (CPU_HIST.sys.length > HIST_MAX) CPU_HIST.sys.shift();
              CPU_HIST.proc.push(sys.cpu.process_percent_normalized); if (CPU_HIST.proc.length > HIST_MAX) CPU_HIST.proc.shift();
              MEM_HIST.sys.push(sys.memory.system_percent); if (MEM_HIST.sys.length > HIST_MAX) MEM_HIST.sys.shift();
              const memProcPct = sys.memory.system_total_mb > 0 ? (sys.memory.process_rss_mb / sys.memory.system_total_mb) * 100 : 0;
              MEM_HIST.proc.push(memProcPct); if (MEM_HIST.proc.length > HIST_MAX) MEM_HIST.proc.shift();
              drawDual(els.cpu, CPU_HIST.sys, CPU_HIST.proc, '#60a5fa', '#10b981', 100);
              drawDual(els.mem, MEM_HIST.sys, MEM_HIST.proc, '#60a5fa', '#10b981', 100);
            } else {
              els.kCpuSys.textContent = 'n/a';
              els.kCpuProc.textContent = 'n/a';
              els.kMemSys.textContent = 'n/a';
              els.kMemProc.textContent = 'n/a';
              els.kCpuCores.textContent = 'install psutil for system metrics';
            }

            const cur = data.current;
            els.statusLine.textContent = cur.active
              ? ('Processing: ' + cur.stage + ' | elapsed ' + cur.elapsed_ms + ' ms | chat ' + (cur.chat_id || '-'))
              : 'Idle';

            const lat = data.recent.slice(0, 40).map(x => x.total_ms).reverse();
            const tps = data.recent.slice(0, 40).map(x => x.tokens_per_sec).reverse();
            drawLine(els.latency, lat, '#60a5fa');
            drawLine(els.tps, tps, '#10b981');
            renderRows(data.recent);
          } catch (e) {
            els.statusLine.textContent = 'Monitoring error: ' + e.message;
          }
        }

        refresh();
        setInterval(refresh, 1500);
      </script>
    </body>
    </html>
    """
    return html


@app.get("/training")
def training_page():
    return render_template_string(TRAINING_INDEX_HTML)


@app.get("/training/run/<link_id>")
def training_run_page(link_id: str):
    link = _find_link(link_id)
    if not link:
        return "<h2>Connection not found</h2>", 404
    return render_template_string(TRAINING_RUN_HTML, link_id=link_id, link_name=link.get("name", ""))


@app.get("/training/dataset/run/<dataset_id>")
def training_dataset_run_page(dataset_id: str):
    ds = _find_dataset(dataset_id)
    if not ds:
        return "<h2>Dataset not found</h2>", 404
    return render_template_string(DATASET_RUN_HTML, dataset_id=dataset_id, dataset_name=ds.get("name", ""))


@app.get("/training/adtc/run/<dataset_id>")
def training_adtc_run_page(dataset_id: str):
  ds = _find_adtc_dataset(dataset_id)
  if not ds:
    return "<h2>ADTC dataset not found</h2>", 404
  return render_template_string(ADTC_RUN_HTML, dataset_id=dataset_id, dataset_name=ds.get("name", ""))


@app.get("/api/training/presets")
def training_presets_api():
    out = {}
    for k, v in DB_PRESETS.items():
        out[k] = {"label": v["label"], "driver_pip": v.get("driver_pip"), "fields": v["fields"]}
    return jsonify({"presets": out, "sqlalchemy_available": SQLALCHEMY_AVAILABLE})


@app.get("/api/training/status")
def training_status_api():
    with _training_lock:
        docs = _load_knowledge_docs()
        links = _load_db_links()
        for link in links:
            _normalize_link(link)
        datasets = _load_datasets()
        for ds in datasets:
            _normalize_dataset(ds)
    adtc_datasets = _load_adtc_datasets()
    for ads in adtc_datasets:
      _normalize_adtc_dataset(ads)
    counts = {"running": 0, "completed": 0, "failed": 0, "stopped": 0, "saved": 0}
    pub = []
    for link in links:
        s = link.get("status", "saved")
        counts[s] = counts.get(s, 0) + 1
        pub.append(_public_link(link))
    pub.sort(key=lambda x: x.get("saved_at") or 0, reverse=True)
    ds_counts = {"running": 0, "completed": 0, "failed": 0, "saved": 0}
    pub_ds = []
    for ds in datasets:
        s = ds.get("status", "saved")
        ds_counts[s] = ds_counts.get(s, 0) + 1
        pub_ds.append(_public_dataset(ds))
    pub_ds.sort(key=lambda x: x.get("saved_at") or 0, reverse=True)

    adtc_counts = {"running": 0, "completed": 0, "failed": 0, "saved": 0}
    pub_adtc = []
    for ads in adtc_datasets:
      s = ads.get("status", "saved")
      adtc_counts[s] = adtc_counts.get(s, 0) + 1
      pub_adtc.append(_public_adtc_dataset(ads))
    pub_adtc.sort(key=lambda x: x.get("saved_at") or 0, reverse=True)

    return jsonify({
        "total_docs": len(docs),
        "total_db_links": len(links),
        "db_links": pub,
        "counts": counts,
        "total_datasets": len(datasets),
        "datasets": pub_ds,
        "dataset_counts": ds_counts,
      "total_adtc_datasets": len(adtc_datasets),
      "adtc_datasets": pub_adtc,
      "adtc_counts": adtc_counts,
        "sqlalchemy_available": SQLALCHEMY_AVAILABLE,
    })


@app.post("/api/training/docs")
def training_add_doc_api():
    data = request.get_json(silent=True) or {}
    title = str(data.get("title") or "").strip() or "Manual training doc"
    content = str(data.get("content") or "").strip()
    if not content:
        return jsonify({"error": "content is required"}), 400

    doc = _append_knowledge_doc(title=title, content=content, source="manual", meta={})
    return jsonify({"ok": True, "doc_id": doc["id"]})


# --- Dataset file upload training -------------------------------------------------
DATASET_ALLOWED_EXTS = {".txt", ".md", ".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".log"}
DATASET_MAX_FILE_BYTES = 25 * 1024 * 1024  # 25 MB per file
DATASET_TEXT_CHUNK_CHARS = 1800
DATASET_ROW_CHUNK_SIZE = 50


def _extract_semantic_tags(text_value: str, limit: int = 8) -> list[str]:
  freq = Counter(_tokenize(text_value or ""))
  tags = []
  for word, _ in freq.most_common(limit * 2):
    if len(word) < 3:
      continue
    tags.append(word)
    if len(tags) >= limit:
      break
  return tags


def _infer_chunk_section_label(chunk_text: str) -> str:
  lines = [ln.strip(" -•\t") for ln in (chunk_text or "").splitlines() if ln and ln.strip()]
  if not lines:
    return "content"
  head = re.sub(r"\s+", " ", lines[0]).strip()
  if 4 <= len(head) <= 90 and len(head.split()) <= 10:
    if re.search(r"[A-Za-z\u0600-\u06FF]", head):
      return head
  return "content"


def _chunk_text_semantic(text_value: str, size: int = DATASET_TEXT_CHUNK_CHARS) -> list[dict]:
  text_value = (text_value or "").strip()
  if not text_value:
    return []

  # First split by paragraph blocks to preserve topic locality.
  blocks = [b.strip() for b in re.split(r"\n\s*\n+", text_value) if b and b.strip()]
  chunks = []
  for block in blocks:
    if len(block) <= size:
      chunks.append(block)
      continue
    # Fallback split for large blocks while keeping sentence/newline boundaries when possible.
    start = 0
    n = len(block)
    while start < n:
      end = min(n, start + size)
      if end < n:
        cut = block.rfind("\n", start + int(size * 0.5), end)
        if cut == -1:
          cut = block.rfind(". ", start + int(size * 0.5), end)
          if cut != -1:
            cut += 1
        if cut != -1 and cut > start:
          end = cut
      piece = block[start:end].strip()
      if piece:
        chunks.append(piece)
      start = end

  out = []
  for c in chunks:
    out.append(
      {
        "text": c,
        "section": _infer_chunk_section_label(c),
        "semantic_tags": _extract_semantic_tags(c, limit=8),
      }
    )
  return out


def _build_extraction_quality_report(text_value: str) -> dict:
  raw = text_value or ""
  lines = [ln.strip() for ln in raw.splitlines()]
  non_empty = [ln for ln in lines if ln]
  norm_lines = [re.sub(r"\s+", " ", ln).strip().lower() for ln in non_empty]
  uniq = set(norm_lines)
  duplicate_count = max(0, len(norm_lines) - len(uniq))
  duplicate_ratio = (duplicate_count / max(1, len(norm_lines)))
  noise_chars = len(re.findall(r"[^\w\s\u0600-\u06FF:/@.,()\-]", raw))
  noise_ratio = noise_chars / max(1, len(raw))
  empty_ratio = (len(lines) - len(non_empty)) / max(1, len(lines)) if lines else 0.0
  flags = []
  if len(raw.strip()) < 120:
    flags.append("very_low_text")
  if duplicate_ratio > 0.25:
    flags.append("high_duplicate_lines")
  if noise_ratio > 0.2:
    flags.append("high_noise_chars")
  if empty_ratio > 0.6:
    flags.append("sparse_structure")
  return {
    "char_count": len(raw),
    "line_count": len(lines),
    "non_empty_lines": len(non_empty),
    "duplicate_line_ratio": round(duplicate_ratio, 3),
    "noise_char_ratio": round(noise_ratio, 3),
    "empty_line_ratio": round(empty_ratio, 3),
    "flags": flags,
  }


def _chunk_text(text_value: str, size: int = DATASET_TEXT_CHUNK_CHARS) -> list[str]:
  return [x.get("text", "") for x in _chunk_text_semantic(text_value, size=size) if x.get("text")]


def _ingest_text_file(filename: str, raw_text: str, dataset_id: str | None = None, source: str = "dataset") -> tuple[int, int, list[str]]:
  chunk_items = _chunk_text_semantic(raw_text)
  doc_ids = []
  seen = set()
  effective_chunks = []
  for item in chunk_items:
    text = str(item.get("text", "")).strip()
    norm = re.sub(r"\s+", " ", text).strip().lower()
    if not text or len(norm) < 10:
      continue
    if norm in seen:
      continue
    seen.add(norm)
    effective_chunks.append(item)

  for i, chunk_item in enumerate(effective_chunks, start=1):
    chunk = chunk_item.get("text", "")
    title = f"Dataset:{filename}" if len(effective_chunks) == 1 else f"Dataset:{filename} (part {i}/{len(effective_chunks)})"
    d = _append_knowledge_doc(
      title=title,
      content=chunk,
      source=source,
      meta={
        "filename": filename,
        "kind": "text",
        "part": i,
        "parts_total": len(effective_chunks),
        "dataset_id": dataset_id,
        "section": chunk_item.get("section", "content"),
        "semantic_tags": chunk_item.get("semantic_tags", []),
      },
    )
    doc_ids.append(d["id"])
  return len(effective_chunks), len(raw_text), doc_ids


def _ingest_csv_file(filename: str, raw_text: str, delimiter: str = ",", dataset_id: str | None = None, source: str = "dataset") -> tuple[int, int, list[str]]:
    csv = __import__("csv")
    import io as _io
    reader = csv.reader(_io.StringIO(raw_text), delimiter=delimiter)
    rows = [r for r in reader]
    if not rows:
        return 0, 0, []
    header = rows[0]
    data_rows = rows[1:] if len(rows) > 1 else []
    total_rows = len(data_rows)
    doc_ids = []
    if total_rows == 0:
        d = _append_knowledge_doc(
            title=f"Dataset:{filename}",
            content="Columns: " + ", ".join(header),
            source=source,
            meta={"filename": filename, "kind": "csv", "row_count": 0, "columns": header, "dataset_id": dataset_id},
        )
        return 1, total_rows, [d["id"]]

    docs_created = 0
    chunk_size = DATASET_ROW_CHUNK_SIZE
    for start in range(0, total_rows, chunk_size):
        chunk_rows = data_rows[start : start + chunk_size]
        lines = ["Columns: " + ", ".join(header), f"Rows {start + 1}–{start + len(chunk_rows)} of {total_rows}:"]
        for row in chunk_rows:
            pairs = []
            for col_idx, val in enumerate(row):
                col_name = header[col_idx] if col_idx < len(header) else f"col_{col_idx}"
                pairs.append(f"{col_name}={val}")
            lines.append("- " + ", ".join(pairs))
        title = (
            f"Dataset:{filename} (rows {start + 1}-{start + len(chunk_rows)})"
            if total_rows > chunk_size
            else f"Dataset:{filename}"
        )
        d = _append_knowledge_doc(
            title=title,
            content="\n".join(lines),
            source=source,
            meta={
                "filename": filename,
                "kind": "csv",
                "row_count": total_rows,
                "rows_in_chunk": len(chunk_rows),
                "row_start": start + 1,
                "row_end": start + len(chunk_rows),
                "columns": header,
                "dataset_id": dataset_id,
            },
        )
        doc_ids.append(d["id"])
        docs_created += 1
    return docs_created, total_rows, doc_ids


def _ingest_jsonl_file(filename: str, raw_text: str, dataset_id: str | None = None, source: str = "dataset") -> tuple[int, int, list[str]]:
    json = __import__("json")
    records = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except Exception:
            continue
    total_rows = len(records)
    if total_rows == 0:
        return 0, 0, []
    docs_created = 0
    doc_ids = []
    chunk_size = DATASET_ROW_CHUNK_SIZE
    for start in range(0, total_rows, chunk_size):
        chunk = records[start : start + chunk_size]
        lines = [f"Records {start + 1}–{start + len(chunk)} of {total_rows}:"]
        for rec in chunk:
            try:
                lines.append("- " + json.dumps(rec, ensure_ascii=False))
            except Exception:
                lines.append("- " + str(rec))
        title = (
            f"Dataset:{filename} (records {start + 1}-{start + len(chunk)})"
            if total_rows > chunk_size
            else f"Dataset:{filename}"
        )
        d = _append_knowledge_doc(
            title=title,
            content="\n".join(lines),
            source=source,
            meta={
                "filename": filename,
                "kind": "jsonl",
                "row_count": total_rows,
                "rows_in_chunk": len(chunk),
                "row_start": start + 1,
                "row_end": start + len(chunk),
                "dataset_id": dataset_id,
            },
        )
        doc_ids.append(d["id"])
        docs_created += 1
    return docs_created, total_rows, doc_ids


def _ingest_json_file(filename: str, raw_text: str, dataset_id: str | None = None, source: str = "dataset") -> tuple[int, int, list[str]]:
    json = __import__("json")
    try:
        data = json.loads(raw_text)
    except Exception as exc:
        raise ValueError(f"Invalid JSON: {exc}")
    if isinstance(data, list):
        joined = "\n".join(json.dumps(item, ensure_ascii=False) for item in data)
        return _ingest_jsonl_file(filename, joined, dataset_id=dataset_id, source=source)
    pretty = json.dumps(data, ensure_ascii=False, indent=2)
    d = _append_knowledge_doc(
        title=f"Dataset:{filename}",
        content=pretty[: DATASET_TEXT_CHUNK_CHARS * 4],
        source=source,
        meta={"filename": filename, "kind": "json", "row_count": 1, "dataset_id": dataset_id},
    )
    return 1, 1, [d["id"]]


@app.post("/api/training/dataset/upload")
def training_dataset_upload_api():
    """Save uploaded files as a dataset (no ingestion yet). Training is run later via /start."""
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files uploaded. Use field name 'files'."}), 400

    name = (request.form.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Dataset name is required (form field 'name')."}), 400

    dataset_id = _new_dataset_id()
    ds_dir = DATASET_FILES_DIR / dataset_id
    ds_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    errors = []
    for f in files:
        original_name = f.filename or "upload.bin"
        ext = ("." + original_name.rsplit(".", 1)[-1].lower()) if "." in original_name else ""
        if ext not in DATASET_ALLOWED_EXTS:
            errors.append({"file": original_name, "error": f"Unsupported file type '{ext}'."})
            continue
        try:
            raw_bytes = f.read()
        except Exception as exc:
            errors.append({"file": original_name, "error": f"Read failed: {exc}"})
            continue
        if not raw_bytes:
            errors.append({"file": original_name, "error": "File is empty"})
            continue
        if len(raw_bytes) > DATASET_MAX_FILE_BYTES:
            errors.append({"file": original_name, "error": f"File too large ({len(raw_bytes)} bytes; max {DATASET_MAX_FILE_BYTES})"})
            continue

        # Sanitize filename to prevent path traversal
        safe_name = re.sub(r"[^\w.\- ]+", "_", original_name).strip() or "upload.bin"
        target = ds_dir / safe_name
        # Ensure unique within dataset folder
        counter = 1
        while target.exists():
            stem, dot, e = safe_name.rpartition(".")
            target = ds_dir / (f"{stem}_{counter}.{e}" if dot else f"{safe_name}_{counter}")
            counter += 1
        try:
            target.write_bytes(raw_bytes)
        except Exception as exc:
            errors.append({"file": original_name, "error": f"Write failed: {exc}"})
            continue

        kind_map = {".txt": "text", ".md": "text", ".log": "text",
                    ".csv": "csv", ".tsv": "tsv",
                    ".json": "json", ".jsonl": "jsonl", ".ndjson": "jsonl"}
        saved_files.append({
            "name": target.name,
            "original_name": original_name,
            "ext": ext,
            "kind": kind_map.get(ext, "text"),
            "size_bytes": len(raw_bytes),
        })

    if not saved_files:
        # Cleanup empty dir
        _delete_dataset_storage(dataset_id)
        return jsonify({"error": "No valid files saved", "errors": errors}), 400

    dataset = _normalize_dataset({
        "id": dataset_id,
        "name": name,
        "files": saved_files,
        "status": "saved",
        "saved_at": _now_ts(),
    })
    _persist_dataset(dataset)
    _append_audit_event("dataset.upload", {
      "dataset_id": dataset_id,
      "name": name,
      "files": len(saved_files),
      "errors": len(errors),
    })

    return jsonify({"ok": True, "dataset": _public_dataset(dataset), "errors": errors})


@app.get("/api/training/datasets")
def training_datasets_list_api():
    with _training_lock:
        items = _load_datasets()
        for ds in items:
            _normalize_dataset(ds)
    pub = [_public_dataset(d) for d in items]
    pub.sort(key=lambda x: x.get("saved_at") or 0, reverse=True)
    return jsonify({"datasets": pub})


@app.get("/api/training/datasets/<dataset_id>")
def training_dataset_get_api(dataset_id: str):
    ds = _find_dataset(dataset_id)
    if not ds:
        return jsonify({"error": "dataset not found"}), 404
    return jsonify(_public_dataset(ds))


@app.delete("/api/training/datasets/<dataset_id>")
def training_dataset_delete_api(dataset_id: str):
    with _training_lock:
        items = _load_datasets()
        items = [d for d in items if d.get("id") != dataset_id]
        _save_datasets(items)
    _delete_dataset_storage(dataset_id)
    _append_audit_event("dataset.delete", {"dataset_id": dataset_id})
    return jsonify({"ok": True})


@app.post("/api/training/datasets/<dataset_id>/start")
def training_dataset_start_api(dataset_id: str):
    ds = _find_dataset(dataset_id)
    if not ds:
        return jsonify({"error": "dataset not found"}), 404

    ds["status"] = "running"
    ds["last_error"] = None
    _persist_dataset(ds)
    _append_audit_event("dataset.train.start", {"dataset_id": dataset_id, "name": ds.get("name")})

    ds_dir = DATASET_FILES_DIR / dataset_id
    results = []
    total_docs = 0
    all_doc_ids = []
    errors = []

    try:
        for fmeta in ds.get("files", []):
            fname = fmeta.get("name")
            if not fname:
                continue
            fpath = ds_dir / fname
            if not fpath.exists():
                errors.append({"file": fname, "error": "Stored file missing"})
                continue
            try:
                raw_bytes = fpath.read_bytes()
            except Exception as exc:
                errors.append({"file": fname, "error": f"Read failed: {exc}"})
                continue
            try:
                raw_text = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    raw_text = raw_bytes.decode("utf-8-sig")
                except UnicodeDecodeError:
                    try:
                        raw_text = raw_bytes.decode("cp1256")
                    except UnicodeDecodeError:
                        raw_text = raw_bytes.decode("latin-1", errors="replace")

            ext = fmeta.get("ext", "")
            display_name = fmeta.get("original_name") or fname
            try:
                if ext in (".txt", ".md", ".log"):
                    docs_added, units, ids = _ingest_text_file(display_name, raw_text, dataset_id=dataset_id)
                    kind = "text"
                elif ext == ".csv":
                    docs_added, units, ids = _ingest_csv_file(display_name, raw_text, delimiter=",", dataset_id=dataset_id)
                    kind = "csv"
                elif ext == ".tsv":
                    docs_added, units, ids = _ingest_csv_file(display_name, raw_text, delimiter="\t", dataset_id=dataset_id)
                    kind = "tsv"
                elif ext in (".jsonl", ".ndjson"):
                    docs_added, units, ids = _ingest_jsonl_file(display_name, raw_text, dataset_id=dataset_id)
                    kind = "jsonl"
                elif ext == ".json":
                    docs_added, units, ids = _ingest_json_file(display_name, raw_text, dataset_id=dataset_id)
                    kind = "json"
                else:
                    docs_added, units, ids = _ingest_text_file(display_name, raw_text, dataset_id=dataset_id)
                    kind = "text"
            except ValueError as exc:
                errors.append({"file": display_name, "error": str(exc)})
                continue
            except Exception as exc:
                errors.append({"file": display_name, "error": f"Ingest failed: {exc}"})
                continue

            total_docs += docs_added
            all_doc_ids.extend(ids)
            results.append({
                "file": display_name,
                "kind": kind,
                "size_bytes": fmeta.get("size_bytes", 0),
                "docs_added": docs_added,
                "units": units,
            })

        ds["status"] = "completed"
        ds["last_trained_at"] = _now_ts()
        ds["last_added_docs"] = total_docs
        ds["last_results"] = results
        ds["last_error"] = "; ".join(f"{e['file']}: {e['error']}" for e in errors) if errors else None
        existing_ids = list(ds.get("doc_ids", []))
        ds["doc_ids"] = existing_ids + all_doc_ids
    except Exception as exc:
        ds["status"] = "failed"
        ds["last_error"] = str(exc)

    _persist_dataset(ds)
    _append_audit_event("dataset.train.finish", {
      "dataset_id": dataset_id,
      "status": ds.get("status"),
      "docs_added": total_docs,
      "errors": len(errors),
    })
    return jsonify({
        "ok": ds["status"] == "completed",
        "status": ds["status"],
        "docs_added": total_docs,
        "results": results,
        "errors": errors,
        "dataset": _public_dataset(ds),
    })


@app.get("/api/training/datasets/<dataset_id>/progress")
def training_dataset_progress_api(dataset_id: str):
    ds = _find_dataset(dataset_id)
    if not ds:
        return jsonify({"error": "dataset not found"}), 404
    return jsonify(_public_dataset(ds))


# --- ADTC training (advanced multi-file understanding) ---------------------------
ADTC_ALLOWED_EXTS = {
  ".pdf", ".docx", ".xlsx", ".xls",
  ".txt", ".md", ".log", ".csv", ".tsv", ".json", ".jsonl", ".ndjson",
}
ADTC_MAX_FILE_BYTES = 45 * 1024 * 1024  # 45 MB per file


@app.post("/api/training/adtc/upload")
def training_adtc_upload_api():
  files = request.files.getlist("files")
  if not files:
    return jsonify({"error": "No files uploaded. Use field name 'files'."}), 400

  name = (request.form.get("name") or "").strip()
  topic = (request.form.get("topic") or "").strip() or "general"
  if not name:
    return jsonify({"error": "Dataset name is required (form field 'name')."}), 400

  dataset_id = _new_adtc_id()
  ds_dir = ADTC_FILES_DIR / dataset_id
  ds_dir.mkdir(parents=True, exist_ok=True)

  saved_files = []
  errors = []
  for f in files:
    original_name = f.filename or "upload.bin"
    ext = ("." + original_name.rsplit(".", 1)[-1].lower()) if "." in original_name else ""
    if ext not in ADTC_ALLOWED_EXTS:
      errors.append({"file": original_name, "error": f"Unsupported file type '{ext}'."})
      continue
    try:
      raw_bytes = f.read()
    except Exception as exc:
      errors.append({"file": original_name, "error": f"Read failed: {exc}"})
      continue
    if not raw_bytes:
      errors.append({"file": original_name, "error": "File is empty"})
      continue
    if len(raw_bytes) > ADTC_MAX_FILE_BYTES:
      errors.append({"file": original_name, "error": f"File too large ({len(raw_bytes)} bytes; max {ADTC_MAX_FILE_BYTES})"})
      continue

    safe_name = re.sub(r"[^\w.\- ]+", "_", original_name).strip() or "upload.bin"
    target = ds_dir / safe_name
    counter = 1
    while target.exists():
      stem, dot, e = safe_name.rpartition(".")
      target = ds_dir / (f"{stem}_{counter}.{e}" if dot else f"{safe_name}_{counter}")
      counter += 1
    try:
      target.write_bytes(raw_bytes)
    except Exception as exc:
      errors.append({"file": original_name, "error": f"Write failed: {exc}"})
      continue

    kind_map = {
      ".pdf": "pdf", ".docx": "word", ".xlsx": "excel", ".xls": "excel",
      ".txt": "text", ".md": "text", ".log": "text",
      ".csv": "csv", ".tsv": "tsv", ".json": "json", ".jsonl": "jsonl", ".ndjson": "jsonl",
    }
    saved_files.append({
      "name": target.name,
      "original_name": original_name,
      "ext": ext,
      "kind": kind_map.get(ext, "text"),
      "size_bytes": len(raw_bytes),
    })

  if not saved_files:
    _delete_adtc_storage(dataset_id)
    return jsonify({"error": "No valid files saved", "errors": errors}), 400

  ds = _normalize_adtc_dataset({
    "id": dataset_id,
    "name": name,
    "topic": topic,
    "files": saved_files,
    "status": "saved",
    "saved_at": _now_ts(),
  })
  _persist_adtc_dataset(ds)
  _append_audit_event("adtc.upload", {
    "dataset_id": dataset_id,
    "name": name,
    "topic": topic,
    "files": len(saved_files),
    "errors": len(errors),
  })
  return jsonify({"ok": True, "dataset": _public_adtc_dataset(ds), "errors": errors})


@app.get("/api/training/adtc/datasets")
def training_adtc_datasets_api():
  with _training_lock:
    items = _load_adtc_datasets()
    for ds in items:
      _normalize_adtc_dataset(ds)
  pub = [_public_adtc_dataset(d) for d in items]
  pub.sort(key=lambda x: x.get("saved_at") or 0, reverse=True)
  return jsonify({"datasets": pub})


@app.get("/api/training/adtc/datasets/<dataset_id>")
def training_adtc_dataset_get_api(dataset_id: str):
  ds = _find_adtc_dataset(dataset_id)
  if not ds:
    return jsonify({"error": "dataset not found"}), 404
  return jsonify(_public_adtc_dataset(ds))


@app.delete("/api/training/adtc/datasets/<dataset_id>")
def training_adtc_dataset_delete_api(dataset_id: str):
  ds = _find_adtc_dataset(dataset_id)
  if not ds:
    return jsonify({"error": "dataset not found"}), 404
  dataset_doc_ids = list(ds.get("doc_ids", []))
  with _training_lock:
    items = _load_adtc_datasets()
    items = [d for d in items if d.get("id") != dataset_id]
    _save_adtc_datasets(items)
  removed_docs = _delete_adtc_related_knowledge(dataset_id, dataset_doc_ids)
  _delete_adtc_storage(dataset_id)
  _append_audit_event("adtc.delete", {
    "dataset_id": dataset_id,
    "removed_docs": removed_docs,
  })
  return jsonify({
    "ok": True,
    "removed_dataset": dataset_id,
    "removed_docs": removed_docs,
    "removed_storage_dir": str(ADTC_FILES_DIR / dataset_id),
  })


@app.post("/api/training/adtc/datasets/<dataset_id>/start")
def training_adtc_dataset_start_api(dataset_id: str):
  ds = _find_adtc_dataset(dataset_id)
  if not ds:
    return jsonify({"error": "dataset not found"}), 404

  ds["status"] = "running"
  ds["last_error"] = None
  _persist_adtc_dataset(ds)
  _append_audit_event("adtc.train.start", {
    "dataset_id": dataset_id,
    "name": ds.get("name"),
    "topic": ds.get("topic"),
  })

  ds_dir = ADTC_FILES_DIR / dataset_id
  results = []
  errors = []
  total_docs = 0
  all_doc_ids = []
  file_profiles = []
  learned_details = []
  all_unclear_points = []
  all_understood_points = []
  all_probe_stats = []
  topic_coverage_totals = {"location": 0, "contacts": 0, "services": 0, "schedule": 0, "insurance": 0}
  confidence_scores = []

  for fmeta in ds.get("files", []):
    fname = fmeta.get("name")
    if not fname:
      continue
    fpath = ds_dir / fname
    if not fpath.exists():
      errors.append({"file": fname, "error": "Stored file missing"})
      continue

    ext = (fmeta.get("ext") or "").lower()
    display_name = fmeta.get("original_name") or fname
    payload_meta = {}
    try:
      if ext in (".txt", ".md", ".log"):
        raw = fpath.read_text(encoding="utf-8", errors="replace")
        docs_added, units, ids = _ingest_text_file(display_name, raw, dataset_id=dataset_id, source="adtc")
        schema_keys = []
        kind = "text"
        raw_text_for_stats = raw
      elif ext == ".csv":
        raw = fpath.read_text(encoding="utf-8", errors="replace")
        docs_added, units, ids = _ingest_csv_file(display_name, raw, delimiter=",", dataset_id=dataset_id, source="adtc")
        rows = [r.split(",") for r in raw.splitlines()[:1] if r.strip()]
        schema_keys = [_norm_key(c) for c in rows[0]] if rows else []
        kind = "csv"
        raw_text_for_stats = raw
      elif ext == ".tsv":
        raw = fpath.read_text(encoding="utf-8", errors="replace")
        docs_added, units, ids = _ingest_csv_file(display_name, raw, delimiter="\t", dataset_id=dataset_id, source="adtc")
        rows = [r.split("\t") for r in raw.splitlines()[:1] if r.strip()]
        schema_keys = [_norm_key(c) for c in rows[0]] if rows else []
        kind = "tsv"
        raw_text_for_stats = raw
      elif ext in (".jsonl", ".ndjson"):
        raw = fpath.read_text(encoding="utf-8", errors="replace")
        docs_added, units, ids = _ingest_jsonl_file(display_name, raw, dataset_id=dataset_id, source="adtc")
        schema_keys = []
        kind = "jsonl"
        raw_text_for_stats = raw
      elif ext == ".json":
        raw = fpath.read_text(encoding="utf-8", errors="replace")
        docs_added, units, ids = _ingest_json_file(display_name, raw, dataset_id=dataset_id, source="adtc")
        schema_keys = []
        kind = "json"
        raw_text_for_stats = raw
      elif ext == ".pdf":
        payload = _extract_pdf_payload(fpath)
        docs_added, units, ids = _ingest_text_file(display_name, payload["text"], dataset_id=dataset_id, source="adtc")
        schema_keys = payload.get("schema_keys", [])
        kind = payload.get("kind", "pdf")
        raw_text_for_stats = payload.get("text", "")
        payload_meta = payload
      elif ext == ".docx":
        payload = _extract_docx_payload(fpath)
        docs_added, units, ids = _ingest_text_file(display_name, payload["text"], dataset_id=dataset_id, source="adtc")
        schema_keys = payload.get("schema_keys", [])
        kind = payload.get("kind", "docx")
        raw_text_for_stats = payload.get("text", "")
      elif ext in (".xlsx", ".xls"):
        if ext == ".xls":
          raise ValueError("Legacy .xls is not supported in this build. Please convert to .xlsx")
        payload = _extract_xlsx_payload(fpath)
        docs_added, units, ids = _ingest_text_file(display_name, payload["text"], dataset_id=dataset_id, source="adtc")
        schema_keys = payload.get("schema_keys", [])
        kind = payload.get("kind", "xlsx")
        raw_text_for_stats = payload.get("text", "")
      else:
        raise ValueError(f"Unsupported extension {ext}")
    except Exception as exc:
      errors.append({"file": display_name, "error": str(exc)})
      continue

    total_docs += docs_added
    all_doc_ids.extend(ids)
    file_profiles.append({"file": display_name, "schema_keys": [k for k in schema_keys if k]})

    understand = _build_understanding_report(raw_text_for_stats, kind, payload_meta=payload_meta)
    extraction_quality = _build_extraction_quality_report(raw_text_for_stats)
    coverage = _build_topic_coverage(raw_text_for_stats, str(ds.get("topic") or ""))
    qa_probes = _build_qa_probes(raw_text_for_stats, str(ds.get("topic") or ""))
    training_confidence = _score_training_confidence(understand, coverage, qa_probes)
    all_unclear_points.extend([f"{display_name}: {x}" for x in understand.get("unclear_points", [])])
    for fact in understand.get("key_facts", [])[:3]:
      all_understood_points.append(f"{display_name}: {fact}")
    for k in topic_coverage_totals.keys():
      topic_coverage_totals[k] += int(coverage.get(k) or 0)
    confidence_scores.append(training_confidence)
    all_probe_stats.extend([
      {
        "file": display_name,
        "probe_id": p.get("id"),
        "answerable": bool(p.get("answerable")),
        "confidence": int(p.get("confidence") or 0),
      }
      for p in qa_probes
    ])

    # Store a compact, human-readable understanding document to improve natural answers.
    understanding_lines = [
      f"File: {display_name}",
      f"Kind: {kind}",
      f"What was understood: {understand.get('understanding_summary', '-')}",
    ]
    if understand.get("key_facts"):
      understanding_lines.append("Key facts extracted:")
      for fact in understand.get("key_facts", [])[:8]:
        understanding_lines.append(f"- {fact}")
    csignals = understand.get("contact_signals", {})
    understanding_lines.append("Detected contact signals:")
    understanding_lines.append("- URLs: " + (" | ".join(csignals.get("urls", [])[:5]) or "None detected"))
    understanding_lines.append("- Emails: " + (" | ".join(csignals.get("emails", [])[:5]) or "None detected"))
    understanding_lines.append("- Phones: " + (" | ".join(csignals.get("phones", [])[:5]) or "None detected"))
    if understand.get("unclear_points"):
      understanding_lines.append("Unclear or low-confidence parts:")
      for item in understand.get("unclear_points", [])[:6]:
        understanding_lines.append(f"- {item}")
    understanding_lines.append("Extraction quality diagnostics:")
    understanding_lines.append(
      f"- chars={extraction_quality.get('char_count')} lines={extraction_quality.get('line_count')} non_empty={extraction_quality.get('non_empty_lines')}"
    )
    understanding_lines.append(
      f"- duplicate_line_ratio={extraction_quality.get('duplicate_line_ratio')} noise_char_ratio={extraction_quality.get('noise_char_ratio')} empty_line_ratio={extraction_quality.get('empty_line_ratio')}"
    )
    if extraction_quality.get("flags"):
      understanding_lines.append("- quality_flags=" + ", ".join(extraction_quality.get("flags", [])))
    understanding_lines.append(f"Training confidence: {training_confidence}/100")
    understanding_lines.append("Topic coverage signals:")
    understanding_lines.append(
      "- " + ", ".join(f"{k}={coverage.get(k, 0)}" for k in ("location", "contacts", "services", "schedule", "insurance"))
    )
    understanding_lines.append("Question-answerability probes:")
    for p in qa_probes[:8]:
      ev = p.get("evidence") or []
      understanding_lines.append(
        f"- {p.get('id')}: answerable={p.get('answerable')} confidence={p.get('confidence')} evidence={(ev[0] if ev else 'none')}"
      )

    understanding_doc = _append_knowledge_doc(
      title=f"ADTC Understanding: {display_name}",
      content="\n".join(understanding_lines),
      source="adtc",
      meta={
        "dataset_id": dataset_id,
        "dataset_name": ds.get("name"),
        "topic": ds.get("topic"),
        "file": display_name,
        "kind": kind,
        "quality_score": understand.get("quality_score", 0),
        "extraction_quality": extraction_quality,
      },
    )
    all_doc_ids.append(understanding_doc["id"])
    total_docs += 1

    # Executive summary per file: concise and directly usable for user-facing answers.
    summary_lines = [
      f"Executive summary for {display_name}",
      f"Topic: {ds.get('topic')}",
      understand.get("understanding_summary", "Basic extraction completed."),
      f"Training confidence: {training_confidence}/100",
      f"Coverage score: {coverage.get('coverage_score', 0)}/100",
    ]
    if understand.get("key_facts"):
      summary_lines.append("Top key facts:")
      for fact in understand.get("key_facts", [])[:5]:
        summary_lines.append(f"- {fact}")
    summary_lines.append("Section hints:")
    for sec in _extract_section_titles(raw_text_for_stats, limit=8):
      summary_lines.append(f"- {sec}")

    summary_doc = _append_knowledge_doc(
      title=f"ADTC Executive Summary: {display_name}",
      content="\n".join(summary_lines),
      source="adtc",
      meta={
        "dataset_id": dataset_id,
        "dataset_name": ds.get("name"),
        "topic": ds.get("topic"),
        "file": display_name,
        "kind": kind,
        "summary_kind": "executive",
        "training_confidence": training_confidence,
      },
    )
    all_doc_ids.append(summary_doc["id"])
    total_docs += 1

    learned_details.append({
      "file": display_name,
      "kind": kind,
      "char_count": len(raw_text_for_stats or ""),
      "content_preview": re.sub(r"\s+", " ", (raw_text_for_stats or "")).strip()[:1600],
      "units_read": units,
      "chunks_created": docs_added,
      "schema_keys": [k for k in schema_keys if k][:20],
      "top_terms": understand.get("top_terms", []),
      "understanding_summary": understand.get("understanding_summary", ""),
      "key_facts": understand.get("key_facts", []),
      "contact_signals": understand.get("contact_signals", {}),
      "unclear_points": understand.get("unclear_points", []),
      "quality_score": understand.get("quality_score", 0),
      "extraction_quality": extraction_quality,
      "signal_metrics": understand.get("signal_metrics", {}),
      "topic_coverage": coverage,
      "qa_probes": qa_probes,
      "training_confidence": training_confidence,
      "understanding_pipeline": {
        "step_1_extract": {
          "kind": kind,
          "chars": len(raw_text_for_stats or ""),
          "units": units,
          "chunks_written": docs_added,
        },
        "step_2_analyze": {
          "key_facts": len(understand.get("key_facts", [])),
          "top_terms": len(understand.get("top_terms", [])),
          "contacts_detected": {
            "urls": len((understand.get("contact_signals", {}) or {}).get("urls", [])),
            "emails": len((understand.get("contact_signals", {}) or {}).get("emails", [])),
            "phones": len((understand.get("contact_signals", {}) or {}).get("phones", [])),
          },
          "unclear_points": len(understand.get("unclear_points", [])),
        },
        "step_3_answerability": {
          "coverage_score": int(coverage.get("coverage_score") or 0),
          "qa_probe_count": len(qa_probes),
          "qa_probe_answerable": sum(1 for p in qa_probes if p.get("answerable")),
          "confidence": training_confidence,
        },
      },
      "section_titles": _extract_section_titles(raw_text_for_stats, limit=12),
      "pdf_stats": {
        "pages_total": payload_meta.get("pages_total"),
        "pages_with_text": payload_meta.get("pages_with_text"),
        "pages_low_text": payload_meta.get("pages_low_text"),
        "avg_chars_per_page": payload_meta.get("avg_chars_per_page"),
        "page_char_counts": payload_meta.get("page_char_counts", []),
        "page_samples": payload_meta.get("page_samples", []),
      } if kind == "pdf" else {},
      "doc_ids": ids,
      "understanding_doc_id": understanding_doc.get("id"),
      "summary_doc_id": summary_doc.get("id"),
    })
    results.append({
      "file": display_name,
      "kind": kind,
      "size_bytes": fmeta.get("size_bytes", 0),
      "docs_added": docs_added,
      "units": units,
      "quality_score": understand.get("quality_score", 0),
      "training_confidence": training_confidence,
      "extraction_quality_flags": extraction_quality.get("flags", []),
    })

  relationships = _analyze_relationships(file_profiles)
  insight_lines = [
    f"ADTC dataset '{ds.get('name')}' topic '{ds.get('topic')}' processed {len(results)} files.",
    f"Total ADTC knowledge docs added: {total_docs}",
    f"Detected cross-file relationships: {len(relationships)}",
  ]
  if all_understood_points:
    insight_lines.append("Top understood facts:")
    for item in all_understood_points[:10]:
      insight_lines.append(f"- {item}")
  if all_unclear_points:
    insight_lines.append("Unclear/low-confidence findings:")
    for item in all_unclear_points[:10]:
      insight_lines.append(f"- {item}")
  if relationships:
    for rel in relationships[:8]:
      insight_lines.append(
        f"- {rel.get('from')} ↔ {rel.get('to')} via keys: {', '.join(rel.get('shared_keys', [])[:6])}"
      )

  insight_doc = _append_knowledge_doc(
    title=f"ADTC Summary: {ds.get('name')}",
    content="\n".join(insight_lines),
    source="adtc",
    meta={
      "dataset_id": dataset_id,
      "dataset_name": ds.get("name"),
      "topic": ds.get("topic"),
      "relationship_count": len(relationships),
      "source_trace": [r.get("file") for r in results],
    },
  )
  all_doc_ids.append(insight_doc["id"])
  total_docs += 1

  ds["status"] = "completed" if not errors else "failed"
  ds["last_trained_at"] = _now_ts()
  ds["last_added_docs"] = total_docs
  ds["last_results"] = results
  ds["relationships"] = relationships
  ds["insights"] = insight_lines
  ds["learned_details"] = learned_details
  knowledge_links = _build_dataset_knowledge_links(dataset_id, source="adtc", limit=24)
  ds["integration_report"] = {
    "knowledge_source": "adtc",
    "dataset_id": dataset_id,
    "dataset_name": ds.get("name"),
    "topic": ds.get("topic"),
    "docs_written": total_docs,
    "doc_ids": all_doc_ids,
    "retrieval_note": "All ADTC docs are appended to knowledge memory and become retrievable by the same semantic overlap pipeline used by chat answers.",
    "cross_file_reasoning_note": "Relationships are inferred from shared schema keys and summarized into ADTC summary docs for multi-file question answering.",
    "understood_points": all_understood_points[:30],
    "unclear_points": all_unclear_points[:30],
    "qa_probe_summary": {
      "total_probes": len(all_probe_stats),
      "answerable_probes": sum(1 for p in all_probe_stats if p.get("answerable")),
      "avg_probe_confidence": round(sum(int(p.get("confidence") or 0) for p in all_probe_stats) / max(1, len(all_probe_stats)), 1),
      "sample": all_probe_stats[:20],
    },
    "topic_coverage_totals": topic_coverage_totals,
    "knowledge_links": knowledge_links,
    "knowledge_link_count": len(knowledge_links),
    "quality_overview": {
      "avg_quality_score": round(sum((r.get("quality_score") or 0) for r in results) / max(1, len(results)), 1),
      "avg_training_confidence": round(sum(confidence_scores) / max(1, len(confidence_scores)), 1),
      "files_analyzed": len(results),
      "files_with_unclear_parts": sum(1 for ld in learned_details if ld.get("unclear_points")),
    },
  }
  ds["last_error"] = "; ".join(f"{e['file']}: {e['error']}" for e in errors) if errors else None
  ds["doc_ids"] = list(ds.get("doc_ids", [])) + all_doc_ids

  _persist_adtc_dataset(ds)
  _append_audit_event("adtc.train.finish", {
    "dataset_id": dataset_id,
    "status": ds.get("status"),
    "docs_added": total_docs,
    "errors": len(errors),
    "knowledge_links": len(knowledge_links),
  })
  return jsonify({
    "ok": ds["status"] == "completed",
    "status": ds["status"],
    "docs_added": total_docs,
    "results": results,
    "errors": errors,
    "dataset": _public_adtc_dataset(ds),
  })


@app.get("/api/training/adtc/datasets/<dataset_id>/progress")
def training_adtc_dataset_progress_api(dataset_id: str):
  ds = _find_adtc_dataset(dataset_id)
  if not ds:
    return jsonify({"error": "dataset not found"}), 404
  return jsonify(_public_adtc_dataset(ds))


def _resolve_url_from_payload(data: dict) -> tuple[str, str, dict]:
    db_type = str(data.get("type") or "").strip().lower()
    fields = data.get("fields") or {}
    raw_url = str(data.get("url") or "").strip()
    if db_type and db_type in DB_PRESETS:
        url = _build_url_from_preset(db_type, fields)
        return url, db_type, fields
    if raw_url:
        return raw_url, db_type or "custom", fields
    raise ValueError("Either preset 'type' with 'fields' or a raw 'url' is required")


@app.post("/api/training/db/test")
def training_db_test_api():
    if not SQLALCHEMY_AVAILABLE:
        return jsonify({"error": "SQLAlchemy is not installed. Install with: pip install sqlalchemy"}), 400
    data = request.get_json(silent=True) or {}
    try:
        url, db_type, _ = _resolve_url_from_payload(data)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    try:
        engine = create_engine(url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return jsonify({"ok": True, "type": db_type, "url_masked": _mask_url(url)})
    except SQLAlchemyError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        msg = str(exc)
        preset = DB_PRESETS.get(db_type)
        if preset and preset.get("driver_pip") and "no module" in msg.lower():
            msg += f" — install driver: pip install {preset['driver_pip']}"
        return jsonify({"error": msg}), 400


@app.post("/api/training/db/save")
def training_db_save_api():
    data = request.get_json(silent=True) or {}
    name = str(data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Connection name is required"}), 400
    try:
        url, db_type, fields = _resolve_url_from_payload(data)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    tables = str(data.get("tables") or "").strip()
    sample_rows = int(data.get("sample_rows") or 5)
    sample_rows = max(1, min(50, sample_rows))

    with _training_lock:
        links = _load_db_links()
        for link in links:
            _normalize_link(link)
        existing = next((x for x in links if x.get("name") == name), None)
        if existing:
            existing.update({
                "type": db_type,
                "fields": fields,
                "url": url,
                "tables": tables,
                "sample_rows": sample_rows,
                "saved_at": _now_ts(),
                "status": existing.get("status", "saved") if existing.get("status") in ("running",) else "saved",
            })
            link_id = existing["id"]
        else:
            link = {
                "id": _new_link_id(),
                "name": name,
                "type": db_type,
                "fields": fields,
                "url": url,
                "tables": tables,
                "sample_rows": sample_rows,
                "saved_at": _now_ts(),
                "status": "saved",
                "last_trained_at": None,
                "last_error": None,
                "last_added_docs": 0,
                "last_table_count": 0,
            }
            links.append(link)
            link_id = link["id"]
        _save_db_links(links)

    return jsonify({"ok": True, "link_id": link_id})


@app.delete("/api/training/db/links/<link_id>")
def training_db_delete_api(link_id: str):
    with _jobs_lock:
        job = TRAINING_JOBS.get(link_id)
    if job and job.get("status") in ("starting", "connecting", "listing", "ingesting", "saving"):
        return jsonify({"error": "Training is currently running for this connection. Stop it first."}), 400
    if not _delete_link(link_id):
        return jsonify({"error": "connection not found"}), 404
    with _jobs_lock:
        TRAINING_JOBS.pop(link_id, None)
    return jsonify({"ok": True})


@app.get("/api/training/db/links/<link_id>")
def training_db_get_link_api(link_id: str):
    link = _find_link(link_id)
    if not link:
        return jsonify({"error": "connection not found"}), 404
    return jsonify({"link": _public_link(link)})


@app.post("/api/training/db/links/<link_id>/test")
def training_db_test_link_api(link_id: str):
    if not SQLALCHEMY_AVAILABLE:
        return jsonify({"error": "SQLAlchemy is not installed. Install with: pip install sqlalchemy"}), 400
    link = _find_link(link_id)
    if not link:
        return jsonify({"error": "connection not found"}), 404
    url = link.get("url", "")
    if not url:
        return jsonify({"error": "connection has no URL"}), 400
    try:
        engine = create_engine(url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return jsonify({"ok": True, "url_masked": _mask_url(url)})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/training/db/links/<link_id>/start")
def training_db_start_api(link_id: str):
    if not SQLALCHEMY_AVAILABLE:
        return jsonify({"error": "SQLAlchemy is not installed. Install with: pip install sqlalchemy"}), 400
    link = _find_link(link_id)
    if not link:
        return jsonify({"error": "connection not found"}), 404
    with _jobs_lock:
        existing = TRAINING_JOBS.get(link_id)
        if existing and existing.get("status") in ("starting", "connecting", "listing", "ingesting", "saving"):
            return jsonify({"error": "Training already running"}), 400

    thread = threading.Thread(target=_run_training_job, args=(link_id,), daemon=True)
    thread.start()
    return jsonify({"ok": True, "link_id": link_id})


@app.post("/api/training/db/links/<link_id>/stop")
def training_db_stop_api(link_id: str):
    with _jobs_lock:
        job = TRAINING_JOBS.get(link_id)
        if not job:
            return jsonify({"error": "no running job for this connection"}), 404
        job["stop_requested"] = True
    return jsonify({"ok": True})


@app.get("/api/training/db/links/<link_id>/progress")
def training_db_progress_api(link_id: str):
    link = _find_link(link_id)
    if not link:
        return jsonify({"error": "connection not found"}), 404
    with _jobs_lock:
        job = TRAINING_JOBS.get(link_id)
        job_snapshot = deepcopy(job) if job else None
    return jsonify({
        "link": _public_link(link),
        "job": job_snapshot,
    })


# Backward-compat: legacy ingest endpoint kept for older clients/tests.
@app.post("/api/training/db/ingest")
def training_db_ingest_api():
    if not SQLALCHEMY_AVAILABLE:
        return jsonify({"error": "SQLAlchemy is not installed. Install with: pip install sqlalchemy"}), 400
    data = request.get_json(silent=True) or {}
    name = str(data.get("name") or "").strip() or "DB Link"
    url = str(data.get("url") or "").strip()
    tables_raw = str(data.get("tables") or "").strip()
    sample_rows = int(data.get("sample_rows") or 5)
    sample_rows = max(1, min(50, sample_rows))
    if not url:
        return jsonify({"error": "url is required"}), 400

    # Save/update link then run synchronously (legacy behavior)
    with _training_lock:
        links = _load_db_links()
        for link in links:
            _normalize_link(link)
        existing = next((x for x in links if x.get("name") == name), None)
        if existing:
            existing.update({"url": url, "tables": tables_raw, "sample_rows": sample_rows, "saved_at": _now_ts()})
            link_id = existing["id"]
        else:
            link = {
                "id": _new_link_id(), "name": name, "type": "custom", "fields": {},
                "url": url, "tables": tables_raw, "sample_rows": sample_rows,
                "saved_at": _now_ts(), "status": "saved",
            }
            links.append(link)
            link_id = link["id"]
        _save_db_links(links)

    _run_training_job(link_id)
    link = _find_link(link_id) or {}
    return jsonify({
        "ok": link.get("status") == "completed",
        "added_docs": link.get("last_added_docs", 0),
        "table_count": link.get("last_table_count", 0),
        "status": link.get("status"),
        "error": link.get("last_error"),
    })


TRAINING_INDEX_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Phi-3 Training Center</title>
  <style>
    :root { --bg:#0f172a; --panel:#1e293b; --line:#334155; --text:#e2e8f0; --muted:#94a3b8; --accent:#10b981; --warn:#ef4444; --info:#38bdf8; --amber:#f59e0b; }
    *{box-sizing:border-box}
    body{margin:0;background:var(--bg);color:var(--text);font-family:Segoe UI,system-ui,sans-serif}
    .wrap{max-width:1200px;margin:0 auto;padding:16px}
    .head{margin-bottom:12px;display:flex;justify-content:space-between;align-items:center;gap:12px}
    .grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
    .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px}
    .k{font-size:12px;color:var(--muted)}
    label{display:block;font-size:12px;color:var(--muted);margin:8px 0 4px}
    input,select,textarea{width:100%;border:1px solid var(--line);background:#0b1220;color:var(--text);border-radius:10px;padding:10px;font-family:inherit;font-size:13px}
    textarea{min-height:140px;resize:vertical}
    button{border:1px solid var(--line);background:#0b1220;color:var(--text);border-radius:10px;padding:9px 12px;cursor:pointer;font-size:13px}
    button.primary{background:var(--accent);border-color:var(--accent);color:#072012;font-weight:700}
    button.danger{background:#3b0d0d;border-color:#7f1d1d;color:#fecaca}
    button.icon{padding:6px 10px;font-size:14px}
    button:disabled{opacity:.5;cursor:not-allowed}
    table{width:100%;border-collapse:collapse}
    th,td{padding:9px 8px;border-bottom:1px solid var(--line);font-size:12px;text-align:left;vertical-align:middle}
    th{color:var(--muted);font-weight:600}
    .row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
    .mono{font-family:Consolas,monospace;font-size:12px}
    .badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:700}
    .b-saved{background:#1e3a8a;color:#dbeafe}
    .b-running{background:#7c2d12;color:#fed7aa}
    .b-completed{background:#064e3b;color:#a7f3d0}
    .b-failed{background:#7f1d1d;color:#fecaca}
    .b-stopped{background:#374151;color:#e5e7eb}
    .stat{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:8px}
    .pill{background:#0b1220;border:1px solid var(--line);padding:6px 10px;border-radius:10px;font-size:12px}
    .req{color:var(--warn);font-weight:700}
    .hint{color:var(--muted);font-size:11px;margin-top:6px}
    .toolbar{display:flex;gap:8px;margin-top:8px;flex-wrap:wrap}
    @media(max-width:980px){.grid{grid-template-columns:1fr}}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="head">
      <div>
        <h2 style="margin:0">Training Center</h2>
        <div class="k">Pick a database type, fill in connection details, test, save, then run training from the Training Status table.</div>
      </div>
      <div class="row">
        <button onclick="window.location.href='/'">← Back to Chat</button>
        <button onclick="refreshStatus()">⟳ Refresh</button>
      </div>
    </div>

    <div class="grid">
      <div class="card">
        <h3 style="margin-top:0">Text Knowledge Training</h3>
        <div class="k">Builds a persistent knowledge memory used during answers. No model fine-tuning.</div>
        <label>Title</label>
        <input id="docTitle" placeholder="e.g. Accounting policies v1" />
        <label>Content</label>
        <textarea id="docContent" placeholder="Paste documentation, rules, procedures, product details..."></textarea>
        <div class="toolbar">
          <button class="primary" id="addDocBtn">Add to Knowledge Memory</button>
          <span class="k" id="docMsg"></span>
        </div>
      </div>

      <div class="card">
        <h3 style="margin-top:0">Database Connection</h3>
        <div class="k">Choose a preset to see required fields. Test the connection, then Save it. Run training from the table below.</div>

        <label>Connection Name <span class="req">*</span></label>
        <input id="dbName" placeholder="e.g. ERP Production" />

        <label>Database Type <span class="req">*</span></label>
        <select id="dbType"></select>
        <div class="hint" id="driverHint"></div>

        <div id="dbFields"></div>

        <label>Include Tables (optional, comma separated)</label>
        <input id="dbTables" placeholder="customers, orders, invoices (leave empty for all)" />

        <label>Sample rows per table</label>
        <input id="dbRows" type="number" value="5" min="1" max="50" />

        <div class="toolbar">
          <button id="testDbBtn">Test Connection</button>
          <button class="primary" id="saveDbBtn" disabled>💾 Save Connection</button>
          <span class="k" id="dbMsg"></span>
        </div>
      </div>
    </div>

    <div class="card" style="margin-top:14px">
      <h3 style="margin-top:0">Dataset Files Training</h3>
      <div class="k">Add a dataset by uploading one or more files and saving with a name. Run training later from the Training Status table.</div>
      <div class="k" style="margin-top:6px">Supported: <b>.txt .md .log</b> (free text) · <b>.csv .tsv</b> (tabular) · <b>.json .jsonl .ndjson</b> (records). Max 25 MB per file.</div>

      <label>Dataset Name <span class="req">*</span></label>
      <input id="dsName" placeholder="e.g. Product catalog Q1 2026" />

      <label>Choose files <span class="req">*</span></label>
      <input id="dsFiles" type="file" multiple accept=".txt,.md,.log,.csv,.tsv,.json,.jsonl,.ndjson" />
      <div id="dsFileList" class="hint" style="margin-top:6px"></div>

      <div class="toolbar">
        <button class="primary" id="dsSaveBtn">💾 Save Dataset</button>
        <button id="dsClearBtn">Clear</button>
        <span class="k" id="dsMsg"></span>
      </div>

      <div id="dsResults" style="margin-top:10px"></div>
    </div>

    <div class="card" style="margin-top:14px">
      <h3 style="margin-top:0">ADTC Training</h3>
      <div class="k">Advanced Dataset Training Center: build domain datasets and train richer cross-file understanding for analysis and reasoning.</div>
      <div class="k" style="margin-top:6px">Supported now: <b>.pdf .docx .xlsx .txt .md .csv .tsv .json .jsonl</b>. Files are saved first, then trained from the status table.</div>

      <label>ADTC Dataset Name <span class="req">*</span></label>
      <input id="adtcName" placeholder="e.g. Financial Compliance 2026" />

      <label>Topic / Domain</label>
      <input id="adtcTopic" placeholder="e.g. finance, hr, procurement, healthcare" />

      <label>Choose files <span class="req">*</span></label>
      <input id="adtcFiles" type="file" multiple accept=".pdf,.docx,.xlsx,.xls,.txt,.md,.log,.csv,.tsv,.json,.jsonl,.ndjson" />
      <div id="adtcFileList" class="hint" style="margin-top:6px"></div>

      <div class="toolbar">
        <button class="primary" id="adtcSaveBtn">💾 Save ADTC Dataset</button>
        <button id="adtcClearBtn">Clear</button>
        <span class="k" id="adtcMsg"></span>
      </div>

      <div id="adtcResults" style="margin-top:10px"></div>
    </div>

    <div class="card" style="margin-top:14px">
      <h3 style="margin-top:0">Training Status</h3>
      <div class="stat">
        <div class="pill">Knowledge docs: <b id="kDocs">0</b></div>
        <div class="pill">Saved connections: <b id="kLinks">0</b></div>
        <div class="pill">Saved datasets: <b id="kDatasets">0</b></div>
        <div class="pill">Saved ADTC datasets: <b id="kAdtcDatasets">0</b></div>
        <div class="pill">Running: <b id="cRun">0</b></div>
        <div class="pill">Completed: <b id="cDone">0</b></div>
        <div class="pill">Failed: <b id="cFail">0</b></div>
      </div>

      <h4 style="margin:12px 0 6px">Database Connections</h4>
      <table>
        <thead>
          <tr>
            <th>Name</th><th>Type</th><th>URL (masked)</th>
            <th>Status</th><th>Last trained</th><th>Docs</th><th>Actions</th>
          </tr>
        </thead>
        <tbody id="dbRowsBody"></tbody>
      </table>

      <h4 style="margin:18px 0 6px">Datasets</h4>
      <table>
        <thead>
          <tr>
            <th>Name</th><th>Files</th><th>Size</th>
            <th>Status</th><th>Last trained</th><th>Docs</th><th>Actions</th>
          </tr>
        </thead>
        <tbody id="dsRowsBody"></tbody>
      </table>

      <h4 style="margin:18px 0 6px">ADTC Datasets</h4>
      <table>
        <thead>
          <tr>
            <th>Name</th><th>Topic</th><th>Files</th><th>Size</th>
            <th>Status</th><th>Last trained</th><th>Docs</th><th>Actions</th>
          </tr>
        </thead>
        <tbody id="adtcRowsBody"></tbody>
      </table>
    </div>
  </div>

  <script>
    let PRESETS = {};
    let testedOk = false;

    async function api(url, options){
      const res = await fetch(url, options || {});
      const txt = await res.text();
      let data = {};
      try { data = txt ? JSON.parse(txt) : {}; } catch(e) { data = {raw: txt}; }
      if(!res.ok) throw new Error(data.error || ('HTTP ' + res.status));
      return data;
    }

    function fmtTime(ts){
      if(!ts) return '-';
      try { return new Date(ts*1000).toLocaleString(); } catch(e){ return ts; }
    }

    function badgeFor(s){
      const map = {saved:'b-saved', running:'b-running', completed:'b-completed', failed:'b-failed', stopped:'b-stopped'};
      const cls = map[s] || 'b-saved';
      return '<span class="badge ' + cls + '">' + s + '</span>';
    }

    function renderFields(){
      const t = document.getElementById('dbType').value;
      const preset = PRESETS[t];
      const wrap = document.getElementById('dbFields');
      wrap.innerHTML = '';
      const hint = document.getElementById('driverHint');
      hint.textContent = preset && preset.driver_pip
        ? ('Required driver: pip install ' + preset.driver_pip)
        : (preset ? 'No extra driver needed.' : '');
      if(!preset) return;
      for(const f of preset.fields){
        const lab = document.createElement('label');
        lab.innerHTML = f.label + (f.required ? ' <span class="req">*</span>' : '');
        const inp = document.createElement('input');
        inp.id = 'f_' + f.key;
        inp.dataset.key = f.key;
        if(f.secret) inp.type = 'password';
        if(f.placeholder) inp.placeholder = f.placeholder;
        if(f.default) inp.value = f.default;
        inp.oninput = () => { testedOk = false; document.getElementById('saveDbBtn').disabled = true; };
        wrap.appendChild(lab);
        wrap.appendChild(inp);
      }
      testedOk = false;
      document.getElementById('saveDbBtn').disabled = true;
    }

    function collectPayload(){
      const t = document.getElementById('dbType').value;
      const preset = PRESETS[t] || {fields:[]};
      const fields = {};
      for(const f of preset.fields){
        const el = document.getElementById('f_' + f.key);
        if(el) fields[f.key] = el.value;
      }
      return {
        name: document.getElementById('dbName').value.trim(),
        type: t,
        fields: fields,
        tables: document.getElementById('dbTables').value,
        sample_rows: Number(document.getElementById('dbRows').value || 5),
      };
    }

    async function loadPresets(){
      const r = await api('/api/training/presets');
      PRESETS = r.presets;
      const sel = document.getElementById('dbType');
      sel.innerHTML = '';
      for(const k of Object.keys(PRESETS)){
        const o = document.createElement('option');
        o.value = k; o.textContent = PRESETS[k].label;
        sel.appendChild(o);
      }
      sel.onchange = renderFields;
      renderFields();
      if(!r.sqlalchemy_available){
        document.getElementById('dbMsg').innerHTML = '<span style="color:#fca5a5">SQLAlchemy is not installed. Run: pip install sqlalchemy</span>';
      }
    }

    async function refreshStatus(){
      try {
        const s = await api('/api/training/status');
        document.getElementById('kDocs').textContent = s.total_docs;
        document.getElementById('kLinks').textContent = s.total_db_links;
        document.getElementById('kDatasets').textContent = s.total_datasets || 0;
        document.getElementById('kAdtcDatasets').textContent = s.total_adtc_datasets || 0;
        const linkRun = s.counts.running || 0;
        const linkDone = s.counts.completed || 0;
        const linkFail = s.counts.failed || 0;
        const dsRun = (s.dataset_counts && s.dataset_counts.running) || 0;
        const dsDone = (s.dataset_counts && s.dataset_counts.completed) || 0;
        const dsFail = (s.dataset_counts && s.dataset_counts.failed) || 0;
        const adtcRun = (s.adtc_counts && s.adtc_counts.running) || 0;
        const adtcDone = (s.adtc_counts && s.adtc_counts.completed) || 0;
        const adtcFail = (s.adtc_counts && s.adtc_counts.failed) || 0;
        document.getElementById('cRun').textContent  = linkRun + dsRun + adtcRun;
        document.getElementById('cDone').textContent = linkDone + dsDone + adtcDone;
        document.getElementById('cFail').textContent = linkFail + dsFail + adtcFail;

        const body = document.getElementById('dbRowsBody');
        body.innerHTML = '';
        if(!s.db_links.length){
          body.innerHTML = '<tr><td colspan="7" class="k" style="text-align:center;padding:18px">No saved connections yet.</td></tr>';
        } else {
          for(const l of s.db_links){
            const tr = document.createElement('tr');
            tr.innerHTML =
              '<td><b>' + (l.name || '-') + '</b></td>' +
              '<td>' + (l.type || '-') + '</td>' +
              '<td class="mono">' + (l.url_masked || '-') + '</td>' +
              '<td>' + badgeFor(l.status || 'saved') + '</td>' +
              '<td>' + fmtTime(l.last_trained_at) + '</td>' +
              '<td>' + (l.last_added_docs || 0) + '</td>' +
              '<td>' +
                '<button class="icon" title="Open Training Run" onclick="window.open(\'/training/run/' + l.id + '\', \'_blank\')">🔍 View</button> ' +
                '<button class="icon danger" title="Delete connection" onclick="deleteLink(\'' + l.id + '\')">🗑</button>' +
              '</td>';
            body.appendChild(tr);
          }
        }

        const dsBody = document.getElementById('dsRowsBody');
        dsBody.innerHTML = '';
        const datasets = s.datasets || [];
        if(!datasets.length){
          dsBody.innerHTML = '<tr><td colspan="7" class="k" style="text-align:center;padding:18px">No saved datasets yet.</td></tr>';
        } else {
          for(const d of datasets){
            const tr = document.createElement('tr');
            tr.innerHTML =
              '<td><b>' + (d.name || '-') + '</b></td>' +
              '<td>' + (d.file_count || 0) + '</td>' +
              '<td>' + Math.round((d.total_size||0)/1024) + ' KB</td>' +
              '<td>' + badgeFor(d.status || 'saved') + '</td>' +
              '<td>' + fmtTime(d.last_trained_at) + '</td>' +
              '<td>' + (d.last_added_docs || 0) + '</td>' +
              '<td>' +
                '<button class="icon" title="Open Dataset Training" onclick="window.open(\'/training/dataset/run/' + d.id + '\', \'_blank\')">🔍 View</button> ' +
                '<button class="icon danger" title="Delete dataset" onclick="deleteDataset(\'' + d.id + '\')">🗑</button>' +
              '</td>';
            dsBody.appendChild(tr);
          }
        }

        const adtcBody = document.getElementById('adtcRowsBody');
        adtcBody.innerHTML = '';
        const adtcDatasets = s.adtc_datasets || [];
        if(!adtcDatasets.length){
          adtcBody.innerHTML = '<tr><td colspan="8" class="k" style="text-align:center;padding:18px">No ADTC datasets yet.</td></tr>';
        } else {
          for(const d of adtcDatasets){
            const tr = document.createElement('tr');
            tr.innerHTML =
              '<td><b>' + (d.name || '-') + '</b></td>' +
              '<td>' + (d.topic || 'general') + '</td>' +
              '<td>' + (d.file_count || 0) + '</td>' +
              '<td>' + Math.round((d.total_size||0)/1024) + ' KB</td>' +
              '<td>' + badgeFor(d.status || 'saved') + '</td>' +
              '<td>' + fmtTime(d.last_trained_at) + '</td>' +
              '<td>' + (d.last_added_docs || 0) + '</td>' +
              '<td>' +
                '<button class="icon" title="Open ADTC Training" onclick="window.open(\'/training/adtc/run/' + d.id + '\', \'_blank\')">🔍 View</button> ' +
                '<button class="icon danger" title="Delete ADTC dataset" onclick="deleteAdtcDataset(\'' + d.id + '\')">🗑</button>' +
              '</td>';
            adtcBody.appendChild(tr);
          }
        }
      } catch(e){
        document.getElementById('kDocs').textContent = 'error';
      }
    }

    async function deleteLink(id){
      if(!confirm('Delete this connection? Trained data will remain in knowledge memory.')) return;
      try { await api('/api/training/db/links/' + id, {method:'DELETE'}); await refreshStatus(); }
      catch(e){ alert('Error: ' + e.message); }
    }

    async function deleteDataset(id){
      if(!confirm('Delete this dataset and its stored files? Already-trained docs will remain in knowledge memory.')) return;
      try { await api('/api/training/datasets/' + id, {method:'DELETE'}); await refreshStatus(); }
      catch(e){ alert('Error: ' + e.message); }
    }

    async function deleteAdtcDataset(id){
      if(!confirm('Delete this ADTC dataset and all related artifacts (stored files/folders + linked ADTC knowledge docs)?')) return;
      try { await api('/api/training/adtc/datasets/' + id, {method:'DELETE'}); await refreshStatus(); }
      catch(e){ alert('Error: ' + e.message); }
    }

    document.addEventListener('DOMContentLoaded', () => {
      document.getElementById('addDocBtn').onclick = async () => {
        const title = document.getElementById('docTitle').value;
        const content = document.getElementById('docContent').value;
        const msg = document.getElementById('docMsg');
        msg.textContent = 'Saving...';
        try {
          await api('/api/training/docs', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({title, content})});
          msg.textContent = 'Saved successfully.';
          document.getElementById('docContent').value = '';
          await refreshStatus();
        } catch(e){ msg.textContent = 'Error: ' + e.message; }
      };

      document.getElementById('testDbBtn').onclick = async () => {
        const m = document.getElementById('dbMsg');
        m.innerHTML = 'Testing connection...';
        try {
          const r = await api('/api/training/db/test', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(collectPayload())});
          m.innerHTML = '<span style="color:#a7f3d0">✓ Connection OK ('+ r.url_masked +')</span>';
          testedOk = true;
          document.getElementById('saveDbBtn').disabled = false;
        } catch(e){
          m.innerHTML = '<span style="color:#fca5a5">✗ ' + e.message + '</span>';
          testedOk = false;
          document.getElementById('saveDbBtn').disabled = true;
        }
      };

      document.getElementById('saveDbBtn').onclick = async () => {
        const m = document.getElementById('dbMsg');
        if(!testedOk){ m.innerHTML = '<span style="color:#fca5a5">Run a successful Test before saving.</span>'; return; }
        const payload = collectPayload();
        if(!payload.name){ m.innerHTML = '<span style="color:#fca5a5">Connection name is required.</span>'; return; }
        m.innerHTML = 'Saving connection...';
        try {
          const r = await api('/api/training/db/save', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
          m.innerHTML = '<span style="color:#a7f3d0">✓ Saved. Use 🔍 View in the table below to run training.</span>';
          await refreshStatus();
        } catch(e){ m.innerHTML = '<span style="color:#fca5a5">Error: ' + e.message + '</span>'; }
      };

      // --- Dataset upload handlers ---
      const dsFilesEl = document.getElementById('dsFiles');
      const dsListEl = document.getElementById('dsFileList');
      const dsMsgEl = document.getElementById('dsMsg');
      const dsResultsEl = document.getElementById('dsResults');

      function renderFileList(){
        const files = dsFilesEl.files;
        if(!files || !files.length){ dsListEl.textContent = ''; return; }
        const items = [];
        for(let i=0;i<files.length;i++){
          const f = files[i];
          items.push(f.name + ' (' + Math.round(f.size/1024) + ' KB)');
        }
        dsListEl.innerHTML = 'Selected: ' + items.join(' · ');
      }
      dsFilesEl.onchange = renderFileList;

      document.getElementById('dsClearBtn').onclick = () => {
        dsFilesEl.value = '';
        dsListEl.textContent = '';
        dsMsgEl.textContent = '';
        dsResultsEl.innerHTML = '';
        document.getElementById('dsName').value = '';
      };

      document.getElementById('dsSaveBtn').onclick = async () => {
        const name = document.getElementById('dsName').value.trim();
        const files = dsFilesEl.files;
        if(!name){
          dsMsgEl.innerHTML = '<span style="color:#fca5a5">Dataset name is required.</span>';
          return;
        }
        if(!files || !files.length){
          dsMsgEl.innerHTML = '<span style="color:#fca5a5">Select at least one file.</span>';
          return;
        }
        const fd = new FormData();
        fd.append('name', name);
        for(let i=0;i<files.length;i++){ fd.append('files', files[i]); }
        dsMsgEl.innerHTML = 'Saving dataset (' + files.length + ' file(s))...';
        dsResultsEl.innerHTML = '';
        try {
          const res = await fetch('/api/training/dataset/upload', { method:'POST', body: fd });
          const txt = await res.text();
          let data = {};
          try { data = txt ? JSON.parse(txt) : {}; } catch(e){ data = {raw: txt}; }
          if(!res.ok) throw new Error(data.error || ('HTTP ' + res.status));
          const ds = data.dataset || {};
          dsMsgEl.innerHTML = '<span style="color:#a7f3d0">✓ Saved as "' + (ds.name||name) + '". Use 🔍 View in the Datasets table below to run training.</span>';
          if(data.errors && data.errors.length){
            let h = '<div style="color:#fca5a5;margin-top:6px"><b>Some files were skipped:</b><ul>';
            for(const e of data.errors){ h += '<li>' + e.file + ': ' + e.error + '</li>'; }
            h += '</ul></div>';
            dsResultsEl.innerHTML = h;
          }
          dsFilesEl.value = '';
          dsListEl.textContent = '';
          document.getElementById('dsName').value = '';
          await refreshStatus();
        } catch(e){
          dsMsgEl.innerHTML = '<span style="color:#fca5a5">Error: ' + e.message + '</span>';
        }
      };

      // --- ADTC upload handlers ---
      const adtcFilesEl = document.getElementById('adtcFiles');
      const adtcListEl = document.getElementById('adtcFileList');
      const adtcMsgEl = document.getElementById('adtcMsg');
      const adtcResultsEl = document.getElementById('adtcResults');

      function renderAdtcFileList(){
        const files = adtcFilesEl.files;
        if(!files || !files.length){ adtcListEl.textContent = ''; return; }
        const items = [];
        for(let i=0;i<files.length;i++){
          const f = files[i];
          items.push(f.name + ' (' + Math.round(f.size/1024) + ' KB)');
        }
        adtcListEl.innerHTML = 'Selected: ' + items.join(' · ');
      }
      adtcFilesEl.onchange = renderAdtcFileList;

      document.getElementById('adtcClearBtn').onclick = () => {
        adtcFilesEl.value = '';
        adtcListEl.textContent = '';
        adtcMsgEl.textContent = '';
        adtcResultsEl.innerHTML = '';
        document.getElementById('adtcName').value = '';
        document.getElementById('adtcTopic').value = '';
      };

      document.getElementById('adtcSaveBtn').onclick = async () => {
        const name = document.getElementById('adtcName').value.trim();
        const topic = document.getElementById('adtcTopic').value.trim();
        const files = adtcFilesEl.files;
        if(!name){
          adtcMsgEl.innerHTML = '<span style="color:#fca5a5">ADTC dataset name is required.</span>';
          return;
        }
        if(!files || !files.length){
          adtcMsgEl.innerHTML = '<span style="color:#fca5a5">Select at least one file.</span>';
          return;
        }
        const fd = new FormData();
        fd.append('name', name);
        fd.append('topic', topic || 'general');
        for(let i=0;i<files.length;i++){ fd.append('files', files[i]); }
        adtcMsgEl.innerHTML = 'Saving ADTC dataset (' + files.length + ' file(s))...';
        adtcResultsEl.innerHTML = '';
        try {
          const res = await fetch('/api/training/adtc/upload', { method:'POST', body: fd });
          const txt = await res.text();
          let data = {};
          try { data = txt ? JSON.parse(txt) : {}; } catch(e){ data = {raw: txt}; }
          if(!res.ok) throw new Error(data.error || ('HTTP ' + res.status));
          const ds = data.dataset || {};
          adtcMsgEl.innerHTML = '<span style="color:#a7f3d0">✓ ADTC dataset saved as "' + (ds.name||name) + '". Open via 🔍 View to run advanced training.</span>';
          if(data.errors && data.errors.length){
            let h = '<div style="color:#fca5a5;margin-top:6px"><b>Some files were skipped:</b><ul>';
            for(const e of data.errors){ h += '<li>' + e.file + ': ' + e.error + '</li>'; }
            h += '</ul></div>';
            adtcResultsEl.innerHTML = h;
          }
          adtcFilesEl.value = '';
          adtcListEl.textContent = '';
          document.getElementById('adtcName').value = '';
          document.getElementById('adtcTopic').value = '';
          await refreshStatus();
        } catch(e){
          adtcMsgEl.innerHTML = '<span style="color:#fca5a5">Error: ' + e.message + '</span>';
        }
      };

      loadPresets().then(refreshStatus);
      setInterval(refreshStatus, 4000);
    });
  </script>
</body>
</html>
"""


ADTC_RUN_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ADTC Training — {{ dataset_name }}</title>
  <style>
    :root { --bg:#0f172a; --panel:#1e293b; --line:#334155; --text:#e2e8f0; --muted:#94a3b8; --accent:#10b981; --warn:#ef4444; }
    *{box-sizing:border-box}
    body{margin:0;background:var(--bg);color:var(--text);font-family:Segoe UI,system-ui,sans-serif}
    .wrap{max-width:1100px;margin:0 auto;padding:16px}
    .head{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;gap:12px}
    .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px;margin-bottom:12px}
    .k{color:var(--muted);font-size:12px}
    button{border:1px solid var(--line);background:#0b1220;color:var(--text);border-radius:10px;padding:9px 12px;cursor:pointer;font-size:13px}
    button.primary{background:var(--accent);border-color:var(--accent);color:#072012;font-weight:700}
    button:disabled{opacity:.5;cursor:not-allowed}
    table{width:100%;border-collapse:collapse}
    th,td{padding:8px;border-bottom:1px solid var(--line);font-size:12px;text-align:left}
    th{color:var(--muted);font-weight:600}
    .pill{display:inline-block;background:#0b1220;border:1px solid var(--line);padding:6px 10px;border-radius:10px;font-size:12px;margin-right:8px}
    .badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:700}
    .b-saved{background:#1e3a8a;color:#dbeafe}
    .b-running{background:#7c2d12;color:#fed7aa}
    .b-completed{background:#064e3b;color:#a7f3d0}
    .b-failed{background:#7f1d1d;color:#fecaca}
    .err{color:#fca5a5;font-size:12px;margin-top:6px}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="head">
      <div>
        <h2 style="margin:0">ADTC Training: {{ dataset_name }}</h2>
        <div class="k" id="adtcStatusLine">Loading…</div>
      </div>
      <div>
        <button onclick="window.location.href='/training'">← Training Center</button>
        <button onclick="loadDataset()">⟳ Refresh</button>
      </div>
    </div>

    <div class="card">
      <h3 style="margin-top:0">Files and Sources</h3>
      <div id="filesBox" class="k">Loading…</div>
    </div>

    <div class="card">
      <h3 style="margin-top:0">Advanced Training</h3>
      <div class="k">Runs deep multi-file ingestion with table/structure extraction, cross-file relationship detection, and ADTC summary generation.</div>
      <div style="margin-top:10px;display:flex;gap:8px;flex-wrap:wrap;align-items:center">
        <button class="primary" id="runBtn">▶ Run ADTC Training</button>
        <span class="pill">Status: <b id="dsStatus">-</b></span>
        <span class="pill">Last trained: <b id="dsLast">-</b></span>
        <span class="pill">Docs added (last run): <b id="dsDocs">0</b></span>
      </div>
      <div id="runMsg" class="k" style="margin-top:8px"></div>
      <div id="errBox" class="err"></div>
    </div>

    <div class="card">
      <h3 style="margin-top:0">Cross-file Relationships</h3>
      <div id="relsBox" class="k">No relationships yet.</div>
    </div>

    <div class="card">
      <h3 style="margin-top:0">What The System Learned</h3>
      <div class="k">Per-file extraction profile with deep monitoring: extracted content preview, PDF page diagnostics, analysis pipeline, confidence, and answerability evidence.</div>
      <div id="learnedBox" class="k" style="margin-top:8px">No learned details yet.</div>
    </div>

    <div class="card">
      <h3 style="margin-top:0">How Learning Is Integrated Into The Model</h3>
      <div id="integrationBox" class="k">Integration report will appear after training.</div>
    </div>

    <div class="card">
      <h3 style="margin-top:0">Last Run Results</h3>
      <div id="resultsBox" class="k">No run yet.</div>
    </div>
  </div>

  <script>
    const DATASET_ID = {{ dataset_id|tojson }};
    function fmtTime(ts){ if(!ts) return '-'; try { return new Date(ts*1000).toLocaleString(); } catch(e){ return ts; } }
    function badgeFor(s){
      const map = {saved:'b-saved', running:'b-running', completed:'b-completed', failed:'b-failed'};
      const cls = map[s] || 'b-saved';
      return '<span class="badge ' + cls + '">' + s + '</span>';
    }

    async function loadDataset(){
      try {
        const res = await fetch('/api/training/adtc/datasets/' + DATASET_ID);
        const data = await res.json();
        if(!res.ok) throw new Error(data.error || ('HTTP ' + res.status));
        renderDataset(data);
      } catch(e){
        document.getElementById('adtcStatusLine').innerHTML = '<span style="color:#fca5a5">Error: ' + e.message + '</span>';
      }
    }

    function renderDataset(d){
      document.getElementById('adtcStatusLine').innerHTML =
        badgeFor(d.status) + ' · topic: ' + (d.topic || 'general') + ' · ' + (d.file_count||0) + ' file(s)';
      document.getElementById('dsStatus').textContent = d.status || '-';
      document.getElementById('dsLast').textContent = fmtTime(d.last_trained_at);
      document.getElementById('dsDocs').textContent = d.last_added_docs || 0;

      let fh = '';
      if(d.files && d.files.length){
        fh += '<table><thead><tr><th>File</th><th>Kind</th><th>Size</th></tr></thead><tbody>';
        for(const f of d.files){
          fh += '<tr><td><b>'+f.name+'</b></td><td>'+(f.kind||'-')+'</td><td>'+Math.round((f.size_bytes||0)/1024)+' KB</td></tr>';
        }
        fh += '</tbody></table>';
      } else { fh = 'No files.'; }
      document.getElementById('filesBox').innerHTML = fh;

      const rels = d.relationships || [];
      const relBox = document.getElementById('relsBox');
      if(!rels.length){
        relBox.textContent = 'No relationships detected yet.';
      } else {
        let rh = '<table><thead><tr><th>From</th><th>To</th><th>Shared keys</th><th>Score</th></tr></thead><tbody>';
        for(const r of rels){
          rh += '<tr><td>'+ (r.from || '-') +'</td><td>'+ (r.to || '-') +'</td><td>'+ (r.shared_keys || []).join(', ') +'</td><td>'+ (r.score || 0) +'</td></tr>';
        }
        rh += '</tbody></table>';
        relBox.innerHTML = rh;
      }

      const errBox = document.getElementById('errBox');
      errBox.textContent = d.last_error || '';

      const learned = d.learned_details || [];
      const learnedBox = document.getElementById('learnedBox');
      if(!learned.length){
        learnedBox.textContent = 'No learned details yet.';
      } else {
        let lh = '<table><thead><tr><th>File</th><th>Kind</th><th>Quality</th><th>Confidence</th><th>Chars</th><th>Units</th><th>Chunks</th><th>Schema keys</th><th>Top terms</th><th>Summary</th></tr></thead><tbody>';
        for(const item of learned){
          const contact = item.contact_signals || {};
          const unclear = item.unclear_points || [];
          const pdfs = item.pdf_stats || {};
          const metrics = item.signal_metrics || {};
          const coverage = item.topic_coverage || {};
          const probes = item.qa_probes || [];
          const preview = item.content_preview || '';
          const pipe = item.understanding_pipeline || {};
          const p1 = pipe.step_1_extract || {};
          const p2 = pipe.step_2_analyze || {};
          const p3 = pipe.step_3_answerability || {};
          const pdfSamples = (pdfs.page_samples || []).slice(0, 8);
          const contactsDetected = p2.contacts_detected || {};
          const contactTriplet = (contactsDetected.urls || 0) + '/' + (contactsDetected.emails || 0) + '/' + (contactsDetected.phones || 0);
          const pdfDiagText = pdfSamples.length
            ? pdfSamples.map(s => 'p' + s.page + ': chars=' + (s.chars || 0) + (s.sample ? ', sample="' + s.sample + '"' : '')).join(' | ')
            : 'No page samples';
          const pdfExtra = item.kind === 'pdf'
            ? ('<div style="margin-top:4px"><b>PDF extraction stats:</b> pages with text ' + (pdfs.pages_with_text || 0) + '/' + (pdfs.pages_total || 0) + ', low-text pages ' + (pdfs.pages_low_text || 0) + ', avg chars/page ' + (pdfs.avg_chars_per_page || 0) + '</div>' +
               '<div style="margin-top:4px"><b>PDF page diagnostics:</b> ' + pdfDiagText + '</div>')
            : '';
          lh += '<tr>' +
            '<td><b>' + (item.file || '-') + '</b></td>' +
            '<td>' + (item.kind || '-') + '</td>' +
            '<td>' + (item.quality_score || 0) + '/100</td>' +
            '<td>' + (item.training_confidence || 0) + '/100</td>' +
            '<td>' + (item.char_count || 0) + '</td>' +
            '<td>' + (item.units_read || 0) + '</td>' +
            '<td>' + (item.chunks_created || 0) + '</td>' +
            '<td>' + ((item.schema_keys || []).join(', ') || '-') + '</td>' +
            '<td>' + ((item.top_terms || []).join(', ') || '-') + '</td>' +
            '<td>' + (item.understanding_summary || '-') + '</td>' +
          '</tr>';
          lh += '<tr><td colspan="10" style="color:#94a3b8">' +
            '<div><b>How system sees content:</b><pre style="white-space:pre-wrap;background:#0b1220;border:1px solid #334155;border-radius:8px;padding:8px;max-height:180px;overflow:auto;margin:6px 0">' + (preview || 'No extracted text preview') + '</pre></div>' +
            '<div style="margin-top:4px"><b>Understanding pipeline:</b> extract(chars=' + (p1.chars || 0) + ', units=' + (p1.units || 0) + ', chunks=' + (p1.chunks_written || 0) + ') -> analyze(facts=' + (p2.key_facts || 0) + ', top_terms=' + (p2.top_terms || 0) + ', contacts urls/emails/phones=' + contactTriplet + ', unclear=' + (p2.unclear_points || 0) + ') -> answerability(coverage=' + (p3.coverage_score || 0) + ', answerable probes=' + (p3.qa_probe_answerable || 0) + '/' + (p3.qa_probe_count || 0) + ', confidence=' + (p3.confidence || 0) + ')</div>' +
            '<div><b>Key facts:</b> ' + ((item.key_facts || []).slice(0,6).join(' | ') || 'None extracted') + '</div>' +
            '<div style="margin-top:4px"><b>Section titles:</b> ' + ((item.section_titles || []).slice(0,8).join(' | ') || 'None detected') + '</div>' +
            '<div style="margin-top:4px"><b>Contacts:</b> URLs=' + ((contact.urls || []).length || 0) + ', Emails=' + ((contact.emails || []).length || 0) + ', Phones=' + ((contact.phones || []).length || 0) + '</div>' +
            '<div style="margin-top:4px"><b>Signal metrics:</b> tokens ' + (metrics.token_count || 0) + ', unique terms ' + (metrics.unique_terms_count || 0) + ', lexical density ' + (metrics.lexical_density || 0) + ', digit density ' + (metrics.digit_density || 0) + '</div>' +
            '<div style="margin-top:4px"><b>Topic coverage:</b> location=' + (coverage.location || 0) + ', contacts=' + (coverage.contacts || 0) + ', services=' + (coverage.services || 0) + ', schedule=' + (coverage.schedule || 0) + ', insurance=' + (coverage.insurance || 0) + ', score=' + (coverage.coverage_score || 0) + '/100</div>' +
            '<div style="margin-top:4px"><b>QA probes:</b> ' + (probes.length ? probes.slice(0,6).map(p => p.id + '(' + (p.answerable ? 'Y' : 'N') + ',' + (p.confidence||0) + ')').join(' | ') : 'No probes') + '</div>' +
            pdfExtra +
            '<div style="margin-top:4px"><b>Unclear parts:</b> ' + (unclear.length ? unclear.join(' | ') : 'None') + '</div>' +
          '</td></tr>';
        }
        lh += '</tbody></table>';
        learnedBox.innerHTML = lh;
      }

      const integ = d.integration_report || {};
      const integBox = document.getElementById('integrationBox');
      if(!integ.docs_written){
        integBox.textContent = 'Integration report will appear after training.';
      } else {
        const q = integ.quality_overview || {};
        const probes = integ.qa_probe_summary || {};
        const cov = integ.topic_coverage_totals || {};
        const understood = integ.understood_points || [];
        const unclear = integ.unclear_points || [];
        integBox.innerHTML =
          '<div><b>Knowledge source:</b> ' + (integ.knowledge_source || '-') + '</div>' +
          '<div style="margin-top:4px"><b>Dataset:</b> ' + (integ.dataset_name || '-') + ' (' + (integ.dataset_id || '-') + ')</div>' +
          '<div style="margin-top:4px"><b>Topic:</b> ' + (integ.topic || 'general') + '</div>' +
          '<div style="margin-top:4px"><b>Docs written to knowledge memory:</b> ' + (integ.docs_written || 0) + '</div>' +
          '<div style="margin-top:4px"><b>Average understanding quality:</b> ' + (q.avg_quality_score || 0) + '/100</div>' +
          '<div style="margin-top:4px"><b>Average training confidence:</b> ' + (q.avg_training_confidence || 0) + '/100</div>' +
          '<div style="margin-top:4px"><b>Files with unclear parts:</b> ' + (q.files_with_unclear_parts || 0) + ' of ' + (q.files_analyzed || 0) + '</div>' +
          '<div style="margin-top:4px"><b>QA answerability:</b> ' + (probes.answerable_probes || 0) + '/' + (probes.total_probes || 0) + ' probes, avg confidence ' + (probes.avg_probe_confidence || 0) + '/100</div>' +
          '<div style="margin-top:4px"><b>Topic coverage totals:</b> location=' + (cov.location || 0) + ', contacts=' + (cov.contacts || 0) + ', services=' + (cov.services || 0) + ', schedule=' + (cov.schedule || 0) + ', insurance=' + (cov.insurance || 0) + '</div>' +
          '<div style="margin-top:8px"><b>Retrieval behavior:</b> ' + (integ.retrieval_note || '-') + '</div>' +
          '<div style="margin-top:4px"><b>Cross-file reasoning:</b> ' + (integ.cross_file_reasoning_note || '-') + '</div>' +
          '<div style="margin-top:8px"><b>What was understood:</b> ' + (understood.length ? understood.slice(0,8).join(' | ') : 'No explicit points.') + '</div>' +
          '<div style="margin-top:4px"><b>What remained unclear:</b> ' + (unclear.length ? unclear.slice(0,8).join(' | ') : 'No major unclear items.') + '</div>' +
          '<div style="margin-top:8px"><b>Knowledge doc IDs:</b> ' + ((integ.doc_ids || []).slice(0, 30).join(', ') || '-') + '</div>';
      }

      const rb = document.getElementById('resultsBox');
      if(d.last_results && d.last_results.length){
        let h = '<table><thead><tr><th>File</th><th>Kind</th><th>Units</th><th>Docs added</th></tr></thead><tbody>';
        for(const r of d.last_results){
          h += '<tr><td>'+r.file+'</td><td>'+r.kind+'</td><td>'+(r.units||0)+'</td><td>'+(r.docs_added||0)+'</td></tr>';
        }
        h += '</tbody></table>';
        rb.innerHTML = h;
      } else { rb.textContent = 'No run yet.'; }

      document.getElementById('runBtn').disabled = (d.status === 'running');
    }

    document.getElementById('runBtn').onclick = async () => {
      const btn = document.getElementById('runBtn');
      const msg = document.getElementById('runMsg');
      btn.disabled = true; msg.textContent = 'Running ADTC training… please wait.';
      try {
        const res = await fetch('/api/training/adtc/datasets/' + DATASET_ID + '/start', {method:'POST'});
        const data = await res.json();
        if(!res.ok) throw new Error(data.error || ('HTTP ' + res.status));
        msg.innerHTML = data.ok
          ? '<span style="color:#a7f3d0">✓ ADTC training complete. ' + (data.docs_added||0) + ' doc(s) added.</span>'
          : '<span style="color:#fca5a5">ADTC training finished with errors. Check details below.</span>';
        if(data.dataset) renderDataset(data.dataset);
      } catch(e){
        msg.innerHTML = '<span style="color:#fca5a5">Error: ' + e.message + '</span>';
        btn.disabled = false;
      }
    };

    loadDataset();
    setInterval(loadDataset, 5000);
  </script>
</body>
</html>
"""


DATASET_RUN_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Dataset Training — {{ dataset_name }}</title>
  <style>
    :root { --bg:#0f172a; --panel:#1e293b; --line:#334155; --text:#e2e8f0; --muted:#94a3b8; --accent:#10b981; --warn:#ef4444; --info:#38bdf8; }
    *{box-sizing:border-box}
    body{margin:0;background:var(--bg);color:var(--text);font-family:Segoe UI,system-ui,sans-serif}
    .wrap{max-width:1100px;margin:0 auto;padding:16px}
    .head{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;gap:12px}
    .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px;margin-bottom:12px}
    button{border:1px solid var(--line);background:#0b1220;color:var(--text);border-radius:10px;padding:9px 12px;cursor:pointer;font-size:13px}
    button.primary{background:var(--accent);border-color:var(--accent);color:#072012;font-weight:700}
    button.danger{background:#3b0d0d;border-color:#7f1d1d;color:#fecaca}
    button:disabled{opacity:.5;cursor:not-allowed}
    table{width:100%;border-collapse:collapse}
    th,td{padding:8px;border-bottom:1px solid var(--line);font-size:12px;text-align:left}
    th{color:var(--muted);font-weight:600}
    .k{color:var(--muted);font-size:12px}
    .pill{display:inline-block;background:#0b1220;border:1px solid var(--line);padding:6px 10px;border-radius:10px;font-size:12px;margin-right:8px}
    .badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:700}
    .b-saved{background:#1e3a8a;color:#dbeafe}
    .b-running{background:#7c2d12;color:#fed7aa}
    .b-completed{background:#064e3b;color:#a7f3d0}
    .b-failed{background:#7f1d1d;color:#fecaca}
    .err{color:#fca5a5;font-size:12px;margin-top:6px}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="head">
      <div>
        <h2 style="margin:0">Dataset Training: {{ dataset_name }}</h2>
        <div class="k" id="dsStatusLine">Loading…</div>
      </div>
      <div>
        <button onclick="window.location.href='/training'">← Training Center</button>
        <button onclick="loadDataset()">⟳ Refresh</button>
      </div>
    </div>

    <div class="card">
      <h3 style="margin-top:0">Files in this dataset</h3>
      <div id="filesBox" class="k">Loading…</div>
    </div>

    <div class="card">
      <h3 style="margin-top:0">Training</h3>
      <div class="k">Run training to ingest these files into the knowledge memory used by the chat model. Re-running adds new docs.</div>
      <div style="margin-top:10px;display:flex;gap:8px;flex-wrap:wrap;align-items:center">
        <button class="primary" id="runBtn">▶ Run Training</button>
        <span class="pill">Status: <b id="dsStatus">-</b></span>
        <span class="pill">Last trained: <b id="dsLast">-</b></span>
        <span class="pill">Docs added (last run): <b id="dsDocs">0</b></span>
      </div>
      <div id="runMsg" class="k" style="margin-top:8px"></div>
      <div id="errBox" class="err"></div>
    </div>

    <div class="card">
      <h3 style="margin-top:0">Last Run Results</h3>
      <div id="resultsBox" class="k">No run yet.</div>
    </div>
  </div>

  <script>
    const DATASET_ID = {{ dataset_id|tojson }};
    function fmtTime(ts){ if(!ts) return '-'; try { return new Date(ts*1000).toLocaleString(); } catch(e){ return ts; } }
    function badgeFor(s){
      const map = {saved:'b-saved', running:'b-running', completed:'b-completed', failed:'b-failed'};
      const cls = map[s] || 'b-saved';
      return '<span class="badge ' + cls + '">' + s + '</span>';
    }

    async function loadDataset(){
      try {
        const res = await fetch('/api/training/datasets/' + DATASET_ID);
        const data = await res.json();
        if(!res.ok) throw new Error(data.error || ('HTTP ' + res.status));
        renderDataset(data);
      } catch(e){
        document.getElementById('dsStatusLine').innerHTML = '<span style="color:#fca5a5">Error: ' + e.message + '</span>';
      }
    }

    function renderDataset(d){
      document.getElementById('dsStatusLine').innerHTML =
        badgeFor(d.status) + ' · ' + (d.file_count||0) + ' file(s) · ' + Math.round((d.total_size||0)/1024) + ' KB';
      document.getElementById('dsStatus').textContent = d.status || '-';
      document.getElementById('dsLast').textContent = fmtTime(d.last_trained_at);
      document.getElementById('dsDocs').textContent = d.last_added_docs || 0;

      let html = '';
      if(d.files && d.files.length){
        html += '<table><thead><tr><th>File</th><th>Kind</th><th>Size</th></tr></thead><tbody>';
        for(const f of d.files){
          html += '<tr><td><b>'+f.name+'</b></td><td>'+(f.kind||'-')+'</td><td>'+Math.round((f.size_bytes||0)/1024)+' KB</td></tr>';
        }
        html += '</tbody></table>';
      } else { html = 'No files.'; }
      document.getElementById('filesBox').innerHTML = html;

      const errBox = document.getElementById('errBox');
      errBox.textContent = d.last_error || '';

      const rb = document.getElementById('resultsBox');
      if(d.last_results && d.last_results.length){
        let h = '<table><thead><tr><th>File</th><th>Kind</th><th>Units</th><th>Docs added</th></tr></thead><tbody>';
        for(const r of d.last_results){
          h += '<tr><td>'+r.file+'</td><td>'+r.kind+'</td><td>'+(r.units||0)+'</td><td>'+(r.docs_added||0)+'</td></tr>';
        }
        h += '</tbody></table>';
        rb.innerHTML = h;
      } else { rb.textContent = 'No run yet.'; }

      document.getElementById('runBtn').disabled = (d.status === 'running');
    }

    document.getElementById('runBtn').onclick = async () => {
      const btn = document.getElementById('runBtn');
      const msg = document.getElementById('runMsg');
      btn.disabled = true; msg.textContent = 'Training… please wait.';
      try {
        const res = await fetch('/api/training/datasets/' + DATASET_ID + '/start', {method:'POST'});
        const data = await res.json();
        if(!res.ok) throw new Error(data.error || ('HTTP ' + res.status));
        msg.innerHTML = data.ok
          ? '<span style="color:#a7f3d0">✓ Training complete. ' + (data.docs_added||0) + ' doc(s) added to knowledge memory.</span>'
          : '<span style="color:#fca5a5">Training finished with errors.</span>';
        if(data.dataset) renderDataset(data.dataset);
      } catch(e){
        msg.innerHTML = '<span style="color:#fca5a5">Error: ' + e.message + '</span>';
        btn.disabled = false;
      }
    };

    loadDataset();
    setInterval(loadDataset, 4000);
  </script>
</body>
</html>
"""


TRAINING_RUN_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Training Run — {{ link_name }}</title>
  <style>
    :root { --bg:#0f172a; --panel:#1e293b; --line:#334155; --text:#e2e8f0; --muted:#94a3b8; --accent:#10b981; --warn:#ef4444; --info:#38bdf8; --amber:#f59e0b; }
    *{box-sizing:border-box}
    body{margin:0;background:var(--bg);color:var(--text);font-family:Segoe UI,system-ui,sans-serif}
    .wrap{max-width:1100px;margin:0 auto;padding:16px}
    .head{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;gap:10px}
    .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px;margin-bottom:12px}
    .k{font-size:12px;color:var(--muted)}
    .grid4{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}
    .grid2{display:grid;grid-template-columns:1fr 1fr;gap:10px}
    .stat{background:#0b1220;border:1px solid var(--line);border-radius:10px;padding:10px}
    .stat .v{font-size:20px;font-weight:800}
    .stat .l{font-size:11px;color:var(--muted)}
    button{border:1px solid var(--line);background:#0b1220;color:var(--text);border-radius:10px;padding:9px 12px;cursor:pointer;font-size:13px}
    button.primary{background:var(--accent);border-color:var(--accent);color:#072012;font-weight:700}
    button.danger{background:#3b0d0d;border-color:#7f1d1d;color:#fecaca}
    button:disabled{opacity:.5;cursor:not-allowed}
    .bar{position:relative;height:18px;background:#0b1220;border:1px solid var(--line);border-radius:999px;overflow:hidden}
    .bar > i{display:block;height:100%;background:linear-gradient(90deg,#10b981,#38bdf8);transition:width .4s ease}
    .bar > span{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;color:#e5e7eb}
    .stages{display:flex;gap:6px;flex-wrap:wrap;margin-top:6px}
    .stage{padding:4px 10px;border-radius:999px;border:1px solid var(--line);font-size:11px;color:var(--muted)}
    .stage.active{background:#1e3a8a;color:#dbeafe;border-color:#1e3a8a}
    .stage.done{background:#064e3b;color:#a7f3d0;border-color:#064e3b}
    .stage.fail{background:#7f1d1d;color:#fecaca;border-color:#7f1d1d}
    table{width:100%;border-collapse:collapse}
    th,td{padding:8px;border-bottom:1px solid var(--line);font-size:12px;text-align:left}
    th{color:var(--muted)}
    .badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:700}
    .b-saved{background:#1e3a8a;color:#dbeafe}
    .b-running{background:#7c2d12;color:#fed7aa}
    .b-completed{background:#064e3b;color:#a7f3d0}
    .b-failed{background:#7f1d1d;color:#fecaca}
    .b-stopped{background:#374151;color:#e5e7eb}
    .logs{max-height:280px;overflow:auto;background:#0b1220;border:1px solid var(--line);border-radius:10px;padding:10px;font-family:Consolas,monospace;font-size:12px;line-height:1.5}
    .log-line{white-space:pre-wrap}
    .err{color:#fca5a5}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="head">
      <div>
        <h2 style="margin:0">Training Run — <span id="hName">{{ link_name }}</span></h2>
        <div class="k">Connection ID: <code>{{ link_id }}</code></div>
      </div>
      <div>
        <button onclick="window.opener && window.opener.focus(); window.close();">Close</button>
      </div>
    </div>

    <div class="card">
      <div class="grid2">
        <div>
          <div class="k">Connection</div>
          <div id="connInfo" style="margin-top:6px"></div>
        </div>
        <div>
          <div class="k">Current Status</div>
          <div style="margin-top:6px"><span id="statusBadge" class="badge b-saved">saved</span></div>
          <div class="stages" id="stages">
            <span class="stage" data-s="connecting">1. Connect</span>
            <span class="stage" data-s="listing">2. List tables</span>
            <span class="stage" data-s="ingesting">3. Ingest</span>
            <span class="stage" data-s="profiling">4. Profile</span>
            <span class="stage" data-s="analyzing">5. Relate</span>
            <span class="stage" data-s="summarizing">6. Summarize</span>
            <span class="stage" data-s="saving">7. Save</span>
            <span class="stage" data-s="done">8. Done</span>
          </div>
        </div>
      </div>
      <div style="margin-top:12px">
        <button class="primary" id="startBtn">▶ Start Training</button>
        <button class="danger" id="stopBtn" disabled>■ Stop</button>
        <span class="k" id="msg" style="margin-left:8px"></span>
      </div>
    </div>

    <div class="card">
      <div class="grid4">
        <div class="stat"><div class="l">Progress</div><div class="v"><span id="pct">0</span>%</div></div>
        <div class="stat"><div class="l">Tables done</div><div class="v"><span id="tDone">0</span>/<span id="tTotal">0</span></div></div>
        <div class="stat"><div class="l">Docs added</div><div class="v" id="docsAdded">0</div></div>
        <div class="stat"><div class="l">Rows sampled</div><div class="v" id="rowsIngested">0</div></div>
        <div class="stat"><div class="l">Rows deeply analysed</div><div class="v" id="rowsProfiled">0</div></div>
        <div class="stat"><div class="l">Elapsed</div><div class="v" id="elapsed">0s</div></div>
        <div class="stat"><div class="l">Remaining (ETA)</div><div class="v" id="eta">-</div></div>
        <div class="stat"><div class="l">Current table</div><div class="v" id="curTable" style="font-size:14px">-</div></div>
        <div class="stat"><div class="l">Stage</div><div class="v" id="stageName" style="font-size:14px">-</div></div>
      </div>
      <div style="margin-top:12px" class="bar"><i id="barFill" style="width:0%"></i><span id="barLabel">0%</span></div>
    </div>

    <div class="card">
      <h3 style="margin-top:0">What the assistant learned</h3>
      <div class="k">A per-table breakdown including semantic role, deep column profiles, detected patterns and memorised samples.</div>
      <div id="learnings" style="margin-top:10px"></div>
    </div>

    <div class="card">
      <h3 style="margin-top:0">Inferred relationships & knowledge structure</h3>
      <div class="k">Cross-table reasoning: junction/mapping tables and logical relations the assistant inferred from column names and content (independent of declared foreign keys), plus the global knowledge summary.</div>
      <div id="analysis" style="margin-top:10px" class="k">Will appear after the analysis stage completes.</div>
    </div>

    <div class="card">
      <h3 style="margin-top:0">How it learned & model integration</h3>
      <div id="integration" class="k">Waiting for training to finish...</div>
    </div>

    <div class="card">
      <h3 style="margin-top:0">Live Logs</h3>
      <div class="logs" id="logs"></div>
      <div id="errBox" class="err" style="margin-top:8px;display:none"></div>
    </div>
  </div>

  <script>
    const LINK_ID = "{{ link_id }}";
    const STAGE_ORDER = ["connecting","listing-tables","ingesting-table","profiling-table","analyzing-relations","semantic-summary","saving","done"];
    const STAGE_LABELS = {
      "connecting":"connecting", "listing":"listing", "listing-tables":"listing",
      "ingesting":"ingesting", "ingesting-table":"ingesting",
      "profiling":"profiling", "profiling-table":"profiling",
      "analyzing":"analyzing", "analyzing-relations":"analyzing",
      "summarizing":"summarizing", "semantic-summary":"summarizing",
      "saving":"saving", "done":"done", "starting":"starting",
      "completed":"done", "failed":"failed", "stopped":"stopped"
    };

    async function api(url, options){
      const res = await fetch(url, options || {});
      const txt = await res.text();
      let data = {};
      try { data = txt ? JSON.parse(txt) : {}; } catch(e){ data = {raw:txt}; }
      if(!res.ok) throw new Error(data.error || ('HTTP ' + res.status));
      return data;
    }

    function fmtMs(ms){
      if(ms == null) return '-';
      const s = Math.max(0, ms/1000);
      if(s < 60) return s.toFixed(1)+'s';
      const m = Math.floor(s/60), r = Math.floor(s%60);
      if(m < 60) return m+'m '+r+'s';
      const h = Math.floor(m/60), rm = m%60;
      return h+'h '+rm+'m';
    }

    function setBadge(s){
      const map = {saved:'b-saved', starting:'b-running', connecting:'b-running', listing:'b-running', ingesting:'b-running', profiling:'b-running', analyzing:'b-running', summarizing:'b-running', saving:'b-running', running:'b-running', completed:'b-completed', failed:'b-failed', stopped:'b-stopped'};
      const el = document.getElementById('statusBadge');
      el.className = 'badge ' + (map[s] || 'b-saved');
      el.textContent = s;
    }

    function setStages(currentStatus){
      const order = ["connecting","listing","ingesting","profiling","analyzing","summarizing","saving","done"];
      const idx = order.indexOf(currentStatus);
      document.querySelectorAll('#stages .stage').forEach(el => {
        el.classList.remove('active','done','fail');
      });
      if(currentStatus === 'failed' || currentStatus === 'stopped'){
        document.querySelectorAll('#stages .stage').forEach((el,i) => {
          if(i < Math.max(0, idx)) el.classList.add('done');
        });
        return;
      }
      if(currentStatus === 'completed'){
        document.querySelectorAll('#stages .stage').forEach(el => el.classList.add('done'));
        return;
      }
      document.querySelectorAll('#stages .stage').forEach((el,i) => {
        if(i < idx) el.classList.add('done');
        else if(i === idx) el.classList.add('active');
      });
    }

    function renderConn(link){
      const fields = link.fields || {};
      const parts = Object.entries(fields).map(([k,v]) => k+': <b>'+(v ?? '')+'</b>');
      document.getElementById('connInfo').innerHTML =
        '<div><b>'+link.name+'</b> <span class="k">('+link.type+')</span></div>' +
        '<div class="k" style="margin-top:4px">'+ (link.url_masked || '') +'</div>' +
        '<div class="k" style="margin-top:4px">'+ parts.join(' • ') +'</div>' +
        '<div class="k" style="margin-top:4px">Tables filter: '+ (link.tables || '(all)') +' • Sample rows: '+ link.sample_rows +'</div>';
    }

    function renderLogs(logs){
      const box = document.getElementById('logs');
      box.innerHTML = '';
      for(const l of (logs || [])){
        const t = new Date(l.t * 1000).toLocaleTimeString();
        const div = document.createElement('div');
        div.className = 'log-line';
        div.textContent = '[' + t + '] ' + l.msg;
        box.appendChild(div);
      }
      box.scrollTop = box.scrollHeight;
    }

    function esc(s){
      return String(s == null ? '' : s)
        .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
        .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
    }

    function renderLearnings(items){
      const box = document.getElementById('learnings');
      if(!items || !items.length){
        box.innerHTML = '<div class="k">No tables ingested yet.</div>';
        return;
      }
      let html = '';
      for(const it of items){
        const profiles = it.column_profiles || [];
        const profRows = profiles.map(p => {
          const tops = (p.top_values||[]).map(t => esc(t.value)+'×'+t.count).join(', ');
          const ns = p.numeric_summary;
          const ts = p.text_summary;
          const pat = p.patterns_detected && Object.keys(p.patterns_detected).length
            ? Object.entries(p.patterns_detected).map(([k,v])=>k+':'+v).join(', ')
            : '-';
          const extra = (ns ? ('min='+ns.min+', max='+ns.max+', avg='+(ns.avg).toFixed(2)) : '')
                       + (ts ? (' • avg_len='+ts.avg_len.toFixed(1)) : '');
          return '<tr>' +
            '<td>'+esc(p.column)+'</td>' +
            '<td class="mono">'+esc(p.declared_type)+'<br><span class="k">→ '+esc(p.inferred_kind)+'</span></td>' +
            '<td>'+(p.distinct_count||0)+'</td>' +
            '<td>'+(p.null_pct||0).toFixed(1)+'%</td>' +
            '<td>'+(p.uniqueness_pct||0).toFixed(1)+'%</td>' +
            '<td class="mono" style="font-size:11px">'+(tops||'-')+'</td>' +
            '<td class="mono" style="font-size:11px">'+esc(pat)+'</td>' +
            '<td class="mono" style="font-size:11px">'+esc(extra||'-')+'</td>' +
          '</tr>';
        }).join('');
        const cols = (it.columns || []).map(c =>
          '<tr><td>'+esc(c.name)+'</td><td class="mono">'+esc(c.type)+'</td>' +
          '<td>'+(c.primary_key?'PK':'')+'</td>' +
          '<td>'+(c.nullable?'nullable':'NOT NULL')+'</td></tr>'
        ).join('');
        const fk = (it.foreign_keys || []).map(f =>
          '('+(f.columns||[]).join(',')+') → '+esc(f.referred_table)+'('+(f.referred_columns||[]).join(',')+')'
        ).join('; ') || '(none)';
        const sample = (it.sample_preview || []).map(r =>
          '<div class="mono" style="font-size:11px;color:#cbd5e1">' + esc(JSON.stringify(r)) + '</div>'
        ).join('') || '<div class="k">(no sample rows)</div>';
        const patternsTbl = it.patterns_detected && Object.keys(it.patterns_detected).length
          ? Object.entries(it.patterns_detected).map(([k,v])=>esc(k)+': '+v).join(' • ')
          : '(none)';
        html +=
          '<div style="border:1px solid var(--line);border-radius:10px;padding:10px;margin-bottom:10px">' +
            '<div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:6px">' +
              '<div><b>📋 ' + esc(it.table) + '</b> ' +
                '<span class="k">• role: <b style="color:#a7f3d0">' + esc(it.semantic_role || '?') + '</b> • ' +
                (it.columns||[]).length + ' cols • ' +
                'rows in source: ' + (it.row_count_in_source==null?'?':it.row_count_in_source) + ' • ' +
                'deeply analysed: ' + (it.rows_profiled||0) + ' • ' +
                'memorised: ' + (it.rows_sampled||0) + '</span></div>' +
              '<div class="k">doc: <code>' + esc(it.doc_id || '-') + '</code> • ' +
                (it.doc_chars||0) + ' chars</div>' +
            '</div>' +
            '<div class="k" style="margin-top:6px"><b>Understood:</b> ' + esc(it.understood) + '</div>' +
            '<div class="k" style="margin-top:6px"><b>Patterns detected across all rows:</b> ' + patternsTbl + '</div>' +
            '<div class="k" style="margin-top:6px"><b>Foreign keys (declared):</b> ' + fk + '</div>' +
            (profRows ? (
              '<div style="margin-top:8px"><div class="k">Deep column profiles:</div>' +
              '<table style="margin-top:4px"><thead><tr>' +
                '<th>Column</th><th>Type</th><th>Distinct</th><th>Null %</th><th>Unique %</th>' +
                '<th>Top values</th><th>Patterns</th><th>Numeric / text</th>' +
              '</tr></thead><tbody>' + profRows + '</tbody></table></div>'
            ) : '') +
            '<div style="margin-top:8px"><div class="k">Schema columns:</div>' +
              '<table style="margin-top:4px"><thead><tr><th>Name</th><th>Type</th><th>Key</th><th>Null</th></tr></thead>' +
              '<tbody>' + cols + '</tbody></table></div>' +
            '<div style="margin-top:8px"><div class="k">Sample rows memorised (preview):</div>' + sample + '</div>' +
            '<div class="k" style="margin-top:6px">Stored in: <code>' + esc(it.stored_in || '-') + '</code></div>' +
          '</div>';
      }
      box.innerHTML = html;
    }

    function renderAnalysis(an){
      const box = document.getElementById('analysis');
      if(!an){
        box.innerHTML = '<div class="k">Will appear after the analysis stage completes.</div>';
        return;
      }
      const rels = (an.inferred_relations || []).map(r =>
        '<tr><td>'+esc(r.from_table)+'.'+esc(r.from_column)+'</td>' +
        '<td>→</td>' +
        '<td>'+esc(r.to_table)+'.'+esc(r.to_column)+'</td>' +
        '<td>'+esc(r.confidence)+'</td>' +
        '<td class="k">'+esc(r.evidence)+'</td></tr>'
      ).join('');
      const jt = (an.junction_tables || []).map(j =>
        '<li><b>'+esc(j.table)+'</b> — links '+esc((j.links||[]).join(', '))+'<div class="k">'+esc(j.explanation)+'</div></li>'
      ).join('');
      box.innerHTML =
        '<div class="grid2">' +
          '<div>' +
            '<div><b>Inferred logical relationships:</b> ' + (an.inferred_relations||[]).length + '</div>' +
            (rels ?
              '<table style="margin-top:6px"><thead><tr><th>From</th><th></th><th>To</th><th>Conf.</th><th>Evidence</th></tr></thead><tbody>'+rels+'</tbody></table>'
              : '<div class="k" style="margin-top:6px">No additional relations were inferred beyond declared FKs.</div>') +
          '</div>' +
          '<div>' +
            '<div><b>Junction / mapping tables:</b> ' + (an.junction_tables||[]).length + '</div>' +
            (jt ? '<ul style="margin:6px 0 0 18px;padding:0">'+jt+'</ul>'
                : '<div class="k" style="margin-top:6px">No mapping/junction tables detected.</div>') +
          '</div>' +
        '</div>' +
        '<div style="margin-top:10px"><b>Knowledge structure summary:</b>' +
          '<pre class="mono" style="white-space:pre-wrap;background:#0b1220;border:1px solid var(--line);border-radius:8px;padding:10px;margin-top:6px;font-size:12px">'+esc(an.knowledge_summary||'')+'</pre>' +
        '</div>';
    }

    function renderIntegration(integration, job, link){
      const box = document.getElementById('integration');
      const docs = job ? (job.added_docs || 0) : 0;
      const status = job ? job.status : (link && link.status) || 'saved';
      if(!integration){
        if(status === 'completed'){
          box.innerHTML = '<div>Knowledge was saved but integration details are unavailable.</div>';
        } else if(status === 'failed' || status === 'stopped'){
          box.innerHTML = '<div class="err">Training did not finish — nothing was added to the assistant.</div>';
        } else {
          box.innerHTML = '<div>Training in progress... so far <b>'+docs+'</b> doc(s) added to the knowledge memory.</div>';
        }
        return;
      }
      box.innerHTML =
        '<div class="grid2">' +
          '<div>' +
            '<div><b>Method:</b> ' + esc(integration.method) + '</div>' +
            '<div style="margin-top:6px"><b>Depth level:</b><br><span class="k">' + esc(integration.depth_level || '-') + '</span></div>' +
            '<div style="margin-top:6px"><b>Model weights changed:</b> ' +
              (integration.model_weights_changed ? '<span style="color:#fca5a5">YES</span>'
                                                 : '<span style="color:#a7f3d0">NO — base model file is untouched</span>') + '</div>' +
            '<div style="margin-top:6px"><b>Base model file:</b> <code>' + esc(integration.model_file || '-') + '</code></div>' +
            '<div style="margin-top:6px"><b>Knowledge base file:</b> <code>' + esc(integration.knowledge_base_path || '-') + '</code></div>' +
          '</div>' +
          '<div>' +
            '<div><b>Tables analysed:</b> ' + (integration.tables_analysed || 0) + '</div>' +
            '<div><b>Rows deeply analysed:</b> ' + (integration.rows_profiled_this_run || 0) + '</div>' +
            '<div><b>Inferred relations:</b> ' + (integration.inferred_relations_count || 0) +
              ' • <b>Junction tables:</b> ' + (integration.junction_tables_count || 0) + '</div>' +
            '<div><b>Docs added this run:</b> ' + (integration.docs_added_this_run || 0) + '</div>' +
            '<div><b>Total docs in knowledge base:</b> ' + (integration.total_docs_in_kb || 0) + '</div>' +
            '<div style="margin-top:6px"><b>What it understood:</b><br><span class="k">' + esc(integration.what_it_understood) + '</span></div>' +
            '<div style="margin-top:6px"><b>How it is used at chat time:</b><br><span class="k">' + esc(integration.how_it_is_used) + '</span></div>' +
          '</div>' +
        '</div>';
    }

    let pollTimer = null;
    let running = false;

    async function poll(){
      try {
        const r = await api('/api/training/db/links/' + LINK_ID + '/progress');
        const link = r.link || {};
        const job = r.job;
        renderConn(link);
        if(job){
          setBadge(job.status);
          setStages(job.stage in {connecting:1,listing:1,ingesting:1,saving:1,done:1}
                    ? job.stage
                    : (STAGE_LABELS[job.stage] || job.status));
          document.getElementById('pct').textContent = job.percent || 0;
          document.getElementById('tDone').textContent = job.tables_done || 0;
          document.getElementById('tTotal').textContent = job.tables_total || 0;
          document.getElementById('docsAdded').textContent = job.added_docs || 0;
          document.getElementById('rowsIngested').textContent = job.rows_ingested || 0;
          document.getElementById('rowsProfiled').textContent = job.rows_profiled || 0;
          document.getElementById('elapsed').textContent = fmtMs(job.elapsed_ms);
          document.getElementById('eta').textContent = job.eta_ms == null ? '-' : fmtMs(job.eta_ms);
          document.getElementById('curTable').textContent = job.current_table || '-';
          document.getElementById('stageName').textContent = job.stage || '-';
          document.getElementById('barFill').style.width = (job.percent || 0) + '%';
          document.getElementById('barLabel').textContent = (job.percent || 0) + '%';
          renderLogs(job.logs);
          renderLearnings(job.learnings);
          renderAnalysis(job.analysis);
          renderIntegration(job.model_integration, job, link);
          const err = document.getElementById('errBox');
          if(job.error){ err.style.display='block'; err.textContent = 'Error: ' + job.error; }
          else { err.style.display='none'; err.textContent=''; }
          const active = ["starting","connecting","listing","ingesting","profiling","analyzing","summarizing","saving"].indexOf(job.status) >= 0;
          running = active;
          document.getElementById('startBtn').disabled = active;
          document.getElementById('stopBtn').disabled  = !active;
          if(!active && pollTimer){ clearInterval(pollTimer); pollTimer = null; }
        } else {
          setBadge(link.status || 'saved');
          setStages(link.status === 'completed' ? 'completed' : '');
          renderLearnings([]);
          renderAnalysis(null);
          renderIntegration(null, null, link);
          document.getElementById('startBtn').disabled = false;
          document.getElementById('stopBtn').disabled  = true;
        }
      } catch(e){
        document.getElementById('msg').innerHTML = '<span class="err">'+ e.message +'</span>';
      }
    }

    document.addEventListener('DOMContentLoaded', () => {
      document.getElementById('startBtn').onclick = async () => {
        const m = document.getElementById('msg');
        m.textContent = 'Verifying connection...';
        try {
          // Test connection first using saved server-side URL
          const t = await api('/api/training/db/links/' + LINK_ID + '/test', {method:'POST'});
          m.innerHTML = '<span style="color:#a7f3d0">Connection verified ('+ (t.url_masked || '') +'). Starting training...</span>';
          await api('/api/training/db/links/' + LINK_ID + '/start', {method:'POST'});
          m.textContent = 'Training started.';
          if(!pollTimer) pollTimer = setInterval(poll, 800);
          poll();
        } catch(e){
          m.innerHTML = '<span class="err">Failed: ' + e.message + '</span>';
        }
      };
      document.getElementById('stopBtn').onclick = async () => {
        try { await api('/api/training/db/links/' + LINK_ID + '/stop', {method:'POST'}); }
        catch(e){ alert(e.message); }
      };
      poll();
      pollTimer = setInterval(poll, 1500);
    });
  </script>
</body>
</html>
"""


@app.get("/api/monitoring")
def monitoring_api():
    return jsonify(_monitor_snapshot())


@app.get("/api/chats")
def list_chats_api():
    with _chat_lock:
        return jsonify({"chats": _list_chats()})


@app.post("/api/chats")
def create_chat_api():
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "").strip() or None
    with _chat_lock:
        chat = _create_chat(title)
    return jsonify({"chat": _chat_summary(chat)})


@app.get("/api/chats/<chat_id>")
def get_chat_api(chat_id: str):
    with _chat_lock:
        chat = _read_chat(chat_id)
    if not chat:
        return jsonify({"error": "chat not found"}), 404
    return jsonify({"chat": chat})


@app.post("/api/chats/<chat_id>/clear")
def clear_chat_api(chat_id: str):
    with _chat_lock:
        chat = _read_chat(chat_id)
        if not chat:
            return jsonify({"error": "chat not found"}), 404
        chat["messages"] = []
        chat["title"] = _default_title()
        chat["updated_at"] = _now_ts()
        _write_chat(chat)
    return jsonify({"chat": _chat_summary(chat)})


@app.delete("/api/chats/<chat_id>")
def delete_chat_api(chat_id: str):
    with _chat_lock:
        chat = _read_chat(chat_id)
        if not chat:
            return jsonify({"error": "chat not found"}), 404
        _delete_chat(chat_id)
    return jsonify({"ok": True})


@app.get("/api/docs")
def api_docs():
    docs_html = """
    <html>
    <head>
      <title>Phi-3 API Documentation</title>
      <style>
        body { font-family: Segoe UI, sans-serif; max-width: 980px; margin: 24px auto; padding: 0 12px; }
        code, pre { background: #f2f2f2; padding: 2px 6px; border-radius: 5px; }
        pre { padding: 12px; overflow: auto; }
      </style>
    </head>
    <body>
      <h1>Phi-3 API Documentation</h1>
      <h2>Chat Memory Endpoints</h2>
      <ul>
        <li><code>GET /api/chats</code> - list all chat memories.</li>
        <li><code>POST /api/chats</code> - create a new chat memory.</li>
        <li><code>GET /api/chats/&lt;chat_id&gt;</code> - get full chat memory.</li>
        <li><code>POST /api/chats/&lt;chat_id&gt;/clear</code> - clear chat messages but keep chat.</li>
        <li><code>DELETE /api/chats/&lt;chat_id&gt;</code> - delete chat and its memory.</li>
      </ul>
      <h2>Generation Endpoints</h2>
      <ul>
        <li><code>POST /api/chat</code> - append user message and generate assistant reply using that chat memory.</li>
        <li><code>GET /api/config</code> - read generation settings.</li>
        <li><code>POST /api/config</code> - update generation settings.</li>
      </ul>
      <h2>Example: send message to a chat</h2>
      <pre>{
  "chat_id": "abc123def456",
  "message": "Continue from where we stopped yesterday"
}</pre>
      <p><a href="/">Back to chat UI</a></p>
    </body>
    </html>
    """
    return docs_html


@app.get("/api/docs/json")
def api_docs_json():
    schema = {
        "openapi": "3.0.0",
        "info": {"title": "Phi-3 Local Chat API", "version": "2.0.0"},
        "paths": {
            "/api/chats": {
                "get": {"summary": "List chats"},
                "post": {"summary": "Create chat"},
            },
            "/api/chats/{chat_id}": {
                "get": {"summary": "Get chat"},
                "delete": {"summary": "Delete chat"},
            },
            "/api/chats/{chat_id}/clear": {
                "post": {"summary": "Clear chat messages"},
            },
            "/api/chat": {
                "post": {
                    "summary": "Generate assistant response in chat memory context",
                }
            },
            "/api/knowledge/retrieve": {
              "get": {
                "summary": "Debug knowledge retrieval with scoring trace",
              }
            },
            "/api/eval/cases": {
              "get": {"summary": "List evaluation benchmark cases"},
              "post": {"summary": "Replace evaluation benchmark cases"},
            },
            "/api/eval/run": {
              "post": {"summary": "Run evaluation benchmark"},
            },
            "/api/chat/feedback": {
              "post": {"summary": "Store user feedback for continuous learning"},
            },
            "/api/learning/backlog": {
              "get": {"summary": "List learning backlog items"},
            },
            "/api/learning/retrain/suggest": {
              "get": {"summary": "Suggest datasets to retrain from backlog"},
            },
            "/api/ops/daily_summary": {
              "get": {"summary": "Operational summary for last 24h"},
            },
            "/api/config": {
                "get": {"summary": "Read generation config"},
                "post": {"summary": "Update generation config"},
            },
        },
    }
    return jsonify(schema)


@app.get("/api/config")
def get_config():
    with _config_lock:
        cfg = deepcopy(RUNTIME_CONFIG)
    return jsonify(cfg)


@app.get("/api/knowledge/retrieve")
def knowledge_retrieve_debug_api():
    query = str(request.args.get("q") or "").strip()
    source = str(request.args.get("source") or "").strip().lower() or None
    try:
      limit = int(request.args.get("limit") or 8)
    except Exception:
      limit = 8
    limit = max(1, min(30, limit))

    if not query:
      return jsonify({"error": "q query parameter is required"}), 400

    docs, trace = _retrieve_knowledge_with_trace(query, limit=limit, source_filter=source)
    return jsonify({
      "query": query,
      "source_filter": source,
      "results": [
        {
          "id": d.get("id"),
          "title": d.get("title"),
          "source": d.get("source"),
          "meta": d.get("meta", {}),
        }
        for d in docs
      ],
      "trace": trace,
    })


@app.get("/api/eval/cases")
def eval_cases_get_api():
    return jsonify({"cases": _load_eval_cases()})


@app.post("/api/eval/cases")
def eval_cases_set_api():
    data = request.get_json(silent=True) or {}
    cases = data.get("cases")
    if not isinstance(cases, list):
      return jsonify({"error": "cases must be an array"}), 400

    normalized = []
    for i, c in enumerate(cases):
      if not isinstance(c, dict):
        continue
      question = str(c.get("question") or "").strip()
      if not question:
        continue
      normalized.append({
        "id": str(c.get("id") or f"case_{i+1}"),
        "question": question,
        "expect_keywords": [str(x).strip().lower() for x in (c.get("expect_keywords") or []) if str(x).strip()],
        "knowledge_mode": str(c.get("knowledge_mode") or "all").strip().lower() or "all",
      })
    _save_eval_cases(normalized)
    return jsonify({"ok": True, "count": len(normalized)})


@app.post("/api/eval/run")
def eval_run_api():
    data = request.get_json(silent=True) or {}
    limit = int(data.get("limit") or 0)
    response_mode = str(data.get("response_mode") or "quick").strip().lower()
    if response_mode not in {"quick", "analytical"}:
      response_mode = "quick"

    cases = _load_eval_cases()
    if not cases:
      return jsonify({"error": "No eval cases configured. Use /api/eval/cases first."}), 400
    if limit > 0:
      cases = cases[:limit]

    rows = []
    pass_count = 0
    for case in cases:
      question = case.get("question", "")
      knowledge_mode = case.get("knowledge_mode", "all")
      if knowledge_mode == "adtc_only":
        docs, _ = _retrieve_knowledge_with_trace(question, limit=5, source_filter="adtc")
      else:
        docs, _ = _retrieve_knowledge_with_trace(question, limit=5)

      reasoning = _build_reasoning_bundle(question, docs)
      grounded = _answer_from_knowledge(question, docs)
      if grounded:
        reply = grounded
      else:
        reply, _ = model_reply(
          question,
          history=[],
          knowledge_docs=docs,
          force_synthesis=True,
          reasoning_bundle=reasoning,
          response_mode=response_mode,
        )

      expect_keywords = case.get("expect_keywords") or []
      low_reply = (reply or "").lower()
      hit_keywords = [k for k in expect_keywords if k and k in low_reply]
      keyword_score = int(round((len(hit_keywords) / max(1, len(expect_keywords))) * 100)) if expect_keywords else 100
      confidence = _build_reply_confidence(_response_quality_metrics(question, reply, docs), reasoning, docs)
      passed = keyword_score >= 60 and confidence.get("score", 0) >= 45
      if passed:
        pass_count += 1

      rows.append({
        "id": case.get("id"),
        "question": question,
        "knowledge_mode": knowledge_mode,
        "keyword_score": keyword_score,
        "confidence_score": confidence.get("score", 0),
        "passed": passed,
        "matched_keywords": hit_keywords,
        "expect_keywords": expect_keywords,
      })

    total = len(rows)
    summary = {
      "total_cases": total,
      "passed": pass_count,
      "failed": total - pass_count,
      "pass_rate": round((pass_count / max(1, total)) * 100, 2),
      "avg_keyword_score": round(sum(r.get("keyword_score", 0) for r in rows) / max(1, total), 2),
      "avg_confidence_score": round(sum(r.get("confidence_score", 0) for r in rows) / max(1, total), 2),
    }
    _append_audit_event("eval.run", summary)
    return jsonify({"summary": summary, "results": rows})


@app.post("/api/chat/feedback")
def chat_feedback_api():
    data = request.get_json(silent=True) or {}
    chat_id = str(data.get("chat_id") or "").strip()
    rating = data.get("rating")
    note = str(data.get("note") or "").strip()
    request_id = str(data.get("request_id") or "").strip()
    category = str(data.get("category") or "general").strip().lower() or "general"

    try:
      rating_val = int(rating)
    except Exception:
      return jsonify({"error": "rating must be an integer 1..5"}), 400
    if rating_val < 1 or rating_val > 5:
      return jsonify({"error": "rating must be in range 1..5"}), 400

    event = {
      "ts": _now_ts(),
      "chat_id": chat_id,
      "request_id": request_id,
      "rating": rating_val,
      "note": note,
      "category": category,
    }
    _append_jsonl(FEEDBACK_LOG_PATH, event)
    _append_audit_event("chat.feedback", {
      "chat_id": chat_id,
      "request_id": request_id,
      "rating": rating_val,
      "category": category,
    })
    if rating_val <= 2:
      _append_learning_item("negative_feedback", event)
    return jsonify({"ok": True})


@app.get("/api/learning/backlog")
def learning_backlog_api():
    limit = int(request.args.get("limit") or 100)
    limit = max(1, min(500, limit))
    items = _read_jsonl(LEARNING_BACKLOG_PATH, max_lines=limit)
    return jsonify({"count": len(items), "items": items})


@app.get("/api/learning/retrain/suggest")
def learning_retrain_suggest_api():
    items = _read_jsonl(LEARNING_BACKLOG_PATH, max_lines=1000)
    dataset_counts = {}
    for it in items:
      payload = it.get("payload") if isinstance(it.get("payload"), dict) else {}
      dataset_id = str(payload.get("dataset_id") or "").strip()
      if not dataset_id:
        continue
      dataset_counts[dataset_id] = dataset_counts.get(dataset_id, 0) + 1
    ranked = sorted(dataset_counts.items(), key=lambda x: x[1], reverse=True)
    return jsonify({
      "suggestions": [
        {"dataset_id": k, "issue_count": v}
        for k, v in ranked[:20]
      ]
    })


@app.get("/api/ops/daily_summary")
def ops_daily_summary_api():
    now = _now_ts()
    since = now - 86400

    audit = [x for x in _read_jsonl(AUDIT_LOG_PATH, max_lines=5000) if int(x.get("ts") or 0) >= since]
    feedback = [x for x in _read_jsonl(FEEDBACK_LOG_PATH, max_lines=2000) if int(x.get("ts") or 0) >= since]
    backlog = [x for x in _read_jsonl(LEARNING_BACKLOG_PATH, max_lines=2000) if int(x.get("ts") or 0) >= since]

    event_counts = {}
    for e in audit:
      k = str(e.get("event") or "unknown")
      event_counts[k] = event_counts.get(k, 0) + 1

    low_ratings = [f for f in feedback if int(f.get("rating") or 0) <= 2]
    avg_rating = round(sum(int(f.get("rating") or 0) for f in feedback) / max(1, len(feedback)), 2) if feedback else None

    return jsonify({
      "window_hours": 24,
      "audit_events": len(audit),
      "event_counts": event_counts,
      "feedback_total": len(feedback),
      "feedback_avg_rating": avg_rating,
      "feedback_low_ratings": len(low_ratings),
      "learning_backlog_new": len(backlog),
      "server_time": now,
    })


@app.post("/api/config")
def set_config():
    data = request.get_json(silent=True) or {}

    try:
        raw_max_tokens = data.get("max_tokens", RUNTIME_CONFIG["max_tokens"])
        if raw_max_tokens in (None, "", 0, "0"):
            max_tokens = None
        else:
            max_tokens = int(raw_max_tokens)

        temperature = float(data.get("temperature", RUNTIME_CONFIG["temperature"]))
        top_p = float(data.get("top_p", RUNTIME_CONFIG["top_p"]))
        stop = data.get("stop", RUNTIME_CONFIG["stop"])

        if not isinstance(stop, list) or not all(isinstance(x, str) for x in stop):
            raise ValueError("stop must be an array of strings")
        if max_tokens is not None and (max_tokens < 1 or max_tokens > 4096):
            raise ValueError("max_tokens must be null (unlimited) or in range 1..4096")
        if temperature < 0 or temperature > 2:
            raise ValueError("temperature must be in range 0..2")
        if top_p <= 0 or top_p > 1:
            raise ValueError("top_p must be in range (0..1]")
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400

    with _config_lock:
        RUNTIME_CONFIG["max_tokens"] = max_tokens
        RUNTIME_CONFIG["temperature"] = temperature
        RUNTIME_CONFIG["top_p"] = top_p
        RUNTIME_CONFIG["stop"] = stop
        cfg = deepcopy(RUNTIME_CONFIG)

    return jsonify(cfg)


@app.post("/api/chat")
def chat_api():
    data = request.get_json(silent=True) or {}
    chat_id = str(data.get("chat_id") or "").strip()
    message = str(data.get("message") or "").strip()
    knowledge_mode = str(data.get("knowledge_mode") or "all").strip().lower()
    response_mode = str(data.get("response_mode") or "analytical").strip().lower()

    if knowledge_mode not in {"all", "adtc_only"}:
      knowledge_mode = "all"
    if response_mode not in {"analytical", "quick"}:
      response_mode = "analytical"

    if not chat_id:
        return jsonify({"error": "chat_id is required"}), 400
    if not message:
        return jsonify({"error": "message is required"}), 400

    request_id = uuid.uuid4().hex[:10]
    _append_audit_event("chat.request", {
      "request_id": request_id,
      "chat_id": chat_id,
      "knowledge_mode": knowledge_mode,
      "response_mode": response_mode,
      "message_len": len(message),
    })
    t0 = time.perf_counter()
    _monitor_start(request_id, chat_id)

    _monitor_stage("loading-memory")
    load_start = time.perf_counter()
    with _chat_lock:
        chat = _read_chat(chat_id)
        if not chat:
            _monitor_finish(
                {
                    "request_id": request_id,
                    "chat_id": chat_id,
                    "success": False,
                    "error": "chat not found",
                    "last_stage": "loading-memory",
                    "total_ms": (time.perf_counter() - t0) * 1000.0,
                    "infer_ms": 0.0,
                    "output_tokens": 0,
                    "tokens_per_sec": 0.0,
                    "at": int(time.time()),
                },
                is_error=True,
            )
            _append_audit_event("chat.error", {"request_id": request_id, "chat_id": chat_id, "error": "chat not found"})
            return jsonify({"error": "chat not found"}), 404

        history = chat.get("messages", [])

    _monitor_stage("retrieving-knowledge")
    know_start = time.perf_counter()
    if knowledge_mode == "adtc_only":
      knowledge_docs, retrieval_trace = _retrieve_knowledge_with_trace(message, limit=5, source_filter="adtc")
    else:
      knowledge_docs, retrieval_trace = _retrieve_knowledge_with_trace(message, limit=5)
    reasoning_bundle = _build_reasoning_bundle(message, knowledge_docs)
    know_ms = (time.perf_counter() - know_start) * 1000.0

    load_ms = (time.perf_counter() - load_start) * 1000.0

    # Fast-path for real-time date/time questions (disabled in ADTC-only mode).
    if knowledge_mode != "adtc_only" and _is_time_question(message):
      infer_start = time.perf_counter()
      reply = _build_time_answer(message)
      usage = {"completion_tokens": _estimate_tokens(reply)}
      infer_ms = (time.perf_counter() - infer_start) * 1000.0

      _monitor_stage("saving-memory")
      save_start = time.perf_counter()
      with _chat_lock:
        chat = _read_chat(chat_id)
        if not chat:
          _monitor_finish(
            {
              "request_id": request_id,
              "chat_id": chat_id,
              "success": False,
              "error": "chat not found during save",
              "last_stage": "saving-memory",
              "total_ms": (time.perf_counter() - t0) * 1000.0,
              "infer_ms": infer_ms,
              "output_tokens": 0,
              "tokens_per_sec": 0.0,
              "at": int(time.time()),
            },
            is_error=True,
          )
          return jsonify({"error": "chat not found"}), 404

        chat["messages"].append({"role": "user", "content": message})
        chat["messages"].append({"role": "assistant", "content": reply})
        if chat.get("title") == _default_title():
          chat["title"] = _derive_title(chat["messages"])
        chat["updated_at"] = _now_ts()
        _write_chat(chat)

      save_ms = (time.perf_counter() - save_start) * 1000.0
      total_ms = (time.perf_counter() - t0) * 1000.0
      out_tokens = int(usage.get("completion_tokens") or _estimate_tokens(reply))
      tps = out_tokens / (infer_ms / 1000.0) if infer_ms > 0 else 0.0

      _monitor_finish(
        {
          "request_id": request_id,
          "chat_id": chat_id,
          "success": True,
          "last_stage": "done",
          "total_ms": round(total_ms, 2),
          "infer_ms": round(infer_ms, 2),
          "load_ms": round(load_ms, 2),
          "knowledge_ms": round(know_ms, 2),
          "knowledge_hits": len(knowledge_docs),
          "save_ms": round(save_ms, 2),
          "output_tokens": out_tokens,
          "tokens_per_sec": round(tps, 2),
          "at": int(time.time()),
        }
      )
      quality = _response_quality_metrics(message, reply, knowledge_docs=knowledge_docs)
      confidence = _build_reply_confidence(quality, reasoning_bundle, knowledge_docs)
      sources = _build_sources_payload(knowledge_docs, limit=5)
      if reasoning_bundle.get("missing_evidence") or int(reasoning_bundle.get("conflict_count") or 0) > 0 or confidence.get("score", 0) < 50:
        dataset_hint = None
        if knowledge_docs:
          meta0 = knowledge_docs[0].get("meta") if isinstance(knowledge_docs[0].get("meta"), dict) else {}
          dataset_hint = meta0.get("dataset_id")
        _append_learning_item("chat_improvement_needed", {
          "request_id": request_id,
          "chat_id": chat_id,
          "question": message,
          "confidence": confidence.get("score", 0),
          "missing_evidence": bool(reasoning_bundle.get("missing_evidence")),
          "conflicts": reasoning_bundle.get("conflicts", []),
          "knowledge_mode": knowledge_mode,
          "dataset_id": dataset_hint,
        })
      _append_audit_event("chat.reply", {
        "request_id": request_id,
        "chat_id": chat_id,
        "quality_score": quality.get("score", 0),
        "confidence_score": confidence.get("score", 0),
        "knowledge_hits": len(knowledge_docs),
        "reasoning_conflicts": int(reasoning_bundle.get("conflict_count") or 0),
      })
      return jsonify({
        "request_id": request_id,
        "chat_id": chat_id,
        "title": chat["title"],
        "reply": reply,
        "quality": quality,
        "confidence": confidence,
        "sources": sources,
        "retrieval_trace": retrieval_trace,
        "reasoning": reasoning_bundle,
      })

    try:
      infer_start = time.perf_counter()
      grounded_answer = _answer_from_knowledge(message, knowledge_docs)
      if knowledge_mode == "adtc_only":
        if knowledge_docs:
          # ADTC-only now performs real model reasoning using only ADTC retrieved docs.
          _monitor_stage("inference")
          # Prevent repeating stale assistant phrasing/snippet dumps from prior turns.
          adtc_history = [m for m in history if str(m.get("role")) == "user"][-8:]
          reply, usage = model_reply(
            message,
            adtc_history,
            knowledge_docs=knowledge_docs,
            force_synthesis=True,
            reasoning_bundle=reasoning_bundle,
            response_mode=response_mode,
          )
          if _looks_like_refusal(reply):
            # Force a clean synthesis pass instead of dumping raw snippets verbatim.
            synth = _synthesize_from_snippets(message, knowledge_docs, response_mode=response_mode)
            if synth:
              _monitor_stage("grounded-answer")
              reply = synth
              usage = {"completion_tokens": _estimate_tokens(reply)}
            else:
              grounded = _answer_from_knowledge(message, knowledge_docs)
              if grounded:
                _monitor_stage("grounded-answer")
                reply = grounded
                usage = {"completion_tokens": _estimate_tokens(reply)}
        else:
          _monitor_stage("grounded-answer")
          reply = (
            "From trained ADTC data:\n"
            "I could not find a matching answer in ADTC-trained documents for this query. "
            "Please rephrase using terms that exist in your ADTC files."
          )
          usage = {"completion_tokens": _estimate_tokens(reply)}
      elif grounded_answer is not None:
        _monitor_stage("grounded-answer")
        reply = grounded_answer
        usage = {"completion_tokens": _estimate_tokens(reply)}
      else:
        _monitor_stage("inference")
        reply, usage = model_reply(
          message,
          history,
          knowledge_docs=knowledge_docs,
          reasoning_bundle=reasoning_bundle,
          response_mode=response_mode,
        )
      infer_ms = (time.perf_counter() - infer_start) * 1000.0
    except Exception as exc:
        _monitor_finish(
            {
                "request_id": request_id,
                "chat_id": chat_id,
                "success": False,
                "error": str(exc),
                "last_stage": "inference",
                "total_ms": (time.perf_counter() - t0) * 1000.0,
                "infer_ms": 0.0,
                "output_tokens": 0,
                "tokens_per_sec": 0.0,
                "at": int(time.time()),
            },
            is_error=True,
        )
        _append_audit_event("chat.error", {"request_id": request_id, "chat_id": chat_id, "error": str(exc)})
        return jsonify({"error": str(exc)}), 500

    _monitor_stage("saving-memory")
    save_start = time.perf_counter()
    with _chat_lock:
        chat = _read_chat(chat_id)
        if not chat:
            _monitor_finish(
                {
                    "request_id": request_id,
                    "chat_id": chat_id,
                    "success": False,
                    "error": "chat not found during save",
                    "last_stage": "saving-memory",
                    "total_ms": (time.perf_counter() - t0) * 1000.0,
                    "infer_ms": infer_ms,
                    "output_tokens": 0,
                    "tokens_per_sec": 0.0,
                    "at": int(time.time()),
                },
                is_error=True,
            )
            _append_audit_event("chat.error", {"request_id": request_id, "chat_id": chat_id, "error": "chat not found during save"})
            return jsonify({"error": "chat not found"}), 404

        chat["messages"].append({"role": "user", "content": message})
        chat["messages"].append({"role": "assistant", "content": reply})

        if chat.get("title") == _default_title():
            chat["title"] = _derive_title(chat["messages"])

        chat["updated_at"] = _now_ts()
        _write_chat(chat)

    save_ms = (time.perf_counter() - save_start) * 1000.0
    total_ms = (time.perf_counter() - t0) * 1000.0
    out_tokens = int(usage.get("completion_tokens") or _estimate_tokens(reply))
    tps = out_tokens / (infer_ms / 1000.0) if infer_ms > 0 else 0.0

    _monitor_finish(
        {
            "request_id": request_id,
            "chat_id": chat_id,
            "success": True,
            "last_stage": "done",
            "total_ms": round(total_ms, 2),
            "infer_ms": round(infer_ms, 2),
            "load_ms": round(load_ms, 2),
            "knowledge_ms": round(know_ms, 2),
            "knowledge_hits": len(knowledge_docs),
            "save_ms": round(save_ms, 2),
            "output_tokens": out_tokens,
            "tokens_per_sec": round(tps, 2),
            "at": int(time.time()),
        }
    )
    quality = _response_quality_metrics(message, reply, knowledge_docs=knowledge_docs)
    confidence = _build_reply_confidence(quality, reasoning_bundle, knowledge_docs)
    sources = _build_sources_payload(knowledge_docs, limit=5)

    if reasoning_bundle.get("missing_evidence") or int(reasoning_bundle.get("conflict_count") or 0) > 0 or confidence.get("score", 0) < 50:
      dataset_hint = None
      if knowledge_docs:
        meta0 = knowledge_docs[0].get("meta") if isinstance(knowledge_docs[0].get("meta"), dict) else {}
        dataset_hint = meta0.get("dataset_id")
      _append_learning_item("chat_improvement_needed", {
        "request_id": request_id,
        "chat_id": chat_id,
        "question": message,
        "confidence": confidence.get("score", 0),
        "missing_evidence": bool(reasoning_bundle.get("missing_evidence")),
        "conflicts": reasoning_bundle.get("conflicts", []),
        "knowledge_mode": knowledge_mode,
        "dataset_id": dataset_hint,
      })

    _append_audit_event("chat.reply", {
      "request_id": request_id,
      "chat_id": chat_id,
      "quality_score": quality.get("score", 0),
      "confidence_score": confidence.get("score", 0),
      "knowledge_hits": len(knowledge_docs),
      "reasoning_conflicts": int(reasoning_bundle.get("conflict_count") or 0),
    })

    return jsonify({
      "request_id": request_id,
      "chat_id": chat_id,
      "title": chat["title"],
      "reply": reply,
      "quality": quality,
      "confidence": confidence,
      "sources": sources,
      "retrieval_trace": retrieval_trace,
      "reasoning": reasoning_bundle,
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
