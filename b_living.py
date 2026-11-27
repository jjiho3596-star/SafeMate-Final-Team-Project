"""B기능 생활지원 상담과 승인 후 모의 실행."""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


_ROOT = Path(__file__).resolve().parent
LIVING_DATA_DIR = _ROOT / "data" / "living"
LIVING_CHROMA_DIR = _ROOT / ".safemate" / "living_chroma"
DEFAULT_ACTION_LOG = _ROOT / ".safemate" / "mock_action_log.jsonl"

_KST = timezone(timedelta(hours=9))
_LOG_LOCK = threading.Lock()
_ALLOWED_CATEGORIES = {"groceries", "daily_necessities", "other"}
_ALLOWED_URGENCY = {"normal", "soon"}
_PERSONAL_DATA_PATTERN = re.compile(
    r"(?:\b\d{2,3}[- ]?\d{3,4}[- ]?\d{4}\b|"
    r"\b\d{6}[- ]?[1-4]\d{6}\b|"
    r"(?:시|도|구|군|동|읍|면|로|길)\s*\d)"
)
_TOKEN_PATTERN = re.compile(r"[가-힣A-Za-z0-9]+")


def _now_iso() -> str:
    return datetime.now(_KST).isoformat(timespec="seconds")


def _base_result(
    *,
    status: str,
    detail_code: str | None,
    message: str,
    route: str,
    answer: str,
    conversation_state: dict,
    missing_fields: list[str] | None = None,
    clarification: dict | None = None,
    proposed_action: dict | None = None,
    usage: dict | None = None,
    tool_trace: list[dict] | None = None,
    limitations: list[str] | None = None,
    living_evidence: list[dict] | None = None,
) -> dict:
    return {
        "status": status,
        "detail_code": detail_code,
        "message": message,
        "route": route,
        "answer": answer or "죄송해요. 지금은 생활지원 답변을 준비하지 못했어요.",
        "conversation_state": conversation_state,
        "missing_fields": missing_fields or [],
        "clarification": clarification,
        "requires_user_confirmation": proposed_action is not None,
        "proposed_action": proposed_action,
        "usage": usage or {"llm_calls": 0},
        "tool_trace": tool_trace or [],
        "limitations": limitations or [],
        "living_evidence": living_evidence or [],
    }


def _error_result(
    status: str,
    detail_code: str,
    message: str,
    conversation_state: dict,
    *,
    tool_trace: list[dict] | None = None,
    living_evidence: list[dict] | None = None,
    llm_calls: int = 0,
) -> dict:
    return _base_result(
        status=status,
        detail_code=detail_code,
        message=message,
        route="error",
        answer="죄송해요. 지금은 생활지원 답변을 준비하지 못했어요.",
        conversation_state=conversation_state,
        usage={"llm_calls": llm_calls},
        tool_trace=tool_trace,
        living_evidence=living_evidence,
    )


def _request_parts(request: dict | str) -> tuple[str, dict | None]:
    if isinstance(request, str):
        user_message = request.strip()
        context = None
    elif isinstance(request, dict):
        user_message = request.get("user_message")
        context = request.get("context")
        if not isinstance(user_message, str):
            raise ValueError("user_message_must_be_string")
        if context is not None and not isinstance(context, dict):
            raise ValueError("context_must_be_dict")
        if context is not None:
            try:
                json.dumps(context, ensure_ascii=False, allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise ValueError("context_must_be_json_serializable") from exc
        user_message = user_message.strip()
    else:
        raise ValueError("request_must_be_string_or_dict")
    if not user_message:
        raise ValueError("empty_user_message")
    if len(user_message) > 600:
        raise ValueError("user_message_too_long")
    return user_message, context


def _local_ollama_config() -> tuple[str, str]:
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    parsed = urlparse(base_url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("ollama_must_be_local")
    model = (
        os.environ.get("B_OLLAMA_MODEL")
        or os.environ.get("OLLAMA_MODEL")
        or "qwen2.5:3b"
    )
    return base_url, model


def _ollama_chat(system_prompt: str, user_prompt: str) -> str:
    base_url, model = _local_ollama_config()
    body = json.dumps(
        {
            "model": model,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = Request(
        base_url + "/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    content = payload.get("message", {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("ollama_empty_response")
    return content


def _document_values(document: Any) -> dict | None:
    if isinstance(document, dict):
        content = document.get("content", document.get("page_content"))
        metadata = document.get("metadata", document)
    else:
        content = getattr(document, "page_content", None)
        metadata = getattr(document, "metadata", {})
    if not isinstance(content, str) or not content.strip():
        return None
    metadata = metadata if isinstance(metadata, dict) else {}
    page = metadata.get("pdf_page")
    if not isinstance(page, int) or isinstance(page, bool) or page < 1:
        page = None
    return {
        "content": content,
        "title": _optional_string(metadata.get("title", metadata.get("source_name"))),
        "publisher": _optional_string(metadata.get("publisher")),
        "url": _optional_string(metadata.get("url")),
        "pdf_page": page,
    }


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _load_pdf_documents(path: Path) -> list[dict]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("living_pdf_support_requires_pypdf") from exc

    reader = PdfReader(str(path))
    metadata = reader.metadata or {}
    title = _optional_string(metadata.get("/Title")) or path.stem
    publisher = _optional_string(metadata.get("/Publisher"))
    url = _optional_string(metadata.get("/URL"))
    documents = []
    for page_number, page in enumerate(reader.pages, start=1):
        content = page.extract_text()
        if not isinstance(content, str) or not content.strip():
            continue
        documents.append(
            {
                "content": content,
                "title": title,
                "publisher": publisher,
                "url": url,
                "pdf_page": page_number,
            }
        )
    return documents


def _load_local_documents() -> list[dict]:
    documents: list[dict] = []
    if not LIVING_DATA_DIR.exists():
        return documents
    for path in sorted(LIVING_DATA_DIR.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in {".txt", ".md", ".json", ".pdf"}:
            continue
        if suffix == ".pdf":
            documents.extend(_load_pdf_documents(path))
            continue
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise OSError(f"living_document_read_failed:{path.name}") from exc
        if suffix == ".json":
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"living_document_invalid_json:{path.name}") from exc
            entries = parsed if isinstance(parsed, list) else [parsed]
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                normalized = _document_values(entry)
                if normalized:
                    documents.append(normalized)
        elif raw.strip():
            documents.append(
                {
                    "content": raw,
                    "title": path.stem,
                    "publisher": None,
                    "url": None,
                    "pdf_page": None,
                }
            )
    return documents


def _search_living_evidence(query: str, index: Any) -> list[dict]:
    """Search the B-only index, or local living documents when no index is supplied.

    A persistent index passed here should be prepared from LIVING_DATA_DIR and stored
    under LIVING_CHROMA_DIR by the application at startup.
    """
    if index is None:
        documents = _load_local_documents()
        tokens = set(_TOKEN_PATTERN.findall(query.lower()))
        ranked = []
        for document in documents:
            content_tokens = set(_TOKEN_PATTERN.findall(document["content"].lower()))
            score = len(tokens & content_tokens)
            if score:
                ranked.append((score, document))
        ranked.sort(key=lambda result: result[0], reverse=True)
        return [document for _, document in ranked[:4]]

    if callable(getattr(index, "similarity_search", None)):
        results = index.similarity_search(query, k=4)
    elif callable(getattr(index, "search", None)):
        results = index.search(query)
    else:
        raise TypeError("index_must_support_similarity_search_or_search")
    if not isinstance(results, (list, tuple)):
        raise TypeError("living_search_result_must_be_a_list")
    evidence = []
    for result in results:
        normalized = _document_values(result)
        if normalized:
            evidence.append(normalized)
    return evidence[:4]


def _clean_state(state: dict | None) -> dict:
    if state is None:
        return {"turn": 0, "pending_category": None, "pending_items": []}
    if not isinstance(state, dict):
        raise ValueError("conversation_state_must_be_dict")
    turn = state.get("turn", 0)
    if not isinstance(turn, int) or isinstance(turn, bool) or turn < 0:
        raise ValueError("conversation_state_turn_invalid")
    pending_category = state.get("pending_category")
    if pending_category not in _ALLOWED_CATEGORIES:
        pending_category = None
    pending_items = state.get("pending_items", [])
    if not isinstance(pending_items, list):
        pending_items = []
    safe_items = []
    for item in pending_items[:20]:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        quantity = item.get("quantity")
        unit = item.get("unit", "")
        if (
            isinstance(name, str)
            and 1 <= len(name.strip()) <= 40
            and (quantity is None or (isinstance(quantity, int) and not isinstance(quantity, bool) and 1 <= quantity <= 99))
            and isinstance(unit, str)
            and len(unit) <= 10
            and not _PERSONAL_DATA_PATTERN.search(unit)
            and not _PERSONAL_DATA_PATTERN.search(name)
        ):
            safe_items.append({"name": name.strip(), "quantity": quantity, "unit": unit})
    return {"turn": turn, "pending_category": pending_category, "pending_items": safe_items}


def _normalize_items(value: Any) -> list[dict]:
    if not isinstance(value, list):
        raise ValueError("items_must_be_list")
    if len(value) > 20:
        raise ValueError("too_many_items")
    items = []
    for raw_item in value:
        if not isinstance(raw_item, dict):
            raise ValueError("item_must_be_object")
        if set(raw_item) - {"name", "quantity", "unit"}:
            raise ValueError("item_contains_unsupported_fields")
        name = raw_item.get("name")
        quantity = raw_item.get("quantity")
        unit = raw_item.get("unit", "")
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 40:
            raise ValueError("item_name_invalid")
        if _PERSONAL_DATA_PATTERN.search(name):
            raise ValueError("personal_data_not_allowed")
        if quantity is not None and (
            not isinstance(quantity, int)
            or isinstance(quantity, bool)
            or not 1 <= quantity <= 99
        ):
            raise ValueError("item_quantity_invalid")
        if not isinstance(unit, str) or len(unit) > 10:
            raise ValueError("item_unit_invalid")
        if _PERSONAL_DATA_PATTERN.search(unit):
            raise ValueError("personal_data_not_allowed")
        items.append({"name": name.strip(), "quantity": quantity, "unit": unit})
    return items


def _merge_items(previous: list[dict], current: list[dict]) -> list[dict]:
    if not current:
        return previous
    merged = [dict(item) for item in previous]
    for item in current:
        match = next(
            (existing for existing in merged if existing["name"] == item["name"]),
            None,
        )
        if match is None:
            merged.append(dict(item))
        elif item["quantity"] is not None:
            match.update(item)
    return merged[:20]


def build_help_message(
    category: str,
    items: list[dict],
    note: str = "",
) -> str:
    if not isinstance(category, str) or category not in _ALLOWED_CATEGORIES:
        raise ValueError("category_invalid")
    normalized_items = _normalize_items(items)
    if not normalized_items:
        raise ValueError("items_invalid")
    if any(item["quantity"] is None for item in normalized_items):
        raise ValueError("item_quantity_missing")
    if not isinstance(note, str) or len(note) > 200:
        raise ValueError("note_invalid")
    if _PERSONAL_DATA_PATTERN.search(note):
        raise ValueError("personal_data_not_allowed")

    need = {
        "groceries": "장보기",
        "daily_necessities": "생필품 구매",
        "other": "생활",
    }[category]
    listed = ", ".join(
        f"{item['name']} {item['quantity']}{item['unit']}"
        for item in normalized_items
    )
    message = (
        f"어르신께서 {need}에 어려움을 겪고 계십니다. "
        f"필요한 물품은 {listed}입니다. 구매에 도움을 부탁드립니다."
    )
    if note:
        message += f" 참고: {note}"
    if len(message) > 300:
        raise ValueError("message_too_long")
    return message


def _parse_model_output(raw: str) -> dict:
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("model_output_not_json") from exc
    if not isinstance(result, dict):
        raise ValueError("model_output_not_object")
    return result


def run_living_support(
    request: dict | str,
    *,
    device: str | None = None,
    index: Any = None,
    conversation_state: dict | None = None,
) -> dict:
    """생활지원 정보를 상담하고, 필요한 경우 보호자 도움을 승인 대기 상태로 제안한다."""
    del device  # 호출 호환성을 위한 자리이며 B 색인 장치는 앱 시작 시 설정한다.
    try:
        state = _clean_state(conversation_state)
    except ValueError as exc:
        return _error_result(
            "tool_error",
            "conversation_state",
            str(exc),
            {},
        )
    previous_state = dict(state)
    try:
        user_message, context = _request_parts(request)
    except ValueError as exc:
        return _base_result(
            status="needs_input",
            detail_code=str(exc),
            message=str(exc),
            route="needs_clarification",
            answer="어떤 생활지원이 필요하신지 말씀해 주세요.",
            conversation_state=previous_state,
            missing_fields=["request"],
            clarification={
                "fields": ["request"],
                "question": "어떤 생활지원이 필요하신지 말씀해 주세요.",
                "reasons": {"request": "요청 내용을 확인할 수 없습니다."},
            },
        )

    search_terms = [user_message]
    if context:
        search_terms.extend(
            str(value) for value in context.values() if isinstance(value, (str, int, float))
        )
    if state.get("pending_category"):
        search_terms.append(state["pending_category"])
    search_terms.extend(item["name"] for item in state["pending_items"])
    try:
        evidence = _search_living_evidence(" ".join(search_terms), index)
    except Exception as exc:
        return _error_result(
            "tool_error",
            "rag_search",
            str(exc),
            previous_state,
            tool_trace=[{"tool": "living_search", "status": "error"}],
        )

    system_prompt = (
        "너는 SafeMate B 생활지원 상담의 정보 추출 및 답변 도우미다. "
        "반드시 JSON 객체로만 답하고, 아래 키를 사용한다: "
        "intent, category, items, urgency, answer. "
        "intent는 request_caregiver_help 또는 living_support 중 하나다. "
        "conversation_state에 pending_items가 있으면 현재 발화가 그 정보의 보충인지 먼저 확인한다. "
        "물품·수량 보충이면 intent는 request_caregiver_help로 유지한다. "
        "items는 [{name, quantity, unit}] 형식이며 모르는 수량은 null로 둔다. "
        "공식 지원제도·기관·조건에 관한 주장은 제공된 living_evidence에 있는 내용만 말한다. "
        "근거가 없으면 공식 자료에서 확인하지 못했다고 말한다. "
        "요청에 없는 물품·수량·사정은 만들지 않는다. 이름, 전화번호, 주소, 보호자 연락처를 "
        "묻거나 출력하지 않는다. 보호자는 역할명 caregiver로만 표현한다. "
        "승인·실행이 끝난 것처럼 말하지 않는다."
    )
    user_prompt = json.dumps(
        {
            "user_message": user_message,
            "context": context,
            "conversation_state": state,
            "living_evidence": evidence,
        },
        ensure_ascii=False,
    )
    trace = [{"tool": "living_search", "status": "ok", "result_count": len(evidence)}]
    try:
        raw_output = _ollama_chat(system_prompt, user_prompt)
    except URLError as exc:
        return _error_result(
            "llm_unavailable",
            "connection_refused",
            str(exc),
            previous_state,
            tool_trace=trace + [{"tool": "ollama", "status": "unavailable"}],
            living_evidence=evidence,
            llm_calls=1,
        )
    except TimeoutError as exc:
        return _error_result(
            "llm_unavailable",
            "timeout",
            str(exc),
            previous_state,
            tool_trace=trace + [{"tool": "ollama", "status": "unavailable"}],
            living_evidence=evidence,
            llm_calls=1,
        )
    except ValueError as exc:
        return _error_result(
            "tool_error",
            "ollama_configuration",
            str(exc),
            previous_state,
            tool_trace=trace + [{"tool": "ollama", "status": "error"}],
            living_evidence=evidence,
            llm_calls=1,
        )
    except Exception as exc:
        return _error_result(
            "llm_unavailable",
            "ollama_request",
            str(exc),
            previous_state,
            tool_trace=trace + [{"tool": "ollama", "status": "unavailable"}],
            living_evidence=evidence,
            llm_calls=1,
        )

    trace.append({"tool": "ollama", "status": "ok"})
    try:
        parsed = _parse_model_output(raw_output)
        intent = parsed.get("intent")
        if intent not in {"request_caregiver_help", "living_support"}:
            raise ValueError("model_intent_invalid")
        category = parsed.get("category")
        if category not in _ALLOWED_CATEGORIES:
            category = state.get("pending_category") or "other"
        items = _normalize_items(parsed.get("items", []))
        item_source = user_message + " " + " ".join(
            str(value) for value in (context or {}).values() if isinstance(value, str)
        )
        item_source = item_source + " " + " ".join(
            item["name"] for item in state["pending_items"]
        )
        if any(item["name"] not in item_source for item in items):
            raise ValueError("model_item_not_in_user_request")
        items = _merge_items(state["pending_items"], items)
        urgency = parsed.get("urgency", "normal")
        if urgency not in _ALLOWED_URGENCY:
            urgency = "normal"
        answer = parsed.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("model_answer_invalid")
        if _PERSONAL_DATA_PATTERN.search(answer):
            raise ValueError("personal_data_not_allowed")
    except ValueError as exc:
        return _error_result(
            "llm_invalid_output",
            "understanding",
            str(exc),
            previous_state,
            tool_trace=trace,
            living_evidence=evidence,
            llm_calls=1,
        )

    limitations = []
    if not evidence:
        limitations.append("공식 생활지원 자료에서 근거를 찾지 못했습니다.")

    if intent == "request_caregiver_help":
        missing_fields = []
        reasons = {}
        if not items:
            missing_fields.append("items")
            reasons["items"] = "도움을 요청할 물품을 확인하지 못했습니다."
        elif any(item["quantity"] is None for item in items):
            missing_fields.append("quantity")
            reasons["quantity"] = "각 물품의 수량을 확인해야 합니다."
        if missing_fields:
            question = (
                "도움을 요청할 물품을 말씀해 주세요."
                if missing_fields == ["items"]
                else "물품을 각각 얼마나 준비하면 될까요?"
            )
            new_state = {
                "turn": state["turn"] + 1,
                "pending_category": category,
                "pending_items": items,
            }
            return _base_result(
                status="needs_input",
                detail_code=(
                    "missing_items" if missing_fields == ["items"] else "missing_quantity"
                ),
                message="",
                route="needs_clarification",
                answer=question,
                conversation_state=new_state,
                missing_fields=missing_fields,
                clarification={
                    "fields": missing_fields,
                    "question": question,
                    "reasons": reasons,
                },
                usage={"llm_calls": 1},
                tool_trace=trace,
                limitations=limitations,
                living_evidence=evidence,
            )
        try:
            message = build_help_message(category, items)
        except ValueError as exc:
            detail_code = str(exc)
            error_message = (
                "물품 목록이 길어 승인 문장을 300자 이내로 만들 수 없습니다."
                if detail_code == "message_too_long"
                else "보호자 요청 문장을 안전하게 만들 수 없습니다."
            )
            return _error_result(
                "tool_error",
                detail_code,
                error_message,
                previous_state,
                tool_trace=trace,
                living_evidence=evidence,
                llm_calls=1,
            )
        proposed_action = {
            "type": "request_caregiver_help",
            "action_id": uuid.uuid4().hex,
            "summary": f"보호자께 {', '.join(item['name'] for item in items)} 도움을 요청합니다.",
            "requires_approval": True,
            "applied": False,
            "created_at": _now_iso(),
            "source": "living_support",
            "payload": {
                "recipient_role": "caregiver",
                "category": category,
                "items": items,
                "note": "",
                "urgency": urgency,
                "message": message,
            },
        }
        return _base_result(
            status="ok",
            detail_code=None,
            message="",
            route="living_support",
            answer="보호자께 전달할 요청을 준비했어요. 내용을 확인한 뒤 승인해 주세요.",
            conversation_state={
                "turn": state["turn"] + 1,
                "pending_category": None,
                "pending_items": [],
            },
            proposed_action=proposed_action,
            usage={"llm_calls": 1},
            tool_trace=trace,
            limitations=limitations,
            living_evidence=evidence,
        )

    return _base_result(
        status="ok",
        detail_code=None,
        message="",
        route="living_support",
        answer=answer.strip(),
        conversation_state={
            "turn": state["turn"] + 1,
            "pending_category": None,
            "pending_items": [],
        },
        usage={"llm_calls": 1},
        tool_trace=trace,
        limitations=limitations,
        living_evidence=evidence,
    )


def _validate_action(action: Any) -> tuple[bool, str]:
    if not isinstance(action, dict):
        return False, "action_must_be_dict"
    allowed_action_keys = {
        "type",
        "action_id",
        "summary",
        "requires_approval",
        "applied",
        "created_at",
        "source",
        "payload",
        "approval",
    }
    if set(action) - allowed_action_keys:
        return False, "action_contains_unsupported_fields"
    if action.get("type") != "request_caregiver_help":
        return False, "unsupported_action_type"
    if action.get("requires_approval") is not True or action.get("applied") is not False:
        return False, "action_flags_invalid"
    action_id = action.get("action_id")
    if not isinstance(action_id, str) or not action_id.strip():
        return False, "action_id_invalid"
    payload = action.get("payload")
    if not isinstance(payload, dict):
        return False, "payload_invalid"
    if set(payload) != {
        "recipient_role",
        "category",
        "items",
        "note",
        "urgency",
        "message",
    }:
        return False, "payload_fields_invalid"
    if payload.get("recipient_role") != "caregiver":
        return False, "recipient_role_invalid"
    category = payload.get("category")
    if not isinstance(category, str) or category not in _ALLOWED_CATEGORIES:
        return False, "category_invalid"
    items = payload.get("items")
    if not isinstance(items, list) or not 1 <= len(items) <= 20:
        return False, "items_invalid"
    try:
        normalized_items = _normalize_items(items)
    except ValueError as exc:
        return False, str(exc)
    if any(item["quantity"] is None for item in normalized_items):
        return False, "item_quantity_missing"
    note = payload.get("note")
    urgency = payload.get("urgency")
    message = payload.get("message")
    if not isinstance(note, str) or len(note) > 200:
        return False, "note_invalid"
    if _PERSONAL_DATA_PATTERN.search(note):
        return False, "personal_data_not_allowed"
    if not isinstance(urgency, str) or urgency not in _ALLOWED_URGENCY:
        return False, "urgency_invalid"
    if not isinstance(message, str) or not 1 <= len(message) <= 300:
        return False, "message_invalid"
    if _PERSONAL_DATA_PATTERN.search(message):
        return False, "personal_data_not_allowed"
    try:
        expected_message = build_help_message(category, normalized_items, note)
    except ValueError as exc:
        return False, str(exc)
    if message != expected_message:
        return False, "message_item_mismatch"
    if not isinstance(action.get("summary"), str) or not action["summary"].strip():
        return False, "summary_invalid"
    if not _is_kst_timestamp(action.get("created_at")):
        return False, "created_at_invalid"
    if action.get("source") != "living_support":
        return False, "source_invalid"
    return True, ""


def _is_kst_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.utcoffset() == timedelta(hours=9)


def _action_log_path() -> Path:
    configured = os.environ.get("SAFEMATE_ACTION_LOG")
    if configured:
        return Path(configured).expanduser()
    return DEFAULT_ACTION_LOG


@contextmanager
def _locked_log(path: Path) -> Iterator[None]:
    lock_path = path.with_name(path.name + ".lock")
    with _LOG_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as lock_file:
            if os.name == "nt":
                import msvcrt

                lock_file.seek(0, os.SEEK_END)
                if lock_file.tell() == 0:
                    lock_file.write(b"\0")
                    lock_file.flush()
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _find_existing_record(path: Path, action_id: str) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as log_file:
        for line_number, line in enumerate(log_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid_action_log_line:{line_number}") from exc
            if isinstance(record, dict) and record.get("action_id") == action_id:
                return record
    return None


def execute_living_action(proposed_action: dict) -> dict:
    """승인된 요청만 실제 전송 없이 JSONL 모의 기록으로 남긴다."""
    action_id = (
        proposed_action.get("action_id")
        if isinstance(proposed_action, dict)
        else None
    )
    valid, reason = _validate_action(proposed_action)
    if not valid:
        return {
            "status": "invalid_action",
            "message": reason,
            "action_id": action_id if isinstance(action_id, str) else None,
            "applied": False,
            "mock": True,
            "record": None,
            "answer": "요청 내용을 확인할 수 없어 기록하지 않았어요.",
        }
    approval = proposed_action.get("approval")
    if not isinstance(approval, dict) or approval.get("status") != "approved":
        return {
            "status": "rejected",
            "message": "사용자 승인이 확인되지 않았습니다.",
            "action_id": action_id,
            "applied": False,
            "mock": True,
            "record": None,
            "answer": "승인되지 않아 요청을 기록하지 않았어요.",
        }
    approved_at = approval.get("at")
    if not _is_kst_timestamp(approved_at):
        return {
            "status": "rejected",
            "message": "승인 시각이 없어 기록하지 않았습니다.",
            "action_id": action_id,
            "applied": False,
            "mock": True,
            "record": None,
            "answer": "승인 정보를 확인할 수 없어 요청을 기록하지 않았어요.",
        }

    try:
        path = _action_log_path()
        with _locked_log(path):
            existing = _find_existing_record(path, action_id)
            if existing is not None:
                return {
                    "status": "already_executed",
                    "message": "같은 action_id의 모의 기록이 이미 있습니다.",
                    "action_id": action_id,
                    "applied": True,
                    "mock": True,
                    "record": existing,
                    "answer": "이 요청은 이미 모의 실행 기록이 있어요.",
                }
            record = {
                "action_id": action_id,
                "type": proposed_action["type"],
                "payload": proposed_action["payload"],
                "approved_at": approved_at,
                "executed_at": _now_iso(),
                "mock": True,
            }
            with path.open("a", encoding="utf-8", newline="\n") as log_file:
                log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                log_file.flush()
                os.fsync(log_file.fileno())
    except (OSError, TypeError, ValueError) as exc:
        return {
            "status": "error",
            "message": str(exc),
            "action_id": action_id,
            "applied": False,
            "mock": True,
            "record": None,
            "answer": "모의 실행 기록을 저장하지 못했어요.",
        }
    item_summary = ", ".join(
        f"{item['name']} {item['quantity']}{item['unit']}"
        for item in proposed_action["payload"]["items"]
    )
    category_summary = {
        "groceries": "장보기",
        "daily_necessities": "생필품 구매",
        "other": "생활",
    }[proposed_action["payload"]["category"]]
    return {
        "status": "ok",
        "message": "승인된 요청을 모의 실행 기록에 저장했습니다.",
        "action_id": action_id,
        "applied": True,
        "mock": True,
        "record": record,
        "answer": (
            f"보호자께 {category_summary} 도움({item_summary})을 요청한 것으로 기록했어요. "
            "(모의 실행이라 실제 메시지는 보내지 않았어요)"
        ),
    }