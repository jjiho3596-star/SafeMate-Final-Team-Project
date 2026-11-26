"""SafeMate 5-5단계: CUDA·XPU·CPU 연산 장치 자동 선택과 실행 장치 보고

핵심 요약
- 2단계 RAG의 MiniLM 임베딩을 실제로 돌릴 장치를 CUDA → XPU → CPU 순서로 고른다. (auto)
  장치마다 'PyTorch 백엔드 → is_available() → 실제 텐서 연산 → MiniLM 로딩·임베딩 → 장치·384차원 확인'을 모두 통과해야 선택한다.
- cuda·xpu·cpu를 직접 지정하면 그 장치만 확인한다. 실패해도 다른 장치로 조용히 바꾸지 않는다.
- 검사는 2단계 함수(configure_embedding_device·load_embeddings·verify_embedding_device)를 그대로 재사용한다.
  선택한 장치의 모델은 load_embeddings의 캐시(lru_cache)에 남아, 이후 RAG 색인 준비가 같은 모델을 다시 쓴다.
- Ollama(LLM) 실행 장치는 PyTorch 장치와 다른 시스템이므로 따로 보고한다. (ollama_report)
  PyTorch가 XPU라고 Ollama도 Intel GPU를 쓴다고 판단하지 않는다. GPU 종류·추론 백엔드는 확정하지 않는다.

이 파일은 전역 환경변수(SAFEMATE_EMBED_DEVICE, CUDA_VISIBLE_DEVICES 등)를 읽어 바꾸지 않고, 일정 저장소를 만들거나 읽지 않는다.

실행 예 (.venv-xpu)
    & .\\.venv-xpu\\Scripts\\python.exe device_utils.py --device auto
    & .\\.venv-xpu\\Scripts\\python.exe device_utils.py --device cpu --json
    & .\\.venv-xpu\\Scripts\\python.exe device_utils.py --device auto --ollama-probe   # qwen2.5:3b 짧은 호출 뒤 다시 측정
종료 코드: 0 임베딩 장치 확인 성공, 1 임베딩 장치 확인 실패(지정 장치 실패·전체 실패·잘못된 장치), 2 임베딩은 성공했지만 Ollama 서버·모델 문제
"""

# %% 공통 라이브러리와 기존 단계 함수 불러오기
# 핵심 요약: 장치 검사는 2단계(safety_rag), Ollama 설정·상태·호출은 3단계(outing_decision) 함수를 그대로 쓴다.
from __future__ import annotations

import argparse
import gc
import http.client
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Any

import outing_decision as od
import safety_rag as sr


# %% 장치 선택 설정값
# 핵심 요약: 자동 선택 순서, 허용하는 장치 이름, 검사 단계 이름, 기존 MiniLM 출력 차원을 한곳에 모은다.
AUTO_ORDER = ("cuda", "xpu", "cpu")      # 승인 계획 11절: CUDA → XPU → CPU
REQUEST_MODES = ("auto",) + AUTO_ORDER
EMBED_DIMENSIONS = 384                   # 2단계 MiniLM(paraphrase-multilingual-MiniLM-L12-v2) 출력 차원
# verify_embedding_device()가 장치·차원 불일치일 때 내는 ValueError 메시지 머리말 (safety_rag.py는 보호 파일이라 바뀌지 않음)
VERIFY_PREFIXES = ("선택 장치", "임베딩 장치·차원 불일치")
STAGES = ("import_torch", "backend", "is_available", "tensor", "model", "embedding_verify", "report_check")
STAGE_WORDS = {"import_torch": "PyTorch 불러오기", "backend": "PyTorch 백엔드", "is_available": "is_available()",
               "tensor": "실제 텐서 연산", "model": "MiniLM 로딩·임베딩", "embedding_verify": "임베딩 장치·차원 확인",
               "report_check": "검증 결과 재확인"}
AMBIGUOUS_MODEL_NOTE = ("기존 load_embeddings()가 모델 로딩 실패와 임베딩 연산 실패를 같은 오류(model_error)로 감싸므로, "
                        "둘 중 어느 쪽인지 임의로 나누지 않고 원래 예외를 그대로 기록합니다.")
OLLAMA_CLI_TIMEOUT = 10                  # ollama ps 명령 대기 시간(초)


class DeviceSelectionError(Exception):
    """자동 선택·직접 지정에서 쓸 수 있는 장치가 없을 때. selection에 시도 기록이 그대로 들어 있다."""

    def __init__(self, selection: dict[str, Any]):
        super().__init__(selection.get("error") or "임베딩 장치를 선택하지 못했습니다.")
        self.selection = selection


# %% 검사 기록 도우미
# 핵심 요약: 장치·단계마다 성공·실패와 이유를 한 줄로 남긴다. 실패 이유는 원래 예외의 종류와 메시지를 그대로 쓴다.
def _attempt(device: str, stage: str, ok: bool, reason: str = "", **extra: Any) -> dict[str, Any]:
    return {"device": device, "stage": stage, "ok": ok, "reason": reason, **extra}


def _error_text(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"[:500]


def _import_torch() -> Any:
    """PyTorch 모듈. 설치되지 않았으면 ImportError (모든 장치가 같은 이유로 실패)"""
    import torch
    return torch


def _build_info(torch: Any, device: str) -> str | None:
    """빌드 정보(torch.version.cuda·xpu). 버전·빌드마다 속성이 없을 수 있으므로 기록에만 쓰고 판단 기준으로 쓰지 않는다."""
    value = getattr(getattr(torch, "version", None), device, None)
    return str(value) if value else None


# %% 실패한 장치의 메모리 정리
# 핵심 요약: 실패한 장치의 쓰지 않는 캐시 블록만 돌려준다. 다른 기능이 쓰는 모델·캐시된 모델은 해제하지 않는다.
# 정리하다 오류가 나도 원래 실패 이유를 덮지 않고 따로 기록한다.
def _release_failed(torch: Any, device: str) -> str | None:
    """정리 중 오류 메시지 또는 None"""
    try:
        gc.collect()
        backend = getattr(torch, device, None) if device != "cpu" else None
        empty = getattr(backend, "empty_cache", None)
        if callable(empty):
            empty()
    except Exception as error:  # noqa: BLE001 - 정리 실패는 기록만 한다
        return _error_text(error)
    return None


# %% PyTorch 단계 검사 (백엔드 → is_available → 실제 텐서 연산)
# 핵심 요약: 텐서 연산은 2단계 configure_embedding_device()를 그대로 쓴다. 빌드 정보가 없다고 장치를 빼지 않는다.
def _check_torch(torch: Any, device: str, attempts: list[dict[str, Any]]) -> str | None:
    """성공하면 실제 장치 이름('xpu:0', 'cpu'), 실패하면 None (실패 기록은 attempts에 남김)"""
    # 1. 백엔드: cpu는 항상 있다. cuda·xpu는 torch.cuda·torch.xpu 모듈이 있어야 한다.
    if device != "cpu":
        backend = getattr(torch, device, None)
        build = _build_info(torch, device)
        if backend is None:
            attempts.append(_attempt(device, "backend", False, f"이 PyTorch에 torch.{device} 백엔드가 없습니다.", build=build))
            return None
        attempts.append(_attempt(device, "backend", True, build=build))

        # 2. is_available(): 드라이버·장치를 PyTorch가 인식하는지 (예외도 실패 이유로 기록)
        try:
            available = backend.is_available()
        except Exception as error:  # noqa: BLE001
            attempts.append(_attempt(device, "is_available", False, _error_text(error)))
            return None
        if available is not True:
            attempts.append(_attempt(device, "is_available", False, f"torch.{device}.is_available()이 {available!r}입니다."))
            return None
        attempts.append(_attempt(device, "is_available", True))

    # 3. 실제 텐서 연산: 2단계 함수가 작은 텐서를 만들어 장치·결과를 확인한다.
    try:
        actual = sr.configure_embedding_device(device)
    except Exception as error:  # noqa: BLE001 - SafetyError(device_error) 포함
        attempts.append(_attempt(device, "tensor", False, _error_text(error)))
        return None
    attempts.append(_attempt(device, "tensor", True, actual=actual))
    return actual


# %% MiniLM 단계 검사 (모델 로딩·임베딩 → 장치·384차원)
# 핵심 요약: 2단계 load_embeddings()가 모델을 불러오고 verify_embedding_device()로 가중치·토큰·출력 장치와 차원을 확인한다.
# 돌려받은 검증 정보를 한 번 더 확인한다. 실패 원인을 나눌 수 없는 예외는 임의로 나누지 않는다.
def _check_model(device: str, actual: str, attempts: list[dict[str, Any]]) -> dict[str, Any] | None:
    """성공하면 검증 정보(details), 실패하면 None"""
    # 1. 모델 로딩·임베딩 (성공한 모델은 load_embeddings 캐시에 남아 RAG가 다시 쓴다)
    try:
        _, details = sr.load_embeddings(device)
    except Exception as error:  # noqa: BLE001
        cause = error.__cause__
        if isinstance(cause, ValueError) and str(cause).startswith(VERIFY_PREFIXES):
            attempts.append(_attempt(device, "embedding_verify", False, _error_text(cause)))
        else:
            attempts.append(_attempt(device, "model", False, _error_text(error),
                                     cause=_error_text(cause) if cause else None, note=AMBIGUOUS_MODEL_NOTE))
        return None
    attempts.append(_attempt(device, "model", True))
    attempts.append(_attempt(device, "embedding_verify", True))

    # 2. 검증 정보 재확인: 가중치·토큰 입력·토큰 출력·최종 벡터가 모두 텐서 연산 장치와 같고, 384차원이어야 한다.
    if not isinstance(details, dict):
        attempts.append(_attempt(device, "report_check", False, "검증 정보가 딕셔너리가 아닙니다."))
        return None
    places = {key: details.get(key) for key in ("weights", "token_input", "token_output", "embedding")}
    problems = [f"{key}={value}" for key, value in places.items() if value != actual]
    if details.get("dimensions") != EMBED_DIMENSIONS:
        problems.append(f"dimensions={details.get('dimensions')}")
    if details.get("requested") != device:
        problems.append(f"requested={details.get('requested')}")
    if problems:
        attempts.append(_attempt(device, "report_check", False, f"실제 장치 {actual}와 다름: " + ", ".join(problems)))
        return None
    attempts.append(_attempt(device, "report_check", True))
    return dict(details)


def _cached_models() -> int | None:
    """load_embeddings 캐시에 남은 모델 수 (장치를 여러 개 직접 검사하면 장치별로 쌓일 수 있음)"""
    info = getattr(sr.load_embeddings, "cache_info", None)
    size = getattr(info(), "currsize", None) if callable(info) else None
    return size if isinstance(size, int) else None


def _order(requested: Any) -> tuple[str, tuple[str, ...]] | None:
    if not isinstance(requested, str) or requested not in REQUEST_MODES:
        return None
    return ("auto", AUTO_ORDER) if requested == "auto" else ("manual", (requested,))


def _result(requested: Any, mode: str, attempts: list[dict[str, Any]], *, selected: str | None = None,
            device: str | None = None, details: dict[str, Any] | None = None, error: str | None = None,
            torch_version: str | None = None, check: str = "minilm") -> dict[str, Any]:
    switches = [f"{a['device']}: {STAGE_WORDS.get(a['stage'], a['stage'])} 실패 → 다음 장치" for a in attempts
                if not a["ok"] and mode == "auto" and a["device"] != selected]
    return {"requested": requested if isinstance(requested, str) else repr(requested), "mode": mode, "check": check,
            "ok": selected is not None, "selected": selected, "device": device, "attempts": attempts,
            "details": details, "switch_reasons": switches if selected else [], "error": error,
            "torch": torch_version, "cached_models": _cached_models() if check == "minilm" else None}


# %% 임베딩 장치 선택 함수 (승인 계획 11절 계약)
# 핵심 요약: 자동이면 CUDA → XPU → CPU 순서로 모든 검사를 통과한 첫 장치를 고르고, 직접 지정이면 그 장치만 검사한다.
def select_embedding_device(requested: str = "auto") -> dict[str, Any]:
    """{"selected", "mode", "attempts", "details", "ok", "device", "switch_reasons", "error", "torch", "cached_models"}

    - selected: 'cuda'·'xpu'·'cpu' (3단계·그래프가 쓰는 장치 이름) 또는 None(실패)
    - device: 실제 장치 이름('xpu:0') / details: verify_embedding_device() 검증 정보(가중치·토큰·출력 장치, 384차원, GPU 이름)
    - mode: auto | manual | invalid. 예외를 밖으로 던지지 않는다. (그래프는 ok=False이면 DeviceSelectionError를 낸다)"""
    # 1. 요청 장치 확인: auto·cuda·xpu·cpu만 받는다.
    order = _order(requested)
    if order is None:
        return _result(requested, "invalid", [], error="장치는 auto, cuda, xpu, cpu 중 하나로 지정해 주세요.")
    mode, devices = order

    # 2. PyTorch: 없으면 어느 장치도 확인할 수 없다.
    attempts: list[dict[str, Any]] = []
    try:
        torch = _import_torch()
    except Exception as error:  # noqa: BLE001
        attempts.append(_attempt(devices[0], "import_torch", False, _error_text(error)))
        return _result(requested, mode, attempts, error="PyTorch를 불러오지 못해 임베딩 장치를 확인할 수 없습니다.")

    # 3. 장치마다 PyTorch 단계 → MiniLM 단계를 검사한다. 자동이면 실패한 장치만 정리하고 다음 장치로 넘어간다.
    for device in devices:
        actual = _check_torch(torch, device, attempts)
        details = _check_model(device, actual, attempts) if actual else None
        if actual and details is not None:
            return _result(requested, mode, attempts, selected=device, device=actual, details=details,
                           torch_version=str(getattr(torch, "__version__", "")))
        cleanup = _release_failed(torch, device)
        if cleanup:
            attempts[-1]["cleanup_error"] = cleanup

    # 4. 모든 장치 실패: 성공으로 처리하지 않는다. 직접 지정이면 다른 장치를 시도하지 않았다고 알린다.
    error = ("실행 가능한 임베딩 장치가 없습니다. (cuda·xpu·cpu 모두 실패)" if mode == "auto"
             else f"지정한 {devices[0]} 장치에서 임베딩을 실행할 수 없습니다. 다른 장치로 바꾸지 않았습니다.")
    return _result(requested, mode, attempts, error=error, torch_version=str(getattr(torch, "__version__", "")))


# %% PyTorch 장치 선택 함수 (B기능용, MiniLM 검사 없음)
# 핵심 요약: B기능의 임베딩 모델은 아직 없으므로 A의 MiniLM으로 B를 검증하지 않는다. 백엔드·is_available·텐서 연산까지만 본다.
def select_torch_device(requested: str = "auto") -> dict[str, Any]:
    """select_embedding_device()와 같은 모양. check="torch_only", details에 'B 모델 검증 안 함'을 남긴다."""
    # 1. 요청 장치와 PyTorch 확인
    order = _order(requested)
    if order is None:
        return _result(requested, "invalid", [], error="장치는 auto, cuda, xpu, cpu 중 하나로 지정해 주세요.", check="torch_only")
    mode, devices = order
    attempts: list[dict[str, Any]] = []
    try:
        torch = _import_torch()
    except Exception as error:  # noqa: BLE001
        attempts.append(_attempt(devices[0], "import_torch", False, _error_text(error)))
        return _result(requested, mode, attempts, error="PyTorch를 불러오지 못했습니다.", check="torch_only")

    # 2. PyTorch 단계만 검사한다.
    for device in devices:
        actual = _check_torch(torch, device, attempts)
        if actual:
            return _result(requested, mode, attempts, selected=device, device=actual, check="torch_only",
                           details={"model_check": "B기능 임베딩 모델 검증은 B 코드 연결 후에 합니다. (텐서 연산까지만 확인)"},
                           torch_version=str(getattr(torch, "__version__", "")))
        cleanup = _release_failed(torch, device)
        if cleanup:
            attempts[-1]["cleanup_error"] = cleanup
    error = ("텐서 연산이 가능한 장치가 없습니다." if mode == "auto"
             else f"지정한 {devices[0]} 장치에서 텐서 연산을 할 수 없습니다. 다른 장치로 바꾸지 않았습니다.")
    return _result(requested, mode, attempts, error=error, check="torch_only",
                   torch_version=str(getattr(torch, "__version__", "")))


# %% Ollama /api/ps 원래 값 읽기
# 핵심 요약: 3단계 ollama_status()는 값이 없을 때 0으로 바꾸므로(보호 테스트 계약 유지), 판정용 원래 값은 여기서 한 번 더 읽는다.
def _get_json(settings: dict[str, Any], path: str) -> tuple[int | None, Any, str | None]:
    """(HTTP 상태, JSON 또는 None, 오류 메시지)"""
    server = settings["server"]
    connector = http.client.HTTPSConnection if server.scheme == "https" else http.client.HTTPConnection
    conn = connector(server.hostname, settings["port"], timeout=settings["connect"])
    try:
        conn.request("GET", server.path.rstrip("/") + path)
        response = conn.getresponse()
        raw = response.read(1_000_001)
        try:
            return response.status, json.loads(raw or b"null"), None
        except ValueError:
            return response.status, None, "JSON이 아닌 응답"
    except (OSError, http.client.HTTPException) as error:
        return None, None, _error_text(error)
    finally:
        conn.close()


def _is_count(value: Any) -> bool:
    """바이트 수로 인정할 값: bool이 아닌 0 이상의 정수"""
    return type(value) is int and value >= 0


# %% Ollama 적재 상태 판정 함수
# 핵심 요약: 설정한 대상 모델의 항목만 보고, size·size_vram이 정상 측정값일 때만 GPU 메모리 적재 정도를 판정한다.
# size_vram > 0은 'GPU 메모리에 올라갔다'는 뜻일 뿐, 모든 연산이 GPU에서 됐다는 보장이 아니다. GPU 종류·백엔드는 정하지 않는다.
def classify_ps(status: int | None, data: Any, error: str | None, model: str) -> dict[str, Any]:
    """{"state", "message", "size", "size_vram", "name"}"""
    def out(state: str, message: str, **extra: Any) -> dict[str, Any]:
        return {"state": state, "message": message, **extra}

    # 1. 응답 확인: 연결 실패·HTTP 오류·형식 오류를 모델 미적재와 구분한다.
    if status is None:
        return out("server_unreachable", f"Ollama /api/ps에 연결하지 못했습니다. ({error})")
    if status != 200:
        return out("response_error", f"/api/ps가 HTTP {status}로 응답했습니다. 실행 장치를 확인할 수 없습니다.")
    models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models, list):
        return out("response_error", "/api/ps 응답에 models 목록이 없습니다. 실행 장치를 확인할 수 없습니다.")

    # 2. 설정한 대상 모델 항목만 고른다. (다른 모델의 GPU 사용을 대상 모델 결과로 쓰지 않음)
    # (검토 보완) 항목마다 식별 정보(name·model)를 먼저 검사한다. 문자열이 아니거나 둘 다 없거나 서로 다른 모델을 가리키면
    # 그 항목이 대상 모델인지 알 수 없으므로 '미적재'나 측정값으로 판정하지 않는다.
    def ident(value: Any) -> str:
        return value[: -len(":latest")] if value.endswith(":latest") else value

    target = ident(model)
    entries = []
    for item in models:
        names = [item.get(key) for key in ("name", "model")] if isinstance(item, dict) else []
        given = [n for n in names if n is not None]
        if not isinstance(item, dict) or not given or not all(isinstance(n, str) and n for n in given):
            return out("invalid_model_info", "모델 식별 정보(name·model)가 없거나 문자열이 아닌 항목이 있어 실행 장치를 판정하지 않았습니다.")
        if len({ident(n) for n in given}) > 1:
            return out("invalid_model_info", "한 항목의 name과 model이 서로 다른 모델을 가리켜 실행 장치를 판정하지 않았습니다.")
        if ident(given[0]) == target:
            entries.append(item)
    if not entries:
        return out("not_loaded", f"{model} 모델이 메모리에 올라가 있지 않아 실행 장치를 확인할 수 없습니다. (CPU 실행이라는 뜻이 아님)")
    if len(entries) > 1:
        return out("invalid_values", f"{model} 항목이 여러 개라 하나로 판정하지 않았습니다.", entries=len(entries))

    # 3. 측정값 확인: bool·음수·정수가 아닌 값, size가 0, size_vram > size는 정상 측정값으로 보지 않는다.
    entry = entries[0]
    size, vram = entry.get("size"), entry.get("size_vram")
    base = {"name": entry.get("name") or entry.get("model"), "size": size, "size_vram": vram}
    if "size_vram" not in entry or "size" not in entry:
        return out("invalid_values", "size 또는 size_vram 값이 없어 GPU 메모리 사용을 판정하지 않았습니다.", **base)
    if not (isinstance(size, int) and isinstance(vram, int) and _is_count(size) and _is_count(vram)) \
            or size == 0 or vram > size:
        return out("invalid_values", "size·size_vram 값이 정상 측정값이 아니라 GPU 메모리 사용을 판정하지 않았습니다.", **base)

    # 4. 판정
    if vram == 0:
        return out("no_gpu_memory", "GPU 메모리 사용이 보고되지 않았습니다. (size_vram=0, CPU 메모리에 적재된 것으로 보입니다)", **base)
    if vram == size:
        return out("gpu_memory_full", "모델 전체가 GPU 메모리에 적재된 것으로 보고됐습니다. (모든 연산이 GPU에서 됐다는 보장은 아님, "
                                      "GPU 종류·추론 백엔드는 확인하지 않음)", **base)
    return out("gpu_memory_partial", f"모델 일부({vram / size:.0%})만 GPU 메모리에 적재됐습니다. CPU·GPU 혼합 실행 가능성이 있습니다. "
                                     "(GPU 종류·추론 백엔드는 확인하지 않음)", **base)


# %% ollama ps 명령 결과 (보조 정보)
# 핵심 요약: 명령이 설정한 서버와 같은 서버를 보는 경우에만 대상 모델의 PROCESSOR 칸을 읽는다. 판정에는 쓰지 않고 참고로만 남긴다.
def _same_server(settings: dict[str, Any]) -> bool:
    """ollama 명령이 보는 서버(OLLAMA_HOST, 기본 127.0.0.1:11434)와 SafeMate가 쓰는 서버가 같은지
    (검토 보완) 스킴(http)과 경로(없음)까지 같아야 한다. 확실하지 않으면 False → ollama ps를 쓰지 않는다."""
    # 1. SafeMate 설정 서버: 경로 없는 http 주소만 비교 대상으로 본다. (https·프록시 경로는 CLI와 같은 서버인지 알 수 없음)
    server = settings["server"]
    if server.scheme != "http" or server.path not in ("", "/"):
        return False
    # 2. CLI 서버(OLLAMA_HOST): 스킴이 있으면 http여야 하고, 경로가 없어야 한다.
    raw = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434").strip()
    scheme, sep, rest = raw.partition("://")
    if sep and scheme != "http":
        return False
    rest = rest if sep else raw
    if "/" in rest.rstrip("/"):
        return False
    name, _, port = rest.rstrip("/").partition(":")
    local = {"127.0.0.1", "localhost", "0.0.0.0", ""}
    same_host = (server.hostname in local and name in local) or server.hostname == name
    return same_host and str(settings["port"] or 11434) == (port or "11434")


def ollama_cli(settings: dict[str, Any], model: str) -> dict[str, Any]:
    """{"status": ok | skipped | error, "processor", "reason"}"""
    # 1. 명령이 있고 같은 서버를 볼 때만 실행한다.
    exe = shutil.which("ollama")
    if not exe:
        return {"status": "skipped", "processor": None, "reason": "ollama 명령을 찾지 못했습니다."}
    if not _same_server(settings):
        return {"status": "skipped", "processor": None, "reason": "ollama 명령이 보는 서버와 SafeMate 설정 서버가 달라 쓰지 않았습니다."}
    try:
        done = subprocess.run([exe, "ps"], capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=OLLAMA_CLI_TIMEOUT, check=False)
    except (OSError, subprocess.SubprocessError) as error:
        return {"status": "error", "processor": None, "reason": _error_text(error)}
    if done.returncode != 0:
        return {"status": "error", "processor": None, "reason": f"ollama ps 종료 코드 {done.returncode}"}
    text = done.stdout or ""

    # 2. 대상 모델 줄의 PROCESSOR 칸만 꺼낸다. ('100% GPU', '48%/52% CPU/GPU' 같은 값, 없으면 None)
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines or "PROCESSOR" not in lines[0]:
        return {"status": "error", "processor": None, "reason": "ollama ps 출력 형식을 확인하지 못했습니다."}
    start = lines[0].index("PROCESSOR")
    nxt = [lines[0].find(col, start + 1) for col in ("CONTEXT", "UNTIL")]
    end = min([n for n in nxt if n > start] or [len(lines[0]) + 100])
    for line in lines[1:]:
        if line.split()[0] in (model, model + ":latest"):
            return {"status": "ok", "processor": line[start:end].strip() or None, "reason": "참고용 (판정에 쓰지 않음)"}
    return {"status": "ok", "processor": None, "reason": "ollama ps 목록에 대상 모델이 없습니다."}


# %% Ollama 실행 장치 보고 함수
# 핵심 요약: 서버·모델 설치(3단계 ollama_status) → /api/ps 원래 값 판정 → (선택) 아주 짧은 실제 호출 → 호출 직후 다시 판정.
# 호출 전과 호출 직후 측정을 따로 남긴다. 측정에 실패해도 숨기지 않고 실패 상태 그대로 보고한다.
PROBE_SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}


def ollama_report(*, probe: bool = False) -> dict[str, Any]:
    """{"server", "model", "server_running", "model_installed", "before", "probe", "after", "cli", "limits"}"""
    limits = ["Ollama API로는 GPU 종류(NVIDIA·Intel 등)와 추론 백엔드를 확인할 수 없습니다.",
              "PyTorch 임베딩 장치와 Ollama 실행 장치는 서로 다른 시스템입니다."]
    # 1. 설정과 서버·모델 설치 상태 (3단계 함수 재사용)
    try:
        settings = od.ollama_settings()
    except od.DecisionError as error:
        return {"server": None, "model": None, "server_running": False, "model_installed": False,
                "before": {"state": "config_error", "message": str(error)}, "probe": None, "after": None,
                "cli": None, "limits": limits}
    model = settings["model"]
    # (검토 보완) 3단계 ollama_status()는 서버가 목록이 아닌 JSON([]·null 등)을 주면 예외를 낸다. 보호 테스트 계약 때문에 그 함수는
    # 고치지 않고, 여기서 예외를 '응답 형식 오류'로 받는다. 이때는 서버 실행·모델 설치를 확인하지 못한 것이다(미설치·CPU로 보지 않음).
    try:
        status = od.ollama_status(settings)
    except (TypeError, AttributeError, KeyError, ValueError) as error:
        report = {"server": settings["server"].geturl(), "model": model, "server_running": None, "model_installed": None,
                  "before": {"state": "response_error", "measured_at": "호출 전",
                             "message": f"Ollama /api/tags·/api/ps 응답 형식이 올바르지 않아 상태를 확인하지 못했습니다. ({_error_text(error)})"},
                  "probe": None, "after": None, "cli": None, "limits": limits}
        report["verdict"] = ollama_verdict(report, probe)
        return report
    report: dict[str, Any] = {"server": settings["server"].geturl(), "model": model,
                              "server_running": status.get("server_running") is True,
                              "model_installed": status.get("model_installed") is True,
                              "before": None, "probe": None, "after": None, "cli": None, "limits": limits}

    # 2. 호출 전 측정
    report["before"] = dict(classify_ps(*_get_json(settings, "/api/ps"), model), measured_at="호출 전")
    if not report["server_running"]:
        report["before"] = {"state": "server_unreachable", "message": status.get("error") or "Ollama 서버에 연결하지 못했습니다.",
                            "measured_at": "호출 전"}
        report["verdict"] = ollama_verdict(report, probe)
        return report
    if not report["model_installed"]:
        limits.append(f"{model} 모델이 설치되어 있지 않습니다. 모델을 내려받거나 바꾸지 않았습니다.")

    # 3. (선택) 아주 짧은 실제 호출 뒤, 모델이 메모리에 남아 있는 동안 다시 측정한다. (로컬 모델, 비용 없음)
    if probe and report["model_installed"]:
        started = time.perf_counter()
        messages = [{"role": "user", "content": "JSON으로 {\"ok\": true} 만 답하세요."}]
        try:
            _, stats = od.call_ollama(messages, PROBE_SCHEMA, settings, stage="device_probe", max_tokens=16, temperature=0)
            report["probe"] = {"status": "ok", "seconds": round(time.perf_counter() - started, 1),
                               "total_duration_ms": stats.get("total_duration_ms")}
        except od.DecisionError as error:
            report["probe"] = {"status": error.status, "code": error.code, "message": str(error),
                               "seconds": round(time.perf_counter() - started, 1)}
        after = dict(classify_ps(*_get_json(settings, "/api/ps"), model), measured_at="호출 직후")
        if after["state"] == "not_loaded":
            after["message"] = f"호출 직후에도 {model} 모델이 메모리에 없어 실행 장치를 확인할 수 없습니다. (장치 측정 실패)"
        report["after"] = after
    report["cli"] = ollama_cli(settings, model)
    report["verdict"] = ollama_verdict(report, probe)
    return report


# %% Ollama 진단 성공 판정 함수
# 핵심 요약: (검토 보완) 서버·모델 설치만 보고 성공으로 끝내지 않는다. 측정값이 정상 형식인지, 호출을 요청했다면 호출과
# 호출 직후 측정이 모두 성공했는지 본다. 정상적인 미적재(호출을 요청하지 않은 경우)와 CPU 추론은 실패가 아니다.
MEASURED_STATES = ("no_gpu_memory", "gpu_memory_full", "gpu_memory_partial")


def ollama_verdict(report: dict[str, Any], probe: bool) -> dict[str, Any]:
    """{"ok": bool, "reason"}"""
    if report.get("server_running") is not True or report.get("model_installed") is not True:
        return {"ok": False, "reason": "Ollama 서버 또는 대상 모델을 확인하지 못했습니다."}
    before = (report.get("before") or {}).get("state")
    if before not in MEASURED_STATES + ("not_loaded",):
        return {"ok": False, "reason": f"호출 전 측정을 확인하지 못했습니다. ({before})"}
    if not probe:
        return {"ok": True, "reason": "요청한 상태 확인을 마쳤습니다." + (" (모델 미적재: 실행 장치는 확인 불가)" if before == "not_loaded" else "")}
    if (report.get("probe") or {}).get("status") != "ok":
        return {"ok": False, "reason": f"실제 호출에 실패했습니다. ({(report.get('probe') or {}).get('status')})"}
    after = (report.get("after") or {}).get("state")
    if after not in MEASURED_STATES:
        return {"ok": False, "reason": f"호출은 성공했지만 호출 직후 실행 장치를 측정하지 못했습니다. ({after})"}
    return {"ok": True, "reason": "실제 호출과 호출 직후 장치 측정을 마쳤습니다."}


# %% 전체 진단 보고 함수
# 핵심 요약: 임베딩 장치 선택 결과와 Ollama 상태를 한 보고서로 묶고, 확인하지 못한 사항을 따로 적는다.
def device_report(requested: str = "auto", *, ollama: bool = True, probe: bool = False) -> dict[str, Any]:
    """{"embedding", "ollama", "unverified"}"""
    # 1. 임베딩 장치 선택
    embedding = select_embedding_device(requested)
    checked = {a["device"] for a in embedding["attempts"]}
    unverified = [f"{d}: 이번 실행에서 검사하지 않았습니다." for d in AUTO_ORDER if d not in checked]
    if embedding["selected"] != "cuda":
        unverified.append("CUDA 환경 실측은 이 실행 결과에 포함되지 않습니다. (CUDA 장치에서 따로 실행해야 함)")
    if embedding["cached_models"] and embedding["cached_models"] > 1:
        unverified.append(f"임베딩 모델이 장치별로 {embedding['cached_models']}개 캐시되어 있습니다. (같은 프로세스에서 여러 장치를 검사한 경우)")

    # 2. Ollama
    report = ollama_report(probe=probe) if ollama else None
    if report is None:
        unverified.append("Ollama 상태는 확인하지 않았습니다.")
    elif not probe:
        unverified.append("Ollama 실제 호출 직후 상태는 확인하지 않았습니다. (--ollama-probe로 확인)")
    return {"embedding": embedding, "ollama": report, "unverified": unverified}


def _print_human(report: dict[str, Any]) -> None:
    """진단 결과를 읽기 쉬운 줄로 출력한다."""
    emb = report["embedding"]
    print(f"[임베딩] 요청 {emb['requested']} ({emb['mode']}) → 선택 {emb['selected'] or '없음'} | 실제 장치 {emb['device']}")
    for a in emb["attempts"]:
        mark = "O" if a["ok"] else "X"
        print(f"  {mark} {a['device']:<4} {STAGE_WORDS.get(a['stage'], a['stage'])}" + (f" — {a['reason']}" if a["reason"] else ""))
    if emb["details"]:
        d = emb["details"]
        print(f"  임베딩 연산 장치 {d.get('embedding')} | 출력 차원 {d.get('dimensions')} | GPU {d.get('gpu_name')} | torch {d.get('torch')}")
    for reason in emb["switch_reasons"]:
        print(f"  전환: {reason}")
    if emb["error"]:
        print(f"  오류: {emb['error']}")
    oll = report["ollama"]
    if oll:
        print(f"[Ollama] {oll['server']} {oll['model']} | 서버 {oll['server_running']} | 모델 설치 {oll['model_installed']}")
        for key in ("before", "after"):
            if oll.get(key):
                print(f"  {oll[key]['measured_at']}: {oll[key]['state']} — {oll[key]['message']}")
        if oll.get("probe"):
            print(f"  실제 호출: {oll['probe']}")
        if oll.get("cli"):
            print(f"  ollama ps: {oll['cli']}")
        if oll.get("verdict"):
            print(f"  판정: {'성공' if oll['verdict']['ok'] else '실패'} — {oll['verdict']['reason']}")
    for line in report["unverified"]:
        print(f"[확인하지 못함] {line}")


# %% 독립 실행
# 핵심 요약: 'python device_utils.py --device auto'로 진단한다. 임베딩 장치를 확인하지 못하면 정상 종료(0)로 끝내지 않는다.
def main(argv: list[str] | None = None) -> int:
    """종료 코드 0(성공)·1(임베딩 장치 실패)·2(Ollama 서버·모델 문제)"""
    # 1. 옵션 읽기 (잘못된 장치 문자열도 받아서 invalid로 보고한다)
    parser = argparse.ArgumentParser(description="SafeMate 연산 장치 진단 (PyTorch 임베딩 + Ollama)")
    parser.add_argument("--device", default="auto", help="auto, cuda, xpu, cpu 중 하나 (기본 auto)")
    parser.add_argument("--ollama-probe", action="store_true", help="qwen2.5:3b 짧은 실제 호출 뒤 GPU 메모리 상태를 다시 측정")
    parser.add_argument("--no-ollama", action="store_true", help="Ollama 상태는 확인하지 않음")
    parser.add_argument("--json", action="store_true", help="전체 결과를 JSON으로 출력")
    args = parser.parse_args(argv)
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(errors="replace")

    # 2. 진단과 출력
    report = device_report(args.device, ollama=not args.no_ollama, probe=args.ollama_probe)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    else:
        _print_human(report)

    # 3. 종료 코드: 임베딩 장치 실패는 1, Ollama 서버·모델·호출·측정 문제는 2 (--no-ollama면 Ollama는 보지 않음)
    if not report["embedding"]["ok"]:
        return 1
    oll = report["ollama"]
    if oll is not None and not (oll.get("verdict") or ollama_verdict(oll, args.ollama_probe))["ok"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
