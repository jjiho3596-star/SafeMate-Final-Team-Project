r"""SafeMate A기능 2단계. 공식 PDF에서 외출 안전 행동 요령과 출처를 검색한다.

Windows PowerShell (기존 Python 3.14.6, Intel Arc 140T, .venv-xpu)
    $py = ".\.venv-xpu\Scripts\python.exe"
    & $py -c "import sys, torch; print(sys.version, torch.__version__, torch.xpu.is_available())"
    $torchPin = & $py -c "from importlib.metadata import version; print('torch==' + version('torch'))"
    $packages = & $py safety_rag.py --requirements
    & $py -m pip install $torchPin @packages
    & $py -m pip check

새 환경에서만 사용할 PyTorch 설치 예 (Windows x64 표준 CPython 3.14 또는 CUDA 12.6 호환 Linux)
    python -m pip install "torch==2.9.1+xpu" --extra-index-url https://download.pytorch.org/whl/xpu
    python -m pip install "torch==2.9.1" --index-url https://download.pytorch.org/whl/cu126
두 명령 중 해당 환경에 맞는 한 줄을 선택한다. 학교 서버에 작동하는 CUDA 빌드가 있으면 그대로 사용한다.
학교 서버의 Bash에서 RAG 의존성을 설치할 때 다음을 사용할 수 있다.
    python -m pip install -r <(python safety_rag.py --requirements) "torch==$(python -c 'from importlib.metadata import version; print(version("torch"))')"


실행 예시 (활성화한 가상환경에서)
    python safety_rag.py --device xpu --info
    python safety_rag.py --device cuda --info
    python safety_rag.py --device xpu "비가 많이 올 때 하천 근처로 걸어가도 되나요?"

다른 모듈에서는 prepare_safety_index(device)로 검색기를 한 번 준비한 뒤,
질문마다 search_safety_guidance(question, index)를 호출한다.
device는 이 프로세스의 문서·질문 임베딩에 적용된다. 학교 서버의 Ollama 주소·모델과는 별도다.
PDF는 이 파일 옆 data/safety에, 영구 색인은 .safemate/chroma에 둔다.
"""

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from langchain_core.vectorstores import VectorStoreRetriever

# %% 모델 · 문서 설정
RAG_PACKAGES = (
    "langchain-community==0.4.2",
    "langchain-core==1.8.4",
    "langchain-text-splitters==1.1.3",
    "langchain-huggingface==1.2.3",
    "langchain-chroma==1.1.0",
    "chromadb==1.5.9",
    "sentence-transformers==5.2.0",
    "transformers==4.57.6",
    "pypdf==6.21.4",
    "pdfplumber==0.11.11",
    "filelock>=3.16,<5",
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "safety"
CACHE_DIR = ROOT / ".safemate" / "chroma"
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
CHUNK_SIZE, OVERLAP = 500, 50
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
NORMALIZE = True
MIN_SIMILARITY = 0.30
HEAT, STORM, WINTER = "kdca_heat_2026.pdf", "gg_rain_wind_2026.pdf", "gg_cold_snow_2025.pdf"
SOURCES = {
    HEAT: {"title": "2026년 폭염대비와 온열질환 예방을 위한 건강수칙 바로 알기 소책자",
           "publisher": "질병관리청", "url": "https://www.kdca.go.kr/bbs/kdca/46/310939/artclView.do"},
    STORM: {"title": "경기도, 풍수해 행동요령 집중 홍보…“미리 아는 만큼 안전합니다!”",
            "publisher": "경기도 자연재난과", "url": "https://gnews.gg.go.kr/briefing/brief_gongbo_view.do?BS_CODE=s017&number=70536"},
    WINTER: {"title": "“대설·한파, 이렇게 대비하세요!” 겨울철 자연재난 행동 요령",
             "publisher": "경기도 뉴스포털", "url": "https://gnews.gg.go.kr/news/news_detail.do?number=202512091332268777C083&s_code=C083"},
}


# 승인한 판본의 바이트를 Windows와 Colab에서 같은 기준으로 대조한다.
APPROVED_SHA256 = {
    HEAT: "f01c58df0b134b277a5263828075c15ea623dfc118850261947b16005fd574c1",
    STORM: "768a897882aeca32d4a644d8da9698ae75d28f228c2a44ff03992475a0c630e2",
    WINTER: "e13041671967f8e82a6b3fe49f5143889349a4eb2c66a49cc44bfa71f1bd636e",
}

# 일상 표현을 문서의 재난 분류에 연결한다. 문장과 행동 요령의 선택은 벡터 검색이 맡는다.
TOPIC_TERMS = {
    "폭염": r"폭염|온열|무더|더위|덥|더[운워울웠]|뜨거운|한여름|열대야|햇볕|양산",
    "한파": r"한파|혹한|겨울|추위|춥|추[운워울웠]|영하|방한|동파",
    "대설": r"대설|폭설|겨울|눈길|눈(?:이|은|도)?\s*(?:(?:많이|계속|펑펑|조금|잔뜩)\s*)?(?:오|올|왔|와|내리|내려|내릴|쌓|녹)|적설|제설|빙판|결빙|얼[었어]|미끄",
    "호우": r"호우|폭우|비바람|태풍|장마|집중강우|빗물|침수|하천|강변|빗길|(?:^|\s)비(?:가|는|를|\s*(?:오|올|내))",
    "강풍": r"강풍|돌풍|바람|낙하물|태풍",
}
RELATED = {"폭염": ["폭염"], "한파": ["한파", "빙판길", "겨울"],
           "대설": ["대설", "빙판길", "겨울"], "호우": ["호우", "풍수해"], "강풍": ["강풍", "풍수해"]}


# %% PDF 본문 · 출처 · 청킹
class SafetyError(Exception):
    """색인을 준비하지 못한 이유를 status와 한국어 메시지로 전달한다."""

    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status


def matches_approved_pdf(path: Path) -> bool:
    """지정된 이름의 PDF와 승인했을 때 기록한 SHA-256이 일치하는지 확인한다."""
    expected = APPROVED_SHA256.get(path.name)
    if expected is None or not path.is_file():
        return False
    try:
        with path.open("rb") as pdf:
            return hashlib.file_digest(pdf, "sha256").hexdigest() == expected
    except OSError as error:
        raise SafetyError("document_error", f"{path.name} 파일을 읽을 수 없습니다. {error}") from error


def between(text, start, stop=None):
    """확인한 제목·문단 경계가 바뀌면 인용을 중단한다."""
    begin = text.index(start)
    end = text.index(stop, begin + len(start)) if stop else len(text)
    return text[begin:end].strip()


def make_section(filename, pdf_page, heading, body, printed_page=None):
    """한 문단에 발행 기관, 원문 URL, PDF 면과 인쇄 쪽 정보를 붙인다."""
    from langchain_core.documents import Document

    metadata = {
        **SOURCES[filename],
        "filename": filename,
        "pdf_page": pdf_page,
        "heading": heading,
        "topic": heading.split()[0],
    }
    if printed_page is not None:
        metadata["printed_page"] = printed_page
    return Document(page_content=" ".join(body.split()), metadata=metadata)


def heat_sections(path, pages):
    """펼침면은 반쪽씩 읽고, 실제 하단의 쪽수와 PyPDFLoader 면을 대조한다."""
    import pdfplumber

    # (PDF 면, 왼쪽=0/오른쪽=1, 인쇄 쪽, 검색 주제, 본문 시작, 다음 절)
    excerpts = [
        (7, 0, 10, "폭염 노인", "노인\n", "어린이\n"),
        (12, 0, 20, "폭염 외출 복장과 햇볕", "①", "②"),
        (12, 0, 20, "폭염 수분 섭취", "②", "③"),
        (12, 0, 20, "폭염 더운 시간대 야외활동", "③", "9-2"),
        (13, 0, 22, "폭염 야외 복장", "9-5", "9-6"),
        (13, 1, 23, "폭염 취약계층 야외활동", "폭염 시 야외활동을 삼가고", None),
        (15, 0, 26, "폭염 외출 복장과 햇볕", "시원하게 지냅니다.", "물을 자주 마십니다."),
        (15, 0, 26, "폭염 수분 섭취", "물을 자주 마십니다.", "더운 시간대에는"),
        (15, 0, 26, "폭염 더운 시간대 야외활동", "더운 시간대에는", "폭염 관련 특히"),
        (15, 0, 26, "폭염 노인 야외활동", "노인\n", "어린이\n"),
    ]
    halves = {}
    with pdfplumber.open(path) as book:
        for page_no, side, printed, heading, start, end in excerpts:
            key = (page_no, side)
            if key not in halves:
                page = book.pages[page_no - 1]
                mid = page.width / 2
                crop = page.crop((side * mid, 0, (side + 1) * mid, page.height))
                text = crop.extract_text(x_tolerance=2, y_tolerance=3) or ""
                body, footer = text.strip().rsplit("\n", 1)
                if footer.strip() != str(printed) or not pages[page_no - 1].page_content.strip():
                    raise ValueError(f"PDF {page_no}면 / 인쇄 {printed}쪽 대조 실패")
                # PDF 장식용 글리프와 글머리 뒤 첫 음절의 잘못된 공백만 정리한다.
                body = re.sub(r"(?m)^([•\x1a]) ([가-힣]) ([가-힣])", r"\1 \2\3", body)
                halves[key] = body.replace("\x1a", "•")
            yield make_section(HEAT, page_no, heading, between(halves[key], start, end), printed)


def article_sections(filename, pages):
    """기사 원문의 문단·번호별로 나눈다. 각 인용의 실제 PDF 면을 따라간다."""
    if filename == STORM:
        first, second = (p.page_content for p in pages[:2])
        general = between(first, "이에 따라 도는 도민 스스로 기상정보", "이를 위해")
        yield make_section(STORM, 1, "풍수해 기상정보와 대피", general)
        actions = between(first, "주요 행동요령은", "특히 어르신에게는").split("▲")[1:]
        if len(actions) != 5:
            raise ValueError("풍수해 행동요령 다섯 항목을 확인해 주세요.")
        for position, body in enumerate(actions):
            heading = "호우 위험지역과 대피" if position < 3 else "강풍 낙하물 위험"
            yield make_section(STORM, 1, heading, body)
        seniors = between(first, "특히 어르신에게는", "또한 도는")
        yield make_section(STORM, 1, "호우 어르신 이동과 대피", seniors)
        yield make_section(STORM, 2, "풍수해 재난문자와 대피", between(second, "김규식", "위 기사는"))
        return

    for page_no, section, stop in [(2, "대설", "한파주의보 행동 요령"), (3, "한파", None)]:
        text = re.split(r"(?m)^\d{2}\.\s*\d{1,2}\.\s*\d{1,2}\.", pages[page_no - 1].page_content)[0]
        body = between(text, "1. ", stop)
        items = re.split(r"(?m)^(\d+)\.\s+", body)
        if items[1::2] != list("123456"):
            raise ValueError(f"{section} 행동요령 1~6번을 확인해 주세요.")
        for number, item in zip(items[1::2], items[2::2]):
            heading = "빙판길 안전 보행" if section == "한파" and number == "4" else section + " 행동 요령"
            yield make_section(WINTER, page_no, heading, number + ". " + item)
    shelter = between(pages[3].page_content, "대설과 한파는", "경기도는")
    yield make_section(WINTER, 4, "겨울 대설 한파 쉼터", shelter)


def load_sections(data_dir):
    """승인 파일 세 개만 PyPDFLoader로 면별 로드한다."""
    from langchain_community.document_loaders import PyPDFLoader

    sections = []
    for filename in SOURCES:
        path = Path(data_dir) / filename
        try:
            if not path.is_file():
                raise FileNotFoundError(path)
            pages = PyPDFLoader(str(path), mode="page").load()
            selected = heat_sections(path, pages) if filename == HEAT else article_sections(filename, pages)
            sections.extend(selected)
        except Exception as error:
            raise SafetyError("document_error", f"{filename} PDF의 파일·본문·판본을 확인해 주세요. {error}") from error
    return sections


def split_sections(sections):
    """검색용 소제목과 실제 인용할 원문을 구분해 보관한다."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP, separators=SEPARATORS
    )
    chunks = splitter.split_documents(sections)
    for chunk in chunks:
        chunk.metadata["excerpt"] = chunk.page_content
        chunk.page_content = chunk.metadata["heading"] + "\n" + chunk.page_content
    return chunks


# %% 임베딩 장치와 모델
def configure_embedding_device(requested: str) -> str:
    """선택한 장치에서 작은 텐서 연산을 실행하고 실제 장치 이름을 돌려준다."""
    if requested not in ("cuda", "xpu", "cpu"):
        raise SafetyError("device_error", "장치는 cuda, xpu, cpu 중 하나를 지정해 주세요.")
    import torch

    try:
        if requested == "cpu":
            device = torch.device("cpu")
        else:
            backend = getattr(torch, requested, None)
            if backend is None or not backend.is_available():
                raise RuntimeError("PyTorch가 이 GPU를 인식하지 못했습니다.")
            device = torch.device(requested, backend.current_device())
        probe = torch.ones(2, device=device).sum()
        if probe.device != device or probe.item() != 2:
            raise RuntimeError("텐서의 장치 또는 연산 결과가 예상과 다릅니다.")
    except Exception as error:
        raise SafetyError("device_error", f"{requested} 장치 확인 실패. PyTorch 빌드와 드라이버를 확인해 주세요. {error}") from error
    return str(device)


def verify_embedding_device(encoder, selected: str) -> dict:
    """가중치 전체, 토큰 임베딩 연산의 입출력, 최종 문장 벡터를 실제 한 문장으로 확인한다."""
    import torch

    weights = {str(parameter.device) for parameter in encoder.parameters()}
    if weights != {selected}:
        raise ValueError(f"선택 장치 {selected}, 실제 모델 가중치 {sorted(weights)}")

    token_devices = {}

    def record_token_devices(layer, inputs, output):
        token_devices["token_input"] = str(inputs[0].device)
        token_devices["token_output"] = str(output.device)

    # MiniLM의 단어 임베딩 층에 한 번만 관찰 함수를 건다. with를 벗어나면 자동으로 제거된다.
    word_embeddings = encoder[0].auto_model.get_input_embeddings()
    with word_embeddings.register_forward_hook(record_token_devices):
        vector = encoder.encode("외출 안전", convert_to_tensor=True, normalize_embeddings=NORMALIZE)

    expected = {"token_input": selected, "token_output": selected}
    if token_devices != expected or str(vector.device) != selected or vector.shape != (384,):
        raise ValueError(f"임베딩 장치·차원 불일치. 선택 {selected}, 토큰 {token_devices}, 출력 {vector.device} {vector.shape}")

    device = torch.device(selected)
    gpu_name = None
    if device.type != "cpu":
        gpu_name = getattr(torch, device.type).get_device_name(device.index)
    return {
        "requested": device.type, "weights": next(iter(weights)), **token_devices,
        "embedding": str(vector.device), "dimensions": vector.numel(),
        "gpu_name": gpu_name, "torch": str(torch.__version__),
    }


@lru_cache(maxsize=3)
def load_embeddings(device: str):
    """장치별 모델을 한 번 준비하고 모델과 검증 정보를 함께 재사용한다."""
    selected = configure_embedding_device(device)
    from langchain_huggingface import HuggingFaceEmbeddings

    try:
        model = HuggingFaceEmbeddings(
            model_name=MODEL,
            model_kwargs={"device": selected, "revision": REVISION},
            encode_kwargs={"normalize_embeddings": NORMALIZE, "batch_size": 8},
        )
        # 고정한 langchain-huggingface 버전에서 실제 인코더는 _client에 들어 있다.
        details = verify_embedding_device(model._client, selected)
    except Exception as error:
        raise SafetyError("model_error", f"{selected} 모델 로딩·추론 실패. 다운로드·메모리·지원 연산을 확인해 주세요. {error}") from error
    return model, details


# %% Chroma 영구 색인
def corpus_key(data_dir):
    """PDF 바이트·전처리 규칙·임베딩 설정·관련 패키지 버전으로 저장소를 구분한다."""
    config = {"schema": 3, "sources": SOURCES, "model": MODEL, "revision": REVISION,
              "split": [CHUNK_SIZE, OVERLAP], "separators": SEPARATORS, "normalized": NORMALIZE, "metric": "cosine",
              "packages": {p: version(p) for p in ("pypdf", "pdfplumber", "langchain-text-splitters",
                           "sentence-transformers", "transformers", "langchain-huggingface", "chromadb", "langchain-chroma")}}
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    seen = set()
    for filename in SOURCES:
        try:
            with (Path(data_dir) / filename).open("rb") as pdf:
                signature = hashlib.file_digest(pdf, "sha256").digest()
        except OSError as error:
            raise SafetyError("document_error", f"{filename} 파일을 읽을 수 없습니다. {error}") from error
        if signature in seen:
            raise SafetyError("document_error", f"{filename} 자리에 다른 승인 PDF의 복사본이 있습니다.")
        seen.add(signature)
        digest.update(signature)
    return digest.hexdigest()


def cosine_score(distance):
    """Chroma의 코사인 거리를 0~1 범위의 검색 점수로 바꾼다."""
    return max(0.0, min(1.0, 1.0 - distance))


@dataclass
class SafetyIndex:
    """여러 질문에 공용으로 넘길 검색기와 실행 정보."""

    retriever: VectorStoreRetriever
    device: dict
    chunks: int
    reused: bool
    path: Path


@lru_cache(maxsize=3)
def open_safety_store(path, device):
    """동일 경로·장치의 연결을 공유한다. 코사인 점수는 같은 검색에서 한 번만 계산한다."""
    from chromadb.config import Settings
    from langchain_chroma import Chroma

    class ScoredChroma(Chroma):
        """공개 Retriever가 계산한 점수를 같은 검색의 Document에 담는다."""
        def similarity_search_with_relevance_scores(self, query, k=4, **kwargs):
            matches = super().similarity_search_with_relevance_scores(query, k=k, **kwargs)
            for doc, score in matches:
                doc.metadata["similarity"] = round(score, 4)
            return matches

    model, _ = load_embeddings(device)
    return ScoredChroma(
        collection_name="outing_safety",
        persist_directory=str(path),
        embedding_function=model,
        client_settings=Settings(anonymized_telemetry=False),
        collection_configuration={"hnsw": {"space": "cosine"}},
        relevance_score_fn=cosine_score,
    )


def prepare_safety_index(
    device: str, *, data_dir: str | Path | None = None, cache_dir: str | Path | None = None
) -> SafetyIndex:
    """선택한 장치에서 색인을 한 번 준비한다. 문서·모델·저장 오류는 SafetyError로 전달한다."""
    from filelock import FileLock

    data_dir = DATA_DIR if data_dir is None else Path(data_dir)
    cache_dir = CACHE_DIR if cache_dir is None else Path(cache_dir)
    fingerprint = corpus_key(data_dir)
    path = cache_dir.resolve() / fingerprint
    marker = path / "ready.json"
    _, device_info = load_embeddings(device)
    try:
        path.mkdir(parents=True, exist_ok=True)
        with FileLock(str(path / "build.lock"), timeout=120):
            store = open_safety_store(path, device)
            existing = set(store.get(include=[])["ids"])
            try:
                ready = json.loads(marker.read_text(encoding="utf-8"))
            except (FileNotFoundError, ValueError):
                ready = []
            reused = bool(existing) and sorted(existing) == ready
            if not reused:
                chunks = split_sections(load_sections(data_dir))
                if corpus_key(data_dir) != fingerprint:
                    raise SafetyError("document_error", "색인 준비 중 PDF가 변경되었습니다. 다시 시작해 주세요.")
                ids = [f"chunk-{i:04}" for i in range(len(chunks))]
                extra = existing.difference(ids)
                if extra:
                    store.delete(ids=sorted(extra))
                store.add_documents(chunks, ids=ids)
                if set(store.get(include=[])["ids"]) != set(ids):
                    raise ValueError("저장 후 청크 목록이 일치하지 않습니다.")
                temp = marker.with_suffix(".tmp")
                temp.write_text(json.dumps(sorted(ids)), encoding="utf-8")
                temp.replace(marker)
                ready = ids
    except SafetyError:
        raise
    except Exception as error:
        raise SafetyError("index_error", f"색인 준비 실패. 모델 연산·DB 경로·쓰기 권한을 확인해 주세요. {error}") from error
    retriever = store.as_retriever(search_type="similarity_score_threshold",
                                   search_kwargs={"k": 8, "score_threshold": MIN_SIMILARITY})
    return SafetyIndex(retriever, dict(device_info), len(ready), reused, path)


# %% 질문 → 근거 검색 결과
def rank_passages(question, documents):
    """의미 검색 후보 안에서 질문의 구체적인 단어가 원문에도 있는 문단을 우선한다."""
    # 한국어 조사가 붙은 '하천으로'도 '하천'과 비교한다. 두 음절 단서는 보조 지표다.
    common = {"날씨", "오늘", "내일", "너무", "많이", "매우", "어떻", "무엇", "하나",
              "해요", "되나", "될까", "하는", "어떤", "인지", "있나", "근처", "때문"}
    def walking_terms(text):
        return re.sub(r"걸어|걷|보폭", "보행", text)

    stems = {word[:2] for word in re.findall(r"[가-힣]{2,}", walking_terms(question))} - common
    def rank(doc):
        text = walking_terms(doc.metadata["excerpt"])
        hits = sum(term in text for term in stems)
        return hits, doc.metadata["similarity"]
    ranked = sorted(documents, key=rank, reverse=True)
    if not ranked:
        return []
    best_hits = rank(ranked[0])[0]
    return [doc for doc in ranked if rank(doc)[0] == best_hits][:2]


def search_safety_guidance(question: str | None, index: SafetyIndex) -> dict:
    """준비된 색인에 질문하고 status, message, count, results를 반환한다."""
    def result(status, message, items):
        return {"status": status, "message": message, "count": len(items), "results": items}

    if not isinstance(question, str) or not question.strip():
        return result("needs_input", "외출 안전에 관해 궁금한 점을 입력해 주세요.", [])
    topics = [name for name, pattern in TOPIC_TERMS.items() if re.search(pattern, question)]
    if not topics:
        return result("insufficient_evidence", "질문에서 검색할 재난을 식별하기 어렵습니다. 날씨나 도로 상태를 알려 주세요.", [])
    allowed = sorted({tag for topic in topics for tag in RELATED[topic]})
    query = " ".join(topics) + " " + question.strip()
    try:
        from langsmith import tracing_context

        with tracing_context(enabled=False):
            documents = index.retriever.invoke(query, filter={"topic": {"$in": allowed}})
    except Exception as error:
        return result("search_error", f"질문 임베딩 또는 벡터 검색 실패. {error}", [])
    hits = []
    for document in rank_passages(question, documents):
        metadata = document.metadata
        hits.append({
            "content": metadata["excerpt"],
            "title": metadata["title"],
            "publisher": metadata["publisher"],
            "url": metadata["url"],
            "filename": metadata["filename"],
            "pdf_page": metadata["pdf_page"],
            "printed_page": metadata.get("printed_page"),
            "similarity": metadata["similarity"],
        })
    if not hits:
        return result("insufficient_evidence", "승인된 자료에서 이 질문에 충분한 근거를 찾지 못했습니다.", [])
    return result("ok", "외출 안전 관련 원문을 찾았습니다.", hits)


# %% 독립 실행
def main(argv=None):
    """색인 정보를 확인하거나 한 질문의 검색 결과를 JSON으로 출력한다."""
    parser = argparse.ArgumentParser(description="SafeMate 공식 PDF 안전 행동 요령 검색")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("question", nargs="?", help="공식 PDF에서 검색할 외출 안전 질문")
    action.add_argument("--info", action="store_true", help="색인을 준비하고 장치·청크 수·재사용 여부·승인본 일치 여부 출력")
    action.add_argument("--requirements", action="store_true", help="PyTorch 외에 설치할 패키지 출력")
    parser.add_argument("--device", choices=("xpu", "cuda", "cpu"), help="임베딩을 실행할 장치")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    args = parser.parse_args(argv)

    if args.requirements:
        print("\n".join(RAG_PACKAGES))
        return 0
    if args.device is None:
        parser.error("검색하거나 색인을 검사할 때 --device xpu, --device cuda, --device cpu 중 하나를 지정해 주세요.")

    try:
        index = prepare_safety_index(args.device, data_dir=args.data_dir, cache_dir=args.cache_dir)
        if args.info:
            output = {"status": "ok", "device": index.device, "chunks": index.chunks,
                      "reused": index.reused, "path": str(index.path),
                      "approved_pdf_matches": {name: matches_approved_pdf(args.data_dir / name) for name in SOURCES}}
        else:
            output = search_safety_guidance(args.question, index)
    except SafetyError as error:
        output = {"status": error.status, "message": str(error), "count": 0, "results": []}
    except ImportError as error:
        output = {"status": "dependency_error", "message": f"패키지를 확인해 주세요. {error}",
                  "count": 0, "results": []}

    print(json.dumps(output, ensure_ascii=False, indent=2))
    return int(output["status"] not in ("ok", "needs_input", "insufficient_evidence"))


if __name__ == "__main__":
    raise SystemExit(main())
