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
# %% 공통 라이브러리 불러오기
# 핵심 요약: 여러 함수에서 공통으로 사용하는 Python 표준 라이브러리와 LangChain 타입을 불러온다.
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
    # (수정) "비 와요", "비 와서"처럼 '와·왔'으로 이어지는 일상 표현도 호우로 인식한다.
    # 기존에는 '오·올·내'만 허용해 이런 질문이 재난 미식별(insufficient_evidence)로 끝났다.
    # 대설의 "눈이 와요" 표현과 같은 활용형(왔·와)을 맞춘 것이며, 다른 재난 표현은 그대로 둔다.
    "호우": r"호우|폭우|비바람|태풍|장마|집중강우|빗물|침수|하천|강변|빗길|(?:^|\s)비(?:가|는|를|\s*(?:오|올|왔|와|내))",
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




# %% 공식 PDF 원본 무결성 검증 함수
def matches_approved_pdf(path: Path) -> bool:
    """지정된 이름의 PDF와 승인했을 때 기록한 SHA-256이 일치하는지 확인한다."""

    # 1. 승인된 PDF 목록에서 해시값을 가져오고, 파일 존재 여부를 확인함
    # 승인 목록에 없거나 실제 파일에 존재하지 않으면 False를 반환함.
    expected = APPROVED_SHA256.get(path.name)
    if expected is None or not path.is_file():
        return False

    # 2. PDF를 바이너리 모드(rb)로 열어 SHA-256 해시값을 계산함
    # 계산한 해시값이 승인된 해시값과 일치하면 True, 다르면 False를 반환함.
    try:
        with path.open("rb") as pdf:
            return hashlib.file_digest(pdf, "sha256").hexdigest() == expected

    # 3. 파일을 읽는 과정에서 오류가 발생하면 SafeMate의 문서 오류로 반환함
    except OSError as error:
        raise SafetyError("document_error", f"{path.name} 파일을 읽을 수 없습니다. {error}") from error




# %% 문서의 특정 구간 추출 함수
def between(text, start, stop=None):
    """확인한 제목·문단 경계가 바뀌면 인용을 중단한다."""

    # 1. 전체 문서(text)에서 시작 문자열(start)의 위치를 찾는다.
    begin = text.index(start)

    # 2. 종료 문자열(stop)이 지정되면 시작 문자열 이후부터 종료 위치를 찾는다.
    # 종료 문자열이 지정되지 않으면 문서의 마지막까지 추출 범위로 설정한다.
    end = text.index(stop, begin + len(start)) if stop else len(text)
    
    # 3. 시작 위치부터 종료 위치 직전까지의 텍스트를 추출한다.
    # strip()으로 앞뒤의 불필요한 공백과 줄바꿈을 제거한 후 반환한다.
    return text[begin:end].strip()




# %% 문서 구간 생성 및 출처 정보 저장 함수
def make_section(filename, pdf_page, heading, body, printed_page=None):
    """한 문단에 발행 기관, 원문 URL, PDF 면과 인쇄 쪽 정보를 붙인다."""

    # 1. LangChain에서 문서 내용과 출처 정보를 함께 관리하는 Document 클래스를 불러온다.
    from langchain_core.documents import Document

    # 2. SOURCES에서 해당 파일의 기존 출처 정보를 가져온다.
    # 파일명, PDF 페이지 번호, 문단 제목, 주제를 메타데이터에 추가한다.
    # 주제(topic)는 문단 제목을 공백으로 나눈 후 첫 번째 단어를 사용한다
    metadata = {
        **SOURCES[filename],
        "filename": filename,
        "pdf_page": pdf_page,
        "heading": heading,
        "topic": heading.split()[0],
    }

    # 3. 문서에 실제 인쇄된 페이지 번호가 제공된 경우에만 메타데이터에 추가한다.
    # PDF 파일의 페이지 번호와 문서에 인쇄된 페이지 번호를 구분하여 관리한다.
    if printed_page is not None:
        metadata["printed_page"] = printed_page

    # 4. 본문의 불필요한 공백과 줄바꿈을 하나의 공백으로 정리한다.
    # 정리된 본문과 출처 정보를 Document 객체로 묶어 반환한다.
    # 이를 통해 RAG 검색 결과에서 문서의 내용과 출처를 함께 확인할 수 있다.
    return Document(page_content=" ".join(body.split()), metadata=metadata)




# %% 폭염 안전 행동 요령 PDF 문단 추출 함수
# 폭염 안전 수칙 PDF에서 필요한 내용만 추출하고 출처를 검증하여 RAG 검색용 데이터로 변환하는 함수.
def heat_sections(path, pages):
    """펼침면은 반쪽씩 읽고, 실제 하단의 쪽수와 PyPDFLoader 면을 대조한다."""
    # 1. PDF의 특정 영역을 잘라내고 텍스트를 추출하기 위해 pdfplumber를 불러온다.
    import pdfplumber

    # 2. RAG 검색에 사용할 폭염 안전 행동 요령의 추출 범위를 정의한다.
    # 각 튜플은 PDF 면, 왼쪽(0)/오른쪽(1), 인쇄 쪽수,
    # 검색 주제, 본문 시작 문자열, 종료 문자열 순서로 구성한다.
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

    # 3. 이미 추출한 PDF 반쪽 면의 본문을 저장할 딕셔너리를 생성한다.
    # 동일한 페이지를 반복해서 읽는 작업을 방지하여 처리 효율을 높인다.
    halves = {}

    # 4. PDF 파일을 열고 정의된 추출 범위를 순서대로 처리한다.
    # with문을 사용하여 작업이 끝나거나 오류가 발생하면 파일을 자동으로 닫는다.
    with pdfplumber.open(path) as book:
        for page_no, side, printed, heading, start, end in excerpts:
            
            # 5. PDF 면 번호와 좌우 방향을 조합하여 고유한 식별 키를 생성한다.
            # 이미 추출한 영역은 다시 읽지 않고 저장된 본문을 재사용한다.
            key = (page_no, side)

            if key not in halves:
                # 6. PDF의 해당 페이지를 가져오고 가로 너비의 절반을 계산한다.
                # 리스트의 인덱스는 0부터 시작하므로 페이지 번호에서 1을 뺀다.
                page = book.pages[page_no - 1]
                mid = page.width / 2

                # 7. 펼침면을 왼쪽 또는 오른쪽 영역으로 잘라낸다.
                # 두 페이지의 텍스트가 서로 섞여 추출되는 문제를 방지한다.
                crop = page.crop((side * mid, 0, (side + 1) * mid, page.height))

                # 8. 잘라낸 영역에서 텍스트를 추출한다.
                # x_tolerance와 y_tolerance는 글자의 가로·세로 간격 허용 범위를 설정한다.
                # 추출 결과가 None이면 빈 문자열로 처리한다.
                text = crop.extract_text(x_tolerance=2, y_tolerance=3) or ""


                # 9. 추출한 텍스트를 마지막 줄바꿈을 기준으로 본문과 하단 쪽수로 분리한다.
                body, footer = text.strip().rsplit("\n", 1)

                # 10. 실제 하단 쪽수가 지정한 인쇄 쪽수와 일치하는지 확인한다.
                # PyPDFLoader로 읽은 동일한 PDF 면에 본문이 존재하는지도 확인한다.
                # 두 조건 중 하나라도 충족하지 못하면 잘못된 문서 추출을 방지하기 위해
                # ValueError를 발생시킨다.
                if footer.strip() != str(printed) or not pages[page_no - 1].page_content.strip():
                    raise ValueError(f"PDF {page_no}면 / 인쇄 {printed}쪽 대조 실패")

                # 11. 정규표현식(re)을 사용하여 PDF 텍스트 추출 과정에서 발생한
                # 글머리 뒤 한글 음절 사이의 불필요한 공백을 제거한다.
                body = re.sub(r"(?m)^([•\x1a]) ([가-힣]) ([가-힣])", r"\1 \2\3", body)

                # 12. 비정상적으로 추출된 장식용 제어문자를 정상적인 글머리 기호로 변환한다.
                # 정리된 본문을 딕셔너리에 저장하여 이후 검색 구간에서 재사용한다.
                halves[key] = body.replace("\x1a", "•")

            # 13. between() 함수를 통해 지정된 시작과 종료 범위의 본문을 추출한다.
            # make_section() 함수로 본문과 출처 정보를 Document 객체로 구성한다.
            # yield를 사용하여 생성된 문서 객체를 하나씩 반환한다.
            yield make_section(HEAT, page_no, heading, between(halves[key], start, end), printed)




# %% 풍수해·대설·한파 안전 행동 요령 기사 추출 함수
# 재난 안전 관련 기사에서 필요한 행동 요령을 항목별로 추출하고 출처 정보를 포함한 RAG 검색용 데이터로 변환하는 함수.
def article_sections(filename, pages):
    """기사 원문의 문단·번호별로 나눈다. 각 인용의 실제 PDF 면을 따라간다."""

    # 1. 풍수해 기사에서 기상정보 확인 및 대피 관련 내용을 추출한다.
    if filename == STORM:
        first, second = (p.page_content for p in pages[:2])
        general = between(first, "이에 따라 도는 도민 스스로 기상정보", "이를 위해")
        yield make_section(STORM, 1, "풍수해 기상정보와 대피", general)
        
        # 2. 풍수해 행동 요령을 다섯 항목으로 분리하고 검증한 후,
        # 호우와 강풍 관련 안전 수칙으로 분류하여 저장한다.
        actions = between(first, "주요 행동요령은", "특히 어르신에게는").split("▲")[1:]
        if len(actions) != 5:
            raise ValueError("풍수해 행동요령 다섯 항목을 확인해 주세요.")
        for position, body in enumerate(actions):
            heading = "호우 위험지역과 대피" if position < 3 else "강풍 낙하물 위험"
            yield make_section(STORM, 1, heading, body)
        
        # 3. 어르신의 이동 및 대피 수칙과 재난문자 관련 내용을 추가로 추출한다.
        seniors = between(first, "특히 어르신에게는", "또한 도는")
        yield make_section(STORM, 1, "호우 어르신 이동과 대피", seniors)
        yield make_section(STORM, 2, "풍수해 재난문자와 대피", between(second, "김규식", "위 기사는"))
        return

    # 4. 대설·한파 기사에서 불필요한 부가 정보를 제외하고
    # 행동 요령을 번호별로 분리하여 1~6번 항목이 모두 존재하는지 검증한다.
    for page_no, section, stop in [(2, "대설", "한파주의보 행동 요령"), (3, "한파", None)]:
        text = re.split(r"(?m)^\d{2}\.\s*\d{1,2}\.\s*\d{1,2}\.", pages[page_no - 1].page_content)[0]
        body = between(text, "1. ", stop)
        items = re.split(r"(?m)^(\d+)\.\s+", body)

        if items[1::2] != list("123456"):
            raise ValueError(f"{section} 행동요령 1~6번을 확인해 주세요.")
        
        # 5. 추출한 행동 요령을 대설, 한파, 빙판길 안전 보행으로 분류하고
        # 출처 정보를 포함한 RAG 검색용 문서 객체로 반환한다.
        for number, item in zip(items[1::2], items[2::2]):
            heading = "빙판길 안전 보행" if section == "한파" and number == "4" else section + " 행동 요령"
            yield make_section(WINTER, page_no, heading, number + ". " + item)

    # 6. 겨울철 기상정보 및 한파 쉼터 관련 안내를 추출하여 반환한다.
    shelter = between(pages[3].page_content, "대설과 한파는", "경기도는")
    yield make_section(WINTER, 4, "겨울 대설 한파 쉼터", shelter)




# %% 공식 안전 행동 요령 PDF 로드 및 문서 추출 함수
# 승인된 PDF 파일을 불러와 재난 유형별 안전 수칙을 추출하고 RAG 검색용 문서 목록으로 통합하는 함수.
def load_sections(data_dir):
    """승인 파일 세 개만 PyPDFLoader로 면별 로드한다."""
    from langchain_community.document_loaders import PyPDFLoader

    # 1. 공식 PDF에서 추출한 행동 요령 문단을 보관할 리스트를 생성한다.
    sections = []

    # 2. SOURCES에 등록된 공식 문서만 순차적으로 불러오고,
    # 해당 PDF 파일이 실제로 존재하는지 확인한다.
    for filename in SOURCES:
        path = Path(data_dir) / filename
        try:
            if not path.is_file():
                raise FileNotFoundError(path)

            # 2-1. PDF 파일명뿐 아니라 승인한 판본의 SHA-256 해시값도 확인한다.
            # 내용이 변경된 문서를 검색 근거로 사용하지 않도록 문서 로딩을 중단한다.
            if not matches_approved_pdf(path):
                raise SafetyError("document_error", f"{filename} PDF가 승인된 원본과 일치하지 않습니다.")

            # 3. PDF를 페이지 단위로 읽고 문서 유형에 따라
            # 폭염 또는 풍수해·대설·한파 전용 추출 함수를 실행한다.
            # 추출된 문서들은 하나의 리스트로 통합한다.
            pages = PyPDFLoader(str(path), mode="page").load()
            selected = heat_sections(path, pages) if filename == HEAT else article_sections(filename, pages)
            sections.extend(selected)

        # 4. 파일 누락, 본문 추출 실패, 판본 불일치 등의 오류가 발생하면
        # SafetyError를 발생시켜 잘못된 문서가 RAG에 사용되는 것을 방지한다.
        except Exception as error:
            raise SafetyError("document_error", f"{filename} PDF의 파일·본문·판본을 확인해 주세요. {error}") from error

    # 5. 모든 공식 문서에서 추출한 안전 행동 요령을 반환한다.
    return sections




# %% RAG 문서 청크 분할 및 검색 정보 구성 함수
# 핵심 요약: 안전 행동 요령 문서를 검색하기 적절한 크기로 나누고, 검색용 제목과 실제 인용할 원문을 구분하여 저장하는 함수.
def split_sections(sections):
    """검색용 소제목과 실제 인용할 원문을 구분해 보관한다."""
    
    # 1. 문서를 작은 단위로 분할하기 위한 텍스트 분할기를 불러온다.
    # 청크 크기, 중복 범위, 분할 기준을 설정하여 문맥이 최대한 유지되도록 한다.
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP, separators=SEPARATORS
    )
    chunks = splitter.split_documents(sections)
    
    # 2. 실제 인용할 원문은 메타데이터에 별도로 보존하고,
    # 검색용 본문에는 소제목을 추가하여 검색 시 주제를 파악하기 쉽게 만든다.
    for chunk in chunks:
        chunk.metadata["excerpt"] = chunk.page_content
        chunk.page_content = chunk.metadata["heading"] + "\n" + chunk.page_content
    return chunks




# %% 임베딩 장치와 모델
# 핵심 요약: 임베딩에 사용할 CPU 또는 GPU가 정상적으로 작동하는지 실제 연산을 통해 검증하는 함수.
def configure_embedding_device(requested: str) -> str:
    """선택한 장치에서 작은 텐서 연산을 실행하고 실제 장치 이름을 돌려준다."""

    # 1. 사용자가 지정한 연산 장치가 지원되는 종류인지 확인한다.
    # cuda는 NVIDIA GPU, xpu는 Intel GPU, cpu는 CPU 연산을 의미한다.
    if requested not in ("cuda", "xpu", "cpu"):
        raise SafetyError("device_error", "장치는 cuda, xpu, cpu 중 하나를 지정해 주세요.")
    import torch

    # 2. 지정된 장치를 PyTorch에서 사용할 수 있는지 확인하고,
    # 실제 임베딩 연산에 사용할 장치를 설정한다.
    try:
        if requested == "cpu":
            device = torch.device("cpu")
        else:
            backend = getattr(torch, requested, None)
            if backend is None or not backend.is_available():
                raise RuntimeError("PyTorch가 이 GPU를 인식하지 못했습니다.")
            device = torch.device(requested, backend.current_device())

        # 3. 선택한 장치에서 간단한 텐서 연산을 실행하여
        # 실제 연산 장치와 계산 결과가 정상적인지 검증한다.
        probe = torch.ones(2, device=device).sum()
        if probe.device != device or probe.item() != 2:
            raise RuntimeError("텐서의 장치 또는 연산 결과가 예상과 다릅니다.")

    # 4. 장치 인식 또는 연산 과정에서 오류가 발생하면
    # 임베딩 작업을 중단하고 PyTorch와 드라이버 확인을 안내한다.
    except Exception as error:
        raise SafetyError("device_error", f"{requested} 장치 확인 실패. PyTorch 빌드와 드라이버를 확인해 주세요. {error}") from error

    # 5. 검증이 완료된 장치의 이름을 반환한다.
    return str(device)




# %% 임베딩 모델의 실제 연산 장치 검증 함수
# 핵심 요약: 임베딩 모델이 지정된 CPU 또는 GPU에서 실제로 연산하는지 확인하고, 생성된 벡터의 차원과 장치 정보를 검증하는 함수.
def verify_embedding_device(encoder, selected: str) -> dict:
    """가중치 전체, 토큰 임베딩 연산의 입출력, 최종 문장 벡터를 실제 한 문장으로 확인한다."""
    import torch

    # 1. 임베딩 모델의 모든 가중치가 지정된 연산 장치에 존재하는지 검증한다.
    # 실제 장치와 선택한 장치가 일치하지 않으면 오류를 발생시킨다.
    weights = {str(parameter.device) for parameter in encoder.parameters()}
    if weights != {selected}:
        raise ValueError(f"선택 장치 {selected}, 실제 모델 가중치 {sorted(weights)}")

    # 2. 모델 내부의 단어 임베딩 연산에서 입력과 출력이
    # 어떤 장치에서 처리되는지 기록하기 위한 관찰 함수를 정의한다.
    token_devices = {}

    def record_token_devices(layer, inputs, output):
        token_devices["token_input"] = str(inputs[0].device)
        token_devices["token_output"] = str(output.device)

    # 3. MiniLM 모델의 단어 임베딩 층에 관찰 함수(Forward Hook)를 연결한다.
    # 실제 문장을 임베딩하여 연산 장치를 확인하고, 검증이 끝나면 관찰 함수를 자동으로 제거한다.
    word_embeddings = encoder[0].auto_model.get_input_embeddings()
    with word_embeddings.register_forward_hook(record_token_devices):
        vector = encoder.encode("외출 안전", convert_to_tensor=True, normalize_embeddings=NORMALIZE)

    # 4. 토큰 임베딩의 입출력과 최종 문장 벡터가 지정된 장치에 존재하는지 확인한다.
    # 최종 벡터가 모델의 출력 규격인 384차원과 일치하는지도 검증한다.
    expected = {"token_input": selected, "token_output": selected}
    if token_devices != expected or str(vector.device) != selected or vector.shape != (384,):
        raise ValueError(f"임베딩 장치·차원 불일치. 선택 {selected}, 토큰 {token_devices}, 출력 {vector.device} {vector.shape}")

    # 5. 검증된 연산 장치의 종류와 실제 GPU 이름을 확인한다.
    # CPU를 사용하는 경우 GPU 이름은 None으로 유지한다.
    device = torch.device(selected)
    gpu_name = None
    if device.type != "cpu":
        gpu_name = getattr(torch, device.type).get_device_name(device.index)

    # 6. 모델 가중치, 토큰 입출력, 최종 임베딩 장치, 벡터 차원,
    # GPU 이름 및 PyTorch 버전 등 검증 결과를 딕셔너리로 반환한다.
    return {
        "requested": device.type, "weights": next(iter(weights)), **token_devices,
        "embedding": str(vector.device), "dimensions": vector.numel(),
        "gpu_name": gpu_name, "torch": str(torch.__version__),
    }



# %% 임베딩 모델 로딩 및 연산 장치 검증 함수
# 핵심 요약: 지정된 연산 장치에 임베딩 모델을 불러오고 정상 작동 여부를 검증한 후 모델을 재사용할 수 있도록 관리하는 함수.
@lru_cache(maxsize=3)
def load_embeddings(device: str):
    """장치별 모델을 한 번 준비하고 모델과 검증 정보를 함께 재사용한다."""

    # 1. 사용자가 지정한 연산 장치의 정상 작동 여부를 확인하고,
    # HuggingFace 임베딩 모델을 사용하기 위한 라이브러리를 불러온다.
    # lru_cache를 통해 최대 3개의 모델 로딩 결과를 보관하여 반복 로딩을 방지한다.
    selected = configure_embedding_device(device)
    from langchain_huggingface import HuggingFaceEmbeddings

    # 2. 지정된 모델과 버전을 선택한 연산 장치에 불러온다.
    # 임베딩 벡터의 정규화 여부를 설정하고, 한 번에 8개의 데이터를 처리하도록 구성한다.
    try:
        model = HuggingFaceEmbeddings(
            model_name=MODEL,
            model_kwargs={"device": selected, "revision": REVISION},
            encode_kwargs={"normalize_embeddings": NORMALIZE, "batch_size": 8},
        )

        # 3. 모델 내부의 실제 인코더를 대상으로 검증 함수를 실행하여
        # 지정된 장치에서 임베딩 연산이 정상적으로 수행되는지 확인한다.
        # 고정한 langchain-huggingface 버전에서 실제 인코더는 _client에 들어 있다.
        details = verify_embedding_device(model._client, selected)

    # 4. 모델 다운로드, 메모리 부족 또는 지원하지 않는 연산 등의 문제가 발생하면
    # SafetyError를 발생시켜 오류 원인 확인을 안내한다.
    except Exception as error:
        raise SafetyError("model_error", f"{selected} 모델 로딩·추론 실패. 다운로드·메모리·지원 연산을 확인해 주세요. {error}") from error

    # 5. 검증이 완료된 임베딩 모델과 연산 장치 정보를 반환한다.
    return model, details



# %% Chroma 영구 색인
# 핵심 요약: 원본 PDF와 RAG 설정의 고유 식별값을 생성하여 기존 벡터 DB의 재사용 여부를 판단할 수 있도록 하는 함수.
def corpus_key(data_dir):
    """PDF 바이트·전처리 규칙·임베딩 설정·관련 패키지 버전으로 저장소를 구분한다."""

    # 1. 원본 문서 정보, 임베딩 모델, 청크 분할 기준,
    # 유사도 계산 방식 및 관련 라이브러리 버전을 하나의 설정으로 구성한다.
    config = {"schema": 3, "sources": SOURCES, "model": MODEL, "revision": REVISION,
              "split": [CHUNK_SIZE, OVERLAP], "separators": SEPARATORS, "normalized": NORMALIZE, "metric": "cosine",
              "packages": {p: version(p) for p in ("pypdf", "pdfplumber", "langchain-text-splitters",
                           "sentence-transformers", "transformers", "langchain-huggingface", "chromadb", "langchain-chroma")}}

    # 2. SHA-256 해시 알고리즘을 사용하여 설정의 고유 식별값을 생성하고,
    # 중복 PDF를 확인하기 위한 집합을 준비한다.
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    seen = set()

    # 3. 승인된 PDF 파일을 순차적으로 읽어 각 파일의 고유 해시값을 계산한다.
    # 파일이 누락되거나 읽을 수 없는 경우 오류를 발생시킨다.
    for filename in SOURCES:
        try:
            with (Path(data_dir) / filename).open("rb") as pdf:
                signature = hashlib.file_digest(pdf, "sha256").digest()
        except OSError as error:
            raise SafetyError("document_error", f"{filename} 파일을 읽을 수 없습니다. {error}") from error

        # 3-1. 기존 Chroma 색인을 재사용할 때도 승인된 PDF 판본인지 확인한다.
        # 해시값이 다르면 잘못된 공식 문서가 사용되지 않도록 색인 준비를 중단한다.
        if signature.hex() != APPROVED_SHA256[filename]:
            raise SafetyError("document_error", f"{filename} PDF의 SHA-256이 승인된 원본과 일치하지 않습니다.")

        # 4. 동일한 내용의 PDF가 중복 등록되었는지 검증하고,
        # 각 PDF의 해시값을 전체 식별값 계산에 반영한다.
        if signature in seen:
            raise SafetyError("document_error", f"{filename} 자리에 다른 승인 PDF의 복사본이 있습니다.")
        seen.add(signature)
        digest.update(signature)

    # 5. 원본 PDF와 RAG 설정이 모두 반영된 최종 해시값을 반환한다.
    # 문서나 설정이 변경되면 다른 식별값이 생성되어 기존 색인과 구분할 수 있다.
    return digest.hexdigest()




# %% 코사인 거리를 검색 유사도 점수로 변환하는 함수
# 핵심 요약: 벡터 간 거리가 작을수록 더 높은 검색 점수를 부여한다.
def cosine_score(distance):
    """Chroma의 코사인 거리를 0~1 범위의 검색 점수로 바꾼다."""
    # 1. 코사인 거리를 1에서 빼 유사도로 변환하고 0~1 범위를 벗어나지 않게 한다.
    return max(0.0, min(1.0, 1.0 - distance))




@dataclass
class SafetyIndex:
    """여러 질문에 공용으로 넘길 검색기와 실행 정보."""

    retriever: VectorStoreRetriever
    device: dict
    chunks: int
    reused: bool
    path: Path




# %% RAG 벡터 데이터베이스 연결 및 유사도 관리 함수
# 핵심 요약: 임베딩 모델과 벡터 DB를 연결하고, 기존 연결을 재사용하며 검색된 문서의 유사도 점수를 함께 관리하는 함수.

@lru_cache(maxsize=3)
def open_safety_store(path, device):
    """동일 경로·장치의 연결을 공유한다. 코사인 점수는 같은 검색에서 한 번만 계산한다."""

    # 1. 벡터 데이터베이스 연결 및 환경 설정에 필요한 라이브러리를 불러온다.
    # lru_cache를 사용하여 최대 3개의 연결을 캐시에 보관하고 재사용한다.
    from chromadb.config import Settings
    from langchain_chroma import Chroma

    # 2. 기존 Chroma 클래스를 상속하여 검색 기능을 확장한다.
    # 검색 과정에서 계산된 유사도 점수를 문서의 메타데이터에 저장하여
    # 이후 검색 결과를 평가할 때 유사도를 다시 계산하지 않도록 한다.
    class ScoredChroma(Chroma):
        """공개 Retriever가 계산한 점수를 같은 검색의 Document에 담는다."""
        def similarity_search_with_relevance_scores(self, query, k=4, **kwargs):
            matches = super().similarity_search_with_relevance_scores(query, k=k, **kwargs)
            for doc, score in matches:
                doc.metadata["similarity"] = round(score, 4)
            return matches

    # 3. 지정된 연산 장치에서 임베딩 모델을 불러오고 벡터 DB와 연결한다.
    # 익명 사용 통계 수집을 비활성화하고, 코사인 거리 기반 검색을 설정한다.
    # 생성된 DB 연결 객체는 이후 문서 저장 및 RAG 검색에 사용된다.
    model, _ = load_embeddings(device)
    return ScoredChroma(
        collection_name="outing_safety",
        persist_directory=str(path),
        embedding_function=model,
        client_settings=Settings(anonymized_telemetry=False),
        collection_configuration={"hnsw": {"space": "cosine"}},
        relevance_score_fn=cosine_score,
    )



# %% RAG 벡터 데이터베이스 색인 준비 함수
# 핵심 요약: 공식 안전 문서를 임베딩하여 벡터 DB를 구축하고, 기존 색인이 유효하면 재사용하여 RAG 검색을 준비하는 함수.
def prepare_safety_index(
    device: str, *, data_dir: str | Path | None = None, cache_dir: str | Path | None = None
) -> SafetyIndex:
    """선택한 장치에서 색인을 한 번 준비한다. 문서·모델·저장 오류는 SafetyError로 전달한다."""

    # 1. 문서와 캐시 저장 경로를 설정하고, 원본 문서의 고유 식별값을 생성한다.
    # 이를 통해 문서의 변경 여부를 구분하고 임베딩 모델 및 연산 장치를 준비한다.
    from filelock import FileLock

    data_dir = DATA_DIR if data_dir is None else Path(data_dir)
    cache_dir = CACHE_DIR if cache_dir is None else Path(cache_dir)
    fingerprint = corpus_key(data_dir)
    path = cache_dir.resolve() / fingerprint
    marker = path / "ready.json"
    _, device_info = load_embeddings(device)

    # 2. 벡터 DB 저장 공간을 생성하고 파일 잠금 기능을 적용한다.
    # 여러 프로세스가 동시에 색인을 구축하여 데이터가 손상되는 것을 방지한다.
    try:
        path.mkdir(parents=True, exist_ok=True)
        with FileLock(str(path / "build.lock"), timeout=120):
            store = open_safety_store(path, device)

            # 3. 기존 벡터 DB와 색인 완료 기록을 비교하여 재사용 가능 여부를 판단한다.
            # 기존 청크 목록이 기록과 일치하면 불필요한 임베딩 작업을 생략한다.
            existing = set(store.get(include=[])["ids"])
            try:
                ready = json.loads(marker.read_text(encoding="utf-8"))
            except (FileNotFoundError, ValueError):
                ready = []
            reused = bool(existing) and sorted(existing) == ready

            # 4. 기존 색인을 재사용할 수 없다면 공식 문서를 불러와 청크로 분할한다.
            # 색인 준비 도중 원본 PDF가 변경되었는지도 검증한다.
            if not reused:
                chunks = split_sections(load_sections(data_dir))
                if corpus_key(data_dir) != fingerprint:
                    raise SafetyError("document_error", "색인 준비 중 PDF가 변경되었습니다. 다시 시작해 주세요.")

                # 5. 각 청크에 고유 ID를 부여하고 불필요한 기존 데이터를 삭제한다.
                # 새로운 청크를 임베딩하여 벡터 DB에 저장하고 저장 결과를 검증한다.
                ids = [f"chunk-{i:04}" for i in range(len(chunks))]
                extra = existing.difference(ids)
                if extra:
                    store.delete(ids=sorted(extra))
                store.add_documents(chunks, ids=ids)
                if set(store.get(include=[])["ids"]) != set(ids):
                    raise ValueError("저장 후 청크 목록이 일치하지 않습니다.")

                # 6. 색인 구축이 정상적으로 완료되면 청크 목록을 기록한다.
                # 임시 파일에 먼저 저장한 후 완료 기록 파일로 교체하여 기록 손상 위험을 줄인다.
                temp = marker.with_suffix(".tmp")
                temp.write_text(json.dumps(sorted(ids)), encoding="utf-8")
                temp.replace(marker)
                ready = ids

    # 7. 색인 구축 과정에서 발생하는 오류를 처리한다.
    # 문서, 모델 연산, DB 저장 경로 및 쓰기 권한 등의 문제를 안내한다.
    except SafetyError:
        raise
    except Exception as error:
        raise SafetyError("index_error", f"색인 준비 실패. 모델 연산·DB 경로·쓰기 권한을 확인해 주세요. {error}") from error

    # 8. 유사도 기반 검색기를 생성한다.
    # 최대 8개의 검색 후보 중 설정된 최소 유사도 기준을 충족하는 문서를 검색하도록 구성한다.
    retriever = store.as_retriever(search_type="similarity_score_threshold",
                                   search_kwargs={"k": 8, "score_threshold": MIN_SIMILARITY})

    # 9. 검색기, 연산 장치 정보, 청크 개수, 색인 재사용 여부,
    # 저장 경로를 하나의 SafetyIndex 객체로 묶어 반환한다.
    return SafetyIndex(retriever, dict(device_info), len(ready), reused, path)




# %% 질문과 관련된 안전 행동 요령 문단의 우선순위 정렬 함수
# 핵심 요약: 검색된 문단에서 질문과 일치하는 단어를 확인해 관련성이 높은 원문을 우선 선택한다.
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




# %% RAG 기반 외출 안전 행동 요령 검색 함수
# 핵심 요약: 사용자의 질문에서 재난 유형을 파악하고 관련 공식 안전 수칙을 검색하여 원문과 출처 정보를 반환하는 함수.
def search_safety_guidance(question: str | None, index: SafetyIndex) -> dict:
    """준비된 색인에 질문하고 status, message, count, results를 반환한다."""

    # 1. 검색 상태, 안내 메시지, 검색 결과 개수와 내용을
    # 일정한 딕셔너리 형식으로 반환하기 위한 내부 함수를 정의한다.
    def result(status, message, items):
        return {"status": status, "message": message, "count": len(items), "results": items}

    # 2. 사용자 질문의 유효성을 확인하고 재난 유형을 파악한다.
    # 질문이 비어 있거나 재난 유형을 식별하지 못하면 추가 정보 입력을 안내한다.
    if not isinstance(question, str) or not question.strip():
        return result("needs_input", "외출 안전에 관해 궁금한 점을 입력해 주세요.", [])
    topics = [name for name, pattern in TOPIC_TERMS.items() if re.search(pattern, question)]
    if not topics:
        return result("insufficient_evidence", "질문에서 검색할 재난을 식별하기 어렵습니다. 날씨나 도로 상태를 알려 주세요.", [])

    # 3. 식별된 재난 유형을 바탕으로 관련 검색 범위를 설정하고,
    # 기존 질문에 재난 주제를 추가하여 검색용 질문을 구성한다.
    allowed = sorted({tag for topic in topics for tag in RELATED[topic]})
    query = " ".join(topics) + " " + question.strip()

    # 4. LangSmith 추적 기능을 비활성화한 상태에서 벡터 검색을 실행한다.
    # 관련 재난 유형의 문서만 검색하며, 검색 실패 시 오류 메시지를 반환한다.
    try:
        from langsmith import tracing_context

        with tracing_context(enabled=False):
            # 4-1. 기존 Retriever의 유사도 기준과 검색 개수는 그대로 유지한다.
            # invoke()에 필터를 전달하면 검색 도구에서 무시될 수 있으므로,
            # 이번 질문에 맞는 재난 주제 필터를 검색기 생성 옵션에 지정한다.
            filtered_retriever = index.retriever.vectorstore.as_retriever(
                search_type=index.retriever.search_type,
                search_kwargs={
                    **index.retriever.search_kwargs,
                    "filter": {"topic": {"$in": allowed}},
                },
            )
            documents = filtered_retriever.invoke(query)
    except Exception as error:
        return result("search_error", f"질문 임베딩 또는 벡터 검색 실패. {error}", [])

    # 5. 검색된 문서의 관련성을 평가하고 순위를 정리한다.
    # 실제 원문, 발행 기관, URL, 페이지 번호, 유사도 등
    # 답변의 근거와 출처 확인에 필요한 정보를 결과 목록에 저장한다.
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

    # 6. 충분한 검색 근거가 없다면 임의로 정보를 생성하지 않고,
    # 근거 부족 상태를 반환한다. 검색에 성공하면 원문과 출처 정보를 반환한다.
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
