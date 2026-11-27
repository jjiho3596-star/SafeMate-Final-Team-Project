import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import b_living


class LivingRagTests(unittest.TestCase):
    def test_local_txt_md_json_documents_and_empty_results(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            (data_dir / "guide.txt").write_text(
                "기초연금 신청에 필요한 안내", encoding="utf-8"
            )
            (data_dir / "policy.md").write_text(
                "기초연금 신청 정책 안내", encoding="utf-8"
            )
            (data_dir / "source.json").write_text(
                json.dumps(
                    {
                        "content": "기초연금 신청 지원 내용",
                        "metadata": {
                            "title": "복지 자료",
                            "publisher": "공공기관",
                            "url": "https://example.invalid/guide",
                            "pdf_page": 2,
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            with patch.object(b_living, "LIVING_DATA_DIR", data_dir):
                evidence = b_living._search_living_evidence("기초연금 신청", None)
                missing = b_living._search_living_evidence("전혀 다른 검색어", None)

        self.assertEqual(len(evidence), 3)
        self.assertEqual(
            set(evidence[0]),
            {"content", "title", "publisher", "url", "pdf_page"},
        )
        json_evidence = next(item for item in evidence if item["title"] == "복지 자료")
        self.assertEqual(json_evidence["publisher"], "공공기관")
        self.assertEqual(json_evidence["pdf_page"], 2)
        self.assertEqual(missing, [])

    def test_pdf_pages_include_source_page_and_metadata(self):
        class FakePage:
            def extract_text(self):
                return "기초연금 신청 PDF 원문"

        class FakeReader:
            metadata = {"/Title": "PDF 복지 안내", "/Publisher": "공공기관"}
            pages = [FakePage(), FakePage()]

            def __init__(self, _path):
                pass

        fake_pypdf = types.ModuleType("pypdf")
        fake_pypdf.PdfReader = FakeReader
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            (data_dir / "guide.pdf").write_bytes(b"test pdf")
            with (
                patch.object(b_living, "LIVING_DATA_DIR", data_dir),
                patch.dict(sys.modules, {"pypdf": fake_pypdf}),
            ):
                evidence = b_living._search_living_evidence(
                    "기초연금 신청", None
                )

        self.assertEqual([item["pdf_page"] for item in evidence], [1, 2])
        self.assertTrue(all(item["title"] == "PDF 복지 안내" for item in evidence))
        self.assertTrue(all(item["publisher"] == "공공기관" for item in evidence))

    def test_injected_index_is_normalized_to_living_evidence(self):
        class FakeIndex:
            def similarity_search(self, _query, k):
                self.requested_k = k
                return [
                    {
                        "page_content": "검색된 원문",
                        "metadata": {
                            "title": "문서",
                            "publisher": "기관",
                            "url": "https://example.invalid/doc",
                            "pdf_page": 3,
                        },
                    }
                ]

        index = FakeIndex()
        evidence = b_living._search_living_evidence("검색어", index)
        self.assertEqual(index.requested_k, 4)
        self.assertEqual(
            evidence,
            [
                {
                    "content": "검색된 원문",
                    "title": "문서",
                    "publisher": "기관",
                    "url": "https://example.invalid/doc",
                    "pdf_page": 3,
                }
            ],
        )

    def test_search_failure_returns_tool_error_without_calling_llm(self):
        class BrokenIndex:
            def similarity_search(self, _query, k):
                raise OSError("index unavailable")

        with patch.object(
            b_living,
            "_ollama_chat",
            side_effect=AssertionError("LLM must not run after search failure"),
        ):
            result = b_living.run_living_support("기초연금 안내", index=BrokenIndex())

        self.assertEqual(result["status"], "tool_error")
        self.assertEqual(result["route"], "error")
        self.assertIsNone(result["proposed_action"])
        self.assertEqual(result["living_evidence"], [])

    def test_pdf_dependency_failure_is_reported_as_tool_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            (data_dir / "guide.pdf").write_bytes(b"test pdf")
            with (
                patch.object(b_living, "LIVING_DATA_DIR", data_dir),
                patch.dict(sys.modules, {"pypdf": None}),
            ):
                result = b_living.run_living_support("기초연금 안내")

        self.assertEqual(result["status"], "tool_error")
        self.assertEqual(result["route"], "error")
        self.assertEqual(result["living_evidence"], [])


if __name__ == "__main__":
    unittest.main()
