from __future__ import annotations

import asyncio
import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import MethodType
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STREAMLIT_DIR = PROJECT_ROOT / "streamlit"
if str(STREAMLIT_DIR) not in sys.path:
    sys.path.insert(0, str(STREAMLIT_DIR))

import grok_utils
import job_description
import search
import components


def _make_engine() -> search.SearchEngine:
    engine = search.SearchEngine.__new__(search.SearchEngine)
    engine.model = None
    engine.index = None
    engine.resume_metadata = []
    engine.resume_metadata_by_filename = {}
    engine.chunk_metadata = []
    engine.chunk_candidates = []
    engine.chunk_term_frequencies = []
    engine.chunk_doc_lengths = []
    engine.chunk_idf = {}
    engine.chunk_avg_doc_length = 0.0
    engine.parsed_resume_map = {}
    engine.resume_term_frequencies = []
    engine.resume_doc_lengths = []
    engine.resume_idf = {}
    engine.resume_avg_doc_length = 0.0
    engine.members_df = None
    engine._member_index = {}
    engine.last_query_analysis = None
    engine.demo_mode = False
    engine.mode_label = "Live"
    engine.mode_banner = ""
    engine.retrieval_backend = "lexical-chunk"
    engine.reranker = None
    engine.reranker_loaded = False
    engine._page_count_cache = {}
    engine._last_retrieval_details = {}
    engine.strict_startup = False
    engine._startup_issues = []
    return engine


def _build_result(candidate_id: str, score: float = 0.5) -> search.ResumeResult:
    return search.ResumeResult(
        rank=0,
        filename=candidate_id,
        candidate_id=candidate_id,
        score=score,
        semantic_score=score,
        retrieval_score=score,
        must_have_coverage=0.5,
        file_path="",
        local_resume_path="",
        text_preview="preview",
        full_text="Full candidate text",
        full_name=candidate_id,
        top_evidence_chunks=[{"section_type": "experience", "score": score, "text": "Built backend services"}],
        hard_filter_status={"matched_must_have_skills": ["Python"], "matched_preferred_skills": []},
        ranking_details={"retrieval_score": score},
    )


class Option2PipelineTests(unittest.TestCase):
    def test_result_card_html_is_single_block_and_escaped(self):
        html_output = components._result_card_html(
            rank=3,
            display_name='Yi-Hsuan (Tony) Kuo <script>alert(1)</script>',
            major="Computer Science & Engineering",
            score_pct="46%",
            score_col="#eab308",
        )

        self.assertTrue(html_output.startswith('<div class="result-card">'))
        self.assertNotIn("\n", html_output)
        self.assertIn("Yi-Hsuan (Tony) Kuo &lt;script&gt;alert(1)&lt;/script&gt;", html_output)
        self.assertIn("Computer Science &amp; Engineering", html_output)
        self.assertTrue(html_output.endswith("</div>"))

    def test_strict_startup_raises_when_semantic_backend_or_reranker_missing(self):
        engine = _make_engine()
        engine.strict_startup = True
        engine.retrieval_backend = "lexical-chunk"
        engine.reranker_loaded = False
        engine._record_startup_issue("Could not import semantic retrieval dependencies")

        with self.assertRaises(RuntimeError) as ctx:
            engine._enforce_required_backends()

        message = str(ctx.exception)
        self.assertIn("startup validation failed", message.lower())
        self.assertIn("semantic retrieval backend", message.lower())
        self.assertIn("cross-encoder reranker", message.lower())
        self.assertIn("./venv/bin/streamlit run streamlit/app.py", message)

    def test_job_description_parser_extracts_standalone_company_line(self):
        parsed = job_description.parse_job_description(
            "Software Engineering Intern\nAmazon\nSeeking Python and React students"
        )

        self.assertEqual(parsed.job_title, "Software Engineering Intern")
        self.assertEqual(parsed.company, "Amazon")
        self.assertNotIn("Amazon", parsed.must_have_skills)

    def test_recruiter_overrides_take_precedence_over_parsed_jd(self):
        parsed = job_description.parse_job_description(
            "Software Engineering Intern\nUnknown Startup\nSeeking Rivian and Python experience"
        )

        overridden = job_description.apply_recruiter_overrides(
            parsed,
            recruiter_company="Rivian",
            recruiter_job_title="Software Engineer, Vehicle Systems",
        )

        self.assertEqual(overridden.company, "Rivian")
        self.assertEqual(overridden.job_title, "Software Engineer, Vehicle Systems")
        self.assertNotIn("Rivian", overridden.must_have_skills)

    def test_company_normalization_equivalence(self):
        self.assertEqual(search._normalize_company_name("Northrop Grumman Corp."), "northrop grumman")
        self.assertTrue(
            search._company_names_equivalent("Northrop Grumman", "Northrop Grumman Corporation")
        )
        self.assertFalse(search._company_names_equivalent("Northrop Grumman", "Grumman Cafe"))

    def test_page_count_and_penalty(self):
        engine = _make_engine()
        with tempfile.TemporaryDirectory() as tmp_dir:
            one_page = Path(tmp_dir) / "one.pdf"
            two_page = Path(tmp_dir) / "two.pdf"
            one_page.write_text("placeholder", encoding="utf-8")
            two_page.write_text("placeholder", encoding="utf-8")

            class _FakeDoc:
                def __init__(self, page_count: int):
                    self.page_count = page_count

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

            class _FakeFitz:
                @staticmethod
                def open(path: str):
                    if path.endswith("one.pdf"):
                        return _FakeDoc(1)
                    if path.endswith("two.pdf"):
                        return _FakeDoc(2)
                    raise RuntimeError("unreadable")

            with mock.patch.dict(sys.modules, {"fitz": _FakeFitz}):
                self.assertEqual(engine._get_resume_page_count(str(one_page)), 1)
                self.assertEqual(engine._get_resume_page_count(str(two_page)), 2)
                self.assertIsNone(engine._get_resume_page_count(str(Path(tmp_dir) / "missing.pdf")))

            one_page_result = _build_result("one")
            one_page_result.reranker_score = 0.6
            one_page_result.grok_fit_score = 0.8
            one_page_result.grok_resume_quality_score = 0.8
            one_page_result.page_count = 1

            two_page_result = _build_result("two")
            two_page_result.reranker_score = 0.6
            two_page_result.grok_fit_score = 0.8
            two_page_result.grok_resume_quality_score = 0.8
            two_page_result.page_count = 2

            score_delta = engine._compute_final_score(one_page_result) - engine._compute_final_score(two_page_result)
            self.assertAlmostEqual(score_delta, 0.04, places=6)

    def test_search_job_description_uses_larger_candidate_pool_before_final_cut(self):
        engine = _make_engine()
        engine.reranker_loaded = True
        engine.reranker = object()

        seen: dict[str, int] = {}

        def fake_retrieve(self, query, parsed, top_k):
            return [search.ChunkHit(candidate_id="candidate", chunk_id="1", section_type="experience", text="x", score=0.5, source="ds3_members")]

        def fake_aggregate(self, chunk_hits, parsed, top_k, min_score, grad_year_filter, major_filter):
            return [_build_result(f"candidate_{idx}", score=0.4) for idx in range(top_k)]

        def fake_rerank(self, query, results, parsed=None):
            seen["rerank_pool"] = len(results)
            return results

        def fake_apply_grok(self, query, results, parsed, top_n, api_key=None, final_top_k=None, progress_callback=None):
            seen["grok_top_n"] = top_n
            return results

        engine._retrieve_chunks = MethodType(fake_retrieve, engine)
        engine._aggregate_chunk_hits = MethodType(fake_aggregate, engine)
        engine._rerank = MethodType(fake_rerank, engine)
        engine._apply_grok_scores = MethodType(fake_apply_grok, engine)

        results = engine._search_job_description(
            query="Northrop Grumman is seeking a software engineering intern with Python.",
            top_k=10,
            min_score=0.0,
            grad_year_filter=None,
            major_filter=None,
        )

        self.assertEqual(seen["rerank_pool"], 30)
        self.assertEqual(seen["grok_top_n"], 10)
        self.assertEqual(len(results), 10)

    def test_search_job_description_uses_recruiter_company_override(self):
        engine = _make_engine()
        seen: dict[str, str] = {}

        def fake_retrieve(self, query, parsed, top_k):
            seen["company"] = parsed.company
            seen["job_title"] = parsed.job_title
            return []

        def fake_aggregate(self, chunk_hits, parsed, top_k, min_score, grad_year_filter, major_filter):
            return []

        def fake_apply_grok(self, query, results, parsed, top_n, api_key=None, final_top_k=None, progress_callback=None):
            return results

        engine._retrieve_chunks = MethodType(fake_retrieve, engine)
        engine._aggregate_chunk_hits = MethodType(fake_aggregate, engine)
        engine._apply_grok_scores = MethodType(fake_apply_grok, engine)

        engine._search_job_description(
            query="Software Engineering Intern\nUnknown Startup\nSeeking Python experience.",
            top_k=10,
            min_score=0.0,
            grad_year_filter=None,
            major_filter=None,
            recruiter_company="Rivian",
            recruiter_job_title="Vehicle Software Intern",
        )

        self.assertEqual(seen["company"], "Rivian")
        self.assertEqual(seen["job_title"], "Vehicle Software Intern")

    def test_structured_retrieval_query_uses_company_title_and_skills(self):
        engine = _make_engine()
        parsed = job_description.apply_recruiter_overrides(
            job_description.parse_job_description(
                "Unknown title\nUnknown company\nSeeking Python and React experience."
            ),
            recruiter_company="Rivian",
            recruiter_job_title="Software Engineering Intern",
        )

        retrieval_query = engine._build_structured_retrieval_query("raw jd text", parsed)

        self.assertIn("Rivian", retrieval_query)
        self.assertIn("Software Engineering Intern", retrieval_query)
        self.assertIn("Python", retrieval_query)
        self.assertIn("React", retrieval_query)

    def test_candidate_profile_extracts_social_links_from_contact_when_metadata_missing(self):
        engine = _make_engine()
        engine.parsed_resume_map = {
            "member_resume_yi-hsuan_kuo.pdf": {
                "contact": (
                    "Yi-Hsuan (Tony) Kuo\n"
                    "La Jolla | y3kuo@ucsd.edu | https://github.com/YihsuanKuo | linkedin.com/in/yi-hsuan-kuo"
                ),
                "summary": "",
                "skills": [],
                "canonical_skills": [],
                "experience": [],
                "education": [],
                "projects": [],
                "metadata": {},
                "source": "ds3_members",
            }
        }
        engine.resume_metadata_by_filename = {
            "member_resume_yi-hsuan_kuo.pdf": {
                "filename": "member_resume_yi-hsuan_kuo.pdf",
                "full_name": "",
                "linkedin": "",
                "github": "",
                "resume_link": "",
                "source": "ds3_members",
                "file_path": "",
                "text": "",
            }
        }

        profile = engine._get_candidate_profile("member_resume_yi-hsuan_kuo.pdf")

        self.assertEqual(profile["full_name"], "Yi-Hsuan (Tony) Kuo")
        self.assertEqual(profile["github"], "https://github.com/YihsuanKuo")
        self.assertEqual(profile["linkedin"], "https://linkedin.com/in/yi-hsuan-kuo")

    def test_hybrid_retrieval_combines_semantic_and_lexical_hits(self):
        engine = _make_engine()
        engine.retrieval_backend = "semantic-chunk"
        engine.index = object()
        engine.model = object()
        engine.chunk_metadata = [{"candidate_id": "semantic.pdf"}]
        engine.chunk_candidates = [{"candidate_id": "lexical.pdf"}]

        parsed = job_description.apply_recruiter_overrides(
            job_description.parse_job_description("Software Engineering Intern\nRivian\nPython and React"),
            recruiter_company="Rivian",
            recruiter_job_title="Software Engineering Intern",
        )

        engine._semantic_chunk_search = MethodType(
            lambda self, query, top_k: [
                search.ChunkHit("semantic.pdf", "semantic-1", "experience", "semantic hit", 0.8, "ds3_members", "semantic")
            ],
            engine,
        )
        engine._lexical_chunk_search = MethodType(
            lambda self, query, parsed, top_k: [
                search.ChunkHit("lexical.pdf", "lexical-1", "experience", "lexical hit", 0.9, "ds3_members", "lexical")
            ],
            engine,
        )
        engine._build_company_rescue_hits = MethodType(lambda self, parsed, existing_candidate_ids, limit=search.COMPANY_RESCUE_LIMIT: [], engine)

        hits = engine._retrieve_chunks("jd", parsed, top_k=10)

        self.assertEqual({hit.candidate_id for hit in hits}, {"semantic.pdf", "lexical.pdf"})
        self.assertEqual(engine._last_retrieval_details["semantic_chunks"], 1)
        self.assertEqual(engine._last_retrieval_details["lexical_chunks"], 1)
        self.assertEqual(engine._last_retrieval_details["hybrid_unique_chunks"], 2)

    def test_company_rescue_adds_missing_same_company_candidate(self):
        engine = _make_engine()
        engine.parsed_resume_map = {
            "kaii.pdf": {
                "candidate_id": "kaii.pdf",
                "source": "ds3_members",
                "metadata": {"full_name": "Kaii Bijlani", "major": "Data Science", "graduation_year": "2027"},
                "experience": [
                    {
                        "title": "Software Engineering Intern",
                        "company": "Rivian",
                        "company_normalized": "rivian",
                        "dates": ["Jun 2025", "Sep 2025"],
                        "bullets": ["Built Databricks pipelines in Python."],
                        "technologies": ["Python", "Databricks"],
                        "raw_text": "Software Engineering Intern\nRivian\nBuilt Databricks pipelines in Python.",
                    }
                ],
                "projects": [],
                "education": [],
                "summary": "",
                "canonical_skills": ["Python", "React", "TypeScript", "AWS"],
            }
        }
        engine.resume_metadata_by_filename = {
            "kaii.pdf": {"filename": "kaii.pdf", "file_path": "", "source": "ds3_members", "text": "Kaii resume"}
        }
        parsed = job_description.apply_recruiter_overrides(
            job_description.parse_job_description("Software Engineering Intern\nRivian\nPython React TypeScript AWS"),
            recruiter_company="Rivian",
            recruiter_job_title="Software Engineering Intern",
        )

        rescue_hits = engine._build_company_rescue_hits(parsed, existing_candidate_ids=set())

        self.assertEqual(len(rescue_hits), 1)
        self.assertEqual(rescue_hits[0].candidate_id, "kaii.pdf")
        self.assertEqual(rescue_hits[0].retrieval_source, "company_rescue")
        self.assertGreater(rescue_hits[0].score, 0.58)

    def test_same_company_candidate_receives_stronger_retrieval_score(self):
        engine = _make_engine()
        engine.resume_metadata_by_filename = {
            "same.pdf": {"filename": "same.pdf", "file_path": "", "text": "Northrop Grumman experience", "source": "ds3_members"},
            "other.pdf": {"filename": "other.pdf", "file_path": "", "text": "Other company experience", "source": "ds3_members"},
        }
        engine.parsed_resume_map = {
            "same.pdf": {
                "candidate_id": "same.pdf",
                "source": "ds3_members",
                "file_path": "",
                "metadata": {"full_name": "Same Company", "major": "Computer Science", "graduation_year": "2027"},
                "skills": ["Python"],
                "summary": "Backend builder",
                "education": [],
                "projects": [],
                "experience": [
                    {
                        "raw_text": (
                            "Software Engineering Intern Jun 2025 - Aug 2025\n"
                            "Northrop Grumman Corporation San Diego, CA\n"
                            "Built internal tools in Python."
                        ),
                        "dates": ["Jun 2025", "Aug 2025"],
                    }
                ],
            },
            "other.pdf": {
                "candidate_id": "other.pdf",
                "source": "ds3_members",
                "file_path": "",
                "metadata": {"full_name": "Other Company", "major": "Computer Science", "graduation_year": "2027"},
                "skills": ["Python"],
                "summary": "Backend builder",
                "education": [],
                "projects": [],
                "experience": [
                    {
                        "raw_text": (
                            "Software Engineering Intern Jun 2025 - Aug 2025\n"
                            "Another Aerospace Company San Diego, CA\n"
                            "Built internal tools in Python."
                        ),
                        "dates": ["Jun 2025", "Aug 2025"],
                    }
                ],
            },
        }

        parsed = job_description.parse_job_description(
            "Northrop Grumman is seeking a software engineering intern with Python experience."
        )
        hits = [
            search.ChunkHit("same.pdf", "1", "experience", "Built internal tools in Python.", 0.75, "ds3_members"),
            search.ChunkHit("other.pdf", "2", "experience", "Built internal tools in Python.", 0.75, "ds3_members"),
        ]

        results = engine._aggregate_chunk_hits(
            chunk_hits=hits,
            parsed=parsed,
            top_k=10,
            min_score=0.0,
            grad_year_filter=None,
            major_filter=None,
        )

        self.assertEqual(results[0].candidate_id, "same.pdf")
        self.assertEqual(results[0].company_match_status, "exact_experience_match")
        self.assertNotEqual(results[0].retrieval_score, results[1].retrieval_score)
        self.assertIn(results[0].ranking_details["retrieval_source"], {"unknown", "semantic", "lexical", "hybrid", "company_rescue"})

    def test_grok_scores_only_top_ten_candidates(self):
        engine = _make_engine()
        engine._get_candidate_profile = MethodType(
            lambda self, candidate_id: {
                "summary": "Student builder",
                "skills": ["Python", "FastAPI"],
                "education_entries": [],
                "experience_entries": [],
                "project_entries": [],
                "estimated_years_experience": 1.0,
                "employer_names": [],
                "combined_text": "Built services",
            },
            engine,
        )

        results = [_build_result(f"candidate_{idx}", score=0.5 - (idx * 0.001)) for idx in range(15)]
        called_candidates: list[str] = []
        parsed = job_description.parse_job_description("Northrop Grumman seeks a Python intern.")

        with mock.patch.object(
            search,
            "assess_candidate_packet_with_grok",
            side_effect=lambda job_description, parsed_requirements, candidate_packet, api_key=None: (
                called_candidates.append(candidate_packet["candidate_id"])
                or {
                    "status": "ok",
                    "qualification_match_score": 8,
                    "company_relevance_score": 7,
                    "experience_relevance_score": 8,
                    "bullet_quality_score": 7,
                    "project_strength_score": 7,
                    "resume_quality_score": 7,
                    "matched_requirements": ["Python"],
                    "missing_requirements": [],
                    "weakness_flags": [],
                    "summary": "Strong fit.",
                }
            ),
        ):
            scored = engine._apply_grok_scores("Northrop Grumman seeks a Python intern.", results, parsed, top_n=10)

        self.assertEqual(len(called_candidates), 10)
        skipped = [result for result in scored if result.grok_status == "skipped"]
        self.assertEqual(len(skipped), 5)

    def test_grok_fallback_unavailable_and_error(self):
        grok_utils._ASSESSMENT_CACHE.clear()
        with mock.patch.dict("os.environ", {}, clear=True):
            unavailable = grok_utils.assess_candidate_packet_with_grok(
                "jd",
                {},
                {"candidate_id": "candidate_unavailable"},
            )
        self.assertEqual(unavailable["status"], "unavailable")

        grok_utils._ASSESSMENT_CACHE.clear()

        class _FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"choices": [{"message": {"content": "not-json"}}]}

        with mock.patch.dict("os.environ", {"XAI_API_KEY": "test-key"}, clear=True):
            with mock.patch.object(grok_utils.requests, "post", return_value=_FakeResponse()):
                errored = grok_utils.assess_candidate_packet_with_grok(
                    "jd",
                    {},
                    {"candidate_id": "candidate_error"},
                )
        self.assertEqual(errored["status"], "error")

    def test_api_response_includes_new_grok_fields(self):
        with mock.patch.object(search.SearchEngine, "_load", lambda self: None):
            with mock.patch.object(search.SearchEngine, "_enforce_required_backends", lambda self: None):
                if "api" in sys.modules:
                    del sys.modules["api"]
                api = importlib.import_module("api")

        api.engine.last_query_analysis = {"job_title": "Intern"}
        seen: dict[str, str] = {}

        def fake_search(**kwargs):
            seen["recruiter_company"] = kwargs.get("recruiter_company")
            seen["recruiter_job_title"] = kwargs.get("recruiter_job_title")
            return [
                search.ResumeResult(
                    rank=1,
                    filename="candidate.pdf",
                    candidate_id="candidate.pdf",
                    score=0.88,
                    semantic_score=0.77,
                    file_path="/tmp/candidate.pdf",
                    local_resume_path="/tmp/candidate.pdf",
                    text_preview="Preview",
                    full_name="Candidate",
                    major="Computer Science",
                    graduation_year="2027",
                    page_count=2,
                    company_match_status="exact_experience_match",
                    grok_status="ok",
                    grok_fit_score=0.83,
                    grok_resume_quality_score=0.71,
                    grok_summary="Strong fit.",
                    grok_matched_requirements=["Python"],
                    grok_missing_requirements=["unclear: citizenship"],
                    grok_weakness_flags=["weak_or_unquantified_bullets"],
                )
            ]

        api.engine.search = fake_search

        payload = asyncio.run(
            api.search_resumes(
                api.SearchRequest(
                    query="python intern",
                    input_mode="Job Description",
                    recruiter_company="Rivian",
                    recruiter_job_title="Vehicle Software Intern",
                )
            )
        )

        result = payload["results"][0]
        self.assertEqual(result["page_count"], 2)
        self.assertEqual(result["company_match_status"], "exact_experience_match")
        self.assertEqual(result["grok_status"], "ok")
        self.assertEqual(result["grok_matched_requirements"], ["Python"])
        self.assertEqual(result["grok_missing_requirements"], ["unclear: citizenship"])
        self.assertEqual(seen["recruiter_company"], "Rivian")
        self.assertEqual(seen["recruiter_job_title"], "Vehicle Software Intern")

    def test_dedupe_results_prefers_highest_scoring_duplicate_identity(self):
        engine = _make_engine()
        low = _build_result("resume_a.pdf", 0.61)
        low.full_name = "Viet Minh Hieu Nguyen"
        low.linkedin = "https://www.linkedin.com/in/vietminhhieunguyen/"
        high = _build_result("resume_b.pdf", 0.82)
        high.full_name = "Hieu Nguyen"
        high.linkedin = "https://www.linkedin.com/in/vietminhhieunguyen/"

        deduped = engine._dedupe_results([low, high])

        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0].candidate_id, "resume_b.pdf")
        self.assertEqual(deduped[0].ranking_details["deduped_duplicate_count"], 2)

    def test_dedupe_results_does_not_collapse_single_token_names(self):
        engine = _make_engine()
        first = _build_result("daniel_a.pdf", 0.7)
        first.full_name = "Daniel"
        first.graduation_year = "2027"
        first.major = "Computer Science"
        second = _build_result("daniel_b.pdf", 0.69)
        second.full_name = "Daniel"
        second.graduation_year = "2027"
        second.major = "Computer Science"

        deduped = engine._dedupe_results([first, second])

        self.assertEqual(len(deduped), 2)

    def test_candidate_profile_falls_back_to_contact_name_when_metadata_missing(self):
        engine = _make_engine()
        engine.resume_metadata_by_filename = {
            "board_resume_1.pdf": {
                "filename": "board_resume_1.pdf",
                "file_path": "",
                "text": "resume text",
                "source": "ds3_board",
                "full_name": "",
            }
        }
        engine.parsed_resume_map = {
            "board_resume_1.pdf": {
                "candidate_id": "board_resume_1.pdf",
                "source": "ds3_board",
                "file_path": "",
                "metadata": {},
                "contact": "Mohak A. Prakash\nmprakash@ucsd.edu",
                "summary": "",
                "education": [],
                "experience": [],
                "projects": [],
                "skills": [],
                "canonical_skills": [],
            }
        }

        profile = engine._get_candidate_profile("board_resume_1.pdf")

        self.assertEqual(profile["full_name"], "Mohak A. Prakash")


if __name__ == "__main__":
    unittest.main()
