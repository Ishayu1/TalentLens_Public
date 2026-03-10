from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
STREAMLIT_DIR = PROJECT_ROOT / "streamlit"
if str(STREAMLIT_DIR) not in sys.path:
    sys.path.insert(0, str(STREAMLIT_DIR))

import pipeline.ds3_rebuild as ds3_rebuild
import job_description
import search


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
    return engine


class DS3RebuildTests(unittest.TestCase):
    def test_skill_normalization_cleans_dirty_tokens(self):
        raw_skills = ds3_rebuild._split_skill_tokens(
            "Frameworks/Libraries: PyTorch, Node js, Next_js, Sckit Learn, NumPy"
        )
        canonical, dirty = ds3_rebuild._normalize_skill_tokens(
            "Frameworks/Libraries: PyTorch, Node js, Next_js, Sckit Learn, NumPy",
            raw_skills,
        )

        self.assertIn("PyTorch", canonical)
        self.assertIn("Node.js", canonical)
        self.assertIn("Next.js", canonical)
        self.assertIn("Scikit-learn", canonical)
        self.assertIn("NumPy", canonical)
        self.assertTrue(any("Node js" in token for token in dirty))
        self.assertTrue(any("Sckit Learn" in token for token in dirty))

    def test_experience_parser_splits_multiple_roles_and_extracts_companies(self):
        section_text = """
Software Engineering Intern | Northrop Grumman Corporation | San Diego, CA | Jun 2025 - Aug 2025
- Built Python automation for flight test data processing on AWS.

Software Engineer Intern | Amazon | Seattle, WA | Jun 2024 - Sep 2024
- Developed internal tooling in Java and React to reduce onboarding time.

Research Assistant | UC San Diego | La Jolla, CA | Sep 2023 - Present
- Trained PyTorch models for computer vision experiments.
"""
        entries = ds3_rebuild._parse_experience_entries(section_text)

        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0]["title"], "Software Engineering Intern")
        self.assertEqual(entries[0]["company"], "Northrop Grumman Corporation")
        self.assertEqual(entries[1]["company_normalized"], "amazon")
        self.assertIn("Python", entries[0]["technologies"])
        self.assertIn("AWS", entries[0]["technologies"])
        self.assertIn("Java", entries[1]["technologies"])
        self.assertIn("React", entries[1]["technologies"])
        self.assertIn("PyTorch", entries[2]["technologies"])

    def test_project_parser_splits_multiple_projects(self):
        section_text = """
Smart Scheduler | Python, FastAPI, PostgreSQL | Jan 2025 - Mar 2025
- Built a scheduling system with FastAPI and PostgreSQL.
- Deployed the service and improved response time by 35%.

Campus Navigation App | React Native, Firebase | Sep 2024 - Dec 2024
- Built mobile navigation features for campus events.
- Added Firebase auth and analytics for 500+ users.
"""
        entries = ds3_rebuild._parse_project_entries(section_text)

        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["name"], "Smart Scheduler")
        self.assertIn("FastAPI", entries[0]["technologies"])
        self.assertIn("PostgreSQL", entries[0]["technologies"])
        self.assertEqual(entries[1]["name"], "Campus Navigation App")
        self.assertIn("React Native", entries[1]["technologies"])
        self.assertIn("Firebase", entries[1]["technologies"])

    def test_company_normalization_is_stable(self):
        self.assertEqual(ds3_rebuild._normalize_company_name("Northrop Grumman Corp."), "northrop grumman")
        self.assertEqual(ds3_rebuild._normalize_company_name("Northrop Grumman Corporation"), "northrop grumman")

    def test_header_parser_handles_company_then_title_layout(self):
        title, company, dates, location = ds3_rebuild._parse_header_fields(
            [
                "Rivian June 2025 September 2025",
                "Software Engineering Intern Irvine, CA",
            ]
        )

        self.assertEqual(title, "Software Engineering Intern")
        self.assertEqual(company, "Rivian")
        self.assertEqual(dates, ["June 2025", "September 2025"])
        self.assertEqual(location, "Irvine, CA")

    def test_ambiguous_experience_entry_is_flagged_for_entry_repair(self):
        entries = ds3_rebuild._parse_experience_entries(
            """
Rivian June 2025 September 2025
Software Engineering Intern Irvine, CA
- Built Databricks pipelines for vehicle telemetry.
"""
        )

        self.assertEqual(len(entries), 1)
        self.assertIn("company_then_title_layout", entries[0]["entry_parse_warnings"])
        self.assertLess(entries[0]["entry_parse_confidence"], ds3_rebuild.ENTRY_REPAIR_CONFIDENCE_THRESHOLD)
        self.assertTrue(ds3_rebuild._entry_needs_grok_repair(entries[0]))

    def test_well_structured_experience_entry_skips_entry_repair(self):
        entries = ds3_rebuild._parse_experience_entries(
            """
Software Engineering Intern | Amazon | Seattle, WA | Jun 2024 - Sep 2024
- Built internal tooling in Java and React.
"""
        )

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["entry_parse_warnings"], [])
        self.assertGreaterEqual(entries[0]["entry_parse_confidence"], ds3_rebuild.ENTRY_REPAIR_CONFIDENCE_THRESHOLD)
        self.assertFalse(ds3_rebuild._entry_needs_grok_repair(entries[0]))

    def test_grok_enrichment_gating_skips_strong_parse_and_flags_weak_parse(self):
        strong = {
            "canonical_skills": ["Python", "AWS", "FastAPI", "PostgreSQL", "Docker"],
            "experience": [
                {"company_normalized": "northrop grumman"},
                {"company_normalized": "amazon"},
            ],
            "projects": [{"name": "A"}, {"name": "B"}],
            "sections_raw": {"experience": "short", "projects": "short"},
            "parse_warnings": [],
        }
        weak = {
            "canonical_skills": ["Python"],
            "experience": [{"company_normalized": ""}],
            "projects": [],
            "sections_raw": {
                "experience": "Very long experience block " * 20,
                "projects": "Very long project block " * 20,
            },
            "parse_warnings": ["experience_not_split"],
        }

        should_enrich_strong, reasons_strong = ds3_rebuild._should_enrich_with_grok(strong)
        should_enrich_weak, reasons_weak = ds3_rebuild._should_enrich_with_grok(weak)

        self.assertFalse(should_enrich_strong)
        self.assertEqual(reasons_strong, [])
        self.assertTrue(should_enrich_weak)
        self.assertIn("low_skill_count", reasons_weak)
        self.assertIn("undersplit_experience", reasons_weak)
        self.assertIn("undersplit_projects", reasons_weak)
        self.assertIn("missing_company", reasons_weak)

    def test_merge_grok_entry_repair_updates_experience_fields(self):
        entry = {
            "title": "Rivian",
            "company": "Software Engineering",
            "company_normalized": "software engineering",
            "dates": ["June 2025", "September 2025"],
            "location": "Intern Irvine, CA",
            "bullets": ["Built Databricks pipelines for vehicle telemetry."],
            "technologies": ["Spark"],
            "raw_header": "Rivian June 2025 September 2025 | Software Engineering Intern Irvine, CA",
            "raw_text": "Rivian June 2025 September 2025\nSoftware Engineering Intern Irvine, CA\nBuilt Databricks pipelines for vehicle telemetry.",
            "entry_parse_warnings": ["title_company_inversion", "title_like_location"],
            "entry_parse_confidence": 0.32,
            "repair_source": "deterministic",
        }
        payload = {
            "title": "Software Engineering Intern",
            "company": "Rivian",
            "company_normalized": "rivian",
            "dates": ["June 2025", "September 2025"],
            "location": "Irvine, CA",
            "bullets": ["Built Databricks pipelines for vehicle telemetry."],
            "technologies": ["Databricks", "Spark"],
            "confidence": 0.93,
            "reason": "First header line is employer and second line is role plus location.",
        }

        repaired = ds3_rebuild._merge_grok_entry_repair(entry, payload, "experience")

        self.assertEqual(repaired["title"], "Software Engineering Intern")
        self.assertEqual(repaired["company"], "Rivian")
        self.assertEqual(repaired["company_normalized"], "rivian")
        self.assertEqual(repaired["location"], "Irvine, CA")
        self.assertEqual(repaired["repair_source"], "grok_entry_repair")
        self.assertGreaterEqual(repaired["entry_parse_confidence"], 0.93)
        self.assertIn("Databricks", repaired["technologies"])
        self.assertNotIn("title_company_inversion", repaired["entry_parse_warnings"])

    def test_merge_grok_entry_repair_updates_project_fields(self):
        entry = {
            "name": "Project",
            "dates": ["Jan 2025", "Mar 2025"],
            "location": "",
            "bullets": ["Built an analytics dashboard."],
            "technologies": ["Python"],
            "raw_header": "Campus Dashboard | Python, Streamlit | Jan 2025 - Mar 2025",
            "raw_text": "Campus Dashboard | Python, Streamlit | Jan 2025 - Mar 2025\nBuilt an analytics dashboard.",
            "entry_parse_warnings": ["project_name_too_generic"],
            "entry_parse_confidence": 0.54,
            "repair_source": "deterministic",
        }
        payload = {
            "name": "Campus Dashboard",
            "dates": ["Jan 2025", "Mar 2025"],
            "location": "",
            "bullets": ["Built an analytics dashboard."],
            "technologies": ["Python", "Streamlit"],
            "confidence": 0.9,
            "reason": "Header clearly identifies the project name before the technology list.",
        }

        repaired = ds3_rebuild._merge_grok_entry_repair(entry, payload, "projects")

        self.assertEqual(repaired["name"], "Campus Dashboard")
        self.assertEqual(repaired["repair_source"], "grok_entry_repair")
        self.assertGreaterEqual(repaired["entry_parse_confidence"], 0.9)
        self.assertIn("Streamlit", repaired["technologies"])
        self.assertNotIn("project_name_too_generic", repaired["entry_parse_warnings"])

    def test_entry_repair_fallback_keeps_entry_and_adds_warning(self):
        parsed_by_id = {
            "cand.pdf": {
                "candidate_id": "cand.pdf",
                "experience": [
                    {
                        "title": "Rivian",
                        "company": "Software Engineering",
                        "company_normalized": "software engineering",
                        "dates": ["June 2025", "September 2025"],
                        "location": "Intern Irvine, CA",
                        "bullets": ["Built Databricks pipelines."],
                        "technologies": ["Spark"],
                        "raw_header": "Rivian June 2025 September 2025 | Software Engineering Intern Irvine, CA",
                        "raw_text": "Rivian June 2025 September 2025\nSoftware Engineering Intern Irvine, CA\nBuilt Databricks pipelines.",
                        "entry_parse_warnings": ["title_company_inversion", "company_then_title_layout"],
                        "entry_parse_confidence": 0.31,
                        "repair_source": "deterministic",
                    }
                ],
                "projects": [],
                "canonical_skills": ["Python"],
                "parse_warnings": [],
                "parse_confidence": 0.42,
            }
        }
        extracted_by_id = {"cand.pdf": {"filename": "cand.pdf", "text": "resume text"}}

        with mock.patch.object(ds3_rebuild, "_grok_api_key", return_value="key"):
            with mock.patch.object(ds3_rebuild, "_call_non_reasoning_grok", side_effect=RuntimeError("boom")):
                repaired, stats = ds3_rebuild._repair_ambiguous_entries_with_grok(
                    extracted_by_id,
                    parsed_by_id,
                    use_grok="always",
                    grok_workers=1,
                )

        entry = repaired["cand.pdf"]["experience"][0]
        self.assertEqual(stats.entry_grok_candidates, 1)
        self.assertEqual(stats.entry_grok_repaired, 0)
        self.assertEqual(entry["repair_source"], "deterministic")
        self.assertIn("grok_entry_repair_failed:RuntimeError", entry["entry_parse_warnings"])

    def test_unflagged_entries_skip_grok_entry_repair(self):
        parsed_by_id = {
            "cand.pdf": {
                "candidate_id": "cand.pdf",
                "experience": [
                    {
                        "title": "Software Engineer Intern",
                        "company": "Amazon",
                        "company_normalized": "amazon",
                        "dates": ["Jun 2024", "Sep 2024"],
                        "location": "Seattle, WA",
                        "bullets": ["Built internal tooling."],
                        "technologies": ["Java"],
                        "raw_header": "Software Engineer Intern | Amazon | Seattle, WA | Jun 2024 - Sep 2024",
                        "raw_text": "Software Engineer Intern | Amazon | Seattle, WA | Jun 2024 - Sep 2024\nBuilt internal tooling.",
                        "entry_parse_warnings": [],
                        "entry_parse_confidence": 0.94,
                        "repair_source": "deterministic",
                    }
                ],
                "projects": [],
                "canonical_skills": ["Java"],
                "parse_warnings": [],
                "parse_confidence": 0.78,
            }
        }
        extracted_by_id = {"cand.pdf": {"filename": "cand.pdf", "text": "resume text"}}

        with mock.patch.object(ds3_rebuild, "_grok_api_key", return_value="key"):
            with mock.patch.object(ds3_rebuild, "_call_non_reasoning_grok", side_effect=AssertionError("should not run")):
                repaired, stats = ds3_rebuild._repair_ambiguous_entries_with_grok(
                    extracted_by_id,
                    parsed_by_id,
                    use_grok="always",
                    grok_workers=1,
                )

        self.assertEqual(stats.entry_grok_candidates, 0)
        self.assertEqual(stats.entry_grok_repaired, 0)
        self.assertEqual(repaired["cand.pdf"]["experience"][0]["repair_source"], "deterministic")

    def test_merge_grok_enrichment_preserves_schema_and_technologies(self):
        parsed = {
            "candidate_id": "cand.pdf",
            "canonical_skills": ["Python"],
            "technology_sources": {"skills_section": ["Python"], "experience": [], "projects": []},
            "experience": [],
            "projects": [],
            "parse_warnings": [],
        }
        payload = {
            "canonical_skills": ["Node.js", "Scikit-learn"],
            "experience_entries": [
                {
                    "title": "Software Engineer Intern",
                    "company": "Northrop Grumman Corporation",
                    "dates": ["Jun 2025", "Aug 2025"],
                    "location": "San Diego, CA",
                    "bullets": ["Built Python services on AWS."],
                    "technologies": ["Python", "AWS"],
                }
            ],
            "project_entries": [
                {
                    "name": "Smart Scheduler",
                    "dates": ["Jan 2025", "Mar 2025"],
                    "location": "",
                    "bullets": ["Built scheduling API with FastAPI and PostgreSQL."],
                    "technologies": ["FastAPI", "PostgreSQL"],
                }
            ],
            "summary_flags": ["underspecified_projects"],
            "parse_warnings": ["repaired_company_from_context"],
        }

        enriched = ds3_rebuild._merge_grok_enrichment(parsed, payload)

        self.assertIn("Node.js", enriched["canonical_skills"])
        self.assertIn("Scikit-learn", enriched["canonical_skills"])
        self.assertIn("FastAPI", enriched["canonical_skills"])
        self.assertEqual(enriched["experience"][0]["company_normalized"], "northrop grumman")
        self.assertEqual(enriched["experience"][0]["repair_source"], "grok_resume_enrichment")
        self.assertIn("AWS", enriched["technology_sources"]["experience"])
        self.assertIn("PostgreSQL", enriched["technology_sources"]["projects"])
        self.assertIn("underspecified_projects", enriched["parse_warnings"])
        self.assertEqual(enriched["parser_version"], ds3_rebuild.PARSER_VERSION)

    def test_chunk_builder_emits_company_and_technology_metadata(self):
        parsed = {
            "candidate_id": "cand.pdf",
            "source": "ds3_members",
            "contact": "cand@example.com",
            "summary": "Builder",
            "education": [],
            "experience": [
                {
                    "title": "Software Engineer Intern",
                    "company": "Northrop Grumman",
                    "company_normalized": "northrop grumman",
                    "dates": ["Jun 2025", "Aug 2025"],
                    "bullets": ["Built Python services on AWS."],
                    "technologies": ["Python", "AWS"],
                    "raw_header": "Software Engineer Intern | Northrop Grumman",
                }
            ],
            "projects": [
                {
                    "name": "Smart Scheduler",
                    "dates": ["Jan 2025", "Mar 2025"],
                    "bullets": ["Built FastAPI service with PostgreSQL."],
                    "technologies": ["FastAPI", "PostgreSQL"],
                    "raw_header": "Smart Scheduler",
                }
            ],
            "canonical_skills": ["Python", "AWS", "FastAPI", "PostgreSQL"],
            "certifications": [],
            "sections_raw": {},
        }

        chunks = ds3_rebuild._build_chunks(parsed)
        experience_chunk = next(chunk for chunk in chunks if chunk["section_type"] == "experience")
        project_chunk = next(chunk for chunk in chunks if chunk["section_type"] == "projects")
        skills_chunk = next(chunk for chunk in chunks if chunk["section_type"] == "skills")

        self.assertEqual(experience_chunk["metadata"]["company_normalized"], "northrop grumman")
        self.assertEqual(experience_chunk["metadata"]["technologies"], ["Python", "AWS"])
        self.assertIn("Technologies: Python, AWS", experience_chunk["text"])
        self.assertEqual(project_chunk["metadata"]["technologies"], ["FastAPI", "PostgreSQL"])
        self.assertEqual(skills_chunk["metadata"]["technologies"], ["Python", "AWS", "FastAPI", "PostgreSQL"])

    def test_rebuild_includes_member_and_board_sources_and_preserves_train(self):
        extracted_rows = [
            {
                "filename": "member.pdf",
                "source": "ds3_members",
                "file_path": "data/ds3/member_resumes/member.pdf",
                "text": "Skills\nPython, FastAPI\nExperience\nSoftware Engineer Intern | Northrop Grumman | San Diego, CA | Jun 2025 - Aug 2025\nBuilt FastAPI services.",
                "word_count": 42,
                "metadata": {"full_name": "Member Student"},
            },
            {
                "filename": "board.pdf",
                "source": "ds3_board",
                "file_path": "data/ds3/board_resumes/board.pdf",
                "text": "Projects\nResearch Dashboard | Python, Streamlit | Jan 2025 - Mar 2025\nBuilt Streamlit analytics app.",
                "word_count": 30,
                "metadata": {"full_name": "Board Student"},
            },
            {
                "filename": "train.pdf",
                "source": "train",
                "file_path": "data/train/train.pdf",
                "text": "train text",
                "word_count": 12,
                "metadata": {"full_name": "Train Candidate"},
            },
        ]
        existing_parsed = [{"candidate_id": "train.pdf", "source": "train", "skills": ["SQL"]}]
        existing_chunks = [{"candidate_id": "train.pdf", "source": "train", "section_type": "skills", "text": "SQL", "metadata": {}}]
        dumped: dict[Path, list[dict]] = {}

        def fake_load(path: Path):
            if path == ds3_rebuild.EXTRACTED_PATH:
                return extracted_rows
            if path == ds3_rebuild.PARSED_PATH:
                return existing_parsed
            if path == ds3_rebuild.CHUNKS_PATH:
                return existing_chunks
            return []

        def fake_dump(path: Path, rows: list[dict]):
            dumped[path] = rows

        with mock.patch.object(ds3_rebuild, "_load_json", side_effect=fake_load):
            with mock.patch.object(ds3_rebuild, "_dump_json", side_effect=fake_dump):
                with mock.patch.object(ds3_rebuild, "_encode_and_index_chunks", return_value=None):
                    with mock.patch.object(ds3_rebuild, "tqdm", side_effect=lambda iterable=None, **kwargs: iterable):
                        stats = ds3_rebuild.rebuild_ds3_artifacts(use_grok="never")

        self.assertEqual(stats.total_ds3, 2)
        parsed_sources = {row["source"] for row in dumped[ds3_rebuild.PARSED_PATH]}
        chunk_sources = {row["source"] for row in dumped[ds3_rebuild.CHUNKS_PATH]}
        member_meta_sources = {row["source"] for row in dumped[ds3_rebuild.MEMBER_RESUMES_META_PATH]}
        member_chunk_sources = {row["source"] for row in dumped[ds3_rebuild.MEMBER_CHUNKS_PATH]}
        self.assertEqual(parsed_sources, {"ds3_members", "ds3_board", "train"})
        self.assertEqual(chunk_sources, {"ds3_members", "ds3_board", "train"})
        self.assertEqual(member_meta_sources, {"ds3_members", "ds3_board"})
        self.assertEqual(member_chunk_sources, {"ds3_members", "ds3_board"})

    def test_search_loaders_accept_member_and_board_sources(self):
        parsed_rows = [
            {"candidate_id": "member.pdf", "source": "ds3_members", "skills": [], "metadata": {}},
            {"candidate_id": "board.pdf", "source": "ds3_board", "skills": [], "metadata": {}},
            {"candidate_id": "train.pdf", "source": "train", "skills": [], "metadata": {}},
        ]
        chunk_rows = [
            {"candidate_id": "member.pdf", "source": "ds3_members", "section_type": "skills", "text": "Python", "metadata": {}},
            {"candidate_id": "board.pdf", "source": "ds3_board", "section_type": "skills", "text": "Streamlit", "metadata": {}},
            {"candidate_id": "train.pdf", "source": "train", "section_type": "skills", "text": "SQL", "metadata": {}},
        ]
        metadata_rows = [
            {"filename": "member.pdf", "source": "ds3_members", "text": "member"},
            {"filename": "board.pdf", "source": "ds3_board", "text": "board"},
            {"filename": "train.pdf", "source": "train", "text": "train"},
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            parsed_path = tmp / "resumes_parsed.json"
            chunks_path = tmp / "resume_chunks.json"
            meta_path = tmp / "member_resumes_metadata.json"
            parsed_path.write_text(json.dumps(parsed_rows), encoding="utf-8")
            chunks_path.write_text(json.dumps(chunk_rows), encoding="utf-8")
            meta_path.write_text(json.dumps(metadata_rows), encoding="utf-8")

            engine = _make_engine()
            with mock.patch.object(search, "PARSED_RESUMES_PATH", parsed_path):
                with mock.patch.object(search, "RESUME_CHUNKS_PATH", chunks_path):
                    with mock.patch.object(search, "METADATA_PATH", meta_path):
                        engine._load_parsed_resumes()
                        engine._load_chunk_records()
                        engine._load_resume_metadata()

        self.assertEqual(set(engine.parsed_resume_map), {"member.pdf", "board.pdf"})
        self.assertEqual({row["source"] for row in engine.chunk_candidates}, {"ds3_members", "ds3_board"})
        self.assertEqual({row["source"] for row in engine.resume_metadata}, {"ds3_members", "ds3_board"})

    def test_board_candidate_can_surface_in_search_results(self):
        engine = _make_engine()
        engine.resume_metadata_by_filename = {
            "board.pdf": {
                "filename": "board.pdf",
                "file_path": "",
                "text": "Built Streamlit analytics tools in Python.",
                "source": "ds3_board",
            }
        }
        engine.parsed_resume_map = {
            "board.pdf": {
                "candidate_id": "board.pdf",
                "source": "ds3_board",
                "file_path": "",
                "metadata": {"full_name": "Board Candidate", "major": "Data Science", "graduation_year": "2027"},
                "skills": ["Python"],
                "canonical_skills": ["Python", "Streamlit"],
                "summary": "Built dashboards",
                "education": [],
                "projects": [],
                "experience": [
                    {
                        "title": "Data Science Intern",
                        "company": "Northrop Grumman",
                        "company_normalized": "northrop grumman",
                        "dates": ["Jun 2025", "Aug 2025"],
                        "bullets": ["Built Streamlit analytics tools in Python."],
                        "technologies": ["Python", "Streamlit"],
                        "raw_text": "Data Science Intern at Northrop Grumman",
                    }
                ],
            }
        }

        parsed = job_description.parse_job_description("Northrop Grumman is seeking a Python intern.")
        hits = [
            search.ChunkHit("board.pdf", "chunk-1", "experience", "Built Streamlit analytics tools in Python.", 0.81, "ds3_board"),
        ]

        results = engine._aggregate_chunk_hits(
            chunk_hits=hits,
            parsed=parsed,
            top_k=5,
            min_score=0.0,
            grad_year_filter=None,
            major_filter=None,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].source, "ds3_board")
        self.assertEqual(results[0].candidate_id, "board.pdf")

    def test_search_profile_prefers_canonical_skills_and_structured_companies(self):
        engine = _make_engine()
        engine.resume_metadata_by_filename = {
            "cand.pdf": {
                "filename": "cand.pdf",
                "file_path": "",
                "text": "Resume body",
                "source": "ds3_members",
            }
        }
        engine.parsed_resume_map = {
            "cand.pdf": {
                "candidate_id": "cand.pdf",
                "source": "ds3_members",
                "file_path": "",
                "metadata": {"full_name": "Candidate", "major": "Computer Science", "graduation_year": "2027"},
                "skills": ["Frameworks/Libraries: PyTorch", "Node js"],
                "canonical_skills": ["PyTorch", "Node.js"],
                "summary": "Builder",
                "education": [],
                "projects": [
                    {
                        "name": "Smart Scheduler",
                        "bullets": ["Built FastAPI service."],
                        "technologies": ["FastAPI"],
                        "raw_text": "raw project text",
                    }
                ],
                "experience": [
                    {
                        "title": "Software Engineer Intern",
                        "company": "Northrop Grumman Corporation",
                        "company_normalized": "northrop grumman",
                        "bullets": ["Built Python tooling."],
                        "technologies": ["Python"],
                        "raw_text": "raw exp text",
                    }
                ],
            }
        }

        profile = engine._get_candidate_profile("cand.pdf")

        self.assertEqual(profile["skills"], ["PyTorch", "Node.js"])
        self.assertEqual(profile["canonical_skills"], ["PyTorch", "Node.js"])
        self.assertEqual(profile["employer_names"], ["Northrop Grumman Corporation"])
        self.assertIn("Technologies: Python", profile["combined_text"])
        self.assertIn("Technologies: FastAPI", profile["combined_text"])


if __name__ == "__main__":
    unittest.main()
