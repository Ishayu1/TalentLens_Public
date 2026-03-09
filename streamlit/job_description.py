from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re

try:
    from config import SKILL_SUGGESTIONS
except ImportError:
    from streamlit.config import SKILL_SUGGESTIONS


SKILL_ALIASES: dict[str, tuple[str, ...]] = {
    "Python": ("python",),
    "Machine Learning": ("machine learning", "ml"),
    "Data Science": ("data science",),
    "NLP": ("nlp", "natural language processing"),
    "SQL": ("sql", "postgresql", "mysql", "sqlite"),
    "React": ("react", "react.js", "reactjs"),
    "Computer Vision": ("computer vision", "opencv"),
    "Java": ("java",),
    "C++": ("c++", "cpp"),
    "JavaScript": ("javascript", "js"),
    "TypeScript": ("typescript", "ts"),
    "TensorFlow": ("tensorflow",),
    "PyTorch": ("pytorch", "torch"),
    "Deep Learning": ("deep learning",),
    "Statistics": ("statistics", "statistical"),
    "R": (" r ", "r language", "rstudio"),
    "AWS": ("aws", "amazon web services"),
    "Docker": ("docker",),
    "Kubernetes": ("kubernetes", "k8s"),
    "Node.js": ("node.js", "nodejs", "node"),
    "HTML/CSS": ("html", "css"),
    "Git": ("git", "github", "gitlab"),
    "MongoDB": ("mongodb", "mongo"),
    "PostgreSQL": ("postgresql", "postgres"),
    "Tableau": ("tableau",),
    "Power BI": ("power bi", "powerbi"),
    "Spark": ("spark", "apache spark"),
    "Hadoop": ("hadoop",),
    "Scikit-learn": ("scikit-learn", "sklearn"),
    "Pandas": ("pandas",),
    "NumPy": ("numpy",),
    "LLMs": ("llm", "llms", "large language model", "large language models"),
    "RAG": ("rag", "retrieval augmented generation"),
    "Transformers": ("transformers", "transformer"),
    "BERT": ("bert",),
    "GPT": ("gpt",),
    "Flask": ("flask",),
    "Django": ("django",),
    "FastAPI": ("fastapi",),
    "Swift": ("swift",),
    "Kotlin": ("kotlin",),
    "Rust": ("rust",),
    "Go": (" golang ", " go ", "golang"),
    "Scala": ("scala",),
    "Amazon": ("amazon", "amazon.inc"),
}

for skill in SKILL_SUGGESTIONS:
    SKILL_ALIASES.setdefault(skill, (skill.lower(),))

PREFERRED_SECTION_RE = re.compile(
    r"\b(preferred|nice to have|bonus|plus|pluses|ideal|desired)\b", re.I
)
REQUIRED_SECTION_RE = re.compile(
    r"\b(requirements|qualifications|required|must have|minimum qualifications|what you bring)\b",
    re.I,
)
REMOTE_RE = re.compile(r"\b(remote|hybrid|on[- ]site|onsite)\b", re.I)
YEARS_RE = re.compile(
    r"(?:(?:at least|minimum of|min\.?)\s*)?(\d+)\+?\s*(?:\+)?\s*years?(?:\s+of)?\s+experience",
    re.I,
)
DEGREE_RE = re.compile(
    r"\b(bachelor(?:'s)?|master(?:'s)?|phd|doctorate|b\.?s\.?|b\.?a\.?|m\.?s\.?|m\.?a\.?)\b",
    re.I,
)
LOCATION_PREFIX_RE = re.compile(r"^(location|based in|office location)\s*[:\-]\s*(.+)$", re.I)


@dataclass
class ParsedJobDescription:
    raw_text: str
    job_title: str = ""
    company: str = ""
    must_have_skills: list[str] = field(default_factory=list)
    preferred_skills: list[str] = field(default_factory=list)
    minimum_years_experience: int | None = None
    location: str = ""
    remote_policy: str = ""
    degree_requirements: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _clean_lines(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines()]
    return [line for line in lines if line]


def _extract_title(lines: list[str]) -> str:
    for line in lines[:8]:
        match = re.match(r"^(job title|title|position|role)\s*[:\-]\s*(.+)$", line, re.I)
        if match:
            return match.group(2).strip()

    for line in lines[:6]:
        if len(line.split()) <= 10 and not re.match(r"^[\-\*\d]", line):
            if not REQUIRED_SECTION_RE.search(line) and not PREFERRED_SECTION_RE.search(line):
                return line
    return ""
    
def _extract_company(lines: list[str]) -> str:
    # Look for common company markers or known big tech
    for line in lines[:10]:
        val = ""
        # Handle "|" separators
        if "|" in line:
            parts = [p.strip() for p in line.split("|")]
            for p in parts:
                if any(x in p.lower() for x in ["amazon", "google", "meta", "apple", "microsoft", "netflix", "nvidia"]):
                    val = p
                    break
        else:
            # Match "Company: X"
            match = re.search(r"\b(?:company|employer|firm)\s*[:\-]\s*([A-Z][A-Za-z0-9\.\s]+)", line, re.I)
            if match:
                 val = match.group(1).strip()
        
        if val:
            # Clean it up: "Amazon.com Services LLC" -> "Amazon"
            core = val.split(".")[0].split()[0].strip().rstrip(",").rstrip(".")
            # Keep meaningful second words for some
            if core.lower() in ["hey", "the", "a"] and len(val.split()) > 1:
                return " ".join(val.split()[:2])
            return core
    return ""


def _extract_skills(text: str) -> list[str]:
    haystack = f" {text.lower()} "
    matches: list[str] = []
    for skill, aliases in SKILL_ALIASES.items():
        for alias in aliases:
            alias_text = alias.lower()
            if alias_text.strip() in {"r", "go"}:
                pattern = rf"(?<![a-z]){re.escape(alias_text.strip())}(?![a-z])"
                if re.search(pattern, haystack):
                    matches.append(skill)
                    break
            elif alias_text in haystack:
                matches.append(skill)
                break
    return sorted(set(matches))


def _extract_years(text: str) -> int | None:
    matches = [int(match.group(1)) for match in YEARS_RE.finditer(text)]
    return max(matches) if matches else None


def _extract_location(lines: list[str], text: str) -> tuple[str, str]:
    remote_policy = ""
    location = ""

    remote_hits = REMOTE_RE.findall(text)
    if remote_hits:
        policies = {hit.lower().replace(" ", "-") for hit in remote_hits}
        if "remote" in policies:
            remote_policy = "Remote"
        elif "hybrid" in policies:
            remote_policy = "Hybrid"
        elif "on-site" in policies or "onsite" in policies:
            remote_policy = "On-site"

    for line in lines[:20]:
        prefix_match = LOCATION_PREFIX_RE.match(line)
        if prefix_match:
            location = prefix_match.group(2).strip()
            break
        if REMOTE_RE.search(line) and not location:
            location = line

    return location, remote_policy


def _extract_degree_requirements(text: str) -> list[str]:
    found = []
    for match in DEGREE_RE.finditer(text):
        degree = match.group(1)
        normalized = degree.replace(".", "").strip().lower()
        if normalized in {"bs", "ba", "bachelor", "bachelors", "bachelor's"}:
            found.append("Bachelor's")
        elif normalized in {"ms", "ma", "master", "masters", "master's"}:
            found.append("Master's")
        elif normalized in {"phd", "doctorate"}:
            found.append("PhD")
    return sorted(set(found), key=lambda item: ["Bachelor's", "Master's", "PhD"].index(item))


def parse_job_description(text: str) -> ParsedJobDescription:
    lines = _clean_lines(text)
    required_lines: list[str] = []
    preferred_lines: list[str] = []
    current_section = "other"

    for line in lines:
        if PREFERRED_SECTION_RE.search(line):
            current_section = "preferred"
            continue
        if REQUIRED_SECTION_RE.search(line):
            current_section = "required"
            continue

        if current_section == "required":
            required_lines.append(line)
        elif current_section == "preferred":
            preferred_lines.append(line)

    required_text = "\n".join(required_lines)
    preferred_text = "\n".join(preferred_lines)
    full_text = "\n".join(lines)

    must_have_skills = _extract_skills(required_text) or _extract_skills(full_text)
    preferred_skills = [skill for skill in _extract_skills(preferred_text) if skill not in must_have_skills]
    minimum_years = _extract_years(required_text or full_text)
    location, remote_policy = _extract_location(lines, full_text)
    degree_requirements = _extract_degree_requirements(full_text)

    return ParsedJobDescription(
        raw_text=text,
        job_title=_extract_title(lines),
        company=_extract_company(lines),
        must_have_skills=must_have_skills,
        preferred_skills=preferred_skills,
        minimum_years_experience=minimum_years,
        location=location,
        remote_policy=remote_policy,
        degree_requirements=degree_requirements,
    )
