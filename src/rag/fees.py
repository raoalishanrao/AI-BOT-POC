"""Fee parsing and course-page enrichment."""

import re

SLUG_TO_PROGRAM: dict[str, str] = {
    "doctor-of-pharmacy": "Doctor of Pharmacy",
    "doctor-of-physical-therapy": "Doctor of Physical Therapy",
    "bs-human-nutrition-and-dietetics": "BS Human Nutrition & Dietetics",
    "bs-anesthesia-technology": "BS Anaesthesia Technology",
    "bs-radiology-technology": "BS Radiology Technology",
    "bs-medical-laboratory-technology": "BS Medical Laboratory Technology (MLT)",
    "bs-dental-technology": "BS Dental Technology",
    "bs-optometry": "BS Optometry",
    "bs-psychology": "BS Psychology",
    "bs-nursing": "BS Nursing",
    "bs-in-computer-science": "BS Computer Science (BS CS)",
    "bs-in-software-engineering": "BS Software Engineering",
    "bs-in-artificial-intelligence": "BS Artificial Intelligence",
    "ad-in-computing": "AD in Computing",
    "bachelor-of-business-administration": "BBA - H",
    "bs-accounting-and-finance": "BS Accounting & Finance",
    "bs-commerce-fintech": "BS Commerce (Fintech)",
    "adp-business-administration": "BBA 2.5 yrs",
    "adp-in-accounting-and-finance": "AD Accounting & Finance",
    "adp-digital-marketing": "AD Digital Marketing",
    "certified-nursing-assistant": "Certified Nursing Assistant (Female/Male)",
    "copy-of-nursing-diploma-lady-health-v": "Community Midwifery (Female only)",
    "nursing-diploma-lady-health-visitor": "Lady Health Visitor (Female only)",
}

_PROGRAM_HEADER = re.compile(
    r"^######\s+(?:Program|Course Name)\s*:?\s*(.*)$",
    re.M | re.I,
)
_DEPARTMENT = re.compile(r"^##\s+(.+)$", re.M)
_TOTAL_FEE = re.compile(r"Total 1st Semester fee:\s*([^\n]+)", re.I)
_FIRST_SEM_FEE = re.compile(r"1st Semester fee:\s*([^\n]+)", re.I)
_ADMISSION_FEE = re.compile(r"(?:One time )?Admission fee\s*(?:\(one time\))?:\s*([^\n]+)", re.I)
_REGISTRATION_FEE = re.compile(r"(?:Per Semester|Annual Registration Fee)\s*:?\s*([^\n]+)", re.I)
_FEE_PER_YEAR = re.compile(r"Fee per Year:\s*([^\n]+)", re.I)


def normalize_name(name: str) -> str:
    name = name.lower()
    name = re.sub(r"\([^)]*\)", "", name)
    name = re.sub(r"[^a-z0-9]+", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def parse_fees_markdown(markdown: str) -> list[dict]:
    records: list[dict] = []
    headers = list(_PROGRAM_HEADER.finditer(markdown))
    dept_matches = list(_DEPARTMENT.finditer(markdown))

    for i, header in enumerate(headers):
        program = header.group(1).strip()
        block_start = header.end()
        block_end = headers[i + 1].start() if i + 1 < len(headers) else len(markdown)
        block = markdown[block_start:block_end]

        if not program:
            lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
            program = lines[0] if lines else ""

        program = program.strip()
        if not program or program.lower().startswith("note"):
            continue

        department = ""
        for dept in dept_matches:
            if dept.start() < header.start():
                department = dept.group(1).strip()

        total = _TOTAL_FEE.search(block)
        first_sem = _FIRST_SEM_FEE.search(block)
        admission = _ADMISSION_FEE.search(block)
        registration = _REGISTRATION_FEE.search(block)
        fee_per_year = _FEE_PER_YEAR.search(block)

        records.append(
            {
                "program": program,
                "department": department,
                "admission_fee": admission.group(1).strip() if admission else "",
                "registration_fee_per_semester": registration.group(1).strip() if registration else "",
                "first_semester_fee": first_sem.group(1).strip() if first_sem else "",
                "total_first_semester_fee": total.group(1).strip() if total else "",
                "fee_per_year": fee_per_year.group(1).strip() if fee_per_year else "",
            }
        )

    return records


def build_fee_lookup(records: list[dict]) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for record in records:
        keys = {
            normalize_name(record["program"]),
            normalize_name(record["program"].split("(")[0]),
        }
        for key in keys:
            if key:
                lookup[key] = record
    for slug, program in SLUG_TO_PROGRAM.items():
        norm = normalize_name(program)
        for record in records:
            if normalize_name(record["program"]) == norm:
                lookup[normalize_name(slug.replace("-", " "))] = record
                lookup[slug] = record
    return lookup


def _clean_title(title: str) -> str:
    return re.sub(r"\s*\|.*$", "", title).strip()


def match_fee_for_page(title: str, slug: str, lookup: dict[str, dict]) -> dict | None:
    if slug in SLUG_TO_PROGRAM:
        program = SLUG_TO_PROGRAM[slug]
        key = normalize_name(program)
        if key in lookup:
            return lookup[key]
        for k, record in lookup.items():
            if key in k or k in key:
                return record

    candidates = [
        normalize_name(_clean_title(title)),
        normalize_name(slug.replace("-", " ")),
        slug,
    ]

    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
        for key, record in lookup.items():
            if not candidate or not key:
                continue
            if candidate in key or key in candidate:
                return record
            c_tokens = set(candidate.split()) - {"bs", "in", "of", "and", "the"}
            k_tokens = set(key.split()) - {"bs", "in", "of", "and", "the"}
            if len(c_tokens & k_tokens) >= 2:
                return record
    return None


def format_fee_section(record: dict) -> str:
    lines = [
        "## Fee Structure (Iqra University Chak Shahzad Campus)",
        f"**Program:** {record['program']}",
        f"**Department:** {record['department']}",
        f"**Admission fee (one time):** {record['admission_fee']}",
        f"**Registration fee per semester:** {record['registration_fee_per_semester']}",
    ]
    if record.get("fee_per_year"):
        lines.append(f"**Fee per year:** {record['fee_per_year']}")
    lines.extend(
        [
            f"**1st semester fee:** {record['first_semester_fee']}",
            f"**Total 1st semester fee:** {record['total_first_semester_fee']}",
        ]
    )
    return "\n".join(lines) + "\n"


def enrich_page_markdown(page: dict, fee_lookup: dict[str, dict]) -> str:
    if page["slug"] == "fees":
        return page["markdown"]

    fee = match_fee_for_page(page.get("title", ""), page["slug"], fee_lookup)
    if not fee:
        return page["markdown"]

    fee_section = format_fee_section(fee)
    if fee_section.strip() in page["markdown"]:
        return page["markdown"]
    return f"{page['markdown']}\n\n{fee_section}"
