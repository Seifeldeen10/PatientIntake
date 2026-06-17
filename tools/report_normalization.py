"""Shared helpers for deduplicating and ordering final report content."""

from __future__ import annotations

import copy
import re
import unicodedata


REPORT_SECTION_FIELDS = [
    ("Urgent Safety Alerts", "urgent_safety_alerts"),
    ("Medication Safety", "medication_safety"),
    ("Findings", "findings"),
    ("Clinical Findings", "clinical_findings"),
    ("Evidence Summary", "evidence_summary"),
    ("Clinician Actions", "clinician_actions"),
    ("Missing Information", "missing_information"),
    ("Limitations", "limitations"),
]


def as_list(value):
    """Normalize a value into a list of non-empty strings."""
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item or "").strip()]
    if value in (None, ""):
        return []
    return [str(value).strip()]


def _strip_source_suffix(text):
    """Remove common source/citation suffixes so duplicate concepts compare cleanly."""
    text = str(text or "").strip()
    if not text:
        return ""

    text = re.split(r"\s+\|\s+", text, maxsplit=1)[0].strip()
    text = re.sub(
        r"\s*[\(\[][^()\[\]]*(?:pdf|pmid|doi|p\.\s*\d+|score\s*\d|https?://|source)[^()\[\]]*[\)\]]\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    return text


def text_key(text):
    """Build a normalized comparison key for deduplication."""
    text = _strip_source_suffix(text)
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def dedupe_list(values, *, seen=None):
    """Deduplicate a list while preserving order."""
    seen = seen if seen is not None else set()
    unique_values = []
    for item in as_list(values):
        key = text_key(item)
        if not key or key in seen:
            continue
        seen.add(key)
        unique_values.append(item)
    return unique_values


def canonical_report_sections(report):
    """Return non-empty report sections in a stable, deduplicated order."""
    report = report or {}
    seen = set()
    sections = []
    for heading, field_name in REPORT_SECTION_FIELDS:
        items = dedupe_list(report.get(field_name), seen=seen)
        if items:
            sections.append({
                "heading": heading,
                "items": items,
            })
    return sections


def normalize_final_report(report):
    """Return a cleaned final report with repeated concepts removed."""
    report = copy.deepcopy(report or {})

    for field in (
        "report_title",
        "report_type",
        "executive_summary",
        "clinical_summary",
        "confidence",
    ):
        if field in report and report.get(field) is not None:
            report[field] = str(report.get(field)).strip()

    if not report.get("report_title"):
        report["report_title"] = "AI Clinical Evidence Report"

    if not report.get("report_type"):
        report["report_type"] = "full_clinical_evidence_review"

    snapshot = report.get("patient_snapshot") if isinstance(report.get("patient_snapshot"), dict) else {}
    report["patient_snapshot"] = {
        "submission_id": str(snapshot.get("submission_id") or "").strip(),
        "age": str(snapshot.get("age") or "").strip(),
        "sex": str(snapshot.get("sex") or snapshot.get("gender") or "").strip(),
        "presenting_question": str(snapshot.get("presenting_question") or "").strip(),
    }

    list_fields_in_order = [
        "urgent_safety_alerts",
        "medication_safety",
        "findings",
        "clinical_findings",
        "evidence_summary",
        "clinician_actions",
        "missing_information",
        "limitations",
    ]

    seen = set()
    for field_name in list_fields_in_order:
        report[field_name] = dedupe_list(report.get(field_name), seen=seen)

    citations = dedupe_list(report.get("citations"))
    source_citations = dedupe_list(report.get("source_citations"))
    if not citations and source_citations:
        citations = source_citations
    report["citations"] = citations
    report["source_citations"] = []

    report["structured_sections"] = canonical_report_sections(report)

    return report
