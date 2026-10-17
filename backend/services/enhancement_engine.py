from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from typing import Any, Dict, List, Tuple


NORMAL_RANGES = {
    "hemoglobin": (12.0, 17.0),
    "glucose": (70.0, 125.0),
    "cholesterol": (0.0, 200.0),
    "triglycerides": (0.0, 150.0),
    "hdl": (40.0, 999.0),
    "ldl": (0.0, 100.0),
    "wbc": (4.5, 11.0),
    "rbc": (4.0, 5.5),
    "platelets": (150.0, 400.0),
    "creatinine": (0.5, 1.2),
    "urea": (7.0, 20.0),
    "sgpt": (7.0, 56.0),
    "sgot": (10.0, 40.0),
    "tsh": (0.4, 4.0),
    "systolic_bp": (90.0, 120.0),
    "diastolic_bp": (60.0, 80.0),
    "mcv": (80.0, 100.0),
    "mch": (27.0, 33.0),
    "mchc": (32.0, 36.0),
    "haematocrit": (36.0, 48.0),
}

CONDITION_CONTRAINDICATIONS = {
    "kidney disease": {"metformin": "Use caution in advanced kidney disease; review eGFR before continuing."},
    "renal failure": {"metformin": "Metformin may be unsafe in severe renal impairment."},
    "ulcer": {"aspirin": "Aspirin may worsen gastrointestinal bleeding risk."},
    "bleeding disorder": {"aspirin": "Avoid aspirin unless specifically advised by a clinician."},
    "pregnancy": {"warfarin": "Warfarin is contraindicated in pregnancy unless specialist-managed."},
    "asthma": {"propranolol": "Non-selective beta blockers may worsen bronchospasm."},
}

INTERACTION_RULES = {
    frozenset({"warfarin", "aspirin"}): {
        "severity": "contraindicated",
        "message": "Warfarin + Aspirin increases bleeding risk and should be reviewed urgently.",
        "mechanism": "additive anticoagulant / antiplatelet effect",
        "monitoring": ["bleeding signs", "INR", "stool occult blood"],
    },
    frozenset({"warfarin", "ibuprofen"}): {
        "severity": "major",
        "message": "Warfarin + Ibuprofen increases bleeding risk.",
        "mechanism": "NSAID-related platelet inhibition and GI irritation",
        "monitoring": ["bleeding signs", "INR"],
    },
    frozenset({"metformin", "contrast"}): {
        "severity": "major",
        "message": "Metformin with iodinated contrast may need temporary holding based on kidney function.",
        "mechanism": "lactic acidosis risk in vulnerable patients",
        "monitoring": ["creatinine", "eGFR"],
    },
    frozenset({"simvastatin", "clarithromycin"}): {
        "severity": "contraindicated",
        "message": "Simvastatin + Clarithromycin can dangerously raise statin levels.",
        "mechanism": "CYP3A4 inhibition",
        "monitoring": ["muscle pain", "CK"],
    },
}

BRAND_TO_GENERIC = {
    "crocin": "paracetamol",
    "dolo": "paracetamol",
    "ecospirin": "aspirin",
    "combiflam": "ibuprofen",
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _parse_date(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not value:
        return None
    if isinstance(value, str):
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y"):
            try:
                return datetime.strptime(value[:10], fmt)
            except Exception:
                continue
    return None


def normalize_drug_name(name: str) -> str:
    return BRAND_TO_GENERIC.get((name or "").strip().lower(), (name or "").strip().lower())


def analyze_response_confidence(
    ocr_quality: float,
    retrieval_similarity: float,
    conflict_risk: float,
    llm_self_assessment: float,
    threshold: float = 0.62,
) -> Dict[str, Any]:
    """Combine quality signals into a single confidence score with abstention."""
    ocr_quality = _clamp01(ocr_quality)
    retrieval_similarity = _clamp01(retrieval_similarity)
    conflict_risk = _clamp01(conflict_risk)
    llm_self_assessment = _clamp01(llm_self_assessment)

    confidence = (
        0.25 * ocr_quality
        + 0.35 * retrieval_similarity
        + 0.15 * (1.0 - conflict_risk)
        + 0.25 * llm_self_assessment
    )
    confidence = round(confidence, 3)
    abstain = confidence < threshold or conflict_risk >= 0.85

    reasons = []
    if ocr_quality < 0.4:
        reasons.append("OCR quality is low")
    if retrieval_similarity < 0.35:
        reasons.append("retrieval match is weak")
    if conflict_risk >= 0.5:
        reasons.append("retrieved evidence conflicts")
    if llm_self_assessment < 0.5:
        reasons.append("LLM self-assessment is low")

    if abstain:
        message = (
            "I cannot reliably answer this. Please upload a clearer report or consult your doctor."
        )
    else:
        message = "Confidence is sufficient to answer with caution."

    return {
        "confidence": confidence,
        "threshold": threshold,
        "abstain": abstain,
        "message": message,
        "reasons": reasons,
    }


def _classify_metric_change(key: str, previous: float, current: float) -> Tuple[str, bool]:
    lo, hi = NORMAL_RANGES.get(key, (None, None))
    prev_normal = lo is not None and hi is not None and lo <= previous <= hi
    curr_normal = lo is not None and hi is not None and lo <= current <= hi
    changed_status = prev_normal != curr_normal
    significant = abs(current - previous) >= max(0.2 * abs(previous), 1.0)
    if changed_status:
        significant = True
    return ("resolved" if prev_normal and curr_normal else "new_abnormal" if not prev_normal and curr_normal is False and current != previous else "changed", significant)


def compare_reports_temporally(previous_report: Dict[str, Any], current_report: Dict[str, Any]) -> Dict[str, Any]:
    """Compare two reports and generate what-changed statements."""
    previous_metrics = previous_report.get("metrics", {}) or {}
    current_metrics = current_report.get("metrics", {}) or {}

    previous_date = _parse_date(previous_report.get("report_date") or previous_report.get("uploaded_at"))
    current_date = _parse_date(current_report.get("report_date") or current_report.get("uploaded_at"))
    months = 1.0
    if previous_date and current_date:
        days = max((current_date - previous_date).days, 1)
        months = max(days / 30.0, 0.25)

    changes: Dict[str, Dict[str, Any]] = {}
    summary: List[str] = []

    for key in sorted(set(previous_metrics) & set(current_metrics)):
        prev_val = previous_metrics[key]
        curr_val = current_metrics[key]
        if not isinstance(prev_val, (int, float)) or not isinstance(curr_val, (int, float)):
            continue
        delta = round(curr_val - prev_val, 3)
        rate_per_month = round(delta / months, 3)
        prev_lo, prev_hi = NORMAL_RANGES.get(key, (None, None))
        prev_normal = prev_lo is None or prev_hi is None or prev_lo <= prev_val <= prev_hi
        curr_normal = prev_lo is None or prev_hi is None or prev_lo <= curr_val <= prev_hi
        status = "normal"
        if prev_normal and not curr_normal:
            status = "new_abnormal"
        elif not prev_normal and curr_normal:
            status = "resolved"
        elif abs(delta) >= max(0.2 * abs(prev_val), 1.0):
            status = "changed"

        significant = status != "normal"
        changes[key] = {
            "previous": prev_val,
            "current": curr_val,
            "delta": delta,
            "rate_per_month": rate_per_month,
            "significant": significant,
            "status": status,
        }

        if significant:
            direction = "rose" if delta > 0 else "dropped" if delta < 0 else "changed"
            summary.append(
                f"Your {key.replace('_', ' ')} has {direction} from {prev_val} to {curr_val} over {months:.1f} month(s)."
            )
            if status == "new_abnormal":
                summary.append(f"{key.replace('_', ' ').title()} is now abnormal and needs attention.")
            elif status == "resolved":
                summary.append(f"{key.replace('_', ' ').title()} is now back in the normal range.")

    prev_only = sorted(set(previous_metrics) - set(current_metrics))
    curr_only = sorted(set(current_metrics) - set(previous_metrics))
    if curr_only:
        summary.append("New abnormalities or values were added in the latest report.")
    if prev_only:
        summary.append("Some older metrics are no longer present in the current report.")

    return {
        "changed": bool(summary),
        "months_between_reports": round(months, 2),
        "changes": changes,
        "summary": summary,
        "new_metrics": curr_only,
        "missing_metrics": prev_only,
    }


def check_prescription_safety(patient: Dict[str, Any], medicines: List[Dict[str, Any]]) -> Dict[str, Any]:
    patient_conditions = [str(c).strip().lower() for c in patient.get("conditions", []) if c]
    normalized_meds = [normalize_drug_name(m.get("name", "")) for m in medicines if m.get("name")]
    alerts: List[Dict[str, Any]] = []
    severities = []

    for med in normalized_meds:
        for condition in patient_conditions:
            contraindication = CONDITION_CONTRAINDICATIONS.get(condition, {}).get(med)
            if contraindication:
                alerts.append({
                    "severity": "contraindicated",
                    "message": contraindication,
                    "drug": med,
                    "condition": condition,
                    "mechanism": "condition-specific contraindication",
                    "monitoring": [],
                })
                severities.append("contraindicated")

    for first, second in combinations(sorted(set(normalized_meds)), 2):
        rule = INTERACTION_RULES.get(frozenset({first, second}))
        if rule:
            alerts.append({
                "severity": rule["severity"],
                "message": rule["message"],
                "drugs": [first, second],
                "mechanism": rule["mechanism"],
                "monitoring": rule["monitoring"],
            })
            severities.append(rule["severity"])

    if "contraindicated" in severities:
        risk_level = "contraindicated"
    elif "major" in severities:
        risk_level = "major"
    elif "moderate" in severities:
        risk_level = "moderate"
    elif alerts:
        risk_level = "minor"
    else:
        risk_level = "none"

    return {
        "risk_level": risk_level,
        "requires_review": risk_level in {"contraindicated", "major"},
        "alerts": alerts,
        "normalized_medicines": normalized_meds,
    }


def build_dpdp_consent_notice() -> Dict[str, Any]:
    return {
        "title": "Separate Consent Notice",
        "purpose": "This notice explains the data items collected, why they are collected, and how they are used for medical analysis and care coordination.",
        "data_items": [
            "Medical reports",
            "Age",
            "Height",
            "Weight",
            "Blood pressure",
            "Contact details",
        ],
        "rights": "You can withdraw consent, request access, correction, or deletion, subject to applicable legal requirements.",
        "controls": "Role-based access, audit logs, encryption in transit, and retention controls are applied to protect your data.",
        "children": "Children's data is handled only with age verification and, when required, verifiable parental or guardian consent.",
    }


def build_report_confidence(report: Dict[str, Any], previous_report: Dict[str, Any] | None = None) -> Dict[str, Any]:
    metrics = report.get("metrics", {}) or {}
    completeness = len(metrics) / 12.0
    ocr_quality = 1.0 if len((report.get("raw_text") or "").strip()) > 250 else 0.45
    retrieval_similarity = 1.0 if report.get("summary") else 0.55
    conflict_risk = 0.1
    llm_self_assessment = 0.7 if report.get("summary") else 0.35

    if previous_report:
        temporal = compare_reports_temporally(previous_report, report)
        if temporal["changed"]:
            llm_self_assessment = min(1.0, llm_self_assessment + 0.1)
    else:
        temporal = {"changed": False, "summary": []}

    confidence = analyze_response_confidence(
        ocr_quality=ocr_quality,
        retrieval_similarity=min(1.0, completeness * 0.9 + retrieval_similarity * 0.3),
        conflict_risk=conflict_risk,
        llm_self_assessment=llm_self_assessment,
    )
    confidence["temporal"] = temporal
    confidence["metric_completeness"] = round(_clamp01(completeness), 3)
    return confidence
