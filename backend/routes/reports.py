import os
import uuid
import traceback
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from utils.jwt_helper import require_auth
from models.report_model import create_report, get_reports_by_user, get_report_by_id, update_report_metrics
from services.text_extractor import (
    extract_text, allowed_file, extract_medical_metrics,
    detect_report_type, is_medical_report
)
from services.bert_summarizer import summarize_medical_text, simplify_medical_summary, build_recommendations
from services.doctor_service import get_suggested_doctors, get_specialist_advice
from services.enhancement_engine import build_report_confidence, compare_reports_temporally
from config import Config

reports_bp = Blueprint("reports", __name__)


def _safe_remove(path: str):
    """Delete a file without raising if it doesn't exist."""
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


@reports_bp.route("/upload", methods=["POST"])
@require_auth
def upload_report():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if not allowed_file(file.filename, Config.ALLOWED_EXTENSIONS):
        return jsonify({"error": f"File type '{ext}' not allowed. Use PDF, JPG, PNG, or JPEG"}), 400

    unique_name = f"{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(Config.UPLOAD_FOLDER, unique_name)
    file.save(save_path)

    current_app.logger.info(f"[UPLOAD] Saved file: {save_path} (ext={ext})")

    try:
        current_app.logger.info(f"[UPLOAD] Extracting text from {ext} file...")
        try:
            raw_text = extract_text(save_path, ext)
        except Exception as ocr_err:
            current_app.logger.error(f"[UPLOAD] Text extraction failed:\n{traceback.format_exc()}")
            _safe_remove(save_path)
            return jsonify({
                "error": "ocr_failed",
                "message": f"Could not extract text from file: {str(ocr_err)}",
                "details": traceback.format_exc(),
            }), 500

        current_app.logger.info(f"[UPLOAD] Extracted {len(raw_text)} chars")

        if len(raw_text.strip()) < 20:
            _safe_remove(save_path)
            return jsonify({
                "error": "unreadable",
                "message": (
                    "Could not read any text from this file. "
                    "Please upload a clear scan, screenshot, or digital PDF."
                ),
            }), 400

        if not is_medical_report(raw_text):
            _safe_remove(save_path)
            return jsonify({
                "error": "out_of_knowledge_base",
                "message": (
                    "This file does not appear to be a medical lab report. "
                    "RAGnosis supports: CBC, Lipid Panel, LFT, KFT, Thyroid, Diabetes, and BP reports."
                ),
            }), 422

        current_app.logger.info("[UPLOAD] Extracting metrics...")
        try:
            metrics = extract_medical_metrics(raw_text)
            report_type = detect_report_type(raw_text)
            current_app.logger.info(f"[UPLOAD] Metrics: {metrics} | Type: {report_type}")
        except Exception as metrics_err:
            current_app.logger.error(f"[UPLOAD] Metrics extraction failed:\n{traceback.format_exc()}")
            _safe_remove(save_path)
            return jsonify({
                "error": "metrics_failed",
                "message": f"Metric extraction failed: {str(metrics_err)}",
            }), 500

        current_app.logger.info("[UPLOAD] Summarizing...")
        try:
            summary_raw = summarize_medical_text(raw_text)
            summary = simplify_medical_summary(summary_raw, metrics)
            recommendations = build_recommendations(metrics)
            current_app.logger.info("[UPLOAD] Summary complete")
        except Exception:
            summary = "Summary could not be generated for this report."
            recommendations = []
            current_app.logger.warning("[UPLOAD] Using fallback summary due to error")

        current_app.logger.info("[UPLOAD] Saving to DB...")

        suggested_doctors = get_suggested_doctors(report_type)
        is_critical = any(r.get("is_critical") for r in recommendations)
        specialist_advice = get_specialist_advice(report_type)

        try:
            report_id = create_report(request.user_id, {
                "filename": unique_name,
                "original_name": secure_filename(file.filename),
                "file_type": ext,
                "file_path": save_path,
                "raw_text": raw_text,
                "summary": summary,
                "metrics": metrics,
                "report_type": report_type,
                "recommendations": recommendations,
                "is_critical": is_critical,
                "suggested_doctors": suggested_doctors,
                "specialist_advice": specialist_advice,
            })
            current_app.logger.info(f"[UPLOAD] Saved report: {report_id}")
        except Exception as db_err:
            current_app.logger.error(f"[UPLOAD] DB save failed:\n{traceback.format_exc()}")
            _safe_remove(save_path)
            return jsonify({
                "error": "db_error",
                "message": f"Failed to save report: {str(db_err)}",
            }), 500

    except Exception as e:
        _safe_remove(save_path)
        current_app.logger.error(f"[UPLOAD] Unexpected error:\n{traceback.format_exc()}")
        return jsonify({
            "error": "unexpected_error",
            "message": f"Unexpected error: {str(e)}",
            "details": traceback.format_exc(),
        }), 500

    confidence = build_report_confidence({
        "raw_text": raw_text,
        "summary": summary,
        "metrics": metrics,
    })

    return jsonify({
        "message": "Report uploaded and analyzed successfully! ✅",
        "report_id": report_id,
        "report_type": report_type,
        "summary": summary,
        "metrics": metrics,
        "recommendations": recommendations,
        "is_critical": is_critical,
        "suggested_doctors": suggested_doctors,
        "specialist_advice": specialist_advice,
        "analysis_confidence": confidence,
    }), 201


@reports_bp.route("/", methods=["GET"])
@require_auth
def list_reports():
    reports = get_reports_by_user(request.user_id)
    for r in reports:
        r.pop("raw_text", None)
        r.pop("file_path", None)
    return jsonify({"reports": reports}), 200


@reports_bp.route("/<report_id>", methods=["GET"])
@require_auth
def get_report(report_id):
    report = get_report_by_id(report_id)
    if not report or report["user_id"] != request.user_id:
        return jsonify({"error": "Report not found"}), 404
    report.pop("file_path", None)
    if not report.get("recommendations") and report.get("metrics"):
        report["recommendations"] = build_recommendations(report["metrics"])

    if "is_critical" not in report:
        report["is_critical"] = any(r.get("is_critical") for r in (report.get("recommendations") or []))
    if not report.get("suggested_doctors"):
        report["suggested_doctors"] = get_suggested_doctors(report.get("report_type", "General Medical Report"))
    if not report.get("specialist_advice"):
        report["specialist_advice"] = get_specialist_advice(report.get("report_type", "General Medical Report"))

    all_reports = get_reports_by_user(request.user_id)
    previous_report = None
    for idx, r in enumerate(all_reports):
        if r.get("_id") == report_id and idx + 1 < len(all_reports):
            previous_report = all_reports[idx + 1]
            break
    report["analysis_confidence"] = build_report_confidence(report, previous_report)
    if previous_report:
        report["temporal_analysis"] = compare_reports_temporally(previous_report, report)

    return jsonify(report), 200


@reports_bp.route("/<report_id>/reanalyze", methods=["POST"])
@require_auth
def reanalyze_report(report_id):
    report = get_report_by_id(report_id)
    if not report or report["user_id"] != request.user_id:
        return jsonify({"error": "Report not found"}), 404
    file_path = report.get("file_path")
    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "Original file missing"}), 400
    try:
        raw_text = extract_text(file_path, os.path.splitext(file_path)[1].lstrip('.'))
        metrics = extract_medical_metrics(raw_text)
        report_type = detect_report_type(raw_text)
        summary_raw = summarize_medical_text(raw_text)
        summary = simplify_medical_summary(summary_raw, metrics)
        recommendations = build_recommendations(metrics)
        update_report_metrics(report_id, metrics, summary, recommendations, report_type)
    except Exception as e:
        current_app.logger.error(f"[REANALYZE] Error:\n{traceback.format_exc()}")
        return jsonify({"error": f"Reanalysis failed: {str(e)}"}), 500
    return jsonify({
        "message": "Report reanalyzed successfully",
        "metrics": metrics,
        "summary": summary,
        "recommendations": recommendations,
        "report_type": report_type,
    }), 200
