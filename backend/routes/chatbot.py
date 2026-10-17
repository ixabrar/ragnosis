from flask import Blueprint, request, jsonify
from utils.jwt_helper import require_auth
from services.groq_service import get_chatbot_response
from models.report_model import get_report_by_id, get_reports_by_user
from services.enhancement_engine import build_report_confidence, compare_reports_temporally

chatbot_bp = Blueprint("chatbot", __name__)


@chatbot_bp.route("/", methods=["POST"])
@require_auth
def chat():
    data = request.get_json() or {}
    user_message = data.get("message", "").strip()
    report_id = data.get("report_id", None)
    chat_history = data.get("history", [])
    mode = (data.get("mode") or "standard").lower()

    if not user_message:
        return jsonify({"error": "Message cannot be empty"}), 400

    report_text = ""
    report = None
    previous_report = None
    if report_id:
        report = get_report_by_id(report_id)
        if report and report.get("user_id") == request.user_id:
            report_text = report.get("raw_text", "")
            all_reports = get_reports_by_user(request.user_id)
            report_ids = [r.get("_id") for r in all_reports]
            if report_id in report_ids:
                idx = report_ids.index(report_id)
                if idx + 1 < len(all_reports):
                    previous_report = all_reports[idx + 1]

    response = get_chatbot_response(user_message, report_text, chat_history)
    confidence = None
    temporal = None
    if report:
        confidence = build_report_confidence(report, previous_report)
        temporal = confidence.pop("temporal", None)

    payload = {
        "response": response,
        "report_id": report_id,
    }
    if confidence:
        payload["confidence"] = confidence
    if temporal:
        payload["temporal"] = temporal
    if mode == "strict" and confidence and confidence.get("abstain"):
        payload["response"] = confidence.get("message")
        payload["mode"] = "strict"
    return jsonify(payload), 200


@chatbot_bp.route("/suggestions", methods=["GET"])
@require_auth
def get_suggestions():
    """Return helpful default questions for the chatbot."""
    suggestions = [
        "What does my hemoglobin level mean?",
        "Is my blood sugar level normal?",
        "Explain my cholesterol results in simple terms",
        "What changed since my last report?",
        "Are there any abnormal values in my report?",
        "What is the significance of my WBC count?",
        "How can I improve my kidney function?",
        "What does my TSH level indicate about my thyroid?",
    ]
    return jsonify({"suggestions": suggestions}), 200
