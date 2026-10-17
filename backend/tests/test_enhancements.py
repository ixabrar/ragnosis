import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.enhancement_engine import (  # noqa: E402
    analyze_response_confidence,
    compare_reports_temporally,
    check_prescription_safety,
    build_dpdp_consent_notice,
)


def test_analyze_response_confidence_abstains_when_signal_is_weak():
    result = analyze_response_confidence(
        ocr_quality=0.22,
        retrieval_similarity=0.18,
        conflict_risk=0.9,
        llm_self_assessment=0.35,
    )

    assert result['abstain'] is True
    assert result['confidence'] < 0.5
    assert 'cannot reliably answer' in result['message'].lower()


def test_compare_reports_temporally_detects_decline_and_new_abnormality():
    previous = {
        'report_date': '2026-01-01',
        'metrics': {
            'hemoglobin': 13.2,
            'glucose': 92,
        },
    }
    current = {
        'report_date': '2026-04-01',
        'metrics': {
            'hemoglobin': 11.2,
            'glucose': 145,
        },
    }

    result = compare_reports_temporally(previous, current)

    assert result['changed'] is True
    assert any('hemoglobin has dropped' in item.lower() for item in result['summary'])
    assert any('new abnormal' in item.lower() for item in result['summary'])
    assert result['changes']['hemoglobin']['delta'] == -2.0


def test_check_prescription_safety_flags_warfarin_and_aspirin():
    patient = {'conditions': ['atrial fibrillation']}
    medicines = [
        {'name': 'Warfarin'},
        {'name': 'Aspirin'},
    ]

    result = check_prescription_safety(patient, medicines)

    assert result['risk_level'] == 'contraindicated'
    assert any('warfarin' in alert['message'].lower() for alert in result['alerts'])
    assert result['requires_review'] is True


def test_dpdp_consent_notice_includes_required_elements():
    notice = build_dpdp_consent_notice()

    assert 'Separate Consent Notice' in notice['title']
    assert 'data items collected' in notice['purpose']
    assert 'withdrawal' in notice['rights'].lower()
    assert 'audit' in notice['controls'].lower()
