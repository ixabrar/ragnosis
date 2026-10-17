

\# RAGnosis Enhancement Merge Request



\## Summary

Implemented the proposed enhancements:

\- Conflict-aware RAG with uncertainty abstention

\- Temporal "what changed" reasoning across reports

\- Drug interaction and contraindication safety checks

\- DPDP consent notice and privacy workflow support



\## Files changed

\- backend/services/enhancement\_engine.py

\- backend/routes/reports.py

\- backend/routes/chatbot.py

\- backend/routes/hospital.py

\- backend/routes/auth.py

\- frontend/src/pages/Dashboard.jsx

\- backend/tests/test\_enhancements.py



\## Verification

\- Backend and frontend startup checks

\- Health endpoint check

\- Local enhancement benchmark checks



\## Notes

\- Tesseract OCR is required for report upload OCR workflows.

\- Real model fine-tuning has not been performed.

\- Current scores are proxy benchmarks based on rule-based evaluation and workflow checks.



