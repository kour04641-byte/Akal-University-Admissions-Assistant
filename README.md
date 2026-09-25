# Akal University Admission Assistant

An AI-powered chatbot that answers student queries about Akal University admissions — programs, fees, hostel, scholarships, eligibility, and the admission process — using retrieval-augmented generation (RAG) with Groq.

## Features
- Answers in English, Hindi, Hinglish or Punjabi
- Only answers from verified university data (no hallucinated fees or dates)
- "Request a callback" form that saves leads and emails the admission office instantly
- Quick-question buttons for common queries

## Setup
1. `pip install -r requirements.txt`
2. Create a `.env` file with `GROQ_API_KEY`, and optionally `SENDER_EMAIL`, `SENDER_APP_PASSWORD`, `NOTIFY_EMAIL`
3. `streamlit run app.py`

## Status
Currently in testing.
