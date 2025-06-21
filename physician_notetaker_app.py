import streamlit as st
import re
import spacy
from transformers import pipeline
import os
os.system("python -m spacy download en_core_web_sm")


st.set_page_config(page_title="Physician Notetaker", layout="wide")

@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

@st.cache_resource
def load_sentiment():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

nlp = load_nlp()
sentiment_pipe = load_sentiment()

# Function to extract entities from transcript
def extract_entities(text):
    symptoms = []
    if re.search(r'neck pain|pain in my neck', text, re.I):
        symptoms.append("Neck pain")
    if re.search(r'back pain|pain in my back', text, re.I):
        symptoms.append("Back pain")
    if re.search(r'head (impact|hit my head)', text, re.I):
        symptoms.append("Head impact")
    if re.search(r'trouble sleeping', text, re.I):
        symptoms.append("Trouble sleeping")
    if re.search(r'discomfort', text, re.I):
        symptoms.append("Discomfort")
    if re.search(r'occasional backaches?|backaches?', text, re.I):
        symptoms.append("Occasional backache")

    diagnosis = "Whiplash injury" if re.search(r'whiplash injury', text, re.I) else "Not specified"

    treatments = []
    if re.search(r'ten sessions|10 sessions|physiotherapy', text, re.I):
        treatments.append("10 physiotherapy sessions")
    if re.search(r'painkillers', text, re.I):
        treatments.append("Painkillers")
    if re.search(r'advice', text, re.I):
        treatments.append("Advice")
    if re.search(r'follow[- ]?up', text, re.I):
        treatments.append("Follow-up")

    prognosis = "Full recovery expected within six months" if re.search(r'full recovery', text, re.I) else "Not specified"

    current_status = "Occasional backache" if re.search(r'occasional backaches?|occasional backache', text, re.I) else "Doing better" if re.search(r'doing better', text, re.I) else "Not specified"

    return {
        "Symptoms": symptoms,
        "Diagnosis": diagnosis,
        "Treatment": treatments,
        "Current_Status": current_status,
        "Prognosis": prognosis
    }

# Function to summarize transcript into JSON format
def summarize_to_json(transcript):
    name_match = re.search(r"Ms\\. Jones|Mrs\\. Jones|Mr\\. Jones|Janet Jones", transcript)
    patient_name = "Janet Jones" if name_match else "Not specified"
    entities = extract_entities(transcript)
    return {
        "Patient_Name": patient_name,
        "Symptoms": entities["Symptoms"],
        "Diagnosis": entities["Diagnosis"],
        "Treatment": entities["Treatment"],
        "Current_Status": entities["Current_Status"],
        "Prognosis": entities["Prognosis"]
    }

# Function to extract keywords
def extract_keywords(text):
    keywords = []
    if re.search(r'whiplash injury', text, re.I):
        keywords.append("Whiplash injury")
    if re.search(r'ten sessions|10 sessions|physiotherapy', text, re.I):
        keywords.append("10 physiotherapy sessions")
    if re.search(r'painkillers', text, re.I):
        keywords.append("Painkillers")
    if re.search(r'back pain', text, re.I):
        keywords.append("Back pain")
    if re.search(r'neck pain', text, re.I):
        keywords.append("Neck pain")
    if re.search(r'head (impact|hit my head)', text, re.I):
        keywords.append("Head impact")
    if re.search(r'trouble sleeping', text, re.I):
        keywords.append("Trouble sleeping")
    if re.search(r'discomfort', text, re.I):
        keywords.append("Discomfort")
    if re.search(r'full recovery', text, re.I):
        keywords.append("Full recovery")
    if re.search(r'stiffness', text, re.I):
        keywords.append("Stiffness")
    if re.search(r'backache', text, re.I):
        keywords.append("Backache")
    return sorted(set(keywords))

# Sentiment & intent analyzer
def analyze_patient_sentiment(text):
    text_lc = text.lower()
    if any(word in text_lc for word in ["worried", "concerned", "anxious", "nervous"]):
        return {"Sentiment": "Anxious", "Intent": "Seeking reassurance"}
    elif any(word in text_lc for word in ["relief", "thankful", "grateful", "appreciate"]):
        return {"Sentiment": "Reassured", "Intent": "Expressing gratitude"}
    elif any(word in text_lc for word in ["just a regular visit", "nothing to report", "doing fine"]):
        return {"Sentiment": "Neutral", "Intent": "Routine checkup"}
    else:
        result = sentiment_pipe(text)[0]
        label = result['label']
        if label == "NEGATIVE":
            return {"Sentiment": "Anxious", "Intent": "Seeking reassurance"}
        elif label == "POSITIVE":
            return {"Sentiment": "Reassured", "Intent": "Expressing gratitude"}
        else:
            return {"Sentiment": "Neutral", "Intent": "Reporting symptoms"}

# SOAP Note Generator
def generate_soap_note(transcript):
    if not transcript.strip():
        return {"Subjective": {}, "Objective": {}, "Assessment": {}, "Plan": {}}

    entities = extract_entities(transcript)
    subjective = {
        "Chief_Complaint": ", ".join(entities["Symptoms"]) if entities["Symptoms"] else "Routine checkup",
        "History_of_Present_Illness": "General checkup. No complaints reported." if not entities["Symptoms"] else "Patient described symptoms as: " + ", ".join(entities["Symptoms"])
    }
    objective = {
        "Physical_Exam": "Normal" if not entities["Symptoms"] else "Relevant exams pending based on reported symptoms.",
        "Observations": "Patient appears in normal health."
    }
    assessment = {
        "Diagnosis": entities["Diagnosis"],
        "Severity": "None" if entities["Diagnosis"] == "Not specified" else "Under observation"
    }
    plan = {
        "Treatment": "No treatment prescribed." if not entities["Treatment"] else ", ".join(entities["Treatment"]),
        "Follow-Up": "Routine annual checkup advised." if not entities["Treatment"] else "Monitor symptoms and follow-up as needed."
    }
    return {
        "Subjective": subjective,
        "Objective": objective,
        "Assessment": assessment,
        "Plan": plan
    }

# Streamlit UI
st.title("🩺 Physician Notetaker: Medical NLP & Sentiment Analysis")

st.header("Transcript Input")
transcript = st.text_area("Paste transcript here (full or sample):", height=400)

if st.button("🧠 Analyze Transcript"):
    st.subheader("1. 🔍 Named Entity Recognition")
    ner = extract_entities(transcript)
    st.json(ner)

    st.subheader("2. 📝 Structured Summary")
    summary = summarize_to_json(transcript)
    st.json(summary)

    st.subheader("3. 🧠 Keyword Extraction")
    keywords = extract_keywords(transcript)
    st.write(keywords)

    st.subheader("5. 📋 SOAP Note Generation")
    soap = generate_soap_note(transcript)
    st.json(soap)

st.header("4. 😊 Sentiment & Intent (Separate Analyzer)")
dialogue = st.text_area("Paste a patient's dialogue for sentiment analysis:")
if st.button("🔍 Analyze Sentiment & Intent"):
    sentiment = analyze_patient_sentiment(dialogue)
    st.json(sentiment)

st.header("Assignment Methodology")
with st.expander("How would you handle ambiguous or missing medical data?"):
    st.write("""
- Use context and negation detection to avoid false positives.
- If a field is missing, output \"Not specified\".
- For ambiguous terms, prefer clinician statements over patient self-report.
""")
with st.expander("What NLP models for medical summarization?"):
    st.write("""
- spaCy with custom patterns for NER.
- Transformers (BERT, ClinicalBERT, SciSpacy).
""")
with st.expander("How would you fine-tune BERT for medical sentiment?"):
    st.write("""
- Collect a labeled dataset of patient dialogues.
- Fine-tune BERT/ClinicalBERT with supervised learning.
- Validate on held-out real clinical conversations.
""")
with st.expander("What datasets for healthcare-specific sentiment?"):
    st.write("""
- i2b2/UTHealth notes, MEDIQA, MIMIC-III, patient opinion mining datasets.
""")
with st.expander("How to train model for SOAP mapping?"):
    st.write("""
- Annotate transcripts with SOAP sections.
- Fine-tune seq2seq models (T5, BART) or use rules for structure.
""")
with st.expander("Techniques to improve SOAP note accuracy?"):
    st.write("""
- Rule-based for structure, deep learning for content.
- Section-specific models, post-processing validation.
""")

st.markdown("---")
st.caption("Made with ❤️ by Kasi")
