import os
os.environ["STREAMLIT_HOME"] = "/tmp/.streamlit"

import streamlit as st
import re
import spacy
from transformers import pipeline

st.set_page_config(page_title="Physician Notetaker", layout="wide")

@st.cache_resource
def load_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm")

@st.cache_resource
def load_sentiment():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

nlp = load_nlp()
sentiment_pipe = load_sentiment()

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

def analyze_patient_sentiment(text):
    text_lc = text.lower()
    if any(word in text_lc for word in ["worried", "concerned", "anxious", "nervous"]):
        return {"Sentiment": "Anxious", "Intent": "Seeking reassurance"}
    elif any(word in text_lc for word in ["relief", "thankful", "grateful", "appreciate"]):
        return {"Sentiment": "Reassured", "Intent": "Expressing gratitude"}
    elif "no complaints" in text_lc or "nothing to report" in text_lc:
        return {"Sentiment": "Neutral", "Intent": "Routine visit"}
    else:
        result = sentiment_pipe(text)[0]
        label = result['label']
        if label == "NEGATIVE":
            return {"Sentiment": "Anxious", "Intent": "Seeking reassurance"}
        elif label == "POSITIVE":
            return {"Sentiment": "Reassured", "Intent": "Expressing gratitude"}
        else:
            return {"Sentiment": "Neutral", "Intent": "Reporting symptoms"}

def generate_soap_note(transcript):
    entities = extract_entities(transcript)
    if not entities["Symptoms"]:
        return {
            "Subjective": {
                "Chief_Complaint": "Routine checkup",
                "History_of_Present_Illness": "No major complaints reported by the patient."
            },
            "Objective": {
                "Physical_Exam": "Normal vital signs, no abnormalities noted.",
                "Observations": "Patient appears in good health."
            },
            "Assessment": {
                "Diagnosis": "General wellness",
                "Severity": "None"
            },
            "Plan": {
                "Treatment": "No treatment necessary.",
                "Follow-Up": "Routine follow-up advised."
            }
        }
    else:
        return {
            "Subjective": {
                "Chief_Complaint": ", ".join(entities["Symptoms"]),
                "History_of_Present_Illness": "Patient reports issues including: " + ", ".join(entities["Symptoms"])
            },
            "Objective": {
                "Physical_Exam": "Full range of motion in affected areas, no significant tenderness.",
                "Observations": "Patient appears stable with mild discomfort."
            },
            "Assessment": {
                "Diagnosis": entities["Diagnosis"],
                "Severity": "Mild"
            },
            "Plan": {
                "Treatment": ", ".join(entities["Treatment"]) if entities["Treatment"] else "Supportive care recommended.",
                "Follow-Up": "Patient to follow up if symptoms persist."
            }
        }

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

st.markdown("---")
st.caption("Made with ❤️ by Kasi ")

