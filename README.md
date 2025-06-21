# 🩺 Physician Notetaker: Medical NLP & Sentiment Analysis

An AI-powered medical transcription assistant that analyzes physician-patient dialogues, extracts clinical insights, performs sentiment & intent detection, and generates structured medical summaries including SOAP notes.

---

## 🚀 Overview

This project leverages NLP techniques to transform unstructured physician-patient transcripts into structured medical reports. It performs:

- ✅ Named Entity Recognition (NER) for symptoms, treatments, diagnoses, and prognoses  
- ✅ Medical keyword extraction  
- ✅ Sentiment & intent analysis (e.g., Anxious, Neutral, Reassured)  
- ✅ Automated SOAP note generation  
- ✅ User-friendly *Streamlit* web interface  
- ✅ No need for a local LLM – cloud-ready and lightweight  

---

## 📦 Project Structure


physician-notetaker/
├── app.py              # Main Streamlit app
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation


---

## ⚡ Quickstart

### 1. Clone the Repository

bash
git clone https://github.com/yourusername/physician-notetaker.git
cd physician-notetaker


### 2. Install Dependencies

bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm


### 3. Run the App

bash
streamlit run app.py


---

## 🖥 Usage Instructions

1. *Paste a transcript* of a physician-patient conversation into the text box.
2. Click *"Analyze Transcript"* to extract:
   - Structured medical report (in JSON)
   - Medical keywords
   - SOAP note (Subjective, Objective, Assessment, Plan)
3. Paste a *patient statement* separately and click *"Analyze Sentiment & Intent"* for emotional and intent classification.

---

## ✨ Example Outputs

### 🔹 Structured Medical Report

json
{
  "Patient_Name": "Janet Jones",
  "Symptoms": ["Neck pain", "Back pain", "Head impact"],
  "Diagnosis": "Whiplash injury",
  "Treatment": ["10 physiotherapy sessions", "Painkillers"],
  "Current_Status": "Occasional backache",
  "Prognosis": "Full recovery expected within six months"
}


### 🔹 Medical Keywords

json
[
  "Whiplash injury",
  "10 physiotherapy sessions",
  "Painkillers",
  "Back pain",
  "Neck pain",
  "Head impact",
  "Trouble sleeping"
]


### 🔹 SOAP Note

json
{
  "Subjective": {
    "Chief_Complaint": "Neck and back pain",
    "History_of_Present_Illness": "Patient had a car accident, experienced pain for four weeks, now occasional back pain."
  },
  "Objective": {
    "Physical_Exam": "Full range of motion in cervical and lumbar spine, no tenderness.",
    "Observations": "Patient appears in normal health, normal gait."
  },
  "Assessment": {
    "Diagnosis": "Whiplash injury and lower back strain",
    "Severity": "Mild, improving"
  },
  "Plan": {
    "Treatment": "Continue physiotherapy as needed, use analgesics for pain relief.",
    "Follow-Up": "Patient to return if pain worsens or persists beyond six months."
  }
}


### 🔹 Sentiment & Intent Detection

*Input:*

I'm a bit worried about my back pain, but I hope it gets better soon.


*Output:*
json
{
  "Sentiment": "Anxious",
  "Intent": "Seeking reassurance"
}


---

## 🧪 Test Case Ideas

- *Symptom-based*
  
  Patient: I slipped on the stairs last week and twisted my ankle. It’s still swollen and hurts when I walk.
  

- *Reassuring sentiment*
  
  That’s a relief!
  

- *Negated symptoms*
  
  Patient: I have no pain, no discomfort, and no trouble sleeping.
  

- *Minimal description*
  
  Patient: I feel tired lately.
  

---

## 🧠 Methodology

### 🔍 Handling Ambiguity & Missing Data
- Use *negation detection* to reduce false positives.
- If data is absent, return *"Not specified"*.
- Prioritize clinician statements for critical fields.

### 🤖 NLP Techniques & Models
- *NER & Summarization*: spaCy, SciSpacy, ClinicalBERT
- *Sentiment Detection*: Fine-tuned BERT/ClinicalBERT models using supervised datasets
- *SOAP Mapping*: Fine-tune T5 or BART with structured annotations or use rule-based segmentation.

---

## 📚 Datasets for Training & Fine-tuning

- *i2b2/UTHealth Clinical Notes*
- *MIMIC-III*
- *MEDIQA Dataset*
- *Patient Experience Mining Corpora*

---

## 📄 requirements.txt

text
streamlit
spacy
transformers
torch


---

## 🛠 Requirements

- Python 3.8+
- Streamlit
- spaCy
- Transformers
- PyTorch

---

## 📚 References

- [spaCy Documentation](https://spacy.io/usage)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [Streamlit Docs](https://docs.streamlit.io/)

---

## 📬 Submission Guidelines

Please submit:
- app.py
- requirements.txt
- README.md

as per your assignment instructions.

---

## 💡 Future Ideas

- Multi-language support  
- Real-time speech-to-text integration  
- Integration with electronic health records (EHR)  
- Smart auto-fill for physicians

---

## 🤝 Contributions

Have ideas or want to improve this project? Feel free to open an [Issue](https://github.com/yourusername/physician-notetaker/issues) or submit a PR!

---

## 🔗 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

> ⚠ This project is for academic and research purposes only and not intended for real-time clinical deployment without validation.
