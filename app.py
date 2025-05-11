import os
import requests
import streamlit as st
from dotenv import load_dotenv
from transformers import GPT2LMHeadModel, GPT2Tokenizer, BertForMaskedLM, BertTokenizer
import torch

# Load environment variables
load_dotenv()

# --- Function Definitions ---

def predict_next_sentences_gemini(input_text, num_predictions=3):
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{
            "parts": [{"text": input_text}]
        }]
    }
    try:
        response = requests.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        if "candidates" in data:
            return [item['content']['parts'][0]['text'] for item in data['candidates']][:num_predictions]
        else:
            return ["No predictions available."]
    except requests.exceptions.RequestException as e:
        return [f"Error: {str(e)}"]

def predict_next_sentences_gpt2(input_text, num_predictions=3, max_length=50):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    inputs = tokenizer(input_text, return_tensors="pt")

    outputs = model.generate(
        inputs['input_ids'],
        max_length=max_length,
        num_return_sequences=num_predictions,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        no_repeat_ngram_size=2
    )
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

def predict_next_sentences_bert(input_text, num_predictions=3):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")

    input_text = input_text.strip()
    if not input_text.endswith("."):
        input_text += "."
    masked_text = input_text + " [MASK]"
    inputs = tokenizer(masked_text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.topk(outputs.logits[0, -1], num_predictions)
    predicted_ids = predictions.indices
    return [tokenizer.decode([predicted_id]) for predicted_id in predicted_ids]

# --- Streamlit UI Setup ---

st.set_page_config(page_title="🎨 Creative Next Sentence Predictor", page_icon="✨", layout="wide")

# --- Background Styling with Image and Custom Theme ---
st.markdown("""
    <style>
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1746768934151-8c5cb84bcf11?q=80&w=2940&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }

        body {
            background-color: #f7f2fa;
            font-family: 'Segoe UI', sans-serif;
        }

        .title {
            font-size: 50px;
            color: #6a0dad;
            text-align: center;
            margin-top: 20px;
        }

        .subtitle {
            color: #4b0082;
            font-size: 20px;
            text-align: center;
            margin-bottom: 30px;
        }

        .stTextArea textarea {
            background-color: #fffaf0;
            font-size: 16px;
            border-radius: 10px;
        }

        .prediction-box {
            background-color: #f0f8ff;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 2px 2px 10px #e0e0e0;
            margin-bottom: 10px;
        }

        .main > div {
            background-color: rgba(255, 255, 255, 0.85);
            border-radius: 12px;
            padding: 20px;
        }

        .stButton > button {
            background-color: #6a0dad;
            color: white;
            font-size: 16px;
            border-radius: 10px;
        }

        h1, h2, h3 {
            color: #4b0082;
        }
    </style>
""", unsafe_allow_html=True)

# --- Title Section ---
st.markdown("<div class='title'>✨ AI-Powered Next Sentence Predictor ✨</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Unlock the magic of future sentences using Google Gemini, GPT-2, and BERT!</div>", unsafe_allow_html=True)

# --- Sidebar Options ---
st.sidebar.markdown("🛠️ **Settings**")
num_return_sequences = st.sidebar.slider("🎯 Number of Predictions", 1, 5, 3)
max_tokens = st.sidebar.slider("🔢 Max Tokens (Length)", 10, 100, 50)
model_choice = st.sidebar.radio("🤖 Choose a Model", ["Google Gemini", "Hugging Face GPT-2", "Hugging Face BERT"])

# --- Initialize session state for input text ---
if 'input_text' not in st.session_state:
    st.session_state['input_text'] = ""

# --- Main Input ---
input_text = st.text_area("📝 Enter your sentence below:", st.session_state['input_text'], height=150)

# --- Prediction Trigger ---
if st.button("🚀 Predict Next Sentences"):
    if input_text.strip():
        st.subheader(f"🎉 Predictions from {model_choice}")
        with st.spinner("Thinking hard... 🤔"):
            if model_choice == "Google Gemini":
                outputs = predict_next_sentences_gemini(input_text, num_return_sequences)
            elif model_choice == "Hugging Face GPT-2":
                outputs = predict_next_sentences_gpt2(input_text, num_return_sequences, max_tokens)
            elif model_choice == "Hugging Face BERT":
                outputs = predict_next_sentences_bert(input_text, num_return_sequences)

            for i, text in enumerate(outputs):
                st.markdown(f"<div class='prediction-box'><strong>{i + 1}️⃣</strong> {text}</div>", unsafe_allow_html=True)

                # If the text ends with a question, provide a button to continue with it
                if "?" in text and ("What" in text or "How" in text or "Why" in text):
                    follow_up_question = text.strip().split("\n")[-1]
                    if st.button(f"❓ Continue with: {follow_up_question}", key=f"followup_{i}"):
                        st.session_state['input_text'] = follow_up_question
                        st.experimental_rerun()
    else:
        st.warning("🚨 Please enter a sentence before clicking Predict.")

# --- Footer ---
st.markdown("""
<hr style='border: 1px solid #bbb;'/>
<center>
    Made with ❤️ by <strong>Rajeev Ranjan Pratap Singh</strong><br>
    Powered by <em>Gemini API, Hugging Face Transformers, and Streamlit</em>
</center>
""", unsafe_allow_html=True)
