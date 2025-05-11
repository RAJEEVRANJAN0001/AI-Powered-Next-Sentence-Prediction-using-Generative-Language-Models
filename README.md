# AI-Powered-Next-Sentence-Prediction-using-Generative-Language-Models
this project is to develop a Next Sentence Prediction system using multiple Generative AI models. The application takes an input sentence from the user and generates several coherent and context-aware next sentence predictions using state-of-the-art NLP models such as Google Gemini, GPT-2, and BERT.

- **Google Gemini (via API)**
- **GPT-2 (from Hugging Face Transformers)**
- **BERT (Masked Language Modeling)**

Built with **Streamlit**, this app features a polished UI, real-time sentence continuation, and follow-up suggestions to enhance creative writing and conversations.

---

## 🚀 Features

- 🔄 Predict next sentence(s) with:
  - 🧠 **Google Gemini**: Powered by Gemini API from Google.
  - 🤖 **GPT-2**: Hugging Face transformer model for generative text.
  - 🧩 **BERT**: Masked Language Model for word prediction in context.
- 🎨 Rich and responsive UI with styled components and background.
- 📊 Sidebar settings:
  - Select model (Gemini / GPT-2 / BERT)
  - Number of predictions (1–5)
  - Max tokens (for GPT-2)
- ❓ Dynamic follow-up buttons for question-based outputs.
- 🔐 Secure API key handling using `.env`.
- 💡 Session state handling for dynamic reruns.
- ⚙️ Styled with custom HTML/CSS inside Streamlit.

---


## 📦 Tech Stack

- **Python 3.9+**
- **Streamlit** — UI framework
- **Hugging Face Transformers** — GPT-2 & BERT
- **Google Gemini API**
- **dotenv** — Secure environment variables
- **Torch** — Model backend execution
- **Requests** — For API calls

---

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/next-sentence-predictor.git
   cd next-sentence-predictor
