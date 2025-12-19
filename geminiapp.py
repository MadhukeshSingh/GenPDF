import os
import io
import numpy as np
import streamlit as st
from pypdf import PdfReader
import google.generativeai as genai
from PIL import Image

# =====================================================
# GEMINI SETUP (SAFE)
# =====================================================
API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

TEXT_MODEL = genai.GenerativeModel("gemini-2.5-flash")
VISION_MODEL = genai.GenerativeModel("gemini-2.5-flash")

st.set_page_config(page_title="InvoiceGPT — Gemini Vision RAG", layout="wide")

# =====================================================
# UI
# =====================================================
st.markdown("""
<style>
.stApp {
    background:
        linear-gradient(rgba(8,12,25,0.96), rgba(8,12,25,0.99)),
        url("https://images.unsplash.com/photo-1535223289827-42f1e9919769?auto=format&fit=crop&w=2000&q=80");
    background-size: cover;
    color: #e6f2ff;
}

h1 {
    text-align: center;
    font-size: 46px;
    color: #9fdcff;
    text-shadow: 0 0 25px rgba(0,200,255,0.7);
}

.glass {
    background: rgba(15,20,40,0.78);
    backdrop-filter: blur(14px);
    padding: 20px;
    border-radius: 18px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>📄 InvoiceGPT — Gemini Vision RAG</h1>", unsafe_allow_html=True)

# =====================================================
# HELPERS
# =====================================================
def gemini_ocr_from_image(img: Image.Image) -> str:
    """Gemini Vision OCR"""
    response = VISION_MODEL.generate_content(
        [
            "Extract all readable text from this invoice page. Preserve numbers exactly.",
            img
        ]
    )
    return response.text or ""

def embed_text(text: str) -> np.ndarray:
    emb = genai.embed_content(
        model="models/embedding-001",
        content=text
    )
    return np.array(emb["embedding"])

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# =====================================================
# SESSION STATE
# =====================================================
if "pages" not in st.session_state:
    st.session_state.pages = []

if "history" not in st.session_state:
    st.session_state.history = []

# =====================================================
# PDF UPLOAD
# =====================================================
with st.form("upload"):
    pdf_file = st.file_uploader("Upload PDF (scanned or digital)", type=["pdf"])
    process = st.form_submit_button("📄 Process PDF")

if process and pdf_file:
    st.session_state.pages.clear()

    reader = PdfReader(pdf_file)

    with st.spinner("Extracting pages with Gemini Vision..."):
        for i, page in enumerate(reader.pages):
            text = page.extract_text()

            # 🔥 Vision OCR fallback
            if not text or len(text.strip()) < 50:
                # render page as image using PyPDF
                img = page.to_image(resolution=200).original
                text = gemini_ocr_from_image(img)

            if not text or len(text.strip()) < 50:
                continue

            st.session_state.pages.append({
                "page": i + 1,
                "text": text,
                "embedding": embed_text(text)
            })

    st.success(f"Indexed {len(st.session_state.pages)} pages")

# =====================================================
# CHAT
# =====================================================
st.markdown("<div class='glass'>", unsafe_allow_html=True)

question = st.text_input(
    "Ask invoice questions (invoice no, PO, IRN, HSN, compare pages):"
)

if st.button("Ask"):
    if not st.session_state.pages:
        st.error("❌ No PDF indexed. Click **Process PDF** first.")
    else:
        q_emb = embed_text(question)

        ranked = sorted(
            st.session_state.pages,
            key=lambda p: cosine(q_emb, p["embedding"]),
            reverse=True
        )

        top_pages = ranked[:2]

        answers = []

        for p in top_pages:
            prompt = f"""
You are an invoice extraction engine.

RULES:
- Use ONLY the text below
- Extract exact values
- Mention page number
- Never hallucinate

PAGE {p['page']} TEXT:
{p['text']}

QUESTION:
{question}
"""
            response = TEXT_MODEL.generate_content(prompt)
            answers.append(f"📄 Page {p['page']}:\n{response.text}")

        st.session_state.history.append((question, "\n\n".join(answers)))

# =====================================================
# HISTORY
# =====================================================
for q, a in st.session_state.history:
    st.markdown(f"🧑 **You:** {q}")
    st.markdown(f"🤖 **InvoiceGPT:**\n{a}")

st.markdown("</div>", unsafe_allow_html=True)
