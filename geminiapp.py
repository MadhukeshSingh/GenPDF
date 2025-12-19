import io
import numpy as np
import streamlit as st
import fitz  # PyMuPDF
from pypdf import PdfReader
from PIL import Image
from google import genai

# =====================================================
# GEMINI SETUP
# =====================================================
API_KEY = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=API_KEY)

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
    font-size: 44px;
    color: #9fdcff;
    text-shadow: 0 0 20px rgba(0,200,255,0.7);
}
.glass {
    background: rgba(15,20,40,0.78);
    backdrop-filter: blur(14px);
    padding: 20px;
    border-radius: 18px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>📄 InvoiceGPT — Gemini Vision (Scanned + Digital)</h1>", unsafe_allow_html=True)

# =====================================================
# HELPERS
# =====================================================
def gemini_ocr(image: Image.Image) -> str:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            "Extract all readable text from this invoice page. Preserve numbers exactly.",
            image
        ]
    )
    return response.text or ""

def embed_text(text: str) -> np.ndarray:
    result = client.models.embed_content(
        model="text-embedding-004",
        contents=[text]
    )
    return np.array(result.embeddings[0].values)

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

    # Text-based extraction
    reader = PdfReader(pdf_file)

    # Image-based extraction (PyMuPDF)
    doc = fitz.open(stream=pdf_file.getvalue(), filetype="pdf")

    with st.spinner("Processing pages with Gemini Vision OCR..."):
        for i, page in enumerate(reader.pages):
            text = page.extract_text()

            if not text or len(text.strip()) < 50:
                pix = doc[i].get_pixmap(dpi=200)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text = gemini_ocr(img)

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

question = st.text_input("Ask invoice questions:")

if st.button("Ask"):
    if not st.session_state.pages:
        st.error("❌ No PDF indexed. Click Process PDF first.")
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
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            answers.append(f"📄 Page {p['page']}:\n{resp.text}")

        st.session_state.history.append((question, "\n\n".join(answers)))

# =====================================================
# HISTORY
# =====================================================
for q, a in st.session_state.history:
    st.markdown(f"🧑 **You:** {q}")
    st.markdown(f"🤖 **InvoiceGPT:**\n{a}")

st.markdown("</div>", unsafe_allow_html=True)

