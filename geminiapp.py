import time
import numpy as np
import streamlit as st
import fitz  # PyMuPDF
from pypdf import PdfReader
from PIL import Image
from google import genai
from google.genai.errors import ServerError

# =====================================================
# GEMINI SETUP
# =====================================================
API_KEY = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=API_KEY)

TEXT_MODEL = "gemini-2.5-flash"
EMBED_MODEL = "text-embedding-004"

st.set_page_config(page_title="InvoiceGPT — Gemini RAG", layout="wide")

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

st.markdown("<h1>📄 InvoiceGPT — Gemini Vision RAG</h1>", unsafe_allow_html=True)

# =====================================================
# IMAGE PREPROCESSING (CRITICAL)
# =====================================================
def prepare_image(img: Image.Image) -> Image.Image:
    if img.mode != "RGB":
        img = img.convert("RGB")

    max_width = 1600
    if img.width > max_width:
        ratio = max_width / img.width
        img = img.resize(
            (max_width, int(img.height * ratio)),
            Image.LANCZOS
        )
    return img

# =====================================================
# GEMINI OCR (SAFE + RETRY)
# =====================================================
def gemini_ocr(img: Image.Image, retries=3) -> str:
    img = prepare_image(img)

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=TEXT_MODEL,
                contents=[
                    "Extract ALL readable text from this invoice page. "
                    "Preserve invoice numbers, dates, GST, IRN, HSN, quantities, "
                    "amounts EXACTLY. Do not summarize.",
                    img
                ]
            )
            return response.text or ""
        except ServerError:
            if attempt == retries - 1:
                return ""
            time.sleep(2 ** attempt)

# =====================================================
# EMBEDDINGS
# =====================================================
def embed_text(text: str) -> np.ndarray:
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=[text]
    )
    return np.array(result.embeddings[0].values)

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# =====================================================
# SAFE GENERATION (FIXES ClientError)
# =====================================================
def safe_generate_text(prompt: str, retries=3) -> str:
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=TEXT_MODEL,
                contents=[prompt]  # MUST be list
            )
            return resp.text or ""
        except Exception:
            if attempt == retries - 1:
                return "⚠️ Unable to answer due to Gemini limits."
            time.sleep(2 ** attempt)

def limit_text(text: str, max_chars=6000) -> str:
    return text if len(text) <= max_chars else text[:max_chars]

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
    doc = fitz.open(stream=pdf_file.getvalue(), filetype="pdf")

    with st.spinner("Indexing pages (text + OCR)…"):
        for i, page in enumerate(reader.pages):
            text = page.extract_text()

            # OCR fallback for scanned pages
            if not text or len(text.strip()) < 50:
                pix = doc[i].get_pixmap(dpi=150)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text = gemini_ocr(img)

            if not text or len(text.strip()) < 50:
                continue

            # light cleanup
            text = "\n".join(
                line.strip() for line in text.splitlines() if line.strip()
            )

            st.session_state.pages.append({
                "page": i + 1,
                "text": text,
                "embedding": embed_text(text)
            })

    st.success(f"Indexed {len(st.session_state.pages)} pages")

# =====================================================
# CHAT (RAG)
# =====================================================
st.markdown("<div class='glass'>", unsafe_allow_html=True)

question = st.text_input(
    "Ask invoice questions (invoice no, PO, IRN, HSN, compare pages):"
)

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

        top_pages = ranked[:2]  # SMALL CONTEXT

        answers = []
        for p in top_pages:
            page_text = limit_text(p["text"])

            prompt = f"""
You are an invoice extraction engine.

STRICT RULES:
- Use ONLY the text below
- Extract EXACT values
- If multiple values exist, list all
- Mention page number
- Never hallucinate

PAGE {p['page']} TEXT:
{page_text}

QUESTION:
{question}
"""
            answer = safe_generate_text(prompt)
            answers.append(f"📄 Page {p['page']}:\n{answer}")

        st.session_state.history.append((question, "\n\n".join(answers)))

# =====================================================
# HISTORY
# =====================================================
for q, a in st.session_state.history:
    st.markdown(f"🧑 **You:** {q}")
    st.markdown(f"🤖 **InvoiceGPT:**\n{a}")

st.markdown("</div>", unsafe_allow_html=True)
