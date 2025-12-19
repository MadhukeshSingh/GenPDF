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

st.set_page_config(page_title="InvoiceGPT — Pure Gemini RAG", layout="wide")

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
    font-size: 42px;
    color: #9fdcff;
}
.glass {
    background: rgba(15,20,40,0.78);
    padding: 20px;
    border-radius: 18px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>📄 InvoiceGPT — Pure Page-Wise Gemini RAG</h1>", unsafe_allow_html=True)

# =====================================================
# IMAGE PREPROCESSING (VISION SAFE)
# =====================================================
def prepare_image(img: Image.Image) -> Image.Image:
    if img.mode != "RGB":
        img = img.convert("RGB")
    if img.width > 1600:
        r = 1600 / img.width
        img = img.resize((1600, int(img.height * r)), Image.LANCZOS)
    return img

# =====================================================
# GEMINI OCR — ONE PAGE ONLY
# =====================================================
def gemini_ocr_single_page(img: Image.Image, retries=3) -> str:
    img = prepare_image(img)
    for i in range(retries):
        try:
            resp = client.models.generate_content(
                model=TEXT_MODEL,
                contents=[
                    "Extract ALL readable text from this single document page. "
                    "Preserve numbers, IDs, codes, dates, and formatting. "
                    "Do NOT summarize.",
                    img
                ]
            )
            return resp.text or ""
        except ServerError:
            time.sleep(2 ** i)
    return ""

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
# SESSION STATE
# =====================================================
if "pages" not in st.session_state:
    st.session_state.pages = []

if "history" not in st.session_state:
    st.session_state.history = []

# =====================================================
# PDF UPLOAD & PAGE-WISE INDEXING
# =====================================================
with st.form("upload"):
    pdf_file = st.file_uploader("Upload PDF (scanned or digital)", type=["pdf"])
    process = st.form_submit_button("📄 Process PDF")

if process and pdf_file:
    st.session_state.pages.clear()

    reader = PdfReader(pdf_file)
    doc = fitz.open(stream=pdf_file.getvalue(), filetype="pdf")

    with st.spinner("Indexing pages (one page at a time)…"):
        for i, page in enumerate(reader.pages):
            text = page.extract_text()

            if not text or len(text.strip()) < 50:
                pix = doc[i].get_pixmap(dpi=150)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text = gemini_ocr_single_page(img)

            if not text:
                continue

            text = "\n".join(l.strip() for l in text.splitlines() if l.strip())

            st.session_state.pages.append({
                "page": i + 1,
                "text": text,
                "embedding": embed_text(text)
            })

    st.success(f"Indexed {len(st.session_state.pages)} pages")

# =====================================================
# CHAT — PURE PAGE-WISE GEMINI RAG
# =====================================================
st.markdown("<div class='glass'>", unsafe_allow_html=True)

question = st.text_input("Ask anything from the document")

if st.button("Ask"):
    if not st.session_state.pages:
        st.error("No PDF indexed.")
    else:
        q_emb = embed_text(question)

        # rank pages
        ranked_pages = sorted(
            st.session_state.pages,
            key=lambda p: cosine(q_emb, p["embedding"]),
            reverse=True
        )

        answered = False

        for p in ranked_pages:
            prompt = f"""
Answer the question using ONLY the text below.

Rules:
- Use ONLY this page
- If answer is not present, say: "Not found on this page"
- Do NOT guess
- Be precise

PAGE {p['page']} TEXT:
{p['text'][:4000]}

QUESTION:
{question}
"""
            try:
                resp = client.models.generate_content(
                    model=TEXT_MODEL,
                    contents=[prompt]
                )

                if resp.text and "not found" not in resp.text.lower():
                    st.session_state.history.append(
                        (question, f"📄 Page {p['page']}:\n{resp.text}")
                    )
                    answered = True
                    break

            except Exception:
                continue

        if not answered:
            st.session_state.history.append(
                (question, "❌ Answer not found in any page.")
            )

# =====================================================
# HISTORY
# =====================================================
for q, a in st.session_state.history:
    st.markdown(f"🧑 **You:** {q}")
    st.markdown(f"🤖 **InvoiceGPT:**\n{a}")

st.markdown("</div>", unsafe_allow_html=True)
