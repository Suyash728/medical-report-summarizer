import streamlit as st
import PyPDF2
from transformers import pipeline
import torch


# ==========================================
# 1. APP CONFIGURATION & UI
# ==========================================
st.set_page_config(page_title="Med-Assist: AI Report Summarizer", layout="wide", page_icon="🩺")

st.title("🩺 AI-Powered Medical Report Summarizer")
st.markdown("""
**Privacy-First Architecture:** Your reports are processed entirely in-memory and are never saved to any server or disk. 
**Clinical Intelligence:** Powered by a localized BioBERT model for secure medical entity extraction.
""")

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    target_language = st.selectbox("Target Language", ["English", "Hindi", "Marathi"])
    st.markdown("---")
    st.info("Upload a medical report to generate a simplified summary and extract key clinical entities.")

# ==========================================
# 2. CACHE LOCAL NER MODEL
# ==========================================
# @st.cache_resource ensures the model is loaded only once, saving time.
@st.cache_resource
def load_ner_pipeline():
    try:
        # Load from the local directory where you saved d4data/biomedical-ner-all
        ner_pipe = pipeline(
            "ner", 
            model="./local_ner_model", 
            tokenizer="./local_ner_model", 
            aggregation_strategy="simple", # Groups sub-words into full words
            device=0 if torch.cuda.is_available() else -1 # Use GPU if available
        )
        return ner_pipe
    except Exception as e:
        st.error(f"Failed to load local NER model: {e}")
        return None

ner_pipeline = load_ner_pipeline()

# ==========================================
# 3. PRIVACY-FIRST PDF EXTRACTION
# ==========================================
def extract_text_from_pdf(uploaded_file):
    text = ""
    # File is read directly from memory (BytesIO), ensuring privacy
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

# ==========================================
# 4. VISUAL NER HIGHLIGHTING LOGIC
# ==========================================
# Define colors for different medical entities to impress the faculty
ENTITY_COLORS = {
    "Disease_disorder": "#ffcccc",      # Light Red
    "Medication": "#cce5ff",            # Light Blue
    "Sign_symptom": "#ffe5cc",          # Light Orange
    "Diagnostic_procedure": "#ccffcc"   # Light Green
}

def highlight_entities(text, entities):
    """Replaces identified entities in the text with colored HTML tags."""
    # Sort entities in reverse order of appearance to avoid messing up indices when replacing
    sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)
    
    highlighted_text = text
    for ent in sorted_entities:
        word = text[ent['start']:ent['end']]
        group = ent['entity_group']
        
        # Only highlight groups we defined in ENTITY_COLORS
        if group in ENTITY_COLORS:
            color = ENTITY_COLORS[group]
            html_tag = f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 4px; border: 1px solid #ccc; font-weight: 500;" title="{group}">{word} <span style="font-size: 0.6em; color: #555;">[{group}]</span></span>'
            # Replace precisely using string slicing
            highlighted_text = highlighted_text[:ent['start']] + html_tag + highlighted_text[ent['end']:]
            
    return highlighted_text

# ==========================================
# 5. MAIN EXECUTION FLOW
# ==========================================
uploaded_file = st.file_uploader("📄 Upload Medical PDF", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Extracting text securely in-memory..."):
        document_text = extract_text_from_pdf(uploaded_file)
        
    if document_text.strip() == "":
        st.error("Could not extract text. The PDF might be a scanned image.")
    else:
        st.success("PDF processed successfully (Privacy-First compliant).")
        
        # Two-column layout: Left for NER Visualization, Right for Summary (coming later)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🧬 Clinical Entity Extraction")
            st.caption("Powered by Local BioBERT")
            
            with st.spinner("Running Local BioBERT..."):
                # Run NER on the first 1500 chars to avoid overwhelming the UI/Model
                sample_text = document_text[:1500] 
                entities = ner_pipeline(sample_text)
                
                # Apply HTML Highlighting
                visual_text = highlight_entities(sample_text, entities)
                
                # Render the HTML in Streamlit
                st.markdown("""
                <div style="padding: 15px; border: 1px solid #ddd; border-radius: 8px; background-color: #f9f9f9; height: 400px; overflow-y: auto;">
                    {}
                </div>
                """.format(visual_text.replace('\n', '<br>')), unsafe_allow_html=True)
                
                # Display Legend
                st.markdown("**Legend:**")
                legend_html = " ".join([f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 4px; margin-right: 10px;">{group}</span>' for group, color in ENTITY_COLORS.items()])
                st.markdown(legend_html, unsafe_allow_html=True)
                
        with col2:
            st.subheader("📝 RAG Summary & Translation")
            st.info("LangChain + FAISS + Llama-3 integration will go here.")
            # We will implement the RAG pipeline here in the next step!