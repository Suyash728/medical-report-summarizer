import streamlit as st
import PyPDF2
from transformers import pipeline
import torch

# --- MODERN LANGCHAIN (LCEL) IMPORTS ---
# Notice we completely removed the buggy "langchain.chains" dependency!
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

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
# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    # HARDCODE YOUR API KEY HERE (Notice it must be inside "quotation marks")
    groq_api_key = "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" 
    
    target_language = st.selectbox("Target Language",["English", "Hindi", "Marathi"])
    st.markdown("---")
    st.info("Upload a medical report to generate a simplified summary and extract key clinical entities.")

# ==========================================
# 2. CACHE LOCAL NER MODEL
# ==========================================
@st.cache_resource
def load_ner_pipeline():
    try:
        ner_pipe = pipeline(
            "ner", 
            model="./local_ner_model", 
            tokenizer="./local_ner_model", 
            aggregation_strategy="simple",
            device=0 if torch.cuda.is_available() else -1
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
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

# ==========================================
# 4. VISUAL NER HIGHLIGHTING LOGIC
# ==========================================
ENTITY_COLORS = {
    "Disease_disorder": "#ffcccc",      
    "Medication": "#cce5ff",            
    "Sign_symptom": "#ffe5cc",          
    "Diagnostic_procedure": "#ccffcc"   
}

def highlight_entities(text, entities):
    sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)
    highlighted_text = text
    for ent in sorted_entities:
        word = text[ent['start']:ent['end']]
        group = ent['entity_group']
        if group in ENTITY_COLORS:
            color = ENTITY_COLORS[group]
            html_tag = f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 4px; border: 1px solid #ccc; font-weight: 500;" title="{group}">{word} <span style="font-size: 0.6em; color: #555;">[{group}]</span></span>'
            highlighted_text = highlighted_text[:ent['start']] + html_tag + highlighted_text[ent['end']:]
    return highlighted_text

# ==========================================
# 5. MODERN RAG PIPELINE (PURE LCEL)
# ==========================================
def process_rag_pipeline(text, target_lang, api_key):
    # 1. Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = text_splitter.split_text(text)
    
    # 2. Privacy-First Local Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 3. In-Memory Vector DB
    vectorstore = FAISS.from_texts(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # 4. Initialize LLM via Groq
    llm = ChatGroq(
        temperature=0.1, 
        groq_api_key=api_key, 
        model_name="llama-3.1-8b-instant"
    )
    
    # 5. Prompt Engineering
    system_prompt = (
        "You are a helpful, empathetic medical assistant designed to help patients and rural ASHA workers understand medical reports. "
        "Use the provided context to summarize the patient's test results. "
        "Strictly follow these rules: "
        "1. Use simple, non-jargon language. "
        "2. Clearly list any 'Abnormal Values' (too high or too low) and explain what they generally mean. "
        "3. Keep the tone reassuring. Do NOT diagnose the patient. "
        f"4. You MUST provide your final response entirely in {target_lang}. "
        "\n\nContext:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # 6. Helper function to format retrieved documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # 7. Create the Modern LCEL Chain (Bypasses the buggy langchain.chains module)
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # 8. Execute the chain
    response = rag_chain.invoke("Please summarize this medical report, list key findings, and highlight any abnormal values.")
    return response

# ==========================================
# 6. MAIN EXECUTION FLOW
# ==========================================
uploaded_file = st.file_uploader("📄 Upload Medical PDF", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Extracting text securely in-memory..."):
        document_text = extract_text_from_pdf(uploaded_file)
        
    if document_text.strip() == "":
        st.error("Could not extract text. The PDF might be a scanned image.")
    else:
        st.success("PDF processed successfully (Privacy-First compliant).")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🧬 Clinical Entity Extraction")
            st.caption("Powered by Local BioBERT")
            
            with st.spinner("Running Local BioBERT..."):
                sample_text = document_text[:1500] 
                entities = ner_pipeline(sample_text)
                visual_text = highlight_entities(sample_text, entities)
                
                st.markdown("""
                <div style="padding: 15px; border: 1px solid #ddd; border-radius: 8px; background-color: #f9f9f9; height: 500px; overflow-y: auto; font-family: monospace; font-size: 14px;">
                    {}
                </div>
                """.format(visual_text.replace('\n', '<br>')), unsafe_allow_html=True)
                
                st.markdown("**Legend:**")
                legend_html = " ".join([f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 4px; margin-right: 10px; font-size: 12px;">{group}</span>' for group, color in ENTITY_COLORS.items()])
                st.markdown(legend_html, unsafe_allow_html=True)
                
        with col2:
            st.subheader(f"📝 Simplified Summary ({target_language})")
            st.caption("Powered by RAG (FAISS + Llama-3)")
            
            if not groq_api_key:
                st.warning("⚠️ Please enter your Groq API Key in the sidebar to generate the summary.")
            else:
                with st.spinner(f"Generating summary in {target_language}..."):
                    try:
                        summary_output = process_rag_pipeline(document_text, target_language, groq_api_key)
                        
                        st.markdown("""
                        <div style="padding: 15px; border: 1px solid #ddd; border-radius: 8px; background-color: #eef7ff; height: 500px; overflow-y: auto;">
                            {}
                        </div>
                        """.format(summary_output.replace('\n', '<br>')), unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Error in RAG Pipeline: {e}")