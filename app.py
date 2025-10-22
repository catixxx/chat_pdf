import os
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import platform

# --- 🌸 Estilos femeninos en lilas y celestes ---
st.markdown("""
    <style>
        /* Fondo general */
        .stApp {
            background-color: #e8f5ff; /* Celeste muy suave */
            font-family: 'Poppins', sans-serif;
        }

        /* Título principal */
        h1 {
            color: #8e7cc3; /* Lila elegante */
            text-align: center;
            font-weight: 700;
            font-size: 2.3em;
            margin-bottom: 0.2em;
        }

        /* Subtítulos */
        h2, h3 {
            color: #7a6ccf;
            font-weight: 600;
        }

        /* Barra lateral */
        section[data-testid="stSidebar"] {
            background-color: #f2e9fb;
            border-radius: 15px;
            color: #6f52b3;
        }

        /* Cuadros de texto */
        .stTextInput > div > div > input, textarea {
            border-radius: 10px !important;
            border: 1px solid #bda9e0 !important;
            background-color: #ffffff !important;
        }

        /* Botones */
        button[kind="primary"] {
            background-color: #9cc9f0 !important;
            color: white !important;
            border-radius: 10px !important;
            border: none !important;
            font-weight: 600 !important;
        }

        button[kind="primary"]:hover {
            background-color: #8ebbe0 !important;
        }

        /* Imagen centrada */
        [data-testid="stImage"] img {
            display: block;
            margin-left: auto;
            margin-right: auto;
            border-radius: 15px;
            box-shadow: 0px 0px 10px rgba(140, 180, 255, 0.5);
        }

        /* Cuadros de información */
        .stAlert {
            border-radius: 12px !important;
        }

        /* Texto y markdown */
        .stMarkdown p {
            color: #4b4b7a;
            font-size: 1.05em;
        }
    </style>
""", unsafe_allow_html=True)

# --- 💬 Título y descripción ---
st.title('💜 Generación Aumentada por Recuperación (RAG)')
st.write(f"Versión de Python: **{platform.python_version()}**")

# --- Imagen decorativa ---
try:
    image = Image.open('Chat_pdf.png')
    st.image(image, width=350)
except Exception as e:
    st.warning(f"No se pudo cargar la imagen: {e}")

# --- Barra lateral informativa ---
with st.sidebar:
    st.subheader("🌷 Asistente de Análisis de PDF 🌷")
    st.markdown("""
    Este agente inteligente te ayudará a:
    - Analizar el contenido de un documento PDF  
    - Responder preguntas basadas en el texto  
    - Resumir y explicar fragmentos específicos  

    💡 *Solo necesitas subir tu archivo y escribir tu pregunta.*
    """)

# --- Clave de API ---
ke = st.text_input('🔑 Ingresa tu Clave de OpenAI', type="password")
if ke:
    os.environ['OPENAI_API_KEY'] = ke
else:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar 💬")

# --- Cargar PDF ---
pdf = st.file_uploader("📄 Carga el archivo PDF", type="pdf")

# --- Procesamiento del PDF ---
if pdf is not None and ke:
    try:
        # Extraer texto
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        st.info(f"✨ Texto extraído: {len(text)} caracteres")
        
        # Dividir texto
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=20,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        st.success(f"🌸 Documento dividido en {len(chunks)} fragmentos")

        # Crear embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # Interfaz de pregunta
        st.subheader("💬 Escribe qué quieres saber sobre el documento")
        user_question = st.text_area(" ", placeholder="Escribe tu pregunta aquí...")

        # Procesar pregunta
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            llm = OpenAI(temperature=0, model_name="gpt-4o")
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)

            # Mostrar respuesta
            st.markdown("### 🌈 Respuesta del asistente:")
            st.markdown(f"<div style='background-color:#f2e9fb; padding:15px; border-radius:10px; color:#4b4b7a;'>{response}</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"⚠️ Error al procesar el PDF: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

elif pdf is not None and not ke:
    st.warning("🔒 Por favor ingresa tu clave de API de OpenAI para continuar.")
else:
    st.info("💠 Carga un archivo PDF para comenzar tu análisis.")
a comenzar")
