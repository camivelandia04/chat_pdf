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

# App title and presentation
st.title('Asistente Inteligente de Documentos 🤖📄')
st.write("Ejecutándose en Python:", platform.python_version())

# Load and display image
try:
    image = Image.open('Bot.jpg')
    st.image(image, width=350)
except Exception as e:
    st.warning(f"No se pudo cargar la imagen: {e}")

# Sidebar information
with st.sidebar:
    st.subheader("🔍 Analiza tu documento fácilmente")
    st.write("Sube un PDF y haz preguntas para obtener respuestas rápidas basadas en su contenido.")

# Get API key from user
ke = st.text_input('Introduce tu API Key de OpenAI', type="password")
if ke:
    os.environ['OPENAI_API_KEY'] = ke
else:
    st.warning("Necesitas ingresar tu API Key para usar la aplicación")

# PDF uploader
pdf = st.file_uploader("Sube tu archivo PDF aquí", type="pdf")

# Process the PDF if uploaded
if pdf is not None and ke:
    try:
        # Extract text from PDF
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        st.info(f"Se extrajeron {len(text)} caracteres del documento")
        
        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=20,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        st.success(f"El documento fue dividido en {len(chunks)} partes para su análisis")
        
        # Create embeddings and knowledge base
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        # User question interface
        st.subheader("💡 Haz una pregunta sobre tu documento")
        user_question = st.text_area("", placeholder="Ej: ¿Cuál es la idea principal del texto?")
        
        # Process question when submitted
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            
            llm = OpenAI(temperature=0, model_name="gpt-4o-mini-2024-07-18")
            
            chain = load_qa_chain(llm, chain_type="stuff")
            
            response = chain.run(input_documents=docs, question=user_question)
            
            # Display the response
            st.markdown("### 📢 Resultado:")
            st.markdown(response)
                
    except Exception as e:
        st.error(f"Ocurrió un error al analizar el documento: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

elif pdf is not None and not ke:
    st.warning("Debes ingresar tu API Key antes de continuar")

else:
    st.info("Sube un archivo PDF para empezar a interactuar con el asistente")
