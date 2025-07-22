import streamlit as st
import google.generativeai as genai
import fitz # PyMuPDF
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURACIÓN ---
# ¡IMPORTANTE! Reemplaza 'TU_CLAVE_API_DE_GEMINI' con tu clave real.
# Si despliegas en Streamlit Community Cloud, configura esta variable como "Secret".
API_KEY = st.secrets["GEMINI_API_KEY"] if "GEMINI_API_KEY" in st.secrets else "AIzaSyDVSsee1vuaoJTSDVx8AY9_DSH2y-2YQiw"

genai.configure(api_key=API_KEY)

# Modelo para embeddings
EMBEDDING_MODEL = "models/embedding-001"
# Modelo para generación de texto
GENERATION_MODEL = "models/gemini-1.5-flash-latest"

# Tamaño de los fragmentos de texto (chunks)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- RUTA DE LA CARPETA DE PDFs (¡Asegúrate de que esta carpeta exista y contenga tus PDFs!) ---
# Si despliegas, esta carpeta debe estar en la raíz de tu proyecto de GitHub.
PDF_FOLDER_PATH = "documentos_pdf"

# --- FUNCIONES DE PROCESAMIENTO (ligeramente adaptadas) ---

def extract_text_from_pdf(pdf_path):
    """Extrae texto de un archivo PDF dado su ruta."""
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()
        doc.close()
    except Exception as e:
        st.error(f"Error al extraer texto del PDF {pdf_path}: {e}")
        return None
    return text

def chunk_text(text, chunk_size, chunk_overlap):
    """Divide el texto en fragmentos (chunks) con superposición."""
    if not text:
        return []
    chunks = []
    current_position = 0
    while current_position < len(text):
        end_position = min(current_position + chunk_size, len(text))
        chunk = text[current_position:end_position]
        chunks.append(chunk)
        current_position += (chunk_size - chunk_overlap)
        if current_position >= len(text) - chunk_overlap and end_position == len(text):
            break
        elif current_position >= len(text):
            break
    return chunks

def get_embeddings(texts):
    """Genera embeddings para una lista de textos."""
    try:
        response = genai.embed_content(model=EMBEDDING_MODEL, content=texts)
        return np.array(response['embedding'])
    except Exception as e:
        st.error(f"Error al generar embeddings: {e}")
        return np.array([])

def find_relevant_chunks(query_embedding, document_embeddings, document_chunks, top_k=3):
    """
    Encuentra los fragmentos de documentos más relevantes para una consulta
    usando similitud coseno.
    """
    if document_embeddings.size == 0 or query_embedding.size == 0:
        return []
    similarities = cosine_similarity([query_embedding], document_embeddings)[0]
    top_k_indices = np.argsort(similarities)[::-1][:top_k]
    relevant_chunks = [document_chunks[i] for i in top_k_indices]
    return relevant_chunks

# --- NUEVA FUNCIÓN PARA CARGAR Y CACHEAR DATOS ---
@st.cache_data(show_spinner="Cargando y procesando documentos... Esto puede tomar un momento.")
def load_and_process_pdfs(folder_path):
    """
    Carga, extrae texto, divide en chunks y genera embeddings de PDFs
    desde una carpeta específica. Esta función se cachea para eficiencia.
    """
    all_chunks = []
    all_chunk_embeddings_list = []
    
    if not os.path.exists(folder_path):
        st.error(f"Error: La carpeta '{folder_path}' no existe. Por favor, asegúrate de que esté en la raíz de tu proyecto y contenga PDFs.")
        return [], np.array([])

    pdf_files_found = False
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            pdf_files_found = True
            st.info(f"Extrayendo texto de: {filename}")
            text = extract_text_from_pdf(pdf_path)
            if text:
                chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
                all_chunks.extend(chunks)
                
                # Procesa los chunks en lotes para los embeddings
                batch_size = 100 
                for i in range(0, len(chunks), batch_size):
                    batch_chunks = chunks[i:i + batch_size]
                    chunk_embeddings = get_embeddings(batch_chunks)
                    if chunk_embeddings.size > 0:
                        all_chunk_embeddings_list.append(chunk_embeddings)
                    else:
                        st.warning(f"Advertencia: No se pudieron generar embeddings para un lote de {filename}.")
            else:
                st.warning(f"No se pudo extraer texto de {filename}. Se omitirá.")
    
    if not pdf_files_found:
        st.error(f"No se encontraron archivos PDF en la carpeta '{folder_path}'. Por favor, coloca tus PDFs allí.")
        return [], np.array([])

    if not all_chunks or not all_chunk_embeddings_list:
        st.error("No se encontró texto procesable en ningún PDF o no se pudieron generar embeddings válidos.")
        return [], np.array([])

    # Aplanar la lista de arrays de embeddings en un solo array numpy
    all_chunk_embeddings_np = np.vstack(all_chunk_embeddings_list)
    st.success(f"¡Documentos cargados! {len(all_chunks)} fragmentos listos para tu agente.")
    return all_chunks, all_chunk_embeddings_np

# --- FUNCIÓN PRINCIPAL DEL AGENTE DE GEMINI (adaptada para humanización) ---
def ask_gemini_with_context(query, context_chunks):
    """
    Formula una pregunta a Gemini, proporcionando el contexto relevante y un tono más humano.
    """
    model = genai.GenerativeModel(GENERATION_MODEL)

    # --- INSTRUCCIONES PARA HUMANIZAR LA RESPUESTA ---
    system_prompt = (
        "Eres un asistente amigable, cercano y muy útil, especializado en explicar información "
        "de documentos de manera sencilla y amable. Responde siempre en español, con un tono cálido "
        "y comprensivo. Usa un lenguaje natural, como si estuvieras hablando con un amigo."
        "Si la información necesaria NO está explícitamente en el texto proporcionado, "
        "di amablemente que no puedes responder con la información actual y ofrece buscar en tu conocimiento general "
        "SOLO si el usuario lo solicita. No inventes datos. Si la respuesta es corta, sé conciso pero siempre amigable."
    )

    if context_chunks:
        context = "\n\n".join(context_chunks)
        # Aquí combinamos el system_prompt con el contexto y la pregunta
        full_prompt = (f"{system_prompt}\n\n"
                       f"Aquí está la información relevante que encontré para ti:\n\n---\n{context}\n---\n\n"
                       f"Ahora, por favor, responde a esta pregunta: '{query}'")
    else:
        # Si no hay contexto, informa amablemente que buscará en su conocimiento general
        full_prompt = (f"{system_prompt}\n\n"
                       f"No pude encontrar información específica en los documentos que tengo para responder a eso. "
                       f"Sin embargo, puedo intentar responder basándome en mi conocimiento general si lo deseas: '{query}'")

    try:
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        st.error(f"¡Oops! Tuve un pequeño problema técnico al generar la respuesta. Por favor, inténtalo de nuevo o reformula tu pregunta. Error: {e}")
        return "Disculpa, no pude procesar tu solicitud en este momento. ¡Volvamos a intentarlo!"

# --- STREAMLIT APP ---

st.set_page_config(
    page_title="Mi Asistente RAG Amigable",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("📚💬 ¡Hola! Soy Mene tu Asistente de Documentos Amigable")
st.write("Estoy aquí para responder tus preguntas basándome en los documentos PDF que ya he cargado.")

# --- SIDEBAR para la clave API ---
with st.sidebar:
    st.header("Configuración")
    if API_KEY == "TU_CLAVE_API_DE_GEMINI_AQUI":
        st.warning("¡Advertencia! Tu clave API de Gemini no está configurada. La aplicación podría no funcionar. Si despliegas, usa 'st.secrets'.")
    else:
        st.success("Clave API de Gemini configurada.")
    
    st.subheader("Estado de los Documentos")
    # Llama a la función de carga/procesamiento. Streamlit la cacheará.
    all_chunks, all_chunk_embeddings_np = load_and_process_pdfs(PDF_FOLDER_PATH)
    
    if all_chunks and all_chunk_embeddings_np.size > 0:
        st.success(f"¡Listo! Cargué {len(all_chunks)} fragmentos de tus documentos. ¡Pregúntame lo que quieras!")
        rag_agent_ready = True
    else:
        st.error("No se pudieron cargar los documentos. Asegúrate de que la carpeta 'documentos_pdf' exista y contenga PDFs válidos.")
        rag_agent_ready = False

# --- Lógica principal del Chat ---

# Inicializar historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes del historial
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Área de entrada para el usuario
if not rag_agent_ready:
    st.info("Por favor, asegúrate de que tus documentos PDF estén en la carpeta correcta y que tu clave API esté configurada.")
    user_prompt = st.chat_input("... (No disponible hasta que los documentos se carguen)", disabled=True)
else:
    user_prompt = st.chat_input("¿Qué quieres saber sobre los documentos?", disabled=False)

if user_prompt:
    # Añadir mensaje del usuario al historial
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Obtener embedding de la consulta del usuario
    query_embedding_list = get_embeddings([user_prompt])

    if query_embedding_list.size == 0:
        st.error("¡Ups! No pude entender tu pregunta. ¿Podrías reformularla?")
        st.session_state.messages.append({"role": "assistant", "content": "¡Ups! No pude entender tu pregunta. ¿Podrías reformularla?"})
    else:
        # Buscar fragmentos relevantes
        relevant_chunks = find_relevant_chunks(
            query_embedding_list[0],
            all_chunk_embeddings_np, # Usamos los embeddings cargados
            all_chunks                 # Y los chunks cargados
        )

        # Generar respuesta con Gemini
        with st.chat_message("assistant"):
            with st.spinner("¡Dame un segundito! Estoy pensando en la mejor respuesta..."):
                answer = ask_gemini_with_context(user_prompt, relevant_chunks)
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

st.markdown("---")
st.markdown("¡Espero que esta conversación te sea muy útil! 😊")