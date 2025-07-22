import google.generativeai as genai
import fitz # PyMuPDF
import os
import textwrap
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- CONFIGURACIÓN ---
# ¡IMPORTANTE! Reemplaza 'TU_CLAVE_API_DE_GEMINI' con tu clave real.
# Es ALTAMENTE RECOMENDABLE cargarla desde una variable de entorno por seguridad:
# API_KEY = os.getenv("GEMINI_API_KEY")
# Para simplificar este inicio, la pondremos directamente, pero tenlo en cuenta para producción.
API_KEY = "AIzaSyDVSsee1vuaoJTSDVx8AY9_DSH2y-2YQiw" # <-- ¡¡¡CAMBIA ESTO CON TU CLAVE REAL!!!

genai.configure(api_key=API_KEY)

# Modelo para embeddings (representaciones numéricas del texto)
EMBEDDING_MODEL = "models/embedding-001"
# Modelo para generación de texto (el que responde las preguntas)
GENERATION_MODEL = "models/gemini-1.5-flash-latest"

# Tamaño de los fragmentos de texto (chunks)
# Un valor común es entre 500 y 1000. Si tu texto es muy denso, considera un tamaño menor.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200 # Para que los chunks se superpongan un poco y no se pierda contexto importante.

# --- FUNCIONES DE PROCESAMIENTO ---

def extract_text_from_pdf(pdf_path):
    """Extrae texto de un archivo PDF."""
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()
        doc.close()
    except Exception as e:
        print(f"Error al extraer texto del PDF {pdf_path}: {e}")
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
        # Asegurarse de que no nos pasemos del final en el último chunk o si el texto es muy corto
        if current_position >= len(text) - chunk_overlap and end_position == len(text):
            break
        elif current_position >= len(text):
            break
    return chunks

def get_embeddings(texts):
    """Genera embeddings para una lista de textos."""
    # La API de Gemini permite procesar múltiples textos en una sola llamada,
    # lo cual es más eficiente.
    try:
        # Los modelos de embeddings tienen un límite de tokens. Aquí se maneja un caso básico.
        # Para aplicaciones a gran escala, se necesitaría una lógica de batching más robusta.
        response = genai.embed_content(model=EMBEDDING_MODEL, content=texts)
        return np.array(response['embedding'])
    except Exception as e:
        print(f"Error al generar embeddings: {e}")
        return np.array([])

def find_relevant_chunks(query_embedding, document_embeddings, document_chunks, top_k=3):
    """
    Encuentra los fragmentos de documentos más relevantes para una consulta
    usando similitud coseno.
    """
    if document_embeddings.size == 0 or query_embedding.size == 0:
        return []

    # Calcular la similitud coseno entre la consulta y todos los fragmentos
    similarities = cosine_similarity([query_embedding], document_embeddings)[0]

    # Ordenar los fragmentos por similitud de forma descendente y seleccionar los top_k
    top_k_indices = np.argsort(similarities)[::-1][:top_k]

    relevant_chunks = [document_chunks[i] for i in top_k_indices]
    return relevant_chunks

def ask_gemini_with_context(query, context_chunks):
    """
    Formula una pregunta a Gemini, proporcionando el contexto relevante
    para que base su respuesta.
    """
    model = genai.GenerativeModel(GENERATION_MODEL)

    # Construye el prompt con la pregunta y el contexto
    if context_chunks:
        # Unimos los chunks para formar un contexto más amplio
        context = "\n\n".join(context_chunks)
        prompt = (f"Basado EXCLUSIVAMENTE en la siguiente información proporcionada:\n\n---\n{context}\n---\n\n"
                  f"Por favor, responde a la siguiente pregunta: {query}\n"
                  "Si la información no es suficiente para responder la pregunta, "
                  "por favor indica claramente que no puedes responder con el contexto dado."
                  "Asegúrate de responder en español y de forma clara y concisa. "
                  "No inventes información que no esté en el texto proporcionado.")
    else:
        # Fallback si no hay contexto relevante
        prompt = (f"No se encontró información relevante para tu pregunta en los documentos proporcionados. "
                  f"Sin embargo, intentaré responder basado en mi conocimiento general: {query}\n"
                  "Por favor, asegura que tu respuesta sea en español.")

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error al generar respuesta con Gemini: {e}"

# --- FUNCIÓN PRINCIPAL DEL AGENTE ---

def run_rag_agent(pdf_folder_path):
    """
    Procesa PDFs de una carpeta, crea embeddings y permite al usuario hacer preguntas.
    """
    all_chunks = []
    all_chunk_embeddings_list = [] # Usamos una lista para apilar los arrays numpy al final
    print(f"--- Procesando PDFs en la carpeta: {pdf_folder_path} ---")

    # Verifica si la clave API es la predeterminada
    if API_KEY == "TU_CLAVE_API_DE_GEMINI":
        print("\n¡¡ADVERTENCIA!! Por favor, reemplaza 'TU_CLAVE_API_DE_GEMINI' con tu clave API real de Google Gemini.")
        print("Obténla en https://aistudio.google.com/app/apikey")
        return

    pdf_files_found = False
    for filename in os.listdir(pdf_folder_path):
        if filename.lower().endswith(".pdf"): # Asegura que la extensión sea en minúsculas
            pdf_path = os.path.join(pdf_folder_path, filename)
            pdf_files_found = True
            print(f"Extrayendo texto de {filename}...")
            text = extract_text_from_pdf(pdf_path)
            if text:
                chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
                all_chunks.extend(chunks)
                print(f"Generando embeddings para {len(chunks)} fragmentos de {filename}...")
                
                # Procesa los chunks en lotes si son muchos para una sola llamada de embedding
                batch_size = 100 # Puedes ajustar este tamaño de lote
                for i in range(0, len(chunks), batch_size):
                    batch_chunks = chunks[i:i + batch_size]
                    chunk_embeddings = get_embeddings(batch_chunks)
                    if chunk_embeddings.size > 0:
                        all_chunk_embeddings_list.append(chunk_embeddings)
                    else:
                        print(f"Advertencia: No se pudieron generar embeddings para el lote {i//batch_size} de {filename}.")
            else:
                print(f"Advertencia: No se pudo extraer texto de {filename}. Se omitirá.")
    
    if not pdf_files_found:
        print(f"Error: No se encontraron archivos PDF en la carpeta '{pdf_folder_path}'. Asegúrate de que los archivos están allí y tienen la extensión .pdf")
        return

    if not all_chunks:
        print("No se encontró texto procesable en ningún PDF. El agente no podrá responder preguntas.")
        return

    # Aplanar la lista de arrays de embeddings en un solo array numpy
    if all_chunk_embeddings_list:
        all_chunk_embeddings_np = np.vstack(all_chunk_embeddings_list)
    else:
        print("Error: No se pudieron generar embeddings válidos para ningún documento. Revisa tu clave API y la configuración.")
        return


    print(f"\n--- Agente listo. {len(all_chunks)} fragmentos cargados. ---")
    print("Puedes hacer preguntas. Escribe 'salir' para terminar.")

    while True:
        user_query = input("\nTu pregunta: ")
        if user_query.lower() == 'salir':
            print("¡Hasta luego!")
            break

        print("Buscando información relevante...")
        # Get embedding for the query. Note: get_embeddings expects a list.
        query_embedding_list = get_embeddings([user_query])

        if query_embedding_list.size == 0:
            print("No se pudo generar el embedding de la pregunta. Intenta de nuevo.")
            continue
        
        # find_relevant_chunks espera un solo vector de embedding para la consulta
        relevant_chunks = find_relevant_chunks(query_embedding_list[0], all_chunk_embeddings_np, all_chunks)

        if not relevant_chunks:
            print("No se encontraron fragmentos relevantes en los documentos.")
            print(f"Respuesta (conocimiento general de Gemini): {ask_gemini_with_context(user_query, [])}")
        else:
            print("Fragmentos relevantes encontrados. Generando respuesta...")
            answer = ask_gemini_with_context(user_query, relevant_chunks)
            print(f"\nRespuesta del Agente: {answer}")

# --- EJECUCIÓN DEL SCRIPT ---
if __name__ == "__main__":
    # ¡IMPORTANTE! Asegúrate de que esta ruta apunte a la carpeta
    # donde colocaste tus archivos PDF.
    pdf_folder = "documentos_pdf" # <--- ¡CAMBIA ESTO SI TU CARPETA TIENE OTRO NOMBRE O RUTA!

    if not os.path.exists(pdf_folder):
        print(f"Error: La carpeta '{pdf_folder}' no existe. Por favor, créala y coloca tus PDFs dentro.")
    else:
        run_rag_agent(pdf_folder)