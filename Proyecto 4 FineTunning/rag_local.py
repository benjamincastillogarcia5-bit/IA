import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

print("1. Cargando motor RAG y Base de Datos...")
ruta_bd = "./rag_db_violencia" 

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vectorstore = Chroma(persist_directory=ruta_bd, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

print("2. Despertando al Tutor en Ollama...")
llm = Ollama(
    model="TutorViolencia",
    temperature=0.1,       
    num_predict=250,
    stop=["Usuario:", "###", "\n\n\n", "Usuario,"]
)

# === INICIO DE LA LÓGICA DEL CHATBOT ===
print("\n" + "="*60)
print("¡SISTEMA LISTO! EL CHATBOT ESTÁ ACTIVO.")
print("Escribe 'salir', 'exit' o 'quit' para terminar la ejecución.")
print("="*60)

# Aquí guardaremos el historial de la conversación
memoria_chat = []

while True:
    # 1. Esperamos tu pregunta directamente en la terminal
    pregunta = input("\nTú: ")
    
    # Condición de salida del bucle
    if pregunta.lower() in ['salir', 'exit', 'quit']:
        print("\nApagando el Tutor... ¡Hasta la próxima!")
        break

    # Si presionas Enter sin escribir nada, que no haga nada
    if not pregunta.strip():
        continue

    print(f"   [🔍 Buscando en los PDFs...]")
    docs = retriever.invoke(pregunta)
    contexto = "\n\n".join([doc.page_content for doc in docs])
    
    # 2. Formateamos la memoria (Solo tomamos los últimos 2 intercambios para no saturar la RAM/CPU)
    historial_texto = ""
    if memoria_chat:
        historial_texto = "\n".join(memoria_chat[-2:])
    else:
        historial_texto = "Ninguno (esta es la primera pregunta)."

    # 3. El Prompt Modificado (Ahora incluye el bloque de Historial)
    prompt_magico = f"""### Instrucción:
Eres un tutor analítico especializado en seguridad pública. Responde brevemente basándote SOLO en el contexto. Si el usuario hace referencia a algo anterior, usa el Historial de Chat para entender el contexto.

### Entrada:
Contexto RAG:
{contexto}

Historial de Chat:
{historial_texto}

Pregunta Actual:
{pregunta}

### Respuesta:
"""
    
    print("\nTutor: ", end="", flush=True)
    
    respuesta_completa = ""
    
    # 4. Streaming de la respuesta
    for fragmento in llm.stream(prompt_magico):
        print(fragmento, end="", flush=True)
        respuesta_completa += fragmento
        
    print("\n" + "-" * 60)

    # 5. Actualizamos la memoria para la siguiente vuelta
    # Guardamos tu pregunta y su respuesta
    interaccion = f"Usuario: {pregunta}\nTutor: {respuesta_completa}"
    memoria_chat.append(interaccion)