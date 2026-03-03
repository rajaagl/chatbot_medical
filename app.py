from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from src.prompt import *
import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

app = Flask(__name__)

load_dotenv()

# Configuration Pinecone (inchangé)
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# NOUVEAU : Configuration GitHub/Azure
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')
AZURE_ENDPOINT = "https://models.inference.ai.azure.com"

if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN non trouvé dans les variables d'environnement!")

# Initialisation du client Azure avec le token GitHub
client = ChatCompletionsClient(
    endpoint=AZURE_ENDPOINT,
    credential=AzureKeyCredential(GITHUB_TOKEN),
)

# Embeddings (inchangé)
embeddings = download_hugging_face_embeddings()

# Connexion à Pinecone (inchangé)
index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# NOUVELLE FONCTION : Utiliser Azure AI au lieu de ChatOpenAI
def generate_response_with_azure(query, context_docs):
    """
    Génère une réponse en utilisant Azure AI Inference avec le token GitHub
    """
    # Formater les documents récupérés
    context = "\n\n".join([doc.page_content for doc in context_docs])
    
    # Créer le prompt avec le contexte
    full_prompt = f"""
    Vous êtes un assistant médical expert au Maroc.
    
    Contexte médical:
    {context}
    
    Question du patient: {query}
    
    Instructions:
    - Répondez en français uniquement
    - Utilisez uniquement les informations du contexte fourni
    - Si l'information n'est pas dans le contexte, dites que vous ne savez pas
    - Soyez professionnel et empathique
    
    Réponse:
    """
    
    try:
        # Appel à Azure AI avec le modèle disponible
        response = client.complete(
            messages=[
                SystemMessage(content="Vous êtes un assistant médical professionnel."),
                UserMessage(content=full_prompt)
            ],
            model="gpt-4o-mini",  # Ou "gpt-4o", "Meta-Llama-3.1-8B-Instruct", etc.
            temperature=0.3,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Erreur Azure AI: {e}")
        return f"Désolé, une erreur s'est produite. Veuillez réessayer."

# NOUVELLE CLASSE : Version RAG avec Azure
class AzureRAGChain:
    def __init__(self, retriever):
        self.retriever = retriever
    
    def invoke(self, inputs):
        query = inputs.get("input", "")
        
        # 1. Récupérer les documents pertinents
        docs = self.retriever.get_relevant_documents(query)
        
        # 2. Générer la réponse avec Azure AI
        answer = generate_response_with_azure(query, docs)
        
        return {
            "input": query,
            "context": docs,
            "answer": answer
        }

# Créer l'instance de la chaîne RAG
rag_chain = AzureRAGChain(retriever)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print(f"Message reçu: {msg}")
    
    try:
        response = rag_chain.invoke({"input": msg})
        print(f"Réponse: {response['answer'][:100]}...")
        return str(response["answer"])
    except Exception as e:
        print(f" Erreur: {e}")
        return "Désolé, une erreur s'est produite. Veuillez réessayer."

# Route de test pour vérifier la connexion
@app.route("/health", methods=["GET"])
def health():
    try:
        # Test simple de l'API Azure
        test_response = client.complete(
            messages=[UserMessage(content="Test de connexion")],
            model="gpt-4o-mini",
            max_tokens=10
        )
        return jsonify({
            "status": "ok",
            "azure": "connected",
            "pinecone": "connected",
            "model": test_response.choices[0].message.content
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

if __name__ == '__main__':
    print("""
     Chatbot Médical RAG avec Azure AI
    =====================================
     Port: 8081
     Pinecone: Connecté
     Modèle: Azure AI (via GitHub token)
    """)
    app.run(host="0.0.0.0", port=8081, debug=True)