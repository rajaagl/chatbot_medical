from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from src.prompt import *
import os
import csv
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

app = Flask(__name__)
load_dotenv()

# ============================================
# CONFIGURATION (SEULEMENT PINECONE + GITHUB)
# ============================================

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')
AZURE_ENDPOINT = "https://models.inference.ai.azure.com"

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY manquante")
if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN manquant")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Client Azure avec GitHub token
client = ChatCompletionsClient(
    endpoint=AZURE_ENDPOINT,
    credential=AzureKeyCredential(GITHUB_TOKEN),
)

# ============================================
# TARIFS
# ============================================

TARIFS = []
def charger_tarifs():
    global TARIFS
    try:
        with open('../data/tarifs.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for ligne in reader:
                TARIFS.append({
                    "nom": ligne['specialite'],
                    "prix": f"{ligne['prix_min']}-{ligne['prix_max']} DH"
                })
        print(f"{len(TARIFS)} tarifs chargés")
    except FileNotFoundError:
        print("data/tarifs.csv non trouvé")
charger_tarifs()

# ============================================
# EMBEDDINGS HUGGINGFACE LOCAL
# ============================================

print("Chargement des embeddings...")
embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot"
print("Connexion à Pinecone...")

try:
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    print("Pinecone connecté")
except Exception as e:
    print(f"Erreur Pinecone: {e}")
    retriever = None

# ============================================
# FONCTION RAG AVEC GITHUB TOKEN
# ============================================

def generate_response_with_github(query, context_docs):
    context = "\n\n".join([doc.page_content for doc in context_docs])
    
    full_prompt = f"""
    Vous êtes un assistant médical expert au Maroc.
    
    Contexte médical:
    {context}
    
    Question du patient: {query}
    
    Réponse en français:
    """
    
    try:
        response = client.complete(
            messages=[
                SystemMessage(content="Assistant médical professionnel."),
                UserMessage(content=full_prompt)
            ],
            model="gpt-4o-mini",  # Gratuit avec GitHub token
            temperature=0.3,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Erreur: {e}")
        return "Désolé, une erreur s'est produite."

class GitHubRAGChain:
    def __init__(self, retriever):
        self.retriever = retriever
    
    def invoke(self, inputs):
        query = inputs.get("input", "")
        docs = self.retriever.get_relevant_documents(query)
        answer = generate_response_with_github(query, docs)
        return {"answer": answer}

rag_chain = GitHubRAGChain(retriever) if retriever else None

# ============================================
# ROUTES
# ============================================

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    msg_lower = msg.lower()
    
    # TARIFS
    if any(mot in msg_lower for mot in ["tarif", "prix", "combien"]):
        if TARIFS:
            reponse = "TARIFS:\n" + "-"*20 + "\n"
            for t in TARIFS:
                reponse += f"• {t['nom']}: {t['prix']}\n"
            return reponse
    
    # RAG
    if rag_chain:
        response = rag_chain.invoke({"input": msg})
        return response["answer"]
    
    return "Service indisponible"

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8081, debug=True)