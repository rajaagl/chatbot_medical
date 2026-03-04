from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from src.prompt import *
import os
import csv
import threading
import time
from functools import lru_cache
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

app = Flask(__name__)

# ============================================
# CHARGEMENT DES VARIABLES D'ENVIRONNEMENT
# ============================================

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')
AZURE_ENDPOINT = "https://models.inference.ai.azure.com"

if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN non trouve dans les variables d'environnement!")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# ============================================
# INITIALISATION DU CLIENT AZURE
# ============================================

client = ChatCompletionsClient(
    endpoint=AZURE_ENDPOINT,
    credential=AzureKeyCredential(GITHUB_TOKEN),
)

# ============================================
# CHARGEMENT DES TARIFS DEPUIS CSV
# ============================================

def charger_tarifs_depuis_csv():
    """Charge les tarifs depuis data/tarifs.csv"""
    tarifs = []
    chemin_csv = os.path.join('data', 'tarifs.csv')
    
    try:
        with open(chemin_csv, 'r', encoding='utf-8') as fichier:
            reader = csv.DictReader(fichier)
            for ligne in reader:
                tarifs.append({
                    "nom": ligne['specialite'],
                    "prix": f"{ligne['prix_min']}-{ligne['prix_max']} DH",
                    "moyenne": f"{ligne['moyenne']} DH",
                    "convention": "Oui" if ligne['convention'].lower() == 'oui' else "Non",
                    "details": ligne['details']
                })
        print(f"{len(tarifs)} tarifs charges depuis data/tarifs.csv")
        return tarifs
    except FileNotFoundError:
        print("Fichier data/tarifs.csv non trouve")
        return []
    except Exception as e:
        print(f"Erreur lors du chargement des tarifs: {e}")
        return []

TARIFS = charger_tarifs_depuis_csv()

def formater_liste_tarifs():
    """Formatte la liste complete des tarifs (sans icones)"""
    
    if not TARIFS:
        return "Les tarifs ne sont pas disponibles pour le moment."
    
    reponse = "TARIFS DE NOS CONSULTATIONS\n"
    reponse += "=" * 40 + "\n\n"
    
    for spec in TARIFS:
        reponse += f"{spec['nom']}\n"
        reponse += f"   Prix : {spec['prix']}\n"
        reponse += f"   Moyenne : {spec['moyenne']}\n"
        reponse += f"   Conventionne : {spec['convention']}\n\n"
    
    reponse += "=" * 40 + "\n"
    reponse += "Paiement accepte : Especes, CB, Cheques\n"
    reponse += "Pour reserver : https://votre-site.com/rendez-vous\n"
    
    return reponse
# ============================================
# FONCTIONS AZURE POUR LE RAG
# ============================================

def generate_response_with_azure(query, context_docs):
    """Genere une reponse en utilisant Azure AI"""
    context = "\n\n".join([doc.page_content for doc in context_docs])
    
    full_prompt = f"""
    Vous etes un assistant medical expert au Maroc.
    
    Contexte medical:
    {context}
    
    Question du patient: {query}
    
    Instructions:
    - Repondez en francais uniquement
    - Utilisez uniquement les informations du contexte fourni
    - Si l'information n'est pas dans le contexte, dites que vous ne savez pas
    - Soyez professionnel et empathique
    
    Reponse:
    """
    
    try:
        response = client.complete(
            messages=[
                SystemMessage(content="Vous etes un assistant medical professionnel."),
                UserMessage(content=full_prompt)
            ],
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Erreur Azure AI: {e}")
        return f"Desole, une erreur s'est produite. Veuillez reessayer."

class AzureRAGChain:
    def __init__(self, retriever):
        self.retriever = retriever
    
    def invoke(self, inputs):
        query = inputs.get("input", "")
        docs = self.retriever.get_relevant_documents(query)
        answer = generate_response_with_azure(query, docs)
        return {
            "input": query,
            "context": docs,
            "answer": answer
        }

# ============================================
# ROUTES
# ============================================

@app.route("/")
def index():
    """Page d'accueil"""
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    """Route principale du chatbot"""
    msg = request.form["msg"]
    msg_lower = msg.lower()
    
    print(f"Message recu: {msg}")
    
    # 1. DETECTION DES TARIFS
    mots_tarifs = ["tarif", "prix", "combien", "coute", "thman", "ثمن"]
    if any(word in msg_lower for word in mots_tarifs):
        print("Demande de tarifs")
        return formater_liste_tarifs()
    
    # 2. DETECTION DES MEDECINS
    mots_medecins = ["medecin", "docteur", "praticien", "طبيب", "دكتور"]
    if any(word in msg_lower for word in mots_medecins):
        print("Demande de medecins")
        return "NOS MEDECINS :\nConsultez la liste sur : https://votre-site.com/medecins"
    
    # 3. DETECTION DES RENDEZ-VOUS
    mots_rdv = ["rendez-vous", "rdv", "reserver", "حجز", "موعد"]
    if any(word in msg_lower for word in mots_rdv):
        print("Demande de rendez-vous")
        return "PRENDRE RENDEZ-VOUS :\nhttps://votre-site.com/rendez-vous"
    
    # 4. VERIFICATION SI LE RAG EST PRET
    if not embeddings_chargees:
        return "Le systeme se prepare... Veuillez reessayer dans quelques secondes ou utilisez les suggestions ci-dessus."
    
    # 5. UTILISATION DU RAG
    try:
        emb = get_embeddings_cached()
        retriever = pinecone_loader.get_retriever(emb)
        rag_chain = AzureRAGChain(retriever)
        
        response = rag_chain.invoke({"input": msg})
        print(f"Reponse: {response['answer'][:100]}...")
        return str(response["answer"])
    except Exception as e:
        print(f"Erreur: {e}")
        return "Desole, une erreur s'est produite."

@app.route("/api/tarifs", methods=["GET"])
def api_tarifs():
    """API pour recuperer les tarifs en JSON"""
    return jsonify({
        "success": True,
        "tarifs": TARIFS,
        "total": len(TARIFS)
    })

@app.route("/health", methods=["GET"])
def health():
    """Route de test"""
    try:
        test_response = client.complete(
            messages=[UserMessage(content="Test")],
            model="gpt-4o-mini",
            max_tokens=10
        )
        return jsonify({
            "status": "ok",
            "azure": "connected",
            "pinecone": "connected" if embeddings_chargees else "loading",
            "tarifs": len(TARIFS)
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

# ============================================
# LANCEMENT
# ============================================

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8081, debug=True)