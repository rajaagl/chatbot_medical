# src/detection/intent_detector.py
"""
Module de détection des intentions de l'utilisateur
"""

class IntentDetector:
    """
    Détecte l'intention de l'utilisateur à partir de son message
    """
    
    def __init__(self):
        # Mots-clés pour chaque intention
        self.intents = {
            "tarifs": {
                "mots": ["tarif", "prix", "combien", "coûte", "cout", "thman", "ثمن", "💰"],
                "priorite": 10
            },
            "medecins": {
                "mots": ["médecin", "docteur", "praticien", "spécialiste", "طبيب", "دكتور", "👨‍⚕️"],
                "priorite": 8
            },
            "rendez_vous": {
                "mots": ["rendez-vous", "rdv", "réserver", "prendre", "حجز", "موعد", "📅"],
                "priorite": 9
            },
            "adresse": {
                "mots": ["adresse", "où", "situé", "trouver", "عنوان", "فين", "📍"],
                "priorite": 7
            },
            "contact": {
                "mots": ["contact", "téléphone", "appeler", "joindre", "اتصال", "هاتف", "📞"],
                "priorite": 6
            },
            "horaires": {
                "mots": ["horaire", "heure", "ouvert", "fermé", "ساعة", "وقت", "🕒"],
                "priorite": 7
            }
        }
    
    def detecter_intention(self, message):
        """
        Détecte l'intention principale du message
        Retourne le nom de l'intention ou None si non détectée
        """
        if not message:
            return None
        
        message_lower = message.lower()
        intentions_trouvees = []
        
        # Chercher toutes les intentions correspondantes
        for intent_name, intent_data in self.intents.items():
            for mot in intent_data["mots"]:
                if mot in message_lower:
                    intentions_trouvees.append({
                        "nom": intent_name,
                        "priorite": intent_data["priorite"],
                        "mot_trouve": mot
                    })
                    break  # Un seul mot suffit pour cette intention
        
        if not intentions_trouvees:
            return None
        
        # Trier par priorité (plus haute d'abord)
        intentions_trouvees.sort(key=lambda x: x["priorite"], reverse=True)
        
        # Retourner l'intention avec la plus haute priorité
        return intentions_trouvees[0]["nom"]
    
    def detecter_tous(self, message):
        """
        Détecte toutes les intentions possibles (pour débogage)
        """
        resultats = []
        message_lower = message.lower()
        
        for intent_name, intent_data in self.intents.items():
            for mot in intent_data["mots"]:
                if mot in message_lower:
                    resultats.append({
                        "intention": intent_name,
                        "mot": mot,
                        "priorite": intent_data["priorite"]
                    })
                    break
        
        return resultats
    
    def ajouter_intention(self, nom, mots, priorite=5):
        """
        Ajoute une nouvelle intention personnalisée
        """
        self.intents[nom] = {
            "mots": mots,
            "priorite": priorite
        }
        print(f"✅ Intention '{nom}' ajoutée")