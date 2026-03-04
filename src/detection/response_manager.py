# src/detection/response_manager.py
"""
Module de gestion des réponses du chatbot
"""

import os
import csv
from datetime import datetime

class ResponseManager:
    """
    Gère les réponses pour chaque type d'intention
    """
    
    def __init__(self, data_path='data'):
        self.data_path = data_path
        self.tarifs = self._charger_tarifs()
        
    def _charger_tarifs(self):
        """Charge les tarifs depuis le CSV"""
        tarifs = []
        chemin_csv = os.path.join(self.data_path, 'tarifs.csv')
        
        try:
            with open(chemin_csv, 'r', encoding='utf-8') as fichier:
                reader = csv.DictReader(fichier)
                for ligne in reader:
                    tarifs.append({
                        "nom": ligne['specialite'],
                        "prix": f"{ligne['prix_min']}-{ligne['prix_max']} DH",
                        "moyenne": f"{ligne['moyenne']} DH",
                        "convention": "✅ Oui" if ligne['convention'].lower() == 'oui' else "❌ Non",
                        "icone": ligne['icone'],
                        "details": ligne['details']
                    })
            print(f"✅ {len(tarifs)} tarifs chargés")
            return tarifs
        except Exception as e:
            print(f"❌ Erreur chargement tarifs: {e}")
            return []
    
    def reponse_tarifs(self):
        """Réponse pour les demandes de tarifs"""
        if not self.tarifs:
            return "💰 Les tarifs ne sont pas disponibles pour le moment."
        
        reponse = "💰 **TARIFS DE NOS CONSULTATIONS**\n"
        reponse += "═" * 40 + "\n\n"
        
        for spec in self.tarifs:
            reponse += f"{spec['icone']} **{spec['nom']}**\n"
            reponse += f"   • Prix : {spec['prix']}\n"
            reponse += f"   • Moyenne : {spec['moyenne']}\n"
            reponse += f"   • Conventionné : {spec['convention']}\n\n"
        
        reponse += "═" * 40 + "\n"
        reponse += "💳 Paiement accepté : Espèces, CB, Chèques\n"
        reponse += "📅 Pour réserver : https://votre-site.com/rendez-vous\n"
        
        return reponse
    
    def reponse_medecins(self):
        """Réponse pour les demandes de médecins"""
        return """
👨‍⚕️ **NOS MÉDECINS**

Consultez la liste complète de nos praticiens sur notre site :
🔗 https://votre-site.com/medecins

Vous y trouverez :
• ✅ Toutes les spécialités
• ✅ Les horaires de consultation
• ✅ Les disponibilités en temps réel
• ✅ La possibilité de prendre RDV

Cliquez sur le lien ci-dessus ! ⬆️
        """
    
    def reponse_rendez_vous(self):
        """Réponse pour les demandes de rendez-vous"""
        return """
📅 **PRENDRE RENDEZ-VOUS**

Réservez votre consultation en ligne :
🔗 https://votre-site.com/rendez-vous

Ou contactez notre secrétariat :
📞 05XX-XXXXXX
📧 rdv@cabinet-medical.ma

Notre équipe vous répondra dans les plus brefs délais.
        """
    
    def reponse_adresse(self):
        """Réponse pour les demandes d'adresse"""
        return """
📍 **NOTRE ADRESSE**

Cabinet Médical
123 Rue de la Liberté
Casablanca 20000

🅿️ Parking gratuit disponible
🗺️ https://maps.google.com/?q=...

🚗 Accès facile depuis le boulevard principal
        """
    
    def reponse_contact(self):
        """Réponse pour les demandes de contact"""
        return """
📞 **NOUS CONTACTER**

• Téléphone : 05XX-XXXXXX
• WhatsApp : 06XX-XXXXXX
• Email : contact@cabinet-medical.ma

⏱️ Réponse sous 24h maximum
        """
    
    def reponse_horaires(self):
        """Réponse pour les demandes d'horaires"""
        return """
🕒 **HORAIRES D'OUVERTURE**

Lundi - Vendredi : 9h00 - 18h00
Samedi : 9h00 - 12h00
Dimanche : Fermé

🏥 Urgences : appelez le 141
        """
    
    def reponse_par_intention(self, intention):
        """
        Retourne la réponse correspondant à l'intention
        """
        reponses = {
            "tarifs": self.reponse_tarifs,
            "medecins": self.reponse_medecins,
            "rendez_vous": self.reponse_rendez_vous,
            "adresse": self.reponse_adresse,
            "contact": self.reponse_contact,
            "horaires": self.reponse_horaires
        }
        
        if intention in reponses:
            return reponses[intention]()
        return None
    
    def reponse_fallback(self):
        """Réponse quand aucune intention n'est détectée"""
        return None  # Retourne None pour utiliser le RAG
    
    def get_tarifs_data(self):
        """Retourne les données des tarifs (pour API)"""
        return self.tarifs