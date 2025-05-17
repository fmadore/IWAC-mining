from google import genai
from google.genai import types
from google.genai import errors
import json
import os
import time
from tqdm import tqdm
from dotenv import load_dotenv
from pydantic import BaseModel

# Define Pydantic model for structured output
class SentimentAnalysisOutput(BaseModel):
    subjectivite_score: int
    subjectivite_justification: str
    polarite: str
    polarite_justification: str

# Charger les variables d'environnement à partir du fichier .env
load_dotenv()

# Configurez votre clé API (de préférence via une variable d'environnement)
try:
    GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
    # genai.configure(api_key=GOOGLE_API_KEY) # Removed: Client configured locally
except KeyError:
    print("Erreur : La variable d'environnement GOOGLE_API_KEY n'est pas définie.")
    print("Veuillez la définir avant d'exécuter le script.")
    print("Exemple : export GOOGLE_API_KEY='VOTRE_CLE_API'")
    exit()
except Exception as e:
    print(f"Une erreur est survenue lors de la configuration de l'API Gemini : {e}")
    exit()

# Définition du modèle Gemini à utiliser
MODEL_NAME = "gemini-2.5-flash-preview-04-17" # Updated model

# Configuration pour la génération de contenu en JSON
# generation_config = { # Old config
# "response_mime_type": "application/json",
# }

# Le prompt pour l'analyse de sentiment
def create_prompt(article_text):
    prompt = f"""
    Vous êtes un expert en analyse de sentiments spécialisé dans les articles de presse. Votre tâche est d'analyser le texte fourni et de renvoyer une analyse structurée en JSON.

    Pour le texte de l'article suivant :
    ---
    {article_text}
    ---

    Veuillez fournir les informations suivantes au format JSON :
    {{
      "subjectivite_score": <score_de_1_a_5>,
      "subjectivite_justification": "<justification_en_1_2_phrases expliquant pourquoi ce score de subjectivité a été attribué>",
      "polarite": "<Très positif | Positif | Neutre | Négatif | Très négatif>",
      "polarite_justification": "<justification_en_1_2_phrases expliquant pourquoi cette polarité a été attribuée>"
    }}

    Voici les barèmes à utiliser :

    Subjectivité (note de 1 à 5) :
    1 : Très objectif (rapporte des faits vérifiables sans exprimer d'opinions ou de sentiments personnels, style purement informatif).
    2 : Plutôt objectif (principalement factuel, mais peut contenir des traces subtiles d'opinions ou de choix de mots suggérant une perspective limitée).
    3 : Mixte (contient un mélange équilibré de faits et d'opinions/sentiments personnels, ou présente plusieurs points de vue).
    4 : Plutôt subjectif (exprime clairement des opinions, des sentiments ou des jugements, même s'il s'appuie sur certains faits pour les étayer).
    5 : Très subjectif (fortement biaisé, exprime des opinions et des émotions intenses, avec peu ou pas de présentation objective des faits, style éditorial ou billet d'humeur).

    Polarité :
    - Très positif : Sentiment général extrêmement favorable, enthousiaste, élogieux.
    - Positif : Sentiment général favorable, optimiste.
    - Neutre : Pas de sentiment clair ou équilibre entre positif et négatif, ton factuel sans charge émotionnelle marquée.
    - Négatif : Sentiment général défavorable, critique, pessimiste.
    - Très négatif : Sentiment général extrêmement défavorable, alarmiste, très critique.

    Assurez-vous que votre réponse est uniquement le JSON structuré demandé, sans texte ou formatage supplémentaire avant ou après le JSON.
    """
    return prompt

def analyze_sentiment(article_text):
    """
    Analyse le sentiment d'un texte d'article en utilisant l'API Gemini.
    """
    if not article_text or not article_text.strip():
        return {
            "error": "Le texte de l'article est vide.",
            "subjectivite_score": None,
            "subjectivite_justification": None,
            "polarite": None,
            "polarite_justification": None
        }

    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        prompt_content = create_prompt(article_text)
        
        contents = [
            types.Content(
                role="user", # Explicitly set role as in NER example
                parts=[types.Part.from_text(text=prompt_content)]
            )
        ]

        generation_config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2, # Setting a low temperature
            response_schema=SentimentAnalysisOutput # Added Pydantic model as schema
        )
        
        # print(f"--- DEBUG: Sending prompt for article snippet: {article_text[:100]}... ---")
        # print(f"--- DEBUG: Prompt:\n{prompt_content}\n---")

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=generation_config
        )
        # print(f"--- DEBUG: Raw API response text:\n{response.text}\n---")
        
        # Gemini avec response_mime_type="application/json" devrait retourner directement un objet JSON parsable.
        # La propriété .text contient la chaîne JSON.
        analysis_result = json.loads(response.text)
        return analysis_result
    except json.JSONDecodeError as e:
        print(f"Erreur de décodage JSON: {e}")
        print(f"Réponse brute de l'API qui a causé l'erreur: {response.text if 'response' in locals() and hasattr(response, 'text') else 'Pas de réponse textuelle reçue ou l\'objet réponse n\'existe pas.'}")
        return {
            "error": "Erreur de décodage JSON de la réponse de l'API.",
            "raw_response": response.text if 'response' in locals() and hasattr(response, 'text') else 'Pas de réponse textuelle reçue ou l\'objet réponse n\'existe pas.',
            "subjectivite_score": None,
            "subjectivite_justification": None,
            "polarite": None,
            "polarite_justification": None
        }
    except errors.APIError as e: # Catch specific API errors
        print(f"Une erreur API est survenue lors de l'appel à l'API Gemini : {e}")
        # Log prompt feedback if available
        if hasattr(e, 'response') and hasattr(e.response, 'prompt_feedback'):
             print(f"Prompt Feedback: {e.response.prompt_feedback}")
        return {
            "error": f"Gemini API Error: {e}",
            "subjectivite_score": None,
            "subjectivite_justification": None,
            "polarite": None,
            "polarite_justification": None
        }
    except Exception as e: # Catch any other exceptions
        print(f"Une erreur inattendue est survenue : {e}")
        return {
            "error": f"Unexpected error: {str(e)}",
            "subjectivite_score": None,
            "subjectivite_justification": None,
            "polarite": None,
            "polarite_justification": None
        }

def main():
    input_json_file = "data.json"
    output_json_file = "resultats_analyse.json"
    all_results = []

    try:
        with open(input_json_file, 'r', encoding='utf-8') as f:
            articles_data = json.load(f)
    except FileNotFoundError:
        print(f"Erreur : Le fichier {input_json_file} n'a pas été trouvé.")
        return
    except json.JSONDecodeError:
        print(f"Erreur : Le fichier {input_json_file} n'est pas un JSON valide.")
        return

    print(f"Début de l'analyse de {len(articles_data)} articles...")

    for i, article in enumerate(tqdm(articles_data, desc="Traitement des articles")):
        print(f"\nTraitement de l'article {article.get('id', 'N/A')} (Progression: {i+1}/{len(articles_data)})...")
        
        article_text = None
        if "bibo:content" in article and isinstance(article["bibo:content"], list) and len(article["bibo:content"]) > 0:
            content_obj = article["bibo:content"][0]
            if isinstance(content_obj, dict) and "@value" in content_obj:
                article_text = content_obj["@value"]
            else:
                print(f"  Avertissement: La structure de 'bibo:content[0]' n'est pas celle attendue pour l'article ID {article.get('id', 'N/A')}. Contenu: {content_obj}")
        else:
            print(f"  Avertissement: Champ 'bibo:content' manquant ou mal formaté pour l'article ID {article.get('id', 'N/A')}.")

        if article_text:
            analysis = analyze_sentiment(article_text)
            print(f"  Analyse obtenue : {analysis}")
            
            # Enrichir l'objet article original avec les résultats de l'analyse
            article_with_analysis = article.copy() # Crée une copie pour ne pas modifier l'original en cas de réutilisation
            article_with_analysis["sentiment_analysis"] = analysis
            all_results.append(article_with_analysis)
        else:
            print(f"  Impossible d'extraire le texte pour l'article ID {article.get('id', 'N/A')}. Article ignoré.")
            article_with_analysis = article.copy()
            article_with_analysis["sentiment_analysis"] = {
                "error": "Texte de l'article non trouvé ou vide.",
                "subjectivite_score": None,
                "subjectivite_justification": None,
                "polarite": None,
                "polarite_justification": None
            }
            all_results.append(article_with_analysis)

        # Petite pause pour éviter de surcharger l'API (surtout pour les quotas gratuits/limités)
        if i < len(articles_data) - 1 : # Ne pas attendre après le dernier article
             time.sleep(1) # Ajustez cette valeur si nécessaire (ex: 2 secondes pour le modèle Flash gratuit)

    try:
        with open(output_json_file, 'w', encoding='utf-8') as f_out:
            json.dump(all_results, f_out, ensure_ascii=False, indent=2)
        print(f"\nAnalyse terminée. Les résultats ont été sauvegardés dans {output_json_file}")
    except IOError:
        print(f"Erreur : Impossible d'écrire dans le fichier de sortie {output_json_file}.")

if __name__ == "__main__":
    main()