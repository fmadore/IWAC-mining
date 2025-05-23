from google import genai
from google.genai import types
from google.genai import errors
import json
import os
import time
from tqdm import tqdm
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional

# Define Pydantic model for structured output
class SentimentAnalysisOutput(BaseModel):
    centralite_islam_musulmans: str
    centralite_justification: str
    subjectivite_score: Optional[int] = Field(default=None)
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
MODEL_NAME = "gemini-2.5-flash-preview-05-20" # Updated model

# Configuration pour la génération de contenu en JSON
# generation_config = { # Old config
# "response_mime_type": "application/json",
# }

# Le prompt pour l'analyse de sentiment
def create_prompt(article_text):
    prompt = f"""
    Vous êtes un expert en analyse de sentiments, spécialisé dans l'étude des représentations de l'islam et des musulmans dans les médias, notamment en Afrique de l'Ouest francophone. Votre tâche est d'analyser le texte fourni sous cet angle spécifique et de renvoyer une analyse structurée en JSON.

    Votre analyse doit spécifiquement évaluer comment l'islam et/ou les musulmans sont dépeints ou représentés dans l'article. La subjectivité et la polarité doivent être jugées par rapport à cette représentation. Si l'islam et les musulmans ne sont qu'un sujet marginal ou non pertinent dans l'article, indiquez-le clairement.

    Pour le texte de l'article suivant :
    ---
    {article_text}
    ---

    Veuillez fournir les informations suivantes au format JSON :
    {{
      "centralite_islam_musulmans": "<Très central | Central | Secondaire | Marginal | Non abordé>",
      "centralite_justification": "<Courte justification (1 phrase) expliquant le niveau de centralité de l'islam/des musulmans dans l'article>",
      "subjectivite_score": <score_de_1_a_5_ou_null_si_non_aborde>,
      "subjectivite_justification": "<justification_en_1_2_phrases expliquant pourquoi ce score de subjectivité a été attribué concernant la manière dont l'article traite de l'islam et/ou des musulmans, ou 'Non applicable si le sujet n'est pas abordé'>",
      "polarite": "<Très positif | Positif | Neutre | Négatif | Très négatif | Non applicable>",
      "polarite_justification": "<justification_en_1_2_phrases expliquant pourquoi cette polarité a été attribuée en ce qui concerne le portrait de l'islam et/ou des musulmans dans l'article, ou 'Non applicable si le sujet n'est pas abordé'>"
    }}

    Voici les barèmes à utiliser :

    Centralité de l'islam et des musulmans dans l'article :
    - Très central : L'article est principalement ou entièrement consacré à l'islam et/ou aux musulmans.
    - Central : L'islam et/ou les musulmans sont un des sujets principaux de l'article.
    - Secondaire : L'islam et/ou les musulmans sont mentionnés ou discutés, mais ne constituent pas le focus principal de l'article.
    - Marginal : L'islam et/ou les musulmans sont brièvement mentionnés de manière anecdotique ou périphérique.
    - Non abordé : L'article ne traite pas du tout de l'islam ou des musulmans.

    Subjectivité (note de 1 à 5) – Évaluez le degré d'objectivité/subjectivité de l'article DANS SA MANIÈRE DE REPRÉSENTER l'islam et/ou les musulmans (Attribuez 'null' si 'Non abordé' pour la centralité) :
    1 : Très objectif (rapporte des faits vérifiables sur l'islam/les musulmans sans exprimer d'opinions ou de sentiments personnels à leur sujet, style purement informatif sur ce thème).
    2 : Plutôt objectif (principalement factuel concernant l'islam/les musulmans, mais peut contenir des traces subtiles d'opinions ou des choix de mots suggérant une perspective limitée sur ce thème).
    3 : Mixte (contient un mélange équilibré de faits et d'opinions/sentiments personnels concernant l'islam/les musulmans, ou présente plusieurs points de vue sur ce thème).
    4 : Plutôt subjectif (exprime clairement des opinions, des sentiments ou des jugements sur l'islam/les musulmans, même s'il s'appuie sur certains faits pour les étayer).
    5 : Très subjectif (fortement biaisé dans sa représentation de l'islam/des musulmans, exprime des opinions et des émotions intenses à leur sujet, avec peu ou pas de présentation objective des faits, style éditorial ou billet d'humeur sur ce thème).

    Polarité – Évaluez le sentiment général exprimé DANS L'ARTICLE ENVERS l'islam et/ou les musulmans, ou concernant leur représentation (Attribuez 'Non applicable' si 'Non abordé' pour la centralité) :
    - Très positif : Le portrait de l'islam/des musulmans est extrêmement favorable, enthousiaste, élogieux.
    - Positif : Le portrait de l'islam/des musulmans est favorable, optimiste.
    - Neutre : Pas de sentiment clair envers l'islam/des musulmans ou équilibre entre aspects positifs et négatifs dans leur représentation ; ton factuel sans charge émotionnelle marquée à leur égard.
    - Négatif : Le portrait de l'islam/des musulmans est défavorable, critique, pessimiste.
    - Très négatif : Le portrait de l'islam/des musulmans est extrêmement défavorable, alarmiste, très critique.

    Si la centralité est "Non abordé", le "subjectivite_score" doit être null, et "polarite", "subjectivite_justification", et "polarite_justification" doivent être "Non applicable". Le JSON doit toujours être valide.
    Par exemple, si "centralite_islam_musulmans" est "Non abordé":
    {{
      "centralite_islam_musulmans": "Non abordé",
      "centralite_justification": "L'article ne mentionne ni l'islam ni les musulmans.",
      "subjectivite_score": null,
      "subjectivite_justification": "Non applicable car le sujet n'est pas abordé.",
      "polarite": "Non applicable",
      "polarite_justification": "Non applicable car le sujet n'est pas abordé."
    }}

    Assurez-vous que votre réponse est uniquement le JSON structuré demandé, sans texte ou formatage supplémentaire avant ou après le JSON.
    """
    return prompt

def analyze_sentiment(article_text):
    """
    Analyse le sentiment d'un texte d'article en utilisant l'API Gemini.
    """
    if not article_text or not article_text.strip():
        # Retourner la structure attendue même en cas d'erreur de texte vide,
        # conformément au nouveau schéma et à la logique "Non abordé"
        return {
            "error": "Le texte de l'article est vide.",
            "centralite_islam_musulmans": "Non abordé",
            "centralite_justification": "Texte de l'article non fourni ou vide.",
            "subjectivite_score": None,
            "subjectivite_justification": "Non applicable car le texte de l'article est vide.",
            "polarite": "Non applicable",
            "polarite_justification": "Non applicable car le texte de l'article est vide."
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
            "centralite_islam_musulmans": "Non applicable", # Valeur par défaut en cas d'erreur
            "centralite_justification": "Erreur lors de l'analyse.",
            "subjectivite_score": None,
            "subjectivite_justification": "Erreur lors de l'analyse.",
            "polarite": "Non applicable",
            "polarite_justification": "Erreur lors de l'analyse."
        }
    except errors.APIError as e: # Catch specific API errors
        print(f"Une erreur API est survenue lors de l'appel à l'API Gemini : {e}")
        # Log prompt feedback if available
        if hasattr(e, 'response') and hasattr(e.response, 'prompt_feedback'):
             print(f"Prompt Feedback: {e.response.prompt_feedback}")
        return {
            "error": f"Gemini API Error: {e}",
            "centralite_islam_musulmans": "Non applicable", # Valeur par défaut en cas d'erreur
            "centralite_justification": "Erreur API Gemini.",
            "subjectivite_score": None,
            "subjectivite_justification": "Erreur API Gemini.",
            "polarite": "Non applicable",
            "polarite_justification": "Erreur API Gemini."
        }
    except Exception as e: # Catch any other exceptions
        print(f"Une erreur inattendue est survenue : {e}")
        return {
            "error": f"Unexpected error: {str(e)}",
            "centralite_islam_musulmans": "Non applicable", # Valeur par défaut en cas d'erreur
            "centralite_justification": "Erreur inattendue.",
            "subjectivite_score": None,
            "subjectivite_justification": "Erreur inattendue.",
            "polarite": "Non applicable",
            "polarite_justification": "Erreur inattendue."
        }

def main():
    # Determine the data directory relative to the script's location
    script_dir = Path(__file__).parent
    data_dir = script_dir / "../../data"

    json_files = []
    if data_dir.exists() and data_dir.is_dir():
        json_files = sorted([f for f in data_dir.iterdir() if f.is_file() and f.suffix.lower() == '.json'])
    
    if not json_files:
        print(f"Erreur : Aucun fichier .json trouvé dans le dossier {data_dir.resolve()}.")
        print("Veuillez vérifier que le dossier existe et contient des fichiers JSON.")
        return

    print("Veuillez choisir un fichier JSON à analyser :")
    for i, file_path in enumerate(json_files):
        print(f"  {i+1}: {file_path.name}")

    selected_index = -1
    while True:
        try:
            choice = input(f"Entrez le numéro du fichier (1-{len(json_files)}) : ")
            selected_index = int(choice) - 1
            if 0 <= selected_index < len(json_files):
                break
            else:
                print(f"Choix invalide. Veuillez entrer un numéro entre 1 et {len(json_files)}.")
        except ValueError:
            print("Entrée invalide. Veuillez entrer un nombre.")

    input_json_file = json_files[selected_index]
    # Create a unique output file name based on the input file name
    output_json_file = data_dir.parent / f"{input_json_file.stem}_resultats_analyse.json"

    all_results = []

    print(f"Lecture du fichier : {input_json_file}")
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
            
            # Construire l'objet résultat avec les champs spécifiés
            result_entry = {
                "o:id": article.get("o:id"),
                "o:title": article.get("o:title")
            }

            # Extraction pour dcterms:publisher -> display_title
            publisher_list = article.get("dcterms:publisher", [])
            if isinstance(publisher_list, list) and len(publisher_list) > 0:
                publisher_obj = publisher_list[0]
                if isinstance(publisher_obj, dict):
                    result_entry["Newspaper"] = publisher_obj.get("display_title")
                # elif isinstance(publisher_obj, str): # This case seems unlikely given the example
                #     result_entry["Newspaper"] = publisher_obj
                else:
                    result_entry["Newspaper"] = None # ou une valeur par défaut
            else:
                result_entry["Newspaper"] = None # ou une valeur par défaut


            # Extraction pour dcterms:date
            date_list = article.get("dcterms:date", [])
            if isinstance(date_list, list) and len(date_list) > 0:
                date_obj = date_list[0]
                if isinstance(date_obj, dict) and "@value" in date_obj:
                    result_entry["dcterms:date"] = date_obj["@value"]
                elif isinstance(date_obj, str):  # Au cas où ce ne serait pas une liste de dictionnaires
                    result_entry["dcterms:date"] = date_obj
                else:
                    result_entry["dcterms:date"] = None # ou une valeur par défaut
            else:
                result_entry["dcterms:date"] = None # ou une valeur par défaut
            
            result_entry["sentiment_analysis"] = analysis
            all_results.append(result_entry)
        else:
            print(f"  Impossible d'extraire le texte pour l'article ID {article.get('id', 'N/A')}. Article ignoré.")
            # Même si le texte est manquant, nous voulons quand même une entrée minimale
            # et la structure d'analyse de sentiment doit correspondre au cas "Non abordé" / Erreur
            result_entry = {
                "o:id": article.get("o:id"),
                "o:title": article.get("o:title")
            }
            publisher_list = article.get("dcterms:publisher", [])
            if isinstance(publisher_list, list) and len(publisher_list) > 0:
                publisher_obj = publisher_list[0]
                if isinstance(publisher_obj, dict):
                    result_entry["Newspaper"] = publisher_obj.get("display_title")
                # elif isinstance(publisher_obj, str):
                #     result_entry["Newspaper"] = publisher_obj
                else:
                    result_entry["Newspaper"] = None
            else:
                result_entry["Newspaper"] = None

            date_list = article.get("dcterms:date", [])
            if isinstance(date_list, list) and len(date_list) > 0:
                date_obj = date_list[0]
                if isinstance(date_obj, dict) and "@value" in date_obj:
                    result_entry["dcterms:date"] = date_obj["@value"]
                elif isinstance(date_obj, str):
                     result_entry["dcterms:date"] = date_obj
                else:
                    result_entry["dcterms:date"] = None
            else:
                result_entry["dcterms:date"] = None

            result_entry["sentiment_analysis"] = {
                "error": "Texte de l'article non trouvé ou vide.",
                "centralite_islam_musulmans": "Non abordé",
                "centralite_justification": "Texte de l'article non trouvé ou vide.",
                "subjectivite_score": None,
                "subjectivite_justification": "Non applicable car le texte de l'article est non trouvé ou vide.",
                "polarite": "Non applicable",
                "polarite_justification": "Non applicable car le texte de l'article est non trouvé ou vide."
            }
            all_results.append(result_entry)

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