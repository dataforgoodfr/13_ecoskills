import os
import requests
import json
from fpdf import FPDF

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

NB_RESULT = 200

# Configuration API
API_BASE_URL = "https://api.welearn.k8s.lp-i.org"
SEARCH_ENDPOINT = f"{API_BASE_URL}/api/v1/search/by_document?nb_results={NB_RESULT}&influence_factor=2&relevance_factor=1&concatenate=true"
API_KEY = os.environ['WELEARN_API_TOKEN']



HEADERS = {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY
}

payload = {
    "query": "sustainability green human resource management",
    #"language": "fr", #NOtUSED
    #"nb_results": 50  #NOtUSED
}

response = requests.post(SEARCH_ENDPOINT, headers=HEADERS, data=json.dumps(payload))

data = []

if response.status_code == 200:
    results = response.json()
    import pdb
    pdb.set_trace()
    print("Fiche Pratique : Fonction Publique Française et Environnement\n")
    for idx, doc in enumerate(results, 1):
        payload = doc.get("payload", {})
        title = payload.get("document_title", "Titre non disponible")
        summary = payload.get("document_desc", "Résumé non disponible")
        url = payload.get("document_url", "URL non disponible")
        data.append([title, summary, url])
else:
    print(f"Erreur lors de la requête : {response.status_code} - {response.text}")
    exit()

class PDF(FPDF):
    def fit_text_to_page(self, title, summary, url, max_height=260):
        font_size = 12
        while font_size >= 6:
            self.set_font("DejaVu", 'B', size=font_size + 2)
            title_height = self.get_string_width(title) / (self.w - 30) * (font_size + 2) * 2
            self.set_font("DejaVu", size=font_size)
            summary_lines = self.multi_cell(0, font_size * 0.6, f"Résumé :\n{summary}", split_only=True)
            link_lines = self.multi_cell(0, font_size * 0.6, f"Lien : {url}", split_only=True)
            total_height = title_height + len(summary_lines) * font_size * 0.6 + len(link_lines) * font_size * 0.6 + 15
            if total_height <= max_height:
                return font_size
            font_size -= 1
        return font_size  # minimal fallback

# Préparer le PDF
pdf = PDF()
pdf.set_auto_page_break(auto=False)

# Charger les polices (assure-toi que DejaVuSans.ttf est bien dans fonts/)
pdf.add_font("DejaVu", "", "fonts/DejaVuSans.ttf", uni=True)
pdf.add_font("DejaVu", "B", "fonts/DejaVuSans.ttf", uni=True)

for idx, (title, summary, url) in enumerate(data, 1):
    pdf.add_page()
    
    # Calculer la bonne taille de texte pour que tout tienne
    font_size = pdf.fit_text_to_page(title, summary, url)
    
    # Affichage réel avec la bonne taille
    pdf.set_font("DejaVu", 'B', size=font_size + 2)
    pdf.multi_cell(0, font_size * 0.6 + 1, f"{idx}. {title}")
    
    pdf.set_font("DejaVu", size=font_size)
    pdf.ln(3)
    pdf.multi_cell(0, font_size * 0.6, f"Résumé :\n{summary}")
    
    pdf.ln(2)
    pdf.set_text_color(0, 0, 255)
    pdf.multi_cell(0, font_size * 0.6, f"Lien : {url}")
    pdf.set_text_color(0, 0, 0)

pdf.output("fiche_pratique_fonction_publique_environnement.pdf")
print("Les résumés ont été sauvegardés dans 'fiche_pratique_fonction_publique_environnement.pdf'.")
