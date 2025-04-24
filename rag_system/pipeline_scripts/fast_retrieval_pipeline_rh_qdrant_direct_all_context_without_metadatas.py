
import logging

from kotaemon.base import Param, lazy
from kotaemon.embeddings import OpenAIEmbeddings
from kotaemon.indices.vectorindex import VectorRetrieval
from kotaemon.llms.chats.openai import ChatOpenAI
from kotaemon.storages import LanceDBDocumentStore
from kotaemon.storages.vectorstores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

LOG_LEVEL = logging.INFO
# When you set the level, all messages from a higher level of severity are also
# logged. For example, when you set the log level to `INFO`, all `WARNING`,
# `ERROR` and `CRITICAL` messages are also logged, but `DEBUG` messages are not.
# Set a seed to enable reproducibility
SEED = 1
# Set a format to the logs.
LOG_FORMAT = '[%(levelname)s | ' + ' | %(asctime)s] - %(message)s'
# Name of the file to store the logs.
LOG_FILENAME = 'script_execution.log'
# == Set up logging ============================================================
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    force=True,
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(LOG_FILENAME, "a", "utf-8"),
              logging.StreamHandler()]
)

FUNCTIONAL_DOMAIN = "Ressources Humaines (RH)"

MISSION_DICT_FR = {
    "Définir, piloter et mettre en œuvre la stratégie RH": [
        "Élaborer, conduire et évaluer la politique RH, en cohérence avec les orientations nationales et les priorités de la structure.",
        "Définir les plans d'action stratégiques en matière de recrutement, GPEEC, mobilité, formation, qualité de vie au travail, communication, conditions de travail, diversité et politique managériale.",
        "Conduire le changement et accompagner les évolutions organisationnelles, culturelles et technologiques de l'administration."
    ],
    "Gérer les parcours, les emplois et les compétences": [
        "Mettre en œuvre la gestion prévisionnelle des effectifs, des emplois et des compétences (GPEEC).",
        "Analyser les besoins actuels et futurs, cartographier les compétences, identifier les viviers et les métiers sensibles.",
        "Conseiller les agents dans la construction de leur parcours professionnel, accompagner les mobilités, les reconversions, les transitions professionnelles, et les secondes parties de carrière.",
        "Participer aux revues de cadres, à la constitution de viviers de talents, et à l'animation des réseaux professionnels."
    ],
    "Recruter, intégrer et fidéliser les agents": [
        "Organiser les campagnes de recrutement externe et interne, concevoir les dispositifs d'intégration et de suivi (onboarding).",
        "Accompagner les managers dans la définition des besoins en recrutement.",
        "Promouvoir la marque employeur et développer les partenariats pour le sourcing (forums, réseaux, apprentissage).",
        "Mettre en œuvre les concours et examens professionnels, en garantissant la sécurité juridique et l'équité des processus."
    ],
    "Accompagner les personnes et les collectifs de travail": [
        "Conseiller les agents et les services en matière RH (carrière, mobilité, retraite, formation, management).",
        "Réaliser des entretiens individualisés, animer des actions collectives de développement professionnel.",
        "Proposer des dispositifs d'accompagnement : coaching, conseil RH de proximité, soutien à la professionnalisation des pratiques managériales.",
        "Assurer l'accueil, l'écoute, la médiation et l'orientation vers les ressources adaptées."
    ],
    "Assurer la gestion administrative, statutaire et la paie": [
        "Gérer les actes administratifs : positions, contrats, retraites, promotions, cessations de fonction.",
        "Assurer le traitement, la vérification et la fiabilisation des rémunérations, primes et cotisations sociales.",
        "Suivre les absences, le temps de travail, les congés, le télétravail.",
        "Gérer les procédures de contentieux et les dossiers individuels sensibles dans le respect du cadre réglementaire."
    ],
    "Piloter le dialogue social et la qualité de vie au travail": [
        "Organiser les élections professionnelles, les instances de dialogue social et les négociations collectives.",
        "Préparer les ordres du jour, assurer le secrétariat et le suivi des décisions des instances représentatives.",
        "Prévenir et gérer les conflits collectifs et individuels, produire des notes d'analyse du climat social.",
        "Élaborer et mettre en œuvre les politiques de prévention des risques professionnels et psychosociaux, en lien avec les services compétents (DUERP, plans d’action, accompagnement des acteurs)."
    ],
    "Développer les outils, les données et les systèmes d'appui RH": [
        "Concevoir et piloter les systèmes d'information RH (SIRH), produire les données sociales et les indicateurs (RSU, DOETH, bilans, tableaux de bord).",
        "Appuyer la décision stratégique via l'analyse des données RH (emploi, masse salariale, diversité, égalité professionnelle).",
        "Contribuer à la transformation numérique des fonctions RH et au développement d'une gestion fondée sur les données (data RH).",
        "Participer à la veille réglementaire, technologique et sociétale sur les évolutions du métier RH."
    ]
}

OLLAMA_DEPLOYMENT = 'docker'
VECTOR_STORE_DEPLOYMENT = 'docker'

DOCSTORE_PATH = "/app/ktem_app_data/user_data/docstore"

DOC_RESULT_PATH = "/app/pipeline_scripts/results/"

COLLECTION_NAME= 'index_4' # Check here your collection throught QDRANT Database dashboard & collection in docstore (ktem_app_data/user_data/doc_store)
#RETRIEVAL_MODE= 'hybrid'


#PDF_FOLDER = "./data_pdf/other_lit"

# ---- Do not touch (temporary) ------------- #

ollama_host = '172.17.0.1' if OLLAMA_DEPLOYMENT == 'docker' else 'localhost'
qdrant_host = '172.17.0.1' if VECTOR_STORE_DEPLOYMENT == 'docker' else 'localhost'


class RetrievalPipeline(VectorRetrieval):

    # --- Kotaemon db for request embedding & retrieve ----

    #retrieval_mode : str = RETRIEVAL_MODE - unused in this direct qdrant script

    qdrant_client: QdrantClient = Param(
        lazy(QdrantClient).withx(
        url=f"http://{qdrant_host}:6333",
        api_key="None"
        )
    )

    doc_store: LanceDBDocumentStore = Param(
        lazy(LanceDBDocumentStore).withx(
            path=DOCSTORE_PATH,
            collection_name= COLLECTION_NAME,
        ),
        ignore_ui=True,
    )
    embedding: OpenAIEmbeddings = Param(
        lazy(OpenAIEmbeddings).withx(
            # base_url="http://172.17.0.1:11434/v1/",
            base_url=f"http://{ollama_host}:11434/v1/",
            model="snowflake-arctic-embed2",
            api_key="ollama",
        ),
        ignore_ui=True,
    )

    llm: ChatOpenAI = ChatOpenAI.withx(
        base_url=f"http://{ollama_host}:11434/v1/",
        model="gemma3:4b-large-context",
        api_key="ollama"
    )

    
    def run_one_generic_request(self, query, nb_results: int = 5, metadatas_filters: dict | None = None) -> None:

        query_vector = self.embedding(query)[0].embedding

        if metadatas_filters is not None:
            conditions_list = [models.FieldCondition(
                                            key=meta_key,
                                            match=models.MatchValue(
                                                    value=meta_value,
                                                    ),
                                            ) for meta_key, meta_value in metadatas_filters.items()]

            all_results = self.qdrant_client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                limit=nb_results,  # Number of results
                query_filter=models.Filter(
                                        must=conditions_list
                                )
                )
        else:
            all_results = self.qdrant_client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                limit=nb_results
                )
        
        all_scores = [res.score for res in all_results]
        all_doc_content = self.doc_store.get([res.payload.get("doc_id") for res in all_results])
        all_text = [doc.text for doc in all_doc_content]

        #all_results = super().run(query, top_k = 10)
        return all_scores, all_text
    
    def run(self) -> None:
        
        #TODO loop on each missions

        for mission in MISSION_DICT_FR.keys():

            print("----------")
            print(f"Mission : {mission}")
            print("----------")

            retrieval_query = f""" Je travaille dans le domaine de {FUNCTIONAL_DOMAIN}. \n
                \n Dans le cadre de ma mission spécifique qui est : {mission}... \n
                et dont voici la description complète :
                {MISSION_DICT_FR[mission]} \n
                qu'est-ce que je pourrais faire de concret pour agir pour 
                la transition écologique et le développement durable ? """

            query = f""" 
                - Organise ta réponse sous la forme d'une fiche pratique qui comprend 6 à 10 points.
                - Tu peux appuyer chacun de tes points de la fiche par des expériences concrètes du terrain que tu as pris dans le corpus de documents, en étant fidèle à ces expériences de terrain.
                Ces expériences doivent être liées à des entreprises ou des institutions présentes dans les documents de contexte et que tu peux citer.
                - Ne dis jamais 'le document dit que...' ou 'le document met en lumière...' etc. Fais comme si tu affirmais toi-même les choses.
                - Organise ta réponse sous la forme d'une fiche pratique point par point, en langue française, s'il te plaît. \n
                Voici le corpus de documents comme contexte pour ta réponse : \n"""
            print("Query :")
            print(query)

            all_context = []
            print("Raw retrieval context : ")

            scores_ki, texts_ki = self.run_one_generic_request(retrieval_query, nb_results=10)

            print("Key Idea : ")
            print(f"Associated scores : {scores_ki}")

            import pdb
            pdb.set_trace()
            
            for text in texts_ki:
                all_context.append(text)

            context = "\n --- raw doc --- \n".join(all_context)

            llm_response = self.llm("\n".join([retrieval_query, query, context]))

            print(llm_response)

            with open(DOC_RESULT_PATH + 'rag_base_without_metadatas_gemme3:4b-large-context.doc', 'a') as f:
                f.write(f"\n \n \n --------- \n MISSION : {mission}: \n ---------- \n \n ***RAW CHUNKS*** : \n {context}  \n --- ### LLM RESPONSE ### : \n {llm_response} \n \n")

        return None




if __name__ == "__main__":

    retrieval_pipeline = RetrievalPipeline()

    retrieval_pipeline.run()

   