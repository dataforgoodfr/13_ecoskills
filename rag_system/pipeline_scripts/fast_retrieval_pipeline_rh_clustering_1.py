
import logging
import numpy as np

from kotaemon.base import Param, lazy
from kotaemon.embeddings import OpenAIEmbeddings
from kotaemon.indices.vectorindex import VectorRetrieval
from kotaemon.llms.chats.openai import ChatOpenAI
from kotaemon.storages import LanceDBDocumentStore
from qdrant_client import QdrantClient, models

from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans

def trunc_decimal(value, decs=0):
    return np.trunc(value*10**decs)/(10**decs)


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

MISSION_LIST = [
            "Élaborer la politique et la stratégie RH, en cohérence avec les orientations nationales et les priorités de la structure",
            "Gérer les parcours, les emplois et les compétences du personnel, notamment en fonctions des besoins prévisionnels de la structure",
            "Recruter et intégrer les nouveaux candidats, par des campagnes de recrutement externe et interne, la mise en place des examens.",
            "Accompagner les personnes et les collectifs de travail (carrière, mobilité, retraite, formation, management).",
            "Assurer la gestion administrative, statutaire et la paie",
            "Piloter le dialogue social et la qualité de vie au travail, ainsi la prévention des risques professionnels et psychosociaux",
            "Développer les outils, les systèmes d'analyses de données et les systèmes d'appui RH "
        ]

OLLAMA_DEPLOYMENT = 'docker'
VECTOR_STORE_DEPLOYMENT = 'docker'

DOCSTORE_PATH = "/app/ktem_app_data/user_data/docstore"

DOC_RESULT_PATH = "/app/pipeline_scripts/results/"

COLLECTION_NAME= 'index_1' # Check here your collection throught QDRANT Database dashboard & collection in docstore (ktem_app_data/user_data/doc_store)
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
        model="gemma2:2b",
        api_key="ollama"
    )

    
    def run_clustering(self, metadatas_filters: dict | None = None, sample_limit: int = 10000, nb_neighbors_limit: int = 500, n_cluster: int = 6) -> None:

        if metadatas_filters is not None:
            conditions_list = [models.FieldCondition(
                                            key=meta_key,
                                            match=models.MatchValue(
                                                    value=meta_value,
                                                    ),
                                            ) for meta_key, meta_value in metadatas_filters.items()]

            result = self.qdrant_client.search_matrix_offsets(
                collection_name=COLLECTION_NAME,
                sample=sample_limit, 
                limit=nb_neighbors_limit,
                query_filter=models.Filter(
                                        must=conditions_list
                                )
                )
        else:
            result = self.qdrant_client.search_matrix_offsets(
                collection_name=COLLECTION_NAME,
                sample=sample_limit, 
                limit=nb_neighbors_limit
                )
            

        # Convert distances matrix to python-native format 
        matrix = csr_matrix(
        (result.scores, (result.offsets_row, result.offsets_col))
        )

        ids_list = result.ids

        # Make the matrix symmetric, as UMAP expects it.
        # Distance matrix is always symmetric, but qdrant only computes half of it.
        matrix = matrix + matrix.T

        # Initialize KMeans with 10 clusters
        kmeans = KMeans(n_clusters=n_cluster)

        # Generate index of the cluster each sample belongs to
        _ = kmeans.fit_predict(matrix)

        all_text = []

        for centroid in kmeans.cluster_centers_:

            best_id = np.argmax(centroid)

            text = self.doc_store.get([ids_list[best_id]])[0].text

            all_text.append(text)

        #all_results = super().run(query, top_k = 10)
        return all_text
    
    def run(self) -> None:
        
        #TODO loop on each missions

        for mission in MISSION_LIST:
            
            print("----------")
            print(f"Mission : {mission}")
            print("----------")

            query = f""" Je travaille dans le domaine de {FUNCTIONAL_DOMAIN}. \n
                A partir d'une liste de points clé, qui sont des idées sous forme de paragraphes, et uniquement à partir de celles-ci donne moi une fiche pratique clair et argumentée sur le thème suivant :
                Qu'est-ce que je pourrais faire de concret pour agir pour 
                la transition écologique et le développement durable
                dans le cadre de la mission spécifique de mon métier qui est : 
                {mission}... \n
                - Organise ta réponse sous la forme d'une fiche pratique qui comprend 6 points,en reprenant les idées (point clé) listés après.
                - Tu dois appuyer chacun de tes points de la fiche par des expériences concrètes du terrain ou des citations d'articles scientifiques. Récupère les informations précises des entreprises ou des institutions qui sont citées.
                - Ne dis jamais 'le document dit que...' ou 'le document met en lumière...' etc. Fais comme si tu affirmais toi-même les choses.
                - Ne dévie pas des idées déjà présentes dans les points clé.
                - tout doit être écrit en langue française. \n
                Voici les points clé... chaque point clé est séparé par le symbole '--- point clé --- ' : \n"""
            
            print("Query :")
            print(query)
        
            all_text = self.run_clustering(metadatas_filters={"doc_type": "pertinent_extract",
                                                                    "mission" : mission,
                                                                    })

            context = "\n --- point clé --- \n".join(all_text)

            llm_response = self.llm("\n".join([query, context]))

            print(llm_response)

            with open(DOC_RESULT_PATH + 'clustering_1_test.doc', 'a') as f:
                f.write(f"\n \n \n --------- \n MISSION : {mission}: \n ---------- \n \n ***RAW CHUNKS*** : \n {context}  \n --- ### LLM RESPONSE ### : \n {llm_response} \n \n")

        return None




if __name__ == "__main__":

    retrieval_pipeline = RetrievalPipeline()

    retrieval_pipeline.run()

   