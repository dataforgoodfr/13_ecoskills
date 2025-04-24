
import logging
import numpy as np
import time

from pydantic import BaseModel

from enum import Enum

from kotaemon.base import Param, lazy
from kotaemon.embeddings import OpenAIEmbeddings
from kotaemon.indices.vectorindex import VectorRetrieval
from kotaemon.llms.chats.openai import ChatOpenAI
from kotaemon.llms.chats.langchain_based import LCChatMistral
from pipelineblocks.llm.ingestionblock.langchain import LangChainCustomPromptLLMInference
from kotaemon.storages import LanceDBDocumentStore
from kotaemon.storages.vectorstores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from kotaemon.base.schema import HumanMessage, SystemMessage

from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans

from prompts_rh.missions import STRUCTURED_MISSION_DICT

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

OLLAMA_DEPLOYMENT = 'docker'
VECTOR_STORE_DEPLOYMENT = 'docker'

DOCSTORE_PATH = "/app/ktem_app_data/user_data/docstore"

DOC_RESULT_PATH = "/app/pipeline_scripts/results/"

COLLECTION_NAME= 'index_8' # Check here your collection throught QDRANT Database dashboard & collection in docstore (ktem_app_data/user_data/doc_store)
#RETRIEVAL_MODE= 'hybrid'

EXTRA_MAX_RETRIES = 6
DELAY_BETWEEN_REQUEST = 1 #seconds

#PDF_FOLDER = "./data_pdf/other_lit"

# ---- Do not touch (temporary) ------------- #

ollama_host = '172.17.0.1' if OLLAMA_DEPLOYMENT == 'docker' else 'localhost'
qdrant_host = '172.17.0.1' if VECTOR_STORE_DEPLOYMENT == 'docker' else 'localhost'

class Reponse(str, Enum):
    oui = 'Oui'
    non = 'Non'

class ResponseWithJustification(BaseModel):
    reponse : Reponse
    justification : str



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

    custom_prompt_llm_inference : LangChainCustomPromptLLMInference = Param(
        lazy(LangChainCustomPromptLLMInference).withx(
            llm = LCChatMistral(
                model="open-mistral-nemo",
                mistral_api_key="mHJJoBJzKeWylZs28Iy9aTqIxADTD5zK",
                temperature=0.3
                ))
    )

    def run_one_mission_inference(self, chunk : str, mission: str):
        
        sub_missions_descr = '\n - '.join(STRUCTURED_MISSION_DICT[mission]['sub_missions'])

        human_message_content = f"""On cherche à savoir si le document aide **concrètement** à satisfaire
                                la mission spécifique suivante du domaine {FUNCTIONAL_DOMAIN},
                                **dans une perspective de transition écologique, de développement durable ou de solution face aux problèmes de climat.**.
                                 \n\n Mission spécifique ciblée : {mission}.
                                 Cela inclut : {sub_missions_descr}
                                \n\n Ta réponse doit être structurée avec un format json comme ceci: {{'reponse': 'Oui'/'Non', 'justification': [...] }}.
                                Le critère principal est la **présence de leviers mis en œuvre ou outillés**, c'est-à-dire
                                d'expériences concrètes terrains ou métiers, en rapport avec le domaine profesionnel et la mission spécifique.
                                \n\n --- \n """
        
        for i, example_dict in enumerate(STRUCTURED_MISSION_DICT[mission]['examples']):
            expected_resp = example_dict['response']
            justif = example_dict['justification']
            doc = example_dict['document']
            human_message_content += f"""\n **Example {i + 1}** 
                                        \n document : {doc}
                                        \n Réponse attendue :
                                         \n {{ 'reponse' : {expected_resp} \n 'justification' : {justif} }} \n --- \n """
            
        human_message_content += f""" **Maintenant, analyse le document suivant :**
                                    \n Réponds avec un format json comme ceci: {{'reponse': 'Oui'/'Non', 'justification': [...] }} \n 
                                    \n Voici le document : {chunk}"""


        messages = [
                    SystemMessage(content = f"Tu es un assistant spécialisé dans l’évaluation fine des documents." 
                                  "Ta tâche est de déterminer si un document est **pertinent** pour "
                                  f"une **mission professionnelle spécifique du domaine {FUNCTIONAL_DOMAIN},"
                                  "en justifiant rigoureusement ta réponse par des critères concrets."
                                    ),
                    HumanMessage(content = human_message_content)]
            
        temperature = 0

        response = self.custom_prompt_llm_inference.run(messages = messages,
                                                        temperature = temperature,
                                                        language = 'French',
                                                        pydantic_schema = ResponseWithJustification,
                                                        extra_max_retries = EXTRA_MAX_RETRIES)
        
        return response
    
    def run_clustering(self, metadatas_filters: dict | None = None, sample_limit: int = 10000, nb_neighbors_limit: int = 500, n_cluster: int = 6, top_k: int = 10) -> None:

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

        ids = np.array(result.ids)

        logging.info(f"Clustering done with {len(ids)} results founds.")

        # Make the matrix symmetric, as UMAP expects it.
        # Distance matrix is always symmetric, but qdrant only computes half of it.
        matrix = matrix + matrix.T

        # Initialize KMeans with 6 clusters by default
        kmeans = KMeans(n_clusters=n_cluster)

        # Generate index of the cluster each sample belongs to
        _ = kmeans.fit_predict(matrix)

        all_clusters = {}

        seen_idx = []

        for i, centroid in enumerate(kmeans.cluster_centers_):

            best_idx = np.argpartition(centroid, -top_k)[-top_k:]

            selected_idx = [idx for idx in best_idx if idx not in seen_idx]
            seen_idx.extend(selected_idx)

            if len(selected_idx) > 0:

                associated_scores = centroid[selected_idx]

                all_text = self.doc_store.get(list(ids[selected_idx]))

                all_clusters[i] = {'all_text' : [text.content for text in all_text],
                               'all_scores' : associated_scores}
                
            else:
                 logging.warning('Empty cluster after removing the seen chunk.')
                 pass

        #all_results = super().run(query, top_k = 10)
        return all_clusters
    
    def run(self) -> None:
        
        #TODO loop on each missions

        for mission in STRUCTURED_MISSION_DICT:
            
            logging.info("----------")
            logging.info(f"Mission : {mission}")
            logging.info("----------")

            with open(DOC_RESULT_PATH + 'clustering_2_test.doc', 'a') as f:
                    f.write(f"\n \n \n --------- \n MISSION : {mission}: \n ---------- \n ")
            with open(DOC_RESULT_PATH + 'clustering_2_test_synthese.doc', 'a') as f:
                    f.write(f"\n \n \n --------- \n MISSION : {mission}: \n ---------- \n ")
        
            all_clusters = self.run_clustering(metadatas_filters={"doc_type": "chunk",
                                                                    "missions_list[]" : mission
                                                                    },
                                                n_cluster=6,
                                                top_k = 2)
            
            logging.info("Clustering DONE.")

            all_paragraph = []

            title_used = []

            for cluster_idx, cluster_content in all_clusters.items():

                logging.info(f"LLM request for building paragraph associated to the cluster n° {cluster_idx}...")
                logging.info(f"-- with scores : {cluster_content['all_scores']}")

                filtered_text = []

                for text in cluster_content['all_text']:

                    try:
                    
                        response = self.run_one_mission_inference(chunk = text,
                                                                mission = mission)
                        
                        time.sleep(DELAY_BETWEEN_REQUEST) # to not excedding request time
                        
                        if response and response.get('reponse').lower()=='oui':
                            logging.info("relevant mission !...")
                            filtered_text.append(text)

                        elif response:
                            logging.info("non relevant mission...")

                        else:
                            logging.warning("Empty response ?")

                    except Exception as e:
                        logging.warning(f"Inference failed initial error : {e}")
                    
                #sub_missions_descr = '\n - '.join(STRUCTURED_MISSION_DICT[mission]['sub_missions'])

                logging.info(f"Number of filtered text : {len(filtered_text)}")

                if len(filtered_text) > 0:

                    logging.info(f"Paragraph reformulation...")

                    all_docs = "\n\n".join(filtered_text)

                    messages = [
                        SystemMessage(content = "Tu es un assistant spécialisé dans la reformulation fidèle de document." 
                                        "Ta tâche est de reformuler légèrement un document pour améliorer sa clarté en faisant référence exclusivement à ce document fourni."
                                        ),
                        HumanMessage(content = f"""A partir du document suivant,
                                                peux-tu écrire la même chose en 1 à 2 paragraphes sans aucun titre ?
                                                    - Génère un à deux paragraphes seulement avec toutes les références, idées, examples très précis inclus dans le document.
                                                    - Ne mets pas de titre et va vraiment à l'essentiel des idées, références et examples développés.
                                                    - Ne dis jamais 'le document dit que...' ou 'le document met en lumière...' etc. Fais comme si tu affirmais toi-même les choses.
                                                    - N'ajoute aucun nom d'entreprise ou d'institutions qui ne sont pas dans les documents.
                                                    \n \n Voici le document :
                                                        {all_docs}
                                                    \n \n Attention, rappelle-toi ! Reformule très légèrement le document pour améliorer sa clarté, c'est tout.
                                                    Il est interdit de citer d'autres expériences ou éléments que celles des documents.
                                                    Il est interdit de citer d'autres entreprises ou institutions que celles qui sont citées dans les documents.
                                                    """)]
                
                    temperature = 0.1
                    max_token = 512

                    pararaph = self.custom_prompt_llm_inference.run(messages = messages,
                                                                    temperature = temperature,
                                                                    language = 'French',
                                                                    max_token = max_token)
                    
                    time.sleep(DELAY_BETWEEN_REQUEST) # to not excedding request time

                    logging.info(f"PAragraph Title generation...")

                    title_used_str = ", ".join(title_used)

                    messages = [
                        SystemMessage(content = "Tu es un assistant spécialisé dans la création de titre." 
                                        "Ta tâche est d'ajouter un titre pertinent à un paragraphe, entre 5 et 20 mots."
                                        ),
                        HumanMessage(content = f"""A partir du document suivant,
                                                peux-tu écrire un titre ?
                                                    - N'emplois pas l'un de ces titres déjà utilisés ailleurs : {title_used_str}
                                                    - Evite les titres vagues ou généraux. Inspire toi des examples et idées concrètes du document.
                                                    - Ne donne en réponse que le titre choisi et c'est tout.
                                                    \n \n Voici le document :
                                                        {pararaph}
                                                    """)]
                
                    temperature = 0.5
                    max_token = 512

                    title = self.custom_prompt_llm_inference.run(messages = messages,
                                                                    temperature = temperature,
                                                                    language = 'French',
                                                                    max_token = max_token)
                    
                    time.sleep(DELAY_BETWEEN_REQUEST) # to not excedding request time

                    complete_text = f"**{title}** \n {pararaph}"

                    
                    logging.info(f"--- LLM RESPONSE --- \n\n {complete_text}")

                    with open(DOC_RESULT_PATH + 'clustering_2_test.doc', 'a') as f:
                        f.write(f"--- CLUSTER n° {cluster_idx} --- \n ***RAW CHUNKS*** : \n {all_docs}  \n\n --- **** LLM RESPONSE **** --- \n\n {complete_text}")

                    all_paragraph.append(f"\n\n {complete_text}")

            # Final building 
                
            final_text = "\n\n".join(all_paragraph)

            with open(DOC_RESULT_PATH + 'clustering_2_test_synthese.doc', 'a') as f:
                f.write(f"\n \n \n ### LLM FINAL TEXT ### : \n {final_text} \n \n")

        return None




if __name__ == "__main__":

    retrieval_pipeline = RetrievalPipeline()

    retrieval_pipeline.run()

   