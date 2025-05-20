
import logging
import numpy as np
import time

from pydantic import BaseModel

from enum import Enum

from dotenv import dotenv_values

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

EXPORT_FILENAME = "raw_1_test"

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

MISTRAL_API_KEY = dotenv_values()['MISTRAL_API_KEY']

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
                mistral_api_key=MISTRAL_API_KEY,
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
        all_ids = [res.payload.get("doc_id") for res in all_results]
        all_text = [self.doc_store.get(id)[0].text for id in all_ids]

        #all_results = super().run(query, top_k = 10)
        return all_ids, all_scores, all_text
    
    def run(self) -> None:
        
        #TODO loop on each missions

        seen_text_list = []

        for mission in STRUCTURED_MISSION_DICT:
            
            logging.info("----------")
            logging.info(f"Mission : {mission}")
            logging.info("----------")

            with open(DOC_RESULT_PATH + f'{EXPORT_FILENAME}.doc', 'a') as f:
                    f.write(f"\n \n \n --------- \n MISSION : {mission}: \n ---------- \n ")
            with open(DOC_RESULT_PATH + f'{EXPORT_FILENAME}_synthese.doc', 'a') as f:
                    f.write(f"\n \n \n --------- \n MISSION : {mission}: \n ---------- \n ")

            query = mission + ", .".join(STRUCTURED_MISSION_DICT[mission])
        
            all_ids, all_scores, all_texts = self.run_one_generic_request(metadatas_filters={"doc_type": "chunk",
                                                                    "missions_list[]" : mission
                                                                    },
                                                            query=query,
                                                            nb_results = 16)
            
            print(f"All scores : {all_scores}")


            final_text_list = [text for text, id in zip(all_texts, all_ids) if id not in seen_text_list]
            seen_text_list.extend(all_ids)
            
            logging.info(f"Corpus nb texts : {len(final_text_list)}.")

            corpus = "\n".join([f"\n --- Doc n° {i} : \n" + doc for i, doc in enumerate(final_text_list)])

            #mission_descr = "\n - ".join(STRUCTURED_MISSION_DICT[mission])

            messages = [
                SystemMessage(content = "Tu es un assistant spécialisé dans la synthèse de corpus de documents"),
                HumanMessage(content = f"""A partir du corpus de documents suivant qui sont numérotés et séparés par '-- Doc n°... : ',
                                        compose une synthèse autour de la mission : {mission}.
                                        - Le titre de cette synthèse peut avoir le titre de la mission.
                                        - Organise cette synthèse sous la forme de 7 à 10 paragraphes avec sous-titres, de 1000 à 1500 tokens au total. Chaque paragraphe représente donc 100 à 200 tokens.
                                        \n
                                            - Utilise tout ou partie des références, idées et examples précis inclus dans le corpus de documents.
                                            - Ne dis jamais 'le document dit que...' ou 'le document met en lumière...' etc. Fais comme si tu affirmais toi-même les choses.
                                            - N'ajoute aucun nom d'entreprise ou d'institutions qui ne sont pas dans les documents.
                                            \n \n Voici le corpus de document :
                                                {corpus}
                                            \n \n Attention, rappelle-toi :
                                            Il est interdit de citer d'autres expériences ou éléments que celles des documents.
                                            Il est interdit de citer d'autres entreprises ou institutions que celles qui sont citées dans les documents.
                                            Tu peux citer parfois les documents en précisant des références précises comme par exemple son numéro de document indiqué.
                                            """)]
        
            temperature = 0.1
            max_token = 2000

            llm_response = self.custom_prompt_llm_inference.run(messages = messages,
                                                            temperature = temperature,
                                                            language = 'French',
                                                            max_token = max_token)
            
            time.sleep(DELAY_BETWEEN_REQUEST) # to not excedding request time
                
            logging.info(f"\n\n --- LLM RESPONSE --- \n\n {llm_response}")

            with open(DOC_RESULT_PATH + f'{EXPORT_FILENAME}.doc', 'a') as f:
                f.write(f"\n ***RAW CHUNKS*** : \n {corpus}  \n\n\n **** LLM RESPONSE **** --- \n\n {llm_response}")

            with open(DOC_RESULT_PATH + f'{EXPORT_FILENAME}_synthese.doc', 'a') as f:
                f.write(f"\n \n \n ### LLM FINAL TEXT ### : \n {llm_response} \n \n")

        return None




if __name__ == "__main__":

    retrieval_pipeline = RetrievalPipeline()

    retrieval_pipeline.run()

   