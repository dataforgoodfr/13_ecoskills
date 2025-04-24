
import logging

from kotaemon.base import Param, lazy
from kotaemon.embeddings import OpenAIEmbeddings
from kotaemon.indices.vectorindex import VectorRetrieval
from kotaemon.llms.chats.openai import ChatOpenAI
from kotaemon.storages import LanceDBDocumentStore
from kotaemon.storages.vectorstores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from prompts_rh.missions import STRUCTURED_MISSION_DICT

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

COLLECTION_NAME= 'index_1' # Check here your collection throught QDRANT Database dashboard & collection in docstore (ktem_app_data/user_data/doc_store)
#RETRIEVAL_MODE= 'hybrid'


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

    llm: ChatOpenAI = ChatOpenAI.withx(
        base_url=f"http://{ollama_host}:11434/v1/",
        model="gemma3:12b-large-context",
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

        for mission in STRUCTURED_MISSION_DICT:

            print("----------")
            print(f"Mission : {mission}")
            print("----------")

            first_query = "no query"

            

            first_query = "no query"

            all_context = []
            print("Raw retrieval context : ")

            scores_ki, texts_ki = self.run_one_generic_request(first_query, nb_results=99999, metadatas_filters={"doc_type": "chunk",
                                                                                                           "missions_list[]" : mission,
                                                                                                           })

            print("Key Idea : ")
            print(f"Associated scores : {scores_ki}")

            import pdb
            pdb.set_trace()
            
            for text in texts_ki:
                all_context.append(text)


            query = f""" Je travaille dans le domaine de {FUNCTIONAL_DOMAIN}. \n
                A partir du corpus de documents suivant, répond par une fiche pratique clair et argumentée à la question suivante :
                \n La question : Dans le cadre de ma mission spécifique qui est : {mission}... \n 
                Qu'est-ce que je pourrais faire de concret pour agir pour 
                la transition écologique et le développement durable ? \n
                - Organise ta réponse sous la forme d'une fiche pratique qui comprend 6 à 10 points.
                - Tu peux appuyer chacun de tes points de la fiche par des expériences concrètes du terrain que tu as pris dans le corpus de documents, en étant fidèle à ces expériences de terrain.
                - Ne dis jamais 'le document dit que...' ou 'le document met en lumière...' etc. Fais comme si tu affirmais toi-même les choses.
                - Organise ta réponse sous la forme d'une fiche pratique point par point, en langue française, s'il te plaît. \n
                Voici le corpus de documents comme contexte pour ta réponse : \n"""
            print("Query :")
            print(query)

            llm_response = self.llm("\n".join([query, "\n".join(all_context)]))

            print(llm_response)

            import pdb
            pdb.set_trace()

            #TODO export results ?

        return None




if __name__ == "__main__":

    retrieval_pipeline = RetrievalPipeline()

    retrieval_pipeline.run()

   