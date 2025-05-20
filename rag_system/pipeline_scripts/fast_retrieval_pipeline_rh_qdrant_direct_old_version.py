
import logging

from kotaemon.base import Param, lazy
from kotaemon.embeddings import OpenAIEmbeddings
from kotaemon.indices.vectorindex import VectorRetrieval
from kotaemon.llms.chats.openai import ChatOpenAI
from kotaemon.storages import LanceDBDocumentStore
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

OLLAMA_DEPLOYMENT = 'docker'
VECTOR_STORE_DEPLOYMENT = 'docker'

DOCSTORE_PATH = "/app/ktem_app_data/user_data/docstore"

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

        missions = [
            "la gestion des campagnes de recrutement",
            "l'analyse des besoins en compétences et leur adéquation avec les besoins de l'organisation",
            "le suivi des processus de sélection",
            "l'accompagnement des candidats et la promotion de l'attractivité de l'employeur"
        ]

        for mission in missions:

            print("----------")
            print(f"Mission : {mission}")
            print("----------")

            query = f"Je suis Chargé de mission en Ressources Humaines. Dans le cadre de ma mission qui est {mission}, qu'est-ce que je peux faire de concret ?"
            print("Query :")
            print(query)

            all_context = []
            print("Raw retrieval context : ")

            scores_ki, texts_ki = self.run_one_generic_request(query, nb_results=10, metadatas_filters={"doc_type": "key_idea"})

            print("Key Idea : ")
            print(f"Associated scores : {scores_ki}")
            
            for text in texts_ki:
                print(f"\n - {text}")
                all_context.append(text)

            scores_ce, texts_ce = self.run_one_generic_request(query, nb_results=10, metadatas_filters={"doc_type": "concrete_experiment"})

            print("Concrete Experiments : ")
            print(f"Associated scores : {scores_ce}")
            
            for text in texts_ce:
                print(f"\n - {text}")
                all_context.append(text)

            prompt = f"From this following context, please answer the question : {query}. \n Please, give this answer in French. \n Answer the questions by being very close and anchored to the context. \n And this is the context :"

            llm_response = self.llm("\n".join([prompt, "\n".join(all_context)]))

            print(llm_response)

            #TODO export results ?

        return None




if __name__ == "__main__":

    retrieval_pipeline = RetrievalPipeline()

    retrieval_pipeline.run()

   