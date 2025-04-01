from typing import List, Tuple
import logging
import pathlib

from kotaemon.base import Document, Param, lazy
from kotaemon.base.component import BaseComponent
from kotaemon.base.schema import LLMInterface
from kotaemon.embeddings import OpenAIEmbeddings
from kotaemon.indices import VectorIndexing
from kotaemon.indices.vectorindex import VectorRetrieval, TextVectorQA
from kotaemon.llms.chats.openai import ChatOpenAI
from kotaemon.storages import LanceDBDocumentStore
from kotaemon.storages.vectorstores.qdrant import QdrantVectorStore

from pipelineblocks.extraction.pdfextractionblock.pdf_to_markdown import PdfExtractionToMarkdownBlock
from pipelineblocks.llm.ingestionblock.openai import OpenAIMetadatasLLMInference

from taxonomy.document import EntireDocument, ChunkOfDocument

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

#PDF_FOLDER = "./data_pdf/other_lit"

# ---- Do not touch (temporary) ------------- #

ollama_host = '172.17.0.1' if OLLAMA_DEPLOYMENT == 'docker' else 'localhost'
qdrant_host = '172.17.0.1' if VECTOR_STORE_DEPLOYMENT == 'docker' else 'localhost'


class RetrievalPipeline(VectorRetrieval):

    # --- Kotaemon db for request embedding & retrieve ----

    vector_store: QdrantVectorStore = Param(
        lazy(QdrantVectorStore).withx(
            url=f"http://{qdrant_host}:6333",
            api_key="None",
            collection_name= "index_5"
        ),
        ignore_ui=True,  # usefull ?
    )
    doc_store: LanceDBDocumentStore = Param(
        lazy(LanceDBDocumentStore).withx(
            path="./kotaemon-custom/kotaemon/ktem_app_data/user_data/docstore",
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
        api_key="ollama",
    )

    
    def run_one_generic_request(self, query) -> None:
        all_results = super().run(query, top_k = 10)
        return all_results
    
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

            results = self.run_one_generic_request(query)

            print("Raw retrieval context : ")
            all_context = []
            for result in results:
                print(f"\n - {result.content}")
                all_context.append(result.content)

            prompt = f"From this following context, please answer the question : {query}. \n Please, give this answer in French. \n Answer the questions by being very close and anchored to the context. \n And this is the context :"

            llm_response = self.llm("\n".join([prompt, "\n".join(all_context)]))

            print(llm_response)

            #TODO export results ?

        return None




if __name__ == "__main__":

    retrieval_pipeline = RetrievalPipeline()

    retrieval_pipeline.run()

   