from typing import List, Tuple
import logging
import pathlib

from kotaemon.base import Param, lazy
from kotaemon.embeddings import OpenAIEmbeddings
from kotaemon.indices import VectorIndexing
from kotaemon.llms.chats.openai import ChatOpenAI
from kotaemon.storages import LanceDBDocumentStore
from kotaemon.storages.vectorstores.qdrant import QdrantVectorStore

from pipelineblocks.extraction.pdfextractionblock.pdf_to_markdown import PdfExtractionToMarkdownBlock
from pipelineblocks.llm.ingestionblock.openai import OpenAIMetadatasLLMInference, OpenAICustomPromptLLMInference

from taxonomy.document import EntireDocument

from kotaemon.base.schema import HumanMessage, SystemMessage

from pydantic import BaseModel

class RelevanceScore(BaseModel):
    relevance_score : float

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

LANGUAGE = 'French'

DOCSTORE_PATH = "/app/ktem_app_data/user_data/docstore"
COLLECTION_NAME = 'index_1' # Check here your collection throught QDRANT Database dashboard & collection in docstore (ktem_app_data/user_data/doc_store)

PDF_FOLDER = "./data_pdf/test"

FUNCTIONAL_DOMAIN = "Ressources Humaines (RH)"

MISSION_LIST = [
            "management of recruitment campaigns",
            "analyzing skills requirements and matching them to the organization's needs",
            "monitoring candidates selection processes",
            "supporting candidates and promoting the employer's attractiveness"
            ]

MISSION_LIST = [
            "la gestion des campagnes de recrutement des candidats",
            "l'analyse des besoins en compétences et leur adéquation avec les besoins de l'organisation",
            "le suivi des processus de sélection des candidats",
            "l'accompagnement des employés"
            "la promotion de l'attractivité de l'entreprise"
        ]

# ---- Do not touch (temporary) ------------- #

ollama_host = '172.17.0.1' if OLLAMA_DEPLOYMENT == 'docker' else 'localhost'
qdrant_host = '172.17.0.1' if VECTOR_STORE_DEPLOYMENT == 'docker' else 'localhost'


class IndexingPipeline(VectorIndexing):

    # --- Different blocks (pipeline blocks library) --- 

    pdf_extraction_block : PdfExtractionToMarkdownBlock = Param(
        lazy(PdfExtractionToMarkdownBlock).withx(
        )
    )

    # At least, one taxonomy = one llm_inference_block
    # (Multiply the number of llm_inference_block when you need handle more than one taxonomy
    metadatas_llm_inference_block_entire_doc : OpenAIMetadatasLLMInference = Param(
        lazy(OpenAIMetadatasLLMInference).withx(
            llm = ChatOpenAI(
                base_url=f"http://{ollama_host}:11434/v1/",
                model="gemma2:2b",
                api_key="ollama",
                ),
            taxonomy = EntireDocument,
            language = 'French'
            )
    )

    custom_prompt_llm_inference : OpenAICustomPromptLLMInference = Param(
        lazy(OpenAICustomPromptLLMInference).withx(
            llm = ChatOpenAI(
                base_url=f"http://{ollama_host}:11434/v1/",
                model="gemma2:2b",
                api_key="ollama",
                ))
    )
            

    # --- Final Kotaemon ingestion ----

    vector_store: QdrantVectorStore = Param(
        lazy(QdrantVectorStore).withx(
            url=f"http://{qdrant_host}:6333",
            api_key="None",
            collection_name=COLLECTION_NAME,
        ),
        ignore_ui=True,  # usefull ?
    )
    doc_store: LanceDBDocumentStore = Param(
        lazy(LanceDBDocumentStore).withx(
            path=DOCSTORE_PATH,
            collection_name = COLLECTION_NAME
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

    pdf_path: str

    def enrich_metadatas_layer(self, metadatas_base : dict | None = None,
                                    doc_type : str = 'unknow',
                                    inheritance_metadatas : dict | None = None,
                                    inheritance_fields_to_exclude : list | None = None,
                                    reapply_fields_to_root : list | None = None):
        """TODO Convert this function into method with a MetadatasManagement Object"""

        if metadatas_base is None:
            metadatas_base = {}
        metadatas_base['doc_type'] = doc_type

        if inheritance_metadatas is not None:

            applied_inheritance_metadatas = {}
            for key, value in inheritance_metadatas.items():
                if inheritance_fields_to_exclude is not None and key in inheritance_fields_to_exclude:
                    pass
                else:
                    applied_inheritance_metadatas[key] = value

            metadatas_base['extract_from'] = applied_inheritance_metadatas

            if reapply_fields_to_root is not None:

                for field in reapply_fields_to_root:

                    if field not in inheritance_metadatas.keys():
                        logging.warning(f"Sorry, but the field {field} is not present in inheritance metadatas for reapplying :  {inheritance_metadatas.keys()}")
                    else:
                        metadatas_base[field] = inheritance_metadatas[field]
        
        return metadatas_base
    
    def run_one_pdf(self, pdf_path: str) -> None:
        """
        ETL pipeline for a single pdf file
        1. Extract text and taxonomy from pdf
        2. Transform taxonomy (flattening)
        3. Ingest text and taxonomy into the vector store

        Return nothing
        """

        logging.info(f"------------- Extraction of the document : {pdf_path}")

        all_text = []
        all_metadatas = []

        chunks = self.pdf_extraction_block.run(pdf_path, 
                                               method = 'split_by_chunk',
                                               chunk_size = 600,
                                               chunk_overlap = 200)

        entire_doc = "\n".join(chunks)

        logging.info("document extracted... ok.")
        logging.info(f"Doc splitted into {len(chunks)} chunks.")

        # 1) Inference on entire doc
        
        entire_doc_metadatas = self.metadatas_llm_inference_block_entire_doc.run(entire_doc,  
                                                        doc_type  = 'entire_doc', 
                                                        inference_type = 'generic')
        
        logging.info("LLM metadatas inference (on entire doc)... ok.")
        
        metadatas_ed = entire_doc_metadatas.model_dump(mode='json')

        metadatas_ed['doc_type'] = 'entire_doc_summary'

        doc_summarize = metadatas_ed['summary']

        all_text.append(doc_summarize)
        all_metadatas.append(metadatas_ed)

        logging.info("Chunks llm inference for this doc :")
        
        for i, chunk in enumerate(chunks):

            metadatas_chunk = self.enrich_metadatas_layer(doc_type='chunk',
                                                      inheritance_metadatas=metadatas_ed,
                                                      inheritance_fields_to_exclude=['professional_functional_area'],
                                                      reapply_fields_to_root=['professional_functional_area'])
            metadatas_chunk['nb_chunk'] = i + 1

            all_text.append(chunk)
            all_metadatas.append(metadatas_chunk)
            
            for mission in MISSION_LIST :

                # 1 - Extraction close to original -

                logging.info(f"Chunk nb°{i + 1} llm inference - mission {mission} step 1/4... ")

                messages = [SystemMessage(content = "You are an extractor algoritm expert. "
                "On the document, you have to extract some relevant ideas,according to the user demand."
                "- The extraction must be very close to original document. Stay close to the original document, both in its construction and in the words used. "
                "- The extraction must be in the same language than the original document."
                "- Affirm ideas as if they were your own, in a direct style, without any reference to the 'document' or 'the text', etc. "),
                HumanMessage(content = f"In the following document, extract only those sections that are relevant to the professional field of {FUNCTIONAL_DOMAIN} "
                f"and that have a significant bearing on the specific professional mission that is {mission} and according to a sustainable development, without any reformulation."
                f"- Focus on just the specific section that's talk about {mission}, not all the document. About 100 words at maximun."
                " - Include priority some examples, concrete experiments on the field, etc... provided by institutional or corporate, if and only if they are presents in the document. Don't imagine it."
                " - If no section is relevant to this professional domain field and this specific mission, just say 'No section found'. "
                " - And don't forget : no reformulation ! Just the original words or similar words."
                "Here's the document :")]

                temperature = 0    

                response = self.custom_prompt_llm_inference.run(text = chunk,
                                                                    messages = messages,
                                                                    temperature = temperature)
                
                if len(response) > 50:
                
                    # 2 - Reformulation
                    logging.info(f"Chunk nb°{i + 1} llm inference - mission {mission} step 2/4... ")

                    messages = [SystemMessage(content = "You are a teacher expert, for adult audience."
                    "- Your responses must be in the same language than the original document."
                    "- Affirm ideas as if they were your own, in a direct style, without any reference to the 'document' or 'the text', etc. "),
                    HumanMessage(content = f"Rewrite this passage in a way that is more concise and understandable, with 3-6 sentences or about 50-100 words,"
                                 f"according to the objective : provide insights of comprehension relative to the specific mission : {mission}"
                                 "but stays as close as possible to the original wording."
                                 " - Focus on just one idea, not many. Extract the hight very relevant ideas, and be concise."
                                 "- If the original passage is already concise (less than 3 sentences or less than 50 words), keep the same lenght : very concise."
                                " - Include priority some examples, concrete experiments on the field, provided by institutional or corporate, etc. if and only if they are already presents in the original passage. Don't imagine it or take it from another source."
                                "Here's the passage :")]
                    
                    temperature = 0.1

                    response = self.custom_prompt_llm_inference.run(text = response,
                                                                    messages = messages,
                                                                    temperature = temperature)

                    # 3 - Reformulation
                    logging.info(f"Chunk nb°{i + 1} llm inference - mission {mission} step 3/4... ")

                    messages = [
                    HumanMessage(content = f"You are a traductor expert."
                    "Please, analyse the language of this document :"
                    f" - if it's already in {LANGUAGE}, just say 'No translation to do' "
                    f" - if it is not, please, traduce it in {LANGUAGE}, with a very accurate traduction."
                     "Don't refer to the original text, don't add any sentences for your own, just return the traduction and that's all.")]

                    temperature = 0.1
                    final_extract = self.custom_prompt_llm_inference.run(text = response,
                                                                    messages = messages,
                                                                    temperature = temperature)

                    logging.info(f"Chunk nb°{i + 1} llm inference - mission {mission} step 4/4... ")


                    messages = [
                    HumanMessage(content = f"You are an professional expert of {FUNCTIONAL_DOMAIN}"
                    "Please, read this short document and answer to the question :"
                    f"Do you think this short document gives some concrete ideas to the specific mission {mission}, in the context of the professional field {FUNCTIONAL_DOMAIN} ?"
                    f" - Just return a score of pertinence between 0 and 1. "
                    f" - Don't forget the specific mission is : {mission}:"
                     )]

                    temperature = 0

                    score_output = self.custom_prompt_llm_inference.run(text = final_extract,
                                                                    messages = messages,
                                                                    temperature = temperature,
                                                                    pydantic_schema = RelevanceScore)

                    metadatas_extract = self.enrich_metadatas_layer(doc_type='pertinent_extract',
                                                      inheritance_metadatas=metadatas_chunk,
                                                      inheritance_fields_to_exclude=['extract_from', 'professional_functional_area'],
                                                      reapply_fields_to_root=['professional_functional_area'])
                    
                    metadatas_extract['mission'] = mission
                    metadatas_extract['pertinence_score'] = float(score_output.relevance_score)

                    all_text.append(final_extract)
                    all_metadatas.append(metadatas_extract)

                    logging.info(f"Chunk nb°{i + 1} llm inference - mission {mission} DONE ! ")
                
                else:
                    logging.info(f"Extract not found => skip !...")
            

        logging.info("All chunks metadatas inference & special extending DONE.")
        logging.info("Export to vectore store througth Kotaemon....")

        import pdb
        pdb.set_trace()

        super().run(text=all_text, metadatas=all_metadatas)

        logging.info("Export OK")

        return None
    
    def run(self) -> None:
        target_folder = pathlib.Path(self.pdf_path)
        for pdf_file in target_folder.iterdir():

            if not pdf_file.is_dir():
                pdf_path = pdf_file.as_posix()
                if pdf_path.endswith(".pdf"):
                    self.run_one_pdf(pdf_path=pdf_path)
                else:
                    logging.warning(f"This file : {pdf_file} seems to not be a pdf... No extraction.")
            else:
                logging.warning(f"This file : {pdf_file} seems to not be file... A sub-folder ? No extraction.")

        return None
    

if __name__ == "__main__":


    indexing_pipeline = IndexingPipeline(pdf_path=PDF_FOLDER)
    indexing_pipeline.run()
