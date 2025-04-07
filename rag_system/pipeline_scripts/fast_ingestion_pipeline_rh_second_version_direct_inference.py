import logging
import pathlib
import argparse
from datetime import datetime
import time

from kotaemon.base import Param, lazy
from kotaemon.embeddings import OpenAIEmbeddings
from kotaemon.indices import VectorIndexing
from kotaemon.llms.chats.openai import ChatOpenAI
from kotaemon.storages import LanceDBDocumentStore
from kotaemon.storages.vectorstores.qdrant import QdrantVectorStore

from pipelineblocks.extraction.pdfextractionblock.pdf_to_markdown import PdfExtractionToMarkdownBlock
from pipelineblocks.llm.ingestionblock.openai import OpenAIMetadatasLLMInference, OpenAICustomPromptLLMInference

from taxonomy.document import EntireDocument
from ingestion_manager.ingestion_manager import IngestionManager

from kotaemon.base.schema import HumanMessage, SystemMessage

from pydantic import BaseModel


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

parser = argparse.ArgumentParser(
                    prog='Ingestion Pipeline script',
                    description='What the program does',
                    epilog='Text at the bottom of help')

OLLAMA_DEPLOYMENT = 'docker'
VECTOR_STORE_DEPLOYMENT = 'docker'

LANGUAGE = 'French'

DOCSTORE_PATH = "/app/ktem_app_data/user_data/docstore"
COLLECTION_NAME = 'index_1' # Check here your collection throught QDRANT Database dashboard & collection in docstore (ktem_app_data/user_data/doc_store)

PDF_FOLDER = "./data_pdf/academic_lit"

FUNCTIONAL_DOMAIN = "Ressources Humaines (RH)"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 200

MISSION_LIST = [
            "Élaborer la politique et la stratégie RH, en cohérence avec les orientations nationales et les priorités de la structure",
            "Gérer les parcours, les emplois et les compétences du personnel, notamment en fonctions des besoins prévisionnels de la structure",
            "Recruter et intégrer les nouveaux candidats, par des campagnes de recrutement externe et interne, la mise en place des examens.",
            "Accompagner les personnes et les collectifs de travail (carrière, mobilité, retraite, formation, management).",
            "Assurer la gestion administrative, statutaire et la paie",
            "Piloter le dialogue social et la qualité de vie au travail, ainsi la prévention des risques professionnels et psychosociaux",
            "Développer les outils, les systèmes d'analyses de données et les systèmes d'appui RH "
        ]

# ---- Do not touch (temporary) ------------- #

ollama_host = '172.17.0.1' if OLLAMA_DEPLOYMENT == 'docker' else 'localhost'
qdrant_host = '172.17.0.1' if VECTOR_STORE_DEPLOYMENT == 'docker' else 'localhost'


class RelevanceScore(BaseModel):
    relevance_score : float


class ExtractionError(Exception):
    pass

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

    ingestion_manager : IngestionManager = None
    
    # Buffer (for one pdf doc)
    all_text: list = []
    all_metadatas: list = []
    ingestion_report : dict = {'status':'waiting'}
    curr_error_list : list = []

    def _link_to_a_pdf_manager(self):
        self.ingestion_manager = IngestionManager(pdf_path = self.pdf_path)

    def _send_to_vectorstore_and_docstore(self):

        super().run(text=self.all_text, metadatas=self.all_metadatas)

        logging.info("Export to vectorstore & docstore OK")

    def _send_report_to_ingestion_manager(self, pdf_filename : str):

        self.ingestion_manager._write_report(pdf_filename, self.ingestion_report)

    def _clean_buffer_elements_to_ingest(self):

        self.all_text = []
        self.all_metadatas = []
        self.ingestion_report = {'status':'waiting'}
        self.curr_error_list = []

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
    

    def run_one_mission_inference(self, chunk : str, nb_chunk: int, mission: str, metadatas_chunk: dict | None = None):

        # 1 - Extraction close to original -

        logging.info(f"Chunk nb°{nb_chunk} llm inference - mission {mission} step 1/4... ")

        messages = [SystemMessage(content = "You are an extractor algoritm expert. "
            "On the document, you have to extract some relevant ideas,according to the user demand."
            "- The extraction must be very close to original document. Stay close to the original document, both in its construction and in the words used. "
            "- The extraction must be in the same language than the original document."
            "- Affirm ideas as if they were your own, in a direct style, without any reference to the 'document' or 'the text', etc. "),
            HumanMessage(content = f"In the following document, extract only those sections that are relevant to the professional field of {FUNCTIONAL_DOMAIN} "
            f"and that have a significant bearing on the specific professional mission that is {mission} and according to a sustainable development, without any reformulation."
            f"- Focus on just the specific section that's talk about {mission}, not all the document. About 200 words at maximun."
            " - Include priority some examples, concrete experiments on the field, etc... provided by institutional or corporate, if and only if they are presents in the document. Don't imagine it."
            " - Don't include a corporate, an institution, or a company that is not present in the original passage"
            " - If no section is relevant to this professional domain field and this specific mission, just say 'No section found'. "
            " - And don't forget : no reformulation ! Just the original words or similar words."
            "Here's the document :")]

        temperature = 0    

        response = self.custom_prompt_llm_inference.run(text = chunk,
                                                            messages = messages,
                                                            temperature = temperature)
        
        if len(response) > 50:
        
            # 2 - Reformulation
            logging.info(f"Chunk nb°{nb_chunk} llm inference - mission : {mission} - step 2/4... ")

            messages = [SystemMessage(content = "You are a teacher expert, for adult audience."
            "- Your responses must be in the same language than the original document."
            "- Affirm ideas as if they were your own, in a direct style, without any reference to the 'document' or 'the text', etc. "),
            HumanMessage(content = f"Rewrite this passage in a way that is more concise and understandable, with 4-8 sentences or about 80-160 words,"
                            "but stays as close as possible to the original wording."
                            "- If the original passage is already concise (less than 4 sentences or less than 160 words), keep the same lenght, just clarify some bit."
                        " - Include priority some examples, concrete experiments on the field, provided by institutional or corporate, etc. if and only if they are already presents in the original passage. Don't imagine it or take it from another source."
                        " - Don't talk about a corporate, an institution, or a company that is not present in the original passage."
                        "Here's the passage :")]
            
            temperature = 0

            response = self.custom_prompt_llm_inference.run(text = response,
                                                            messages = messages,
                                                            temperature = temperature)

            # 3 - Reformulation
            logging.info(f"Chunk nb°{nb_chunk} llm inference - mission : {mission} - step 3/4... ")

            messages = [
            HumanMessage(content = f"You are a traductor expert."
            "Please, analyse the language of this document :"
            f" - if it's already in {LANGUAGE}, just say 'No translation to do' "
            f" - if it is not, please, traduce it in {LANGUAGE}, with a very accurate traduction."
                "Don't refer to the original text, don't add any sentences for your own, just return the traduction and that's all.")]

            temperature = 0
            final_extract = self.custom_prompt_llm_inference.run(text = response,
                                                            messages = messages,
                                                            temperature = temperature)

            logging.info(f"Chunk nb°{nb_chunk} llm inference - mission : {mission} - step 4/4... ")


            messages = [
            HumanMessage(content = f"You are an professional expert of {FUNCTIONAL_DOMAIN}"
            "Please, read this short document and answer to these questions :"
            f"Do you think this short document gives some concrete ideas to the specific mission {mission}, in the context of the professional field {FUNCTIONAL_DOMAIN} ?"
            f"Do you think this short document is faithful to the original document, given below ?"
            f" - Return a score of pertinence between 0 and 1, combined by the two aspects. "
            f" - If the short document quotes one or any corporate, institution, company that is not quoted in the original document, just return 0.0 as the pertinence score."
            f" - If the short document is not very faithful to the original document, just return 0.0 as the pertinence score."
            f" - Don't forget the specific mission is : {mission}:"
            f" - Here the original document :"
            f" {chunk} \n"
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

            self.all_text.append(final_extract)
            self.all_metadatas.append(metadatas_extract)

            logging.info(f"Chunk nb°{nb_chunk} llm inference - mission : {mission} - DONE ! ")
                
        else:
            logging.info(f"Extract not found => skip !...")

    
    def run_one_chunk(self, chunk : str, nb_chunk: int, metadatas_entire_doc : dict | None = None):

        metadatas_chunk = self.enrich_metadatas_layer(doc_type='chunk',
                                                      inheritance_metadatas=metadatas_entire_doc,
                                                      inheritance_fields_to_exclude=['professional_functional_area'],
                                                      reapply_fields_to_root=['professional_functional_area'])
        metadatas_chunk['nb_chunk'] = nb_chunk

        self.all_text.append(chunk)
        self.all_metadatas.append(metadatas_chunk)
            
        for mission in MISSION_LIST :

            try:
                self.run_one_mission_inference(chunk=chunk,
                                               nb_chunk=nb_chunk,
                                               mission=mission,
                                               metadatas_chunk=metadatas_chunk)
                
            except Exception as e:
                self.curr_error_list.append(f"Extraction error : chunk n°{nb_chunk} - mission : {mission} - error {e}")
                pass

    
    def run_one_pdf(self, pdf_path: str) -> None:
        """
        ETL pipeline for a single pdf file
        1. Extract text and taxonomy from pdf
        2. Transform taxonomy (flattening)
        3. Ingest text and taxonomy into the vector store

        Return nothing
        """

        try:

            logging.info(f"->>>--->>> Extraction of the document : {pdf_path}")

            chunks = self.pdf_extraction_block.run(pdf_path, 
                                                    method = 'split_by_chunk',
                                                    chunk_size = CHUNK_SIZE,
                                                    chunk_overlap = CHUNK_OVERLAP)

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

            self.all_text.append(doc_summarize)
            self.all_metadatas.append(metadatas_ed)

            logging.info("Chunks llm inference for this doc :")
            
            for i, chunk in enumerate(chunks):
                nb_chunk = i + 1
                try:
                    self.run_one_chunk(chunk = chunk,
                                        nb_chunk= nb_chunk,
                                        metadatas_entire_doc=metadatas_ed)
                except Exception as e:
                    self.curr_error_list.append(f"Extraction error : chunk n°{nb_chunk} - error {e}")
                    raise ExtractionError(f"Extraction error (functional) : chunk n°{nb_chunk} - error {e}")
                
        except Exception as e:
            self.curr_error_list.append(f"Complete extraction error for this doc : {e}")
            raise ExtractionError(f"Complete extraction error for this doc : - error {e}")

            
        logging.info("All chunks metadatas inference for this doc : DONE.")
        logging.info("Export to vectore store througth Kotaemon....")

        self._send_to_vectorstore_and_docstore()

        logging.info("Export Done.")

        return None
    

    def determines_pdf_to_skip(self, retry_pdf_error : bool = False, force_reindex : bool = False) -> list:

        if self.ingestion_manager is not None:
            ingestion_report = self.ingestion_manager._get_ingestion_report()
            pdf_report = ingestion_report.get('pdfs', None)

            if pdf_report is not None:
                all_pdf_file = [pdf_file for pdf_file in list(pdf_report.keys())]

                pdf_done = []
                pdf_error = []

                for pdf_file in all_pdf_file:
                    status = pdf_report.get(pdf_file).get('status')
                    if status == 'extracted ok':
                        pdf_done.append(pdf_file)
                    elif status == 'failed':
                        pdf_error.append(pdf_file)

                if force_reindex:
                    pdf_to_skip = []
                elif retry_pdf_error:
                    pdf_to_skip = pdf_done
                else:
                    pdf_to_skip = pdf_done + pdf_error
            else:
                logging.warning('Pdf manager empty. Retry with all the pdf in folder.')
                pdf_to_skip = []
        else:
            logging.warning('Pdf manager not defined. Retry with all the pdf in folder.')
            pdf_to_skip = []

        return pdf_to_skip

    
    def run(self, retry_pdf_error : bool = False, force_reindex : bool = False) -> None:

        target_folder = pathlib.Path(self.pdf_path)

        pdf_to_skip = self.determines_pdf_to_skip(retry_pdf_error=retry_pdf_error,
                                                  force_reindex=force_reindex)

        for pdf_file in target_folder.iterdir():
            pdf_file_str = pdf_file.as_posix().split('/')[-1]

            if pdf_file_str not in pdf_to_skip:

                self._clean_buffer_elements_to_ingest()

                try:

                    if not pdf_file.is_dir():
                        if pdf_file_str.endswith(".pdf"):

                            self.run_one_pdf(pdf_path=pdf_file)

                            self.ingestion_report['status'] = 'extracted ok'
                            self.ingestion_report['last_modif'] = datetime.now().isoformat()
                            self._send_report_to_ingestion_manager(pdf_filename=pdf_file_str)
                            logging.info(f"Extraction pdf : {pdf_file_str} - OK.")
                            time.sleep(seconds=1200) # Wait 20 minutes to reduce laptop temperature

                        else:
                            error_message = f"This file : '{pdf_file_str}' seems to not be a pdf... No extraction."
                            self.ingestion_report['status'] = 'failed'
                            self.ingestion_report['error_log'] = [error_message]
                            self.ingestion_report['last_modif'] = datetime.now().isoformat()
                            self._send_report_to_ingestion_manager(pdf_filename=pdf_file_str)
                            logging.warning(error_message)
                    else:
                        error_message = f"This file : '{pdf_file_str}' seems to not be file... A sub-folder ? No extraction."
                        self.ingestion_report['status'] = 'failed'
                        self.ingestion_report['error_log'] = [error_message]
                        self.ingestion_report['last_modif'] = datetime.now().isoformat()
                        self._send_report_to_ingestion_manager(pdf_filename=pdf_file_str)
                        logging.warning(error_message)

                except ExtractionError as e:
                    error_message = f"Extraction Error for this pdf file : '{pdf_file_str}'"
                    self.ingestion_report['status'] = 'failed'
                    self.ingestion_report['error_log'] = self.curr_error_list
                    self.ingestion_report['last_modif'] = datetime.now().isoformat()
                    self._send_report_to_ingestion_manager(pdf_filename=pdf_file_str)
                    logging.warning(error_message)

                except Exception as e:
                    error_message = f"Unknow Error for this pdf file : '{pdf_file_str}' - error {e}"
                    self.ingestion_report['status'] = 'failed'
                    self.ingestion_report['error_log'] = self.curr_error_list + [error_message]
                    self.ingestion_report['last_modif'] = datetime.now().isoformat()
                    self._send_report_to_ingestion_manager(pdf_filename=pdf_file_str)
                    logging.warning(error_message)
            else:
                logging.info(f"pdf file skipped >> {pdf_file_str}")

        return None
    

if __name__ == "__main__":


    parser.add_argument('-fr','--force_reindex', action="store_true", help='Force to reindex all the pdf files in the folder')           
    parser.add_argument('-re', '--retry_error', action="store_true", help='Retry ingestion all the pdf files with error status')      

    args = parser.parse_args()

    indexing_pipeline = IndexingPipeline(pdf_path=PDF_FOLDER)
    indexing_pipeline._link_to_a_pdf_manager()

    indexing_pipeline.run(retry_pdf_error=args.retry_error, force_reindex=args.force_reindex)
