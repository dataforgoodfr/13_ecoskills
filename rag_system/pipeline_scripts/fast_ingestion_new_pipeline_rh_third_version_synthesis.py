import logging
import pathlib
import argparse
import time
import uuid

from pathlib import Path

from kotaemon.base import Param, lazy, Document
from kotaemon.embeddings import OpenAIEmbeddings
from kotaemon.indices import VectorIndexing
from kotaemon.llms.chats.langchain_based import LCChatMistral

from llama_index.core.readers.file.base import default_file_metadata_func

from pipelineblocks.extraction.pdfextractionblock.pdf_to_markdown import PdfExtractionToMarkdownBlock
from pipelineblocks.llm.ingestionblock.langchain import LangChainMetadatasLLMInference, LangChainCustomPromptLLMInference
from taxonomy.document import EntireDocument

from kotaemon.base.schema import HumanMessage, SystemMessage

from ktem.index.file.index import FileIndex
from ktem.index.file.pipelines import IndexPipeline

from kotaemon.loaders import PDFThumbnailReader

from kotaemon.indices.splitters import TokenSplitter

from pydantic import BaseModel

from enum import Enum

from prompts_rh.missions import STRUCTURED_MISSION_DICT

from dotenv import dotenv_values

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

COLLECTION_ID = 1
USER_ID = '2bd87cee60a5430ca23c84ee80d81cfa'

PDF_FOLDER = "./data_pdf/from_wlearn"

FUNCTIONAL_DOMAIN = "Ressources Humaines (RH)"

METADATA_BASE = {
    "ingestion_method" : 'fast_script',
    "ingestion_origin_folder": PDF_FOLDER,
    "functional_domain" : FUNCTIONAL_DOMAIN
}

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

EXTRA_MAX_RETRIES = 6
DELAY_BETWEEN_REQUEST = 1 #seconds

MISTRAL_API_KEY = dotenv_values()['MISTRAL_API_KEY']


# ---- Do not touch (temporary) ------------- #

ollama_host = '172.17.0.1' if OLLAMA_DEPLOYMENT == 'docker' else 'localhost'
# qdrant_host = '172.17.0.1' if VECTOR_STORE_DEPLOYMENT == 'docker' else 'localhost' -- now, defined in flowsettings.py


class RelevanceScore(BaseModel):
    relevance_score : float


class Reponse(str, Enum):
    oui = 'Oui'
    non = 'Non'

class ResponseWithJustification(BaseModel):
    reponse : Reponse
    justification : str

class ExtractionError(Exception):
    pass

class IndexingPipelineShortCut(IndexPipeline):

    # --- PDF Extraction (optionnal... to replace Kotaemon Loader by default) --- 

    pdf_extraction_block : PdfExtractionToMarkdownBlock = Param(
        lazy(PdfExtractionToMarkdownBlock).withx(
        )
    )

    """
    llm = ChatOpenAI(
                base_url=f"http://{ollama_host}:11434/v1/",
                model="gemma3:4b-large-context",
                api_key="ollama",
                )"""

    # --- LLM MODELS ---
    # At least, one taxonomy = one llm_inference_block
    # (Multiply the number of llm_inference_block when you need handle more than one taxonomy
    metadatas_llm_inference_block_entire_doc : LangChainMetadatasLLMInference = Param(
        lazy(LangChainMetadatasLLMInference).withx(
            llm = LCChatMistral(
                model="open-mistral-nemo",
                mistral_api_key=MISTRAL_API_KEY,
                temperature=0
                ),
            taxonomy = EntireDocument,
            language = 'French'
            )
    )

    custom_prompt_llm_inference : LangChainCustomPromptLLMInference = Param(
        lazy(LangChainCustomPromptLLMInference).withx(
            llm = LCChatMistral(
                model="open-mistral-nemo",
                mistral_api_key=MISTRAL_API_KEY,
                temperature=0.1
                ))
    )

    # --- EMBEDDINGS MODELS ---
    embedding: OpenAIEmbeddings = Param(
        lazy(OpenAIEmbeddings).withx(
            # base_url="http://172.17.0.1:11434/v1/",
            base_url=f"http://{ollama_host}:11434/v1/",
            model="snowflake-arctic-embed2",
            api_key="ollama",
        ),
        ignore_ui=True,
    )
            

    # --- Others params ---

    file_index_associated = FileIndex(app=None,
                            id= COLLECTION_ID,
                            name= 'ecoskills_rh',
                            config= {
                                'embedding': 'default',
                                'supported_file_types': '.png, .jpeg, .jpg, .tiff, .tif, .pdf, .xls, .xlsx, .doc, .docx, .pptx, .csv, .html, .mhtml, .txt, .md, .zip', 'max_file_size': 1000, 'max_number_of_files': 0, 'private': True, 'chunk_size': 0, 'chunk_overlap': 0})

    file_index_associated.on_start()

    
    pdf_path: str

    #ingestion_manager : IngestionManager = None

    Index : None
    Source : None
    collection_name : None
    FSPath : None
    user_id : None
    loader : None
    splitter : None
    vector_indexing : None
    
    
    

    """
    def _send_to_vectorstore_and_docstore(self):

        super().run(text=self.all_text, metadatas=self.all_metadatas)

        logging.info("Export to vectorstore & docstore OK")
    """

    def get_resources_set_up(self):

        self.VS = self.file_index_associated._vs
        self.DS = self.file_index_associated._docstore
        self.FSPath = self.file_index_associated._fs_path
        self.Index = self.file_index_associated._resources.get('Index')
        self.Source = self.file_index_associated._resources.get('Source')
        self.collection_name = f"index_{self.file_index_associated.id}"
        self.user_id = USER_ID
        self.loader = PDFThumbnailReader()
        self.splitter = TokenSplitter(
                chunk_size=CHUNK_SIZE or 1024,
                chunk_overlap=CHUNK_OVERLAP or 256,
                separator="\n\n",
                backup_separators=["\n", ".", "\u200B"],
            )
        self.vector_indexing = VectorIndexing(
            vector_store=self.VS, doc_store=self.DS, embedding=self.embedding
        )

    # --- CUSTOM PIPELINE LOGIC ----

    def concat__metadatas_layer(self, metadatas_base: dict, metadatas_root: dict):
        for key, value in metadatas_root.items():
            metadatas_base[key] = value
        return metadatas_base

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
    

    def run_one_mission_inference(self, chunk : str, mission: str):
        
        sub_missions_descr = '\n - '.join(STRUCTURED_MISSION_DICT[mission]['sub_missions'])

        human_message_content = f"""On cherche à savoir si le document aide **concrètement** à satisfaire
                                la mission spécifique suivante du domaine {FUNCTIONAL_DOMAIN},
                                **dans une perspective de transition écologique, de développement durable ou de solution face aux problèmes de climat.**.
                                 \n\n Mission spécifique ciblée : {mission}.
                                 Cela inclut : {sub_missions_descr}
                                \n\n Ta réponse doit être structurée avec un format json comme ceci: {{'reponse': 'Oui'/'Non', 'justification': [...] }}.
                                Le critère principal est la **présence de leviers mis en œuvre ou outillés**, c'est-à-dire
                                d'expériences concrètes terrains ou métiers, en rapport avec le domaine profesionnel et la mission.
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

    
    def inference_on_one_chunk(self, chunk : str, nb_chunk: int):

        missions_list = []
        missions_justification_answers = {}
        extraction_error = {}
            
        for mission in STRUCTURED_MISSION_DICT.keys() :

            logging.info(f"Chunk nb°{nb_chunk} llm inference - mission {mission} ... ")

            try:
                response = self.run_one_mission_inference(chunk=chunk,
                                               mission=mission)

                time.sleep(DELAY_BETWEEN_REQUEST) # to not excedding request time
               
                if response and response.get('reponse').lower()=='oui':
                    logging.info("relevant mission !...")
                    missions_list.append(mission)
                    missions_justification_answers[mission] = response

                elif response:
                    logging.info("non relevant mission...")
                    missions_justification_answers[mission] = response

                else:
                    logging.warning("Empty response ?")
                    extraction_error[mission] = "Error : empty response"
                
            except Exception as e:
                logging.info(f"Extraction error : chunk n°{nb_chunk} - mission : {mission} - error {e}")
                extraction_error[mission] = f"{e}"
                pass

        return missions_list, missions_justification_answers, extraction_error
        

    def inference_and_summarize_entire_doc(self, entire_text: str, metadata_vs_base: dict | None = None):

        metadatas_ed = None

        try:

            metadatas_ed = self.metadatas_llm_inference_block_entire_doc.run(entire_text,  
                                                            doc_type  = 'entire_doc', 
                                                            inference_type = 'generic')
            
            if not isinstance(metadatas_ed, dict):
                raise ExtractionError("Sorry, but response from llm is not well formated...")
            
        except Exception as e:
            logging.warning(f"Error summarization... doc with a too large context ? Trying to reduce it... Initial error : {e}")
            for _ in range(3):
                try:
                    lenght_et = len(entire_text)
                    entire_text = entire_text[lenght_et // 3 : - lenght_et // 3]
                    logging.warning(f"Failed summarization & metadatas inference on entire doc - lenght {lenght_et} letters. Retry with lenght {len(entire_text)} letters...")
                    metadatas_ed = self.metadatas_llm_inference_block_entire_doc.run(entire_text,  
                                                                doc_type  = 'entire_doc', 
                                                                inference_type = 'generic')
                    
                    if isinstance(metadatas_ed, dict):
                        break
                    
                except Exception as e:
                    pass

        
        if not isinstance(metadatas_ed, dict):
            raise RuntimeError("Failed summarization inference & metadatas for this doc...")
        
        logging.info("LLM metadatas inference (on entire doc)... ok.")

        #metadatas_ed = entire_doc_metadatas.model_dump(mode='json')

        metadatas_ed['doc_type'] = 'entire_doc_summary'

        metadatas_ed = self.concat__metadatas_layer(metadatas_base = metadatas_ed, metadatas_root=metadata_vs_base)

        summary = metadatas_ed['summary']

        return Document(text=summary, id_=str(uuid.uuid4())), metadatas_ed


    
    def inference_on_all_chunks(self, chunks: list[Document], metadata_vs_base: dict | None = None, metadata_entire_doc: dict | None = None):

        all_metadatas = []

        if metadata_vs_base is None:
            metadata_vs_base = {}

        if metadata_entire_doc is None:
            metadata_entire_doc = {}

        try:

            language = metadata_entire_doc.get('language', 'English').lower()

            logging.info("Chunks llm inference for this doc :")
            
            for i, chunk in enumerate(chunks):
                
                # Prepare metadatas base for this chunk :
                metadata_chunk = self.enrich_metadatas_layer(doc_type='chunk',
                                                      inheritance_metadatas=metadata_entire_doc,
                                                      inheritance_fields_to_exclude=['professional_functional_area', 'doc_type'],
                                                      reapply_fields_to_root=['professional_functional_area'])
                
                metadata_chunk = self.concat__metadatas_layer(metadatas_base=metadata_chunk, metadatas_root=metadata_vs_base)
                nb_chunk = i + 1 
                metadata_chunk['nb_chunk'] = nb_chunk

                try:
                    #logging.info(f"Chunk nb°{nb_chunk} llm inference - step 1 (Ecological filter).. ")

                    """messages = [SystemMessage(content = "You are a document evaluation assistant."),
                    HumanMessage(content = f"Is the following document is even remotely related to the theme of ecology, sufficiency, sustainability, environment ... ?"
                        f"- Your answer must be structured with a true affirmative response for 'Yes' or false affirmative response for 'No', with a precise justification."
                        "Here's the document :")]

                    temperature = 0 

                    response = self.custom_prompt_llm_inference.run(text = chunk,
                                                            messages = messages,
                                                            temperature = temperature,
                                                            pydantic_schema = ResponseWithJustification)
                    
                    import pdb
                    pdb.set_trace()
                    
        
                    if response.get('affirmative_response'):

                        metadata_chunk['is_about_ecology'] = True
                        metadata_chunk['is_about_ecology_justification'] = response.get('justification', 'No justification provided')
                        """
                    logging.info(f"Chunk nb°{nb_chunk} llm inference - step 2 - (optional) french traduction... ")

                    if language.lower() != 'french':

                        messages = [
                        SystemMessage(content = "Tu es un expert en traduction française."),
                        HumanMessage(content = "Dans le cas où le document suivant ne soit pas déjà en français, peux-tu me traduire ce document en Français s'il te plaît, le plus fidèlement possible ?" \
                        "- Ne fournis que la traduction, rien d'autres, pas de commentaire supplémentaire avant ou après." \
                        f"\n Voici le document : \n {chunk}")]

                        temperature = 0
                        trad = self.custom_prompt_llm_inference.run(messages = messages,
                                                                    temperature = temperature,
                                                                    language='French')
                        chunk.text = trad

                        logging.info("French traduction done. ")
                    
                    else:
                        logging.info("- No traductino required (doc already in French)...")


                    missions_list, mission_justifications_answers, extraction_error = self.inference_on_one_chunk(chunk = chunk,
                                    nb_chunk= nb_chunk)
                    
                    
                    metadata_chunk['missions_list'] = missions_list
                    metadata_chunk['missions_justifications'] = mission_justifications_answers
                    metadata_chunk['extraction_error'] = extraction_error
                    
                    """else:
                        logging.info(f"No ecology int this chunk ! => skip to the next chunk !...")
                        metadata_chunk['is_about_ecology'] = False"""
            
                    all_metadatas.append(metadata_chunk)

                except Exception as e:
                    logging.warning(f"Extraction error : chunk n°{nb_chunk} - error {e}")
                    raise ExtractionError(f"Extraction error (functional) : chunk n°{nb_chunk} - error {e}") from e
                
        except Exception as e:
            logging.warning(f"Complete extraction error for this doc : {e}")
            raise ExtractionError(f"Complete extraction error for this doc : - error {e}") from e

            
        logging.info("All chunks metadatas inference for this doc : DONE.")

        return all_metadatas
  


    def handle_docs(self, docs, file_id, file_name) -> int:
        s_time = time.time()
        text_docs = []
        non_text_docs = []
        thumbnail_docs = []
        other_vs_metadatas = []

        for doc in docs:
            doc_type = doc.metadata.get("type", "text")
            if doc_type == "text":
                text_docs.append(doc)
            elif doc_type == "thumbnail":
                thumbnail_docs.append(doc)
            else:
                non_text_docs.append(doc)

        
        page_label_to_thumbnail = {
            doc.metadata["page_label"]: doc.doc_id for doc in thumbnail_docs
        }

        if self.splitter:
            all_chunks = self.splitter(text_docs)
        else:
            all_chunks = text_docs

        # add the thumbnails doc_id to the chunks
        for chunk in all_chunks:
            page_label = chunk.metadata.get("page_label", None)
            if page_label and page_label in page_label_to_thumbnail:
                chunk.metadata["thumbnail_doc_id"] = page_label_to_thumbnail[page_label]

        # ------------ CUSTOM LOGIC ---------------------
        # *** Inference metadatas for all chunks ***

        entire_text = "/n".join([doc.text for doc in text_docs])

        summary, metadatas_entire_doc = self.inference_and_summarize_entire_doc(entire_text=entire_text,
                                                                                  metadata_vs_base=METADATA_BASE
                                                                                  )
        # Clone metadata first doc => summary doc
        summary.metadata = text_docs[0].metadata

        text_vs_metadatas = self.inference_on_all_chunks(chunks=all_chunks,
                                                                metadata_vs_base=METADATA_BASE,
                                                                metadata_entire_doc = metadatas_entire_doc)
        
        if len(text_vs_metadatas) != len(all_chunks):
            raise ExtractionError("Error : len(text_vs_metadatas) != len(all_chunks)")

        # ------------ END CUSTOM LOGIC ---------------------

        other_vs_metadatas = [METADATA_BASE for _ in range(len(thumbnail_docs) + len(non_text_docs))]

        # All results to ingestion :

        to_index_chunks = all_chunks + non_text_docs + thumbnail_docs + [summary]
        to_index_metadatas = text_vs_metadatas + other_vs_metadatas + [metadatas_entire_doc]

        # enrich metadata for doctore ingestion (inside the 'core' Document format)
        for chunk, metadatas in zip(to_index_chunks, to_index_metadatas):
            chunk.metadata.update(metadatas)
            
        logging.info(f"Got {len(all_chunks)} text chunks - {len(thumbnail_docs)} page thumbnails - {len(non_text_docs)} other type chunks - 1 summary")
        logging.info(f"And {len(to_index_metadatas)} metadatas list to index.")

        # /// DOC STORE Ingestion 
        chunks = []
        n_chunks = 0
        chunk_size = self.chunk_batch_size * 4
        for start_idx in range(0, len(to_index_chunks), chunk_size):
            chunks = to_index_chunks[start_idx : start_idx + chunk_size]
            self.handle_chunks_docstore(chunks, file_id)
            n_chunks += len(chunks)

        # /// VECTOR STORE Ingestion 
        def insert_chunks_to_vectorstore():
            chunks = []
            n_chunks = 0
            chunk_size = self.chunk_batch_size
            for start_idx in range(0, len(to_index_chunks), chunk_size):
                chunks = to_index_chunks[start_idx : start_idx + chunk_size]
                metadatas = to_index_metadatas[start_idx : start_idx + chunk_size]
                self.handle_chunks_vectorstore(chunks, file_id, metadatas)
                n_chunks += len(chunks)
                
        insert_chunks_to_vectorstore()
        """# run vector indexing in thread if specified
        if self.run_embedding_in_thread:
            print("Running embedding in thread")
            threading.Thread(
                target=lambda: list(insert_chunks_to_vectorstore())
            ).start()
        else:
            yield from insert_chunks_to_vectorstore()"""

        print("indexing step took", time.time() - s_time)
        return n_chunks
    


    
    def run_one_file(
        self, file_path: str | Path, reindex: bool, **kwargs
    ) -> None:
        
        # check if the file is already indexed
        if isinstance(file_path, Path):
            file_path = file_path.resolve()

        file_id = self.get_id_if_exists(file_path)

        if isinstance(file_path, Path):
            if file_id is not None:
                if not reindex:
                    raise ValueError(
                        f"File {file_path.name} already indexed. Please rerun with "
                        "reindex=True to force reindexing."
                    )
                else:
                    # remove the existing records
                    self.delete_file(file_id)
                    file_id = self.store_file(file_path)
                    
            else:
                # add record to db
                file_id = self.store_file(file_path)

        else:
            if file_id is not None:
                raise ValueError(f"URL {file_path} already indexed.")
            else:
                # add record to db
                file_id = self.store_url(file_path)

        logging.info(f"**** Document extraction : {str(file_path)}. ****")

        # extract the file
        if isinstance(file_path, Path):
            extra_info = default_file_metadata_func(str(file_path))
            file_name = file_path.name
        else:
            extra_info = {"file_name": file_path}
            file_name = file_path

        extra_info["file_id"] = file_id
        extra_info["collection_name"] = self.collection_name

        docs = self.loader.load_data(file_path, extra_info=extra_info)
        logging.info("document extracted... ok.")

        nb_chunks = self.handle_docs(docs, file_id, file_name)

        logging.info(f" Ingestion OK ! --- Nb chunks send to docstore & vector store : {nb_chunks}")

    

    def run_all_files(self, reindex: bool = False, just_one_file: str | None = None):

        target_folder = pathlib.Path(self.pdf_path)

        for file in target_folder.iterdir():
            file_str = file.as_posix().split('/')[-1]

            try:

                if not file.is_dir():

                    if just_one_file is not None and file_str==just_one_file:
                        self.run_one_file(file_path=file, reindex=reindex)

                    elif just_one_file is None and file_str.endswith(".pdf"):
                        self.run_one_file(file_path=file, reindex=reindex)

            except Exception as e:
                logging.warning(f"Error with this file : {file_str} - error : {e}")
                pass


if __name__ == "__main__":


    parser.add_argument('-fr','--force_reindex', action="store_true", help='Force to reindex all the pdf files in the folder')           
    parser.add_argument('-jof', '--just_one_file', type=str, help='Retry ingestion all the pdf files with error status')      

    args = parser.parse_args()

    indexing_pipeline = IndexingPipelineShortCut(pdf_path=PDF_FOLDER)

    indexing_pipeline.get_resources_set_up()

    indexing_pipeline.run_all_files(reindex=args.force_reindex, just_one_file=args.just_one_file)
