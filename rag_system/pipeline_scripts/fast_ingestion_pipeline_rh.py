from typing import List, Tuple
import logging
import pathlib

from kotaemon.base import Document, Param, lazy
from kotaemon.base.component import BaseComponent
from kotaemon.base.schema import LLMInterface
from kotaemon.embeddings import OpenAIEmbeddings
from kotaemon.indices import VectorIndexing
from kotaemon.indices.vectorindex import VectorRetrieval
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

PDF_FOLDER = "./data_pdf/academic_lit"

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

    metadatas_llm_inference_block_chunk : OpenAIMetadatasLLMInference = Param(
        lazy(OpenAIMetadatasLLMInference).withx(
            llm = ChatOpenAI(
                base_url=f"http://{ollama_host}:11434/v1/",
                model="gemma2:2b",
                api_key="ollama",
                ),
            taxonomy = ChunkOfDocument,
            language = 'French'
            )
    )

    # --- Final Kotaemon ingestion ----

    vector_store: QdrantVectorStore = Param(
        lazy(QdrantVectorStore).withx(
            url=f"http://{qdrant_host}:6333",
            api_key="None",
            collection_name="ecoskills_test",
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

    pdf_path: str

    def enrich_metadatas_layer(self, metadatas_base : dict = {},
                                    doc_type : str = 'unknow',
                                    inheritance_metadatas : dict | None = None,
                                    reapply_fields_to_root : list | None = None):
        """TODO Convert this function into method with a MetadatasManagement Object"""

        metadatas_base['doc_type'] = doc_type

        if inheritance_metadatas is not None:
            metadatas_base['extract_from'] = inheritance_metadatas
            if reapply_fields_to_root is not None:
                for field in reapply_fields_to_root:

                    if field not in inheritance_metadatas.keys():
                        raise RuntimeError(f"Sorry, but the field {field} is not present in inheritance metadatas for reapplying :  {inheritance_metadatas.keys()}")
                    
                    metadatas_base[field] = inheritance_metadatas[field]
        
        return metadatas_base
    
    def extend_text_and_metadatas_from_special_field(self, all_text : list,
                                                        all_metadatas : list,
                                                        metadatas_base: dict,
                                                        special_metadatas_def : List[Tuple[str, str]],
                                                        reapply_fields_to_root : list | None = None):
        """TODO Convert this function into method with a MetadatasManagement Object"""
        count = 0
        for special_meta_def in special_metadatas_def:
            metadata_key = special_meta_def[0]
            doc_type = special_meta_def[1]
            if metadata_key in metadatas_base.keys():
                content_list = metadatas_base[metadata_key]
                if content_list is not None:
                    if type(content_list)==str:
                        content_list = [content_list]
                    if type(content_list)==list:
                        for text in content_list:
                            all_text.append(text)
                            all_metadatas.append(self.enrich_metadatas_layer(metadatas_base={},
                                                                            doc_type=doc_type,
                                                                            inheritance_metadatas=metadatas_base,
                                                                            reapply_fields_to_root=reapply_fields_to_root))
                            count += 1
                    else:
                        raise RuntimeError(f'Sorry, but this special metadata key {metadata_key} must be convert to a real doc type {doc_type}... \
                                    BUT this field content insdie the doc must be a list or str, not {type(content_list)}')
                else:
                    logging.warning(f'For the doc, the field {metadata_key} of the metadatas is None... No special building from this content.')
            else:
                logging.warning(f"Sorry, but the field {metadata_key} is not present insdie the infered fields from the original doc. ")

        return all_text, all_metadatas, count

    
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
                                               chunk_size = 2000,
                                               chunk_overlap = 200)

        entire_doc = "\n".join(chunks)

        logging.info(f"document extracted... ok.")


        # 1) Inference on entire doc

        entire_doc_metadatas = self.metadatas_llm_inference_block_entire_doc.run(entire_doc,  
                                                        doc_type  = 'entire_doc', 
                                                        inference_type = 'generic')
        
        logging.info(f"LLM metadatas inference (on entire doc)... ok.")
        
        metadatas_ed = entire_doc_metadatas.model_dump()

        metadatas_ed['doc_type'] = 'entire_doc'

        all_text.append(entire_doc)
        all_metadatas.append(metadatas_ed)

        """all_text, all_metadatas, count = self.extend_text_and_metadatas_from_special_field(all_text = all_text,
                                                                                    all_metadatas = all_metadatas,
                                                                                    metadatas_base= metadatas_ed,
                                                                                    special_metadatas_def=[
                                                                                                            ('concrete_experience_in_the_field',
                                                                                                             'concrete_experiment')
                                                                                                            ],
                                                                                    reapply_fields_to_root=['functional_area'])
        
        logging.info(f"Added {count} new special doc extract from the metadas itself..") """

        logging.info(f"Chunks llm inference for this doc :")

        for i, chunk in enumerate(chunks):

            chunk_metadatas = self.metadatas_llm_inference_block_chunk.run(chunk,
                                                                           doc_type = 'chunk',
                                                                           inference_type = 'generic')
            
            logging.info(f"LLM metadatas inference (on chunk {i})... ok.")

            metadatas_c = chunk_metadatas.model_dump()

            metadatas_c = self.enrich_metadatas_layer(metadatas_base=metadatas_c,
                                                      doc_type='chunk',
                                                      inheritance_metadatas=metadatas_ed,
                                                      reapply_fields_to_root=['functional_area'])
            # additional special metadata for chunk
            metadatas_c['nb_chunk'] = i + 1

            all_text.append(chunk)
            all_metadatas.append(metadatas_c)

            all_text, all_metadatas, count = self.extend_text_and_metadatas_from_special_field(all_text = all_text,
                                                                                    all_metadatas = all_metadatas,
                                                                                    metadatas_base= metadatas_c,
                                                                                    special_metadatas_def=[
                                                                                                            ('key_idea_sentences',
                                                                                                             'key_idea'),
                                                                                                             ('concrete_experience_in_the_field',
                                                                                                             'concrete_experiment')
                                                                                                            ],
                                                                                    reapply_fields_to_root=['functional_area'])
            
            logging.info(f"Added {count} new special doc extract from the metadas itself..")

        logging.info(f"All chunks metadatas inference & special extending DONE.")
        logging.info(f"Export to vectore store througth Kotaemon....")

        super().run(text=all_text, metadatas=all_metadatas)

        logging.info(f"Export OK")

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
