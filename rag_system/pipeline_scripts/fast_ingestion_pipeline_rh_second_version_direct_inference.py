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
COLLECTION_NAME= 'index_1' # Check here your collection throught QDRANT Database dashboard & collection in docstore (ktem_app_data/user_data/doc_store)

PDF_FOLDER = "./data_pdf/test"

FUNCTIONAL_DOMAIN = "Ressources Humaines (RH)"

MISSION_LIST = [
            "management of recruitment campaigns",
            "analyzing skills requirements and matching them to the organization's needs",
            "monitoring candidates selection processes",
            "supporting candidates and promoting the employer's attractiveness"
            ]

MISSION_LIST = [
            "la gestion des campagnes de recrutement",
            "l'analyse des besoins en compétences et leur adéquation avec les besoins de l'organisation",
            "le suivi des processus de sélection",
            "l'accompagnement des candidats et la promotion de l'attractivité de l'employeur"
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
                model="gemma3:4b",
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
                model="gemma3:4b",
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
                                    reapply_fields_to_root : list | None = None):
        """TODO Convert this function into method with a MetadatasManagement Object"""

        if metadatas_base is None:
            metadatas_base = {}
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
                    if isinstance(content_list, str):
                        content_list = [content_list]
                    if isinstance(content_list, list):
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
                                               chunk_size = 1024,
                                               chunk_overlap = 150)

        entire_doc = "\n".join(chunks)

        logging.info("document extracted... ok.")
        logging.info(f"Doc splitted into {len(chunks)} chunks.")


        # 1) Inference on entire doc
        
        # Temporary comment
        """entire_doc_metadatas = self.metadatas_llm_inference_block_entire_doc.run(entire_doc,  
                                                        doc_type  = 'entire_doc', 
                                                        inference_type = 'generic')
        
        logging.info("LLM metadatas inference (on entire doc)... ok.")
        
        metadatas_ed = entire_doc_metadatas.model_dump(mode='json')"""

        # Temporary comment
        #metadatas_ed['doc_type'] = 'summary'
        # Temporary comment
        #doc_summarize = metadatas_ed['summary']
        # Temporary comment
        #all_text.append(doc_summarize)
        #all_metadatas.append(metadatas_ed)

        logging.info("Chunks llm inference for this doc :")
        
        for i, chunk in enumerate(chunks):
            # Temporary comment
            """metadatas_chunk = self.enrich_metadatas_layer(doc_type='chunk',
                                                      inheritance_metadatas=metadatas_ed,
                                                      reapply_fields_to_root=['professional_functional_area'])"""
            # Temporary comment
            # additional special metadata for chunk
            #metadatas_chunk['nb_chunk'] = i + 1
            # Temporary comment
            #all_text.append(chunk)
            #all_metadatas.append(metadatas_chunk)

            chunk += """\n Les employées des Ressources Humaines peuvent organiser des campagnes de recrutements
            qui soient en accord avec les grands principes du développement durable. Pour cela, ils peuvent par exemple
            organiser des forums thématiques sur les sujets en accord avec les objectifs répertoriés dasn le GIEC.
            Ces forums peuvent s'organiser chez des lieux partenaires qui partagent les même objectifs et qui soient des lieux de formation, université, etc.
            De cette manière, les candidats recrutés ont plus de chance d'être déjà attentifs aux objectifs environnementaux pronés par l'entreprise.
            A titre d'exemple, nous pouvons citer l'expérience qui a été faite par l'entreprise TotalEnergies qui a noué un partenariat avec l'Université de Lyon
            et a organisé un forum JobConnect pour les métiers du numérique en partenariat avec la FrenchTech de la région Rhone-Alpes où était présent l'entreprise,
            mais aussi d'autres entreprises du bassin. Chaque entreprise faisait une conférence au sujet de la Data et des impacts environnementaux, mais aussi
            d'autres sujets liés. Ce forum a permis de faire se rencontrer des candidats et des entreprises autour d'un thème bien précis, qui favorisait les échanges autour de l'environement.
            Par ailleurs, pour ce qui est du suivi des candidats pendant le processus de recrutement, un bonne idée à mettre en place
            est de jalonner les entretiens par quelques mails qui présentent la politique de l'entreprise en matière d'environnement."""

            import pdb
            pdb.set_trace()

            for mission in MISSION_LIST :
                """Here's, for a short context, the summary of the document from which the text was taken:
                    {doc_summarize}"""
                # related to one of your daily tasks, which is {mission}, and according to an ecological objective.

                print(mission)


                # 1 - Extraction close to original -

                messages = [SystemMessage(content = "You are an extractor algoritm expert. "
                "On the document, you have to extract some relevant ideas,according to the user demand."
                "- The extraction must be very close to original document. "
                "- The extraction must be in the same language than the original document."
                "- Dont' introduce your reponse with something like that : 'The document says...' or 'The document appears to...'. "
                "- Affirm ideas as if they were your own."),
                HumanMessage(content = f"In the following document, extract only those sections that are relevant to the professional field of {FUNCTIONAL_DOMAIN} "
                f"and that have a significant bearing on the specific professional mission that is {mission} and according to a sustainable development, preserving as much original wording as possible."
                "If no section is relevant to the theme, just say 'No section found'. "
                "Here's the document :")]

                temperature = 0    

                response = self.custom_prompt_llm_inference.run(text = chunk,
                                                                    messages = messages,
                                                                    temperature = temperature)
                import pdb 
                pdb.set_trace()
                
                if len(response) > 50:
                
                    # 2 - Reformulation

                    messages = [SystemMessage(content = "You are a teacher expert, for adult audience. Expert of synthetic reformulation. "
                    "- Your responses must be in the same language than the original document."
                    "- Dont' introduce your reponse with something like that : 'The document says...' or 'The document appears to...'. "
                    "- Affirm ideas as if they were your own."),
                    HumanMessage(content = f"If the passage exceeds 3 sentences or 100 words, rewrite this passage in a way that is more concise and understandable,"
                                 "but stays as close as possible to the original wording."
                                 "If the passage doesn't exeed 3 sentences or 100 words, just rewrite this passage in a way that is more understandable,"
                                 "but stays as close as possible to the original wording. "
                    "Here's the passage :")]
                    
                    temperature = 0.3

                    response = self.custom_prompt_llm_inference.run(text = response,
                                                                    messages = messages,
                                                                    temperature = temperature)

                    import pdb
                    pdb.set_trace()

                    # 3 - Reformulation

                    messages = [SystemMessage(content = "You are a reader expert. "
                    "- Dont' introduce your reponse with something like that : 'The document says...' or 'The document appears to...'. "
                    "- Affirm ideas as if they were your own."),
                    HumanMessage(content = f"Compare the following extracted text with the original document from which it is taken."
                                 "Rate how much information is lost and suggest some improvements."
                                 "If the original document contains concrete experiences in the field,"
                                 "such as examples of corporate and institutional initiatives, include them."
                                 " Don't exceed 7 sentences."
                                 f"Then, traduce your response in {LANGUAGE}")]

                    text = f"\n Here the extracted text : {response} \n "
                    f"\n Here the original document : {chunk} \n"
                    
                    langage = LANGUAGE

                    temperature = 0.3

                    response = self.custom_prompt_llm_inference.run(text = text,
                                                                    messages = messages,
                                                                    temperature = temperature)

                    import pdb
                    pdb.set_trace()
               
                
                """output_list_core = output_list.split("[")[1].split("]")[0]
                all_text_list = [text for text in output_list_core.split(",") if len(text)>=100]"""

                metadatas_chunk = {} # temp
                all_text_list = [] # temp

                logging.info(f"LLM key idea inference (on chunk {i})... - mission {mission} - nb_response : {len(all_text_list)} ok.")
                if len(all_text_list) > 0:
                    metadatas_ki = self.enrich_metadatas_layer(
                                                            doc_type='key_idea',
                                                        inheritance_metadatas=metadatas_chunk,
                                                        reapply_fields_to_root=['professional_functional_area'])
            
                    
                    all_text.extend([text for text in all_text_list])
                    all_metadatas.extend([metadatas_ki for _ in range(len(all_text_list))])

        logging.info("All chunks metadatas inference & special extending DONE.")
        logging.info("Export to vectore store througth Kotaemon....")

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
