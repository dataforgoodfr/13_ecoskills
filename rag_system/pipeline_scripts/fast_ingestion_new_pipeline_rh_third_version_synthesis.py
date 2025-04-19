import logging
import pathlib
import argparse
from datetime import datetime
import time
import uuid

from pathlib import Path

from kotaemon.base import Param, lazy, Document
from kotaemon.embeddings import OpenAIEmbeddings
from kotaemon.indices import VectorIndexing
from kotaemon.llms.chats.openai import ChatOpenAI
from kotaemon.storages import LanceDBDocumentStore
from kotaemon.storages.vectorstores.qdrant import QdrantVectorStore

from llama_index.core.readers.file.base import default_file_metadata_func
from llama_index.core.readers.base import BaseReader

from pipelineblocks.extraction.pdfextractionblock.pdf_to_markdown import PdfExtractionToMarkdownBlock
from pipelineblocks.llm.ingestionblock.openai import OpenAIMetadatasLLMInference, OpenAICustomPromptLLMInference

from taxonomy.document import EntireDocument
from ingestion_manager.ingestion_manager import IngestionManager

from kotaemon.base.schema import HumanMessage, SystemMessage

from ktem.db.models import engine
from sqlalchemy.orm import Session
from ktem.index.file.index import FileIndex
from ktem.index.file.pipelines import IndexPipeline

from kotaemon.loaders import PDFThumbnailReader

from kotaemon.indices.splitters import TokenSplitter

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
COLLECTION_ID = 7
USER_ID = '2bd87cee60a5430ca23c84ee80d81cfa'

PDF_FOLDER = "./data_pdf/test"

FUNCTIONAL_DOMAIN = "Ressources Humaines (RH)"

METADATA_BASE = {
    "ingestion_method" : 'fast_script',
    "ingestion_origin_folder": PDF_FOLDER,
    "functional_domain" : FUNCTIONAL_DOMAIN
}

CHUNK_SIZE = 600
CHUNK_OVERLAP = 150

MISSION_DICT_FR = {
    "Définir, piloter et mettre en œuvre la stratégie RH": [
        "Élaborer, conduire et évaluer la politique RH, en cohérence avec les orientations nationales et les priorités de la structure.",
        "Définir les plans d'action stratégiques en matière de recrutement, GPEEC, mobilité, formation, qualité de vie au travail, communication, conditions de travail, diversité et politique managériale.",
        "Conduire le changement et accompagner les évolutions organisationnelles, culturelles et technologiques de l'administration."
    ],
    "Gérer les parcours, les emplois et les compétences": [
        "Mettre en œuvre la gestion prévisionnelle des effectifs, des emplois et des compétences (GPEEC).",
        "Analyser les besoins actuels et futurs, cartographier les compétences, identifier les viviers et les métiers sensibles.",
        "Conseiller les agents dans la construction de leur parcours professionnel, accompagner les mobilités, les reconversions, les transitions professionnelles, et les secondes parties de carrière.",
        "Participer aux revues de cadres, à la constitution de viviers de talents, et à l'animation des réseaux professionnels."
    ],
    "Recruter, intégrer et fidéliser les agents": [
        "Organiser les campagnes de recrutement externe et interne, concevoir les dispositifs d'intégration et de suivi (onboarding).",
        "Accompagner les managers dans la définition des besoins en recrutement.",
        "Promouvoir la marque employeur et développer les partenariats pour le sourcing (forums, réseaux, apprentissage).",
        "Mettre en œuvre les concours et examens professionnels, en garantissant la sécurité juridique et l'équité des processus."
    ],
    "Accompagner les personnes et les collectifs de travail": [
        "Conseiller les agents et les services en matière RH (carrière, mobilité, retraite, formation, management).",
        "Réaliser des entretiens individualisés, animer des actions collectives de développement professionnel.",
        "Proposer des dispositifs d'accompagnement : coaching, conseil RH de proximité, soutien à la professionnalisation des pratiques managériales.",
        "Assurer l'accueil, l'écoute, la médiation et l'orientation vers les ressources adaptées."
    ],
    "Assurer la gestion administrative, statutaire et la paie": [
        "Gérer les actes administratifs : positions, contrats, retraites, promotions, cessations de fonction.",
        "Assurer le traitement, la vérification et la fiabilisation des rémunérations, primes et cotisations sociales.",
        "Suivre les absences, le temps de travail, les congés, le télétravail.",
        "Gérer les procédures de contentieux et les dossiers individuels sensibles dans le respect du cadre réglementaire."
    ],
    "Piloter le dialogue social et la qualité de vie au travail": [
        "Organiser les élections professionnelles, les instances de dialogue social et les négociations collectives.",
        "Préparer les ordres du jour, assurer le secrétariat et le suivi des décisions des instances représentatives.",
        "Prévenir et gérer les conflits collectifs et individuels, produire des notes d'analyse du climat social.",
        "Élaborer et mettre en œuvre les politiques de prévention des risques professionnels et psychosociaux, en lien avec les services compétents (DUERP, plans d’action, accompagnement des acteurs)."
    ],
    "Développer les outils, les données et les systèmes d'appui RH": [
        "Concevoir et piloter les systèmes d'information RH (SIRH), produire les données sociales et les indicateurs (RSU, DOETH, bilans, tableaux de bord).",
        "Appuyer la décision stratégique via l'analyse des données RH (emploi, masse salariale, diversité, égalité professionnelle).",
        "Contribuer à la transformation numérique des fonctions RH et au développement d'une gestion fondée sur les données (data RH).",
        "Participer à la veille réglementaire, technologique et sociétale sur les évolutions du métier RH."
    ]
}


# ---- Do not touch (temporary) ------------- #

ollama_host = '172.17.0.1' if OLLAMA_DEPLOYMENT == 'docker' else 'localhost'
qdrant_host = '172.17.0.1' if VECTOR_STORE_DEPLOYMENT == 'docker' else 'localhost'


class RelevanceScore(BaseModel):
    relevance_score : float


class ExtractionError(Exception):
    pass

class IndexingPipelineShortCut(IndexPipeline):

    # --- PDF Extraction (optionnal... to replace Kotaemon Loader by default) --- 

    pdf_extraction_block : PdfExtractionToMarkdownBlock = Param(
        lazy(PdfExtractionToMarkdownBlock).withx(
        )
    )

    # --- LLM MODELS ---
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

        messages = [SystemMessage(content = f"Tu es un lecteur expert du domaine professionnel : {FUNCTIONAL_DOMAIN}."
                "Sur ce document, tu dois faire une lecture très attentive et méticuleuse."
                "- Réponds seulement par 'oui' ou 'non' "),
                HumanMessage(content = f"Est-ce que le document suivant peut m'aider d'une manière ou d'une autre à conceptualiser ou " 
                "à réaliser la mission spécifique suivante dans le concret, dans l'optique de développement durable ou à l'écologie au sein de cette mission spécifique. "
                f"- Réponds seulement par 'oui' ou 'non' "
                "- si tu réponds 'oui' cela veut dire que le document est très en lien, et de manière très pertinente, avec la mission spécifique suivante."
                "\n Voici la mission spécifique :"
                f"{mission}"
                "et la description complète de cette mission point par point :"
                f"{' - '.join(MISSION_DICT_FR[mission])}"
                "\n Et voici le document :")]
            
        temperature = 0    

        response = self.custom_prompt_llm_inference.run(text = chunk,
                                                            messages = messages,
                                                            temperature = temperature)
        
        if response.lower().startswith('oui') :
            return True
        else:
            return False

    
    def inference_on_one_chunk(self, chunk : str, nb_chunk: int, metadatas_entire_doc : dict | None = None):

        missions_list = []
            
        for mission in MISSION_DICT_FR.keys() :

            logging.info(f"Chunk nb°{nb_chunk} llm inference - mission {mission} ... ")

            try:
                mission_ok = self.run_one_mission_inference(chunk=chunk,
                                               mission=mission)
                
                if mission_ok:
                    missions_list.append(mission)
                
            except Exception as e:
                logging.info(f"Extraction error : chunk n°{nb_chunk} - mission : {mission} - error {e}")
                pass
        
        metadatas_chunk = self.enrich_metadatas_layer(doc_type='chunk',
                                                      inheritance_metadatas=metadatas_entire_doc,
                                                      inheritance_fields_to_exclude=['professional_functional_area', 'doc_type'],
                                                      reapply_fields_to_root=['professional_functional_area'])
        metadatas_chunk['nb_chunk'] = nb_chunk
        metadatas_chunk['missions_list'] = missions_list


        return metadatas_chunk
    

    def inference_and_summarize_entire_doc(self, entire_text: str, metadata_vs_base: dict | None = None):

        entire_doc_metadatas = self.metadatas_llm_inference_block_entire_doc.run(entire_text,  
                                                            doc_type  = 'entire_doc', 
                                                            inference_type = 'generic')
        
        logging.info("LLM metadatas inference (on entire doc)... ok.")

        metadatas_ed = entire_doc_metadatas.model_dump(mode='json')

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
                nb_chunk = i + 1
                try:
                    logging.info(f"Chunk nb°{nb_chunk} llm inference - step 1 (Ecological filter).. ")

                    messages = [SystemMessage(content = "You are a scientific expert of questions related to ecology, sustainability and sufficiency. "
                    "On the document, you have to make a lecture very attentive and critical."
                    "- Just respond with 'yes' or 'no'"),
                    HumanMessage(content = f"Is the following passagge have an evident link with sustainability, sufficiency, or an ecological point of view ?"
                        f"- Just say yes or no."
                        "Here's the document :")]

                    temperature = 0    

                    response = self.custom_prompt_llm_inference.run(text = chunk,
                                                            messages = messages,
                                                            temperature = temperature)
        
                    if response.lower().startswith('yes'):

                        if language!='french':

                            logging.info(f"Chunk nb°{nb_chunk} llm inference - step 2 - (optional) french traduction... ")

                            messages = [
                            HumanMessage(content = f"Tu es un expert en traduction française."
                            "Peux-tu me traduire ce document en Français s'il te plaît, le plus fidèlement possible ?" \
                            "- Ne fournis que la traduction, rien d'autres, pas de commentaire supplémentaire avant ou après.")]

                            temperature = 0
                            chunk = self.custom_prompt_llm_inference.run(text = chunk,
                                                                            messages = messages,
                                                                            temperature = temperature)
                        else:

                            logging.info(f"Chunk nb°{nb_chunk} - chunk already in French language (no traduction) ")


                        metadata_chunk = self.inference_on_one_chunk(chunk = chunk,
                                        nb_chunk= nb_chunk,
                                        metadatas_entire_doc=metadata_entire_doc)
                        
                        metadata_chunk = self.concat__metadatas_layer(metadatas_base=metadata_chunk, metadatas_root=metadata_vs_base)

                        all_metadatas.append(metadata_chunk)   
                        
                    else:
                        logging.info(f"No ecology int this chunk ! => skip !...")

                except Exception as e:
                    logging.warning(f"Extraction error : chunk n°{nb_chunk} - error {e}")
                    raise ExtractionError(f"Extraction error (functional) : chunk n°{nb_chunk} - error {e}")
                
        except Exception as e:
            logging.warning(f"Complete extraction error for this doc : {e}")
            raise ExtractionError(f"Complete extraction error for this doc : - error {e}")

            
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

        # ------------ END CUSTOM LOGIC ---------------------

        other_vs_metadatas = [METADATA_BASE for _ in range(len(thumbnail_docs) + len(non_text_docs))]

        # All results to ingestion :

        to_index_chunks = all_chunks + non_text_docs + thumbnail_docs + [summary]
        to_index_metadatas = text_vs_metadatas + other_vs_metadatas + [metadatas_entire_doc]

        logging.info(f"Got {len(thumbnail_docs)} text chunks - {len(thumbnail_docs)} page thumbnails - {len(non_text_docs)} other type chunks - 1 summary")
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

    

    def run_all_files(self, reindex: bool = False):

        target_folder = pathlib.Path(indexing_pipeline.pdf_path)

        for file in target_folder.iterdir():
            file_str = file.as_posix().split('/')[-1]

            try:

                if not file.is_dir():
                    if file_str.endswith(".pdf"):

                        self.run_one_file(file_path=file, reindex=reindex)

            except Exception as e:
                logging.warning(f"Error with this file : {file_str} - error : {e}")
                pass


if __name__ == "__main__":


    parser.add_argument('-fr','--force_reindex', action="store_true", help='Force to reindex all the pdf files in the folder')           
    parser.add_argument('-re', '--retry_error', action="store_true", help='Retry ingestion all the pdf files with error status')      

    args = parser.parse_args()

    indexing_pipeline = IndexingPipelineShortCut(pdf_path=PDF_FOLDER)

    indexing_pipeline.get_resources_set_up()

    indexing_pipeline.run_all_files(reindex=args.force_reindex)
