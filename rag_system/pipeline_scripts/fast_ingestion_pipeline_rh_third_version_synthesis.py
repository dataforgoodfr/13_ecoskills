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

PDF_FOLDER = "./data_pdf/test"

FUNCTIONAL_DOMAIN = "Ressources Humaines (RH)"

CHUNK_SIZE = 400
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


MISSION_DICT_EN = {
    "Define, steer, and implement the HR strategy": [
        "Develop, lead, and evaluate the HR policy in line with national guidelines and the organization's priorities.",
        "Define strategic action plans for recruitment, workforce planning (GPEEC), mobility, training, quality of work life, communication, working conditions, diversity, and management policy.",
        "Lead change and support organizational, cultural, and technological transformations within the administration."
    ],
    "Manage career paths, jobs, and skills": [
        "Implement forward-looking management of staffing, jobs, and skills (GPEEC).",
        "Analyze current and future needs, map out skills, identify talent pools and critical roles.",
        "Advise employees on building their career paths, support mobility, retraining, professional transitions, and second careers.",
        "Participate in leadership reviews, build talent pools, and facilitate professional networks."
    ],
    "Recruit, onboard, and retain employees": [
        "Organize internal and external recruitment campaigns, and design onboarding and follow-up programs.",
        "Support managers in defining recruitment needs.",
        "Promote employer branding and develop sourcing partnerships (forums, networks, apprenticeships).",
        "Implement competitive exams and professional assessments while ensuring legal security and fairness in the processes."
    ],
    "Support individuals and work groups": [
        "Advise employees and departments on HR matters (career, mobility, retirement, training, management).",
        "Conduct individual interviews and facilitate collective professional development activities.",
        "Propose support mechanisms: coaching, local HR advisory, and support for the professionalization of managerial practices.",
        "Ensure reception, listening, mediation, and guidance to appropriate resources."
    ],
    "Ensure administrative, statutory management and payroll": [
        "Manage administrative acts: status, contracts, retirements, promotions, and terminations.",
        "Process, verify, and secure the reliability of payroll, bonuses, and social contributions.",
        "Track absences, working time, leave, and remote work.",
        "Handle litigation procedures and sensitive individual cases within the regulatory framework."
    ],
    "Steer social dialogue and quality of work life": [
        "Organize professional elections, social dialogue bodies, and collective bargaining.",
        "Prepare agendas, ensure secretariat duties, and follow up on decisions of representative bodies.",
        "Prevent and manage collective and individual conflicts, and produce analyses on social climate.",
        "Develop and implement policies to prevent occupational and psychosocial risks, in collaboration with relevant services (DUERP, action plans, stakeholder support)."
    ],
    "Develop HR tools, data, and support systems": [
        "Design and manage HR information systems (HRIS), and produce HR data and indicators (social report, disability declarations, dashboards, etc.).",
        "Support strategic decision-making through HR data analysis (employment, payroll, diversity, gender equality).",
        "Contribute to the digital transformation of HR functions and the development of data-driven management (HR data).",
        "Participate in regulatory, technological, and societal monitoring related to the evolution of HR professions."
    ]
}


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
        
        import pdb
        pdb.set_trace()
        
        if response.lower().startswith('oui') :
            return True
        else:
            return False

    
    def run_one_chunk(self, chunk : str, nb_chunk: int, metadatas_entire_doc : dict | None = None):

        missions_list = []
            
        for mission in MISSION_DICT_FR.keys() :

            logging.info(f"Chunk nb°{nb_chunk} llm inference - mission {mission} ... ")

            try:
                mission_ok = self.run_one_mission_inference(chunk=chunk,
                                               mission=mission)
                
                if mission_ok:
                    missions_list.append(mission)
                
            except Exception as e:
                self.curr_error_list.append(f"Extraction error : chunk n°{nb_chunk} - mission : {mission} - error {e}")
                pass
        
        metadatas_chunk = self.enrich_metadatas_layer(doc_type='chunk',
                                                      inheritance_metadatas=metadatas_entire_doc,
                                                      inheritance_fields_to_exclude=['professional_functional_area'],
                                                      reapply_fields_to_root=['professional_functional_area'])
        metadatas_chunk['nb_chunk'] = nb_chunk
        metadatas_chunk['missions_list'] = missions_list


        import pdb
        pdb.set_trace()

        self.all_text.append(chunk)
        self.all_metadatas.append(metadatas_chunk)


    
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

            language = metadatas_ed['language'].lower()

            import pdb
            pdb.set_trace()

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
        
                    if response.lower().startswith('yes') and nb_chunk > 4 :

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


                        self.run_one_chunk(chunk = chunk,
                                        nb_chunk= nb_chunk,
                                        metadatas_entire_doc=metadatas_ed)      
                        
                    else:
                        logging.info(f"No ecology int this chunk ! => skip !...")


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
                            time.sleep(1200) # Wait 20 minutes to reduce laptop temperature

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
