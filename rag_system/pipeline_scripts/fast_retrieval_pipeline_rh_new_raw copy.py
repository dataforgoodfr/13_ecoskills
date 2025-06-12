import logging
import numpy as np
import time

from pydantic import BaseModel

from enum import Enum

from dotenv import dotenv_values

from kotaemon.base import Param, lazy
from kotaemon.embeddings import OpenAIEmbeddings
from kotaemon.indices.vectorindex import VectorRetrieval
from kotaemon.llms.chats.langchain_based import LCChatMistral
from pipelineblocks.llm.ingestionblock.langchain import LangChainCustomPromptLLMInference
from kotaemon.storages import LanceDBDocumentStore
from qdrant_client import QdrantClient, models
from kotaemon.base.schema import HumanMessage, SystemMessage


from prompts_rh.missions import STRUCTURED_MISSION_DICT

def trunc_decimal(value, decs=0):
    return np.trunc(value*10**decs)/(10**decs)


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

EXPORT_FILENAME = "raw_4_test"

OLLAMA_DEPLOYMENT = 'docker'
VECTOR_STORE_DEPLOYMENT = 'docker'

DOCSTORE_PATH = "/app/ktem_app_data/user_data/docstore"

DOC_RESULT_PATH = "/app/pipeline_scripts/results/"

COLLECTION_NAME= 'index_8' # Check here your collection throught QDRANT Database dashboard & collection in docstore (ktem_app_data/user_data/doc_store)
#RETRIEVAL_MODE= 'hybrid'

EXTRA_MAX_RETRIES = 6
DELAY_BETWEEN_REQUEST = 1 #seconds

#PDF_FOLDER = "./data_pdf/other_lit"

# ---- Do not touch (temporary) ------------- #

MISTRAL_API_KEY = dotenv_values()['MISTRAL_API_KEY']

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

    custom_prompt_llm_inference : LangChainCustomPromptLLMInference = Param(
        lazy(LangChainCustomPromptLLMInference).withx(
            llm = LCChatMistral(
                model="open-mistral-nemo",
                mistral_api_key=MISTRAL_API_KEY,
                temperature=0.3
                ))
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
        all_ids = [res.payload.get("doc_id") for res in all_results]
        all_text = [self.doc_store.get(id)[0].text for id in all_ids]

        #all_results = super().run(query, top_k = 10)
        return all_ids, all_scores, all_text
    
    def run(self) -> None:
        
        #TODO loop on each missions

        seen_text_list = []

        for mission in STRUCTURED_MISSION_DICT:
            
            logging.info("----------")
            logging.info(f"Mission : {mission}")
            logging.info("----------")

            with open(DOC_RESULT_PATH + f'{EXPORT_FILENAME}.doc', 'a') as f:
                    f.write(f"\n \n \n --------- \n MISSION : {mission}: \n ---------- \n ")
            with open(DOC_RESULT_PATH + f'{EXPORT_FILENAME}_synthese.doc', 'a') as f:
                    f.write(f"\n \n \n --------- \n MISSION : {mission}: \n ---------- \n ")

            query = mission + ", .".join(STRUCTURED_MISSION_DICT[mission])
        
            all_ids, all_scores, all_texts = self.run_one_generic_request(metadatas_filters={"doc_type": "chunk",
                                                                    "missions_list[]" : mission
                                                                    },
                                                            query=query,
                                                            nb_results = 16)
            
            print(f"All scores : {all_scores}")


            final_text_list = [text for text, id in zip(all_texts, all_ids, strict=False) if id not in seen_text_list]
            seen_text_list.extend(all_ids)
            
            logging.info(f"Corpus nb texts : {len(final_text_list)}.")

            corpus = "\n".join([f"\n --- Doc n° {i} : \n" + doc for i, doc in enumerate(final_text_list)])

            #mission_descr = "\n - ".join(STRUCTURED_MISSION_DICT[mission])

            messages = [
                SystemMessage(content = "Tu es un assistant spécialisé dans la synthèse de corpus de documents"),
                HumanMessage(content = f"""A partir du corpus de documents suivant qui sont numérotés et séparés par '-- Doc n°... : ',
                                        compose une synthèse autour de la mission : {mission}.
                                        - Le titre de cette synthèse peut avoir le titre de la mission.
                                        - Organise cette synthèse sous la forme de 7 à 10 paragraphes avec sous-titres, de 1500  à 2000 tokens au total. Chaque paragraphe représente donc 150 à 250 tokens.
                                        \n
                                            - Utilise tout ou partie des références, idées et examples précis inclus dans le corpus de documents.
                                            - Ne dis jamais 'le document dit que...' ou 'le document met en lumière...' etc. Fais comme si tu affirmais toi-même les choses.
                                            - N'ajoute aucun nom d'entreprise ou d'institutions qui ne sont pas dans les documents.
                                            \n \n Voici le corpus de document :
                                                {corpus}
                                            \n \n Attention, rappelle-toi :
                                            Crée des paragraphes de 150 à 250 tokens, avec des sous-titres.
                                            Il est interdit de citer d'autres expériences ou éléments que celles des documents.
                                            Il est interdit de citer d'autres entreprises ou institutions que celles qui sont citées dans les documents.
                                            Tu dois citer les documents avec son numéro de document indiqué en mettant cette référence entre parenthèse directement dans le corps du texte.
                                            """)]
        
            temperature = 0.1
            max_token = 2000

            llm_response = self.custom_prompt_llm_inference.run(messages = messages,
                                                            temperature = temperature,
                                                            language = 'French',
                                                            max_token = max_token)

            
            paragraphs = llm_response.split("\n")
            paragraphs = [p.strip() for p in paragraphs if p.strip() != ""]

            org_paragraphs = [(i, p) for i, p in enumerate(paragraphs)]

            titles = [tupl for tupl in org_paragraphs if len(tupl[1].split(" ")) <= 20 ]
            real_paragraph = [tupl for tupl in org_paragraphs if len(tupl[1].split(" ")) > 20]

            time.sleep(DELAY_BETWEEN_REQUEST) # to not excedding request time

            all_parts = []

            for tupl in real_paragraph:

                index = tupl[0]

                p = tupl[1]
                 
                messages = [
                SystemMessage(content = "Tu es un assistant spécialisé dans la création d'exemple concret à valeur pédagogique pour les professionnels de la fonction publique."),
                HumanMessage(content = f"""A partir du paragraphe suivant,
                                        propose un example concret de dispositif ou d'expérience terrain issu du contexte de la fonction publique (collectivité , municipalité, organisme public, institution...), faisant entre 100 et 400 tokens.
                                        Tu as deux choix pour créer cette example concret :
                                        - soit l'example concret est pris dans le corpus de document suivant. Il s'agit alors d'un dispositif ou d'une expérimentation terrain explicité directement dans le corpus de document.
                                        - soit l'example est inventé. Dans ce cas, commence par dire "Exemple inventé -> " suivi de l'example concret.
                                        Dans tous les cas, cet example concret doit être détaillé et très réaliste. Il doit mettre en scène des protagonistes précis qui mettent en place le dispositif 
                                        ou l'expérimentation. Ensuite tu dois expliquer comment ces protagonistes s'y prennent, quels sont les enjeux, les obstacles rencontrés avec précision,
                                        les résultats obtenus, etc.
                                        \n
                                            - Ne dis jamais 'le document dit que...' ou 'le document met en lumière...' etc. Fais comme si tu affirmais toi-même les choses.
                                            \n \n Voici le corpus de document où récupérer éventuellement les examples :
                                                {corpus}
                                            \n \n Et voici le paragraphe :
                                                {p}
                                            \n \n Attention, rappelle-toi :
                                            Si les examples concrets sont pris dans le corpus de document, alors tu dois citer le document avec son numéro de document indiqué en mettant cette référence entre parenthèse directement dans le corps du texte.
                                            Si tu inventes un example concret, alors commence par dire "Exemple inventé :" suivi de l'example concret.
                                            """)]

                temperature = 0.8
                max_token = 800

                llm_response_paragraph = self.custom_prompt_llm_inference.run(messages = messages,
                                                            temperature = temperature,
                                                            language = 'French',
                                                            max_token = max_token)
            

                part = p + " \n\n " + llm_response_paragraph + "\n\n"
            
                all_parts.append((index, part))

                time.sleep(DELAY_BETWEEN_REQUEST) # to not excedding request time

            all_parts.extend(titles)

            all_parts = sorted(all_parts, key=lambda x: x[0])

            all_parts = [part[1] for part in all_parts]

            final_text = "\n\n".join(all_parts)
                
            logging.info(f"\n\n --- LLM RESPONSE --- \n\n {final_text}")

            with open(DOC_RESULT_PATH + f'{EXPORT_FILENAME}.doc', 'a') as f:
                f.write(f"\n ***RAW CHUNKS*** : \n {corpus}  \n\n\n **** LLM INIT **** --- \n\n {llm_response} \n \n **** LLM FINAL TEST **** --- \n\n {final_text} \n \n")

            with open(DOC_RESULT_PATH + f'{EXPORT_FILENAME}_synthese.doc', 'a') as f:
                f.write(f"\n \n \n ### LLM FINAL TEXT ### : \n {final_text} \n \n")

        return None




if __name__ == "__main__":

    retrieval_pipeline = RetrievalPipeline()

    retrieval_pipeline.run()