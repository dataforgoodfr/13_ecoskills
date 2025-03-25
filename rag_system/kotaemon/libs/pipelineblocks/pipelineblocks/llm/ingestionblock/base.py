from typing import AsyncGenerator, Iterator, Any

from kotaemon.base import BaseComponent, Document
from pydantic import BaseModel
from kotaemon.base.schema import SystemMessage
from pipelineblocks.llm.prompts.scientific_paper import scientific_basic_prompt_entire_doc
from pipelineblocks.llm.prompts.generic_document import generic_extraction_prompt_entire_doc, generic_extraction_prompt_chunk

class BaseLLMIngestionBlock(BaseComponent):

    def stream(self, *args, **kwargs) -> Iterator[Document] | None :
        raise NotImplementedError

    def astream(self, *args, **kwargs) -> AsyncGenerator[Document, None] | None:
        raise NotImplementedError

    def run(self, *args, **kwargs) -> Document | list[Document] | Iterator[Document] | None | Any:
        return NotImplementedError


class MetadatasLLMInfBlock(BaseLLMIngestionBlock):
        
    taxonomy : BaseModel
    language : str = "English"

    def _invoke_json_schema_from_taxo(self):

        return self.taxonomy.model_json_schema()

    def _convert_content_to_pydantic_schema(self, content) -> BaseModel:
            
            return self.taxonomy.model_validate_json(content)
    
    def _adjust_prompt_according_to_doc_type(self, text, doc_type = 'entire_doc', inference_type: str = "generic") -> str:

        if inference_type == 'scientific' and doc_type == 'entire_doc':
            # First combination example
            enriched_prompt = scientific_basic_prompt_entire_doc(text)

        elif inference_type == 'scientific' and doc_type == 'chunk':
            # Other combination Example
            raise NotImplementedError(f"The {inference_type} inference type is not implemented for this doc_type : {doc_type} ")

        elif inference_type == 'generic' and doc_type == 'entire_doc':
            enriched_prompt = generic_extraction_prompt_entire_doc(text, language=self.language)

        elif inference_type == 'generic' and doc_type == 'chunk':
            enriched_prompt = generic_extraction_prompt_chunk(text, language=self.language) 
        
        else:
            raise NotImplementedError(f"The {inference_type} inference type is not implemented for this doc_type : {doc_type} ")
        
        return enriched_prompt
    
    def _build_a_system_message_to_force_language(self, language : str = "English") -> SystemMessage:

        return SystemMessage(content = f"You must respond only in {language}, regardless of the input language.")
    

    def run(self, *args, **kwargs) -> BaseModel:
        return NotImplementedError
    
# TODO --- Exemple
class SummarizationLLMInfBlock(BaseLLMIngestionBlock):

    def __init__(self, *args, **kwargs):

        pass

    def run(self, *args, **kwargs):
        return NotImplementedError