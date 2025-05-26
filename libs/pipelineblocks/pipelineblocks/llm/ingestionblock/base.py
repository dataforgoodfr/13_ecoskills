from typing import AsyncGenerator, Iterator, Any

from kotaemon.base import BaseComponent, Document
from pydantic import BaseModel
<<<<<<< HEAD
from kotaemon.base.schema import SystemMessage
from pipelineblocks.llm.prompts.scientific_paper import scientific_basic_prompt_entire_doc
from pipelineblocks.llm.prompts.generic_document import generic_extraction_prompt_entire_doc, generic_extraction_prompt_chunk

class BaseLLMIngestionBlock(BaseComponent):

    """A parent class for all LLM Ingestion Block"""

    def _build_a_system_message_to_force_language(self, language : str = "English") -> SystemMessage:
        """A common method to force a llm to respond only in a specific language."""
        return SystemMessage(content = f"You must respond only in {language}. Extract key insights as a list of strings.")

=======
from pipelineblocks.llm.prompts.scientific_paper import scientific_basic_prompt

class BaseLLMIngestionBlock(BaseComponent):

>>>>>>> 97a2958 ((feat) kotaemon : add libs taxonomy OK & pipelineblocks OK)
    def stream(self, *args, **kwargs) -> Iterator[Document] | None :
        raise NotImplementedError

    def astream(self, *args, **kwargs) -> AsyncGenerator[Document, None] | None:
        raise NotImplementedError

    def run(self, *args, **kwargs) -> Document | list[Document] | Iterator[Document] | None | Any:
        return NotImplementedError


class MetadatasLLMInfBlock(BaseLLMIngestionBlock):
<<<<<<< HEAD

    """Parent class for LLM Inference blocks that deduce metadatas from a document, according to a pydantic schema object"""
        
    taxonomy : BaseModel
    language : str = "English"
=======
        
    taxonomy : BaseModel
>>>>>>> 97a2958 ((feat) kotaemon : add libs taxonomy OK & pipelineblocks OK)

    def _invoke_json_schema_from_taxo(self):

        return self.taxonomy.model_json_schema()

<<<<<<< HEAD
    def _convert_content_to_pydantic_schema(self, content, mode='json') -> BaseModel:
            
        if mode=='json':
        
            return self.taxonomy.model_validate_json(content)
        
        elif mode=='dict':
            return self.taxonomy.model_validate(content)
        
        else:
            raise NotImplementedError("Please provide a mode implemented for this method '_convert_content_to_pydantic_schema' ")
    
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
=======
    def _convert_content_to_pydantic_schema(self, content) -> BaseModel:
            
            return self.taxonomy.model_validate_json(content)
    
    def _adjust_prompt_according_to_doc_type(self, text, doc_type, inference_type) -> str:

        if inference_type == 'scientific' and doc_type == 'entire_doc':
            # First combination example
            enriched_prompt = scientific_basic_prompt(text)
            
        elif inference_type == 'scientific' and doc_type == 'chunk':
            # Other combination Example
            raise NotImplementedError(f"The {inference_type} inference type is not implemented for this doc_type : {doc_type} ")
>>>>>>> 97a2958 ((feat) kotaemon : add libs taxonomy OK & pipelineblocks OK)
        
        else:
            raise NotImplementedError(f"The {inference_type} inference type is not implemented for this doc_type : {doc_type} ")
        
        return enriched_prompt
<<<<<<< HEAD
=======
    
>>>>>>> 97a2958 ((feat) kotaemon : add libs taxonomy OK & pipelineblocks OK)

    def run(self, *args, **kwargs) -> BaseModel:
        return NotImplementedError
    
<<<<<<< HEAD


class CustomPromptLLMInfBlock(BaseLLMIngestionBlock):

    """Parent class for LLM Inference blocks that respond to a specific custom prompt."""

    def _invoke_json_schema_from_pydantic_schema(self, pydantic_schema) -> dict:

        return pydantic_schema.model_json_schema()

    def _convert_content_to_pydantic_schema(self, content, pydantic_schema) -> BaseModel:
            
        return pydantic_schema.model_validate_json(content)

    def run(self, *args, **kwargs) -> BaseModel:
        return NotImplementedError
    


=======
>>>>>>> 97a2958 ((feat) kotaemon : add libs taxonomy OK & pipelineblocks OK)
# TODO --- Exemple
class SummarizationLLMInfBlock(BaseLLMIngestionBlock):

    def __init__(self, *args, **kwargs):

        pass

    def run(self, *args, **kwargs):
        return NotImplementedError