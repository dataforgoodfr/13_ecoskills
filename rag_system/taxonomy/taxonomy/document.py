from typing import Optional, Union
from pydantic import BaseModel, Field
from enum import Enum

from taxonomy.ressources_humaines import RessourcesHumainesDomain

class Author_gender(str, Enum):
    male = 'Male'
    female = 'Female'
    unknown = 'Unknown'

class Author(BaseModel):
    name: str
    gender: Optional[Author_gender]

class Publication_type(str, Enum):
    research_article  = 'Research article'
    article_in_press  = 'Article-in-Press (AiP)'
    book              = 'Book'
    case_study        = 'Case study'
    chapter           = 'Chapter'
    conference_paper  = 'Conference Paper'
    data_paper        = 'Data paper'
    editorial         = 'Editorial'
    erratum           = 'Erratum'
    letter            = 'Letter'
    note              = 'Note'
    retracted_article = 'Retracted article'
    review            = 'Review'
    short_survey      = 'Short survey'
    commentary        = 'Commentary '
    presentation      = 'Presentation'
    technical_report  = 'Technical report'
    policy_report     = 'Policy report'
    policy_brief      = 'Policy brief'
    factsheet         = 'Factsheet'


class EntireDocument(BaseModel):
    title: str
    functional_area: Union[RessourcesHumainesDomain] = Field(discriminator='functional_area')
    authors: list[Author]
    summary: str
    year_of_publication: int
    publication_type: Publication_type
    language: Optional[str]


class ChunkOfDocument(BaseModel):
    key_idea_sentences: list[str]
    concrete_experience_in_the_field : Optional[list[str]]