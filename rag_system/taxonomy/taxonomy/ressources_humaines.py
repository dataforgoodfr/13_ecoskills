from pydantic import BaseModel
from typing import Literal

"""class RessourcesHumainesMissions(str, Enum):  # Mandatory:	YES, Type:	EXCLUSIVE
    campaign_management = "La gestion des campagnes de recrutement."
    needs_analysis = "L'analyse des besoins en compétences et leur adéquation avec les besoins de l'organisation."
    process_selection = "Le suivi des processus de sélection."
    candidates_support = "L'accompagnement des candidats et la promotion de l'attractivité de l'employeur."
"""

class RessourcesHumainesDomain(BaseModel):
    professional_functional_area : Literal["Ressources Humaines (RH)"]