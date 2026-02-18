"""
Classe de base abstraite dont tous les agents héritent
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from core.message import AgentMessage
from core.agent_context import AgentContext

class BaseAgent(ABC):
    """Classe abstraite de base pour tous les agents"""
    
    def __init__(self, agent_name: str, config: Dict[str, Any]):
        self.agent_name = agent_name
        self.config = config
        self.is_active = True
    
    @abstractmethod
    def can_handle(self, message: AgentMessage, context: AgentContext) -> bool:
        """
        Détermine si cet agent peut traiter le message
        
        Returns:
            bool: True si l'agent peut traiter, False sinon
        """
        pass
    
    @abstractmethod
    def process(self, message: AgentMessage, context: AgentContext) -> AgentMessage:
        """
        Traite le message et retourne une réponse
        
        Returns:
            AgentMessage: Réponse de l'agent
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Retourne les capacités de l'agent
        
        Returns:
            Dict décrivant ce que l'agent peut faire
        """
        pass
    
    def log(self, message: str, level: str = "INFO"):
        """Logger simple"""
        print(f"[{level}] [{self.agent_name}] {message}")