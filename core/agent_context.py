"""
Contexte partagé entre tous les agents
"""

from typing import Dict, List, Any
from datetime import datetime

class AgentContext:
    """Contexte global partagé entre agents"""
    
    def __init__(self):
        self.conversation_history: List[Dict] = []
        self.user_profile: Dict[str, Any] = {}
        self.session_data: Dict[str, Any] = {}
        self.current_agent: str = None
        self.created_at: datetime = datetime.now()
    
    def add_message(self, sender: str, content: str, role: str = "assistant"):
        """Ajoute un message à l'historique"""
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "sender": sender,
            "role": role,
            "content": content
        })
    
    def get_history(self, last_n: int = None) -> List[Dict]:
        """Récupère l'historique (optionnellement les n derniers messages)"""
        if last_n:
            return self.conversation_history[-last_n:]
        return self.conversation_history
    
    def set_session_data(self, key: str, value: Any):
        """Stocke une donnée de session"""
        self.session_data[key] = value
    
    def get_session_data(self, key: str, default=None) -> Any:
        """Récupère une donnée de session"""
        return self.session_data.get(key, default)
    
    def to_dict(self) -> Dict:
        """Export du contexte en dict"""
        return {
            "conversation_history": self.conversation_history,
            "user_profile": self.user_profile,
            "session_data": self.session_data,
            "current_agent": self.current_agent
        }