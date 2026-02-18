"""
Format de messages standardisé pour communication inter-agents
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime
from enum import Enum

class MessageType(Enum):
    QUERY = "query"
    RESPONSE = "response"
    ERROR = "error"
    HANDOFF = "handoff"  # Transfer vers un autre agent

class AgentMessage(BaseModel):
    """Message standardisé entre agents"""
    
    message_id: str
    timestamp: datetime
    sender_agent: str
    receiver_agent: Optional[str] = None
    message_type: MessageType
    content: str
    metadata: Dict[str, Any] = {}
    context: Dict[str, Any] = {}
    
    class Config:
        use_enum_values = True

    @classmethod
    def create_query(cls, sender: str, content: str, **kwargs):
        """Créer un message de type query"""
        import uuid
        return cls(
            message_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            sender_agent=sender,
            message_type=MessageType.QUERY,
            content=content,
            **kwargs
        )
    
    @classmethod
    def create_response(cls, sender: str, content: str, **kwargs):
        """Créer un message de type response"""
        import uuid
        return cls(
            message_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            sender_agent=sender,
            message_type=MessageType.RESPONSE,
            content=content,
            **kwargs
        )