from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List


class ResultHandlerAction(Enum):
    POST_REPLY = "postReply"


class MessageSenderRole(Enum):
    HUMAN = "human"
    ASSISTANT = "assistant"


@dataclass
class HistoryItem:
    content: str
    role: MessageSenderRole

    def __init__(self, item: dict):
        if "content" not in item or "role" not in item:
            raise ValueError("Missing required fields: content, role")

        self.content = item["content"]
        self.role = MessageSenderRole(item["role"])

    def to_dict(self) -> Dict[str, str | MessageSenderRole]:
        return {"content": self.content, "role": self.role.value}


@dataclass
class RoomDetails:
    room_id: str
    thread_id: str
    communication_id: str
    interaction_id: str

    def __init__(self, details: Dict[str, Any]) -> None:
        required_fields = {"roomID", "communicationID"}
        if not all(field in details for field in required_fields):
            raise ValueError(
                f"Missing required fields: {required_fields - set(details.keys())}"
            )
        self.room_id = details["roomID"]
        self.thread_id = details.get("threadID", "")
        self.communication_id = details["communicationID"]
        self.interaction_id = details.get("vcInteractionID", "")

    def to_dict(self) -> Dict[str, str]:
        return {
            "roomID": self.room_id,
            "threadID": self.thread_id,
            "communicationID": self.communication_id,
            "vcInteractionID": self.interaction_id,
        }


@dataclass
class ResultHandler:
    action: ResultHandlerAction
    room_details: RoomDetails

    def __init__(self, config):
        if "action" not in config or "roomDetails" not in config:
            raise ValueError("Missing required fields: action, roomDetails")
        self.action = ResultHandlerAction(config["action"])
        self.room_details = RoomDetails(config["roomDetails"])

    def to_dict(self) -> Dict[str, str | Dict[str, str]]:
        return {"action": self.action.value, "roomDetails": self.room_details.to_dict()}


@dataclass
class Input:
    engine: str
    prompt: str
    user_id: str
    message: str
    bok_id: str
    context_id: str
    history: List[HistoryItem]
    external_metadata: Dict[str, Any]
    display_name: str
    description: str
    external_config: Dict[str, Any]
    result_handler: ResultHandler
    persona_service_id: str
    language: str

    def __init__(self, input_data: Dict[str, Any]) -> None:
        print("\n\n\n")
        print(input_data)
        print("\n\n\n")
        if not isinstance(input_data, dict):
            raise TypeError("Expected dictionary input")

        required_fields = {
            "engine",
            "prompt",
            "userID",
            "message",
            "bodyOfKnowledgeID",
            # "contextID",
            "history",
            "externalMetadata",
            "displayName",
            # "description",
            "externalConfig",
            "resultHandler",
            "personaServiceID",
        }
        if missing := required_fields - set(input_data.keys()):
            raise ValueError(f"Missing required fields: {missing}")

        self.engine = str(input_data["engine"])
        self.prompt = str(input_data["prompt"])
        self.user_id = str(input_data["userID"])
        self.message = str(input_data["message"])
        self.bok_id = str(input_data["bodyOfKnowledgeID"])
        self.context_id = input_data.get("contextID", "")
        self.history = [HistoryItem(item) for item in input_data["history"]]
        self.external_metadata = dict(input_data["externalMetadata"])
        self.display_name = str(input_data["displayName"])
        self.description = input_data.get("description", "")
        self.external_config = dict(input_data["externalConfig"])
        self.result_handler = ResultHandler(input_data["resultHandler"])
        self.persona_service_id = str(input_data["personaServiceID"])
        self.language = str(input_data["language"])

    def to_dict(self):
        return {
            "engine": self.engine,
            "prompt": self.prompt,
            "userID": self.user_id,
            "message": self.message,
            "bodyOfKnowledgeID": self.bok_id,
            "contextID": self.context_id,
            "history": [item.to_dict() for item in self.history],
            "externalMetadata": self.external_metadata,
            "displayName": self.display_name,
            "description": self.description,
            "externalConfig": self.external_config,
            "resultHandler": self.result_handler.to_dict(),
            "personaServiceID": self.persona_service_id,
            "language": self.language,
        }
