from typing import Any, Dict, List, Optional


class Source:
    def __init__(self, source: Dict[str, str]):
        self.chunk_index = None
        self.embedding_type = None
        if "chunkIndex" in source:
            self.chunk_index = source.get("chunkIndex")
        if "embeddingType" in source:
            self.embedding_type = source.get("embeddingType")

        self.document_id = source.get("documentId")
        self.source = source.get("source")
        self.title = source.get("title")
        self.type = source.get("type")
        self.score = source.get("score")
        self.uri = source.get("uri")

        if all(
            [self.document_id, self.source, self.title, self.type, self.score, self.uri]
        ):
            raise ValueError("Missing required fields in source dictionary")

    def to_dict(self):
        result = {
            "documentId": self.document_id,
            "source": self.source,
            "title": self.title,
            "type": self.type,
            "score": self.score,
            "uri": self.uri,
        }

        if self.chunk_index:
            result["chunkIndex"] = self.chunk_index
        if self.embedding_type:
            result["embeddingType"] = self.embedding_type

        return result


class Response:

    # TODO the signature should be like this but validation is too much work
    # def __init__(self, response: Optional[Dict[str, str | List[Dict[str, str]]]] = None):

    def __init__(self, response: Optional[Dict[str, Any]] = None):
        self.result: Optional[str] = None
        self.human_language: Optional[str] = None
        self.result_language: Optional[str] = None
        self.knowledge_language: Optional[str] = None
        self.original_result: Optional[str] = None
        self.sources: List[Source] = []

        if response is not None:
            try:
                self.result = response["result"]
                self.human_language = response["human_language"]
                self.result_language = response["result_language"]
                self.knowledge_language = response["knowledge_language"]
                self.original_result = response["original_result"]
                self.sources = [
                    Source(source) for source in response.get("sources", [])
                ]
            except KeyError as e:
                raise ValueError(f"Missing required field: {e}")

    def to_dict(self):
        return {
            "result": self.result,
            "humanLanguage": self.human_language,
            "resultLanguage": self.result_language,
            "knowledgeLanguage": self.knowledge_language,
            "originalResult": self.original_result,
            "sources": list(map(lambda source: source.to_dict(), self.sources)),
        }
