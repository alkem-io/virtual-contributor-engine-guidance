import re
import time
from alkemio_virtual_contributor_engine import (
    Input,
    Response,
    setup_logger,
    mistral_small,
    query_documents,
    combine_query_results,
    PromptGraph,
    history_as_conversation,
    history_as_dict,
)

logger = setup_logger(__name__)

COLLECTIONS = [
    "alkem.io-knowledge",
    "welcome.alkem.io-knowledge",
    "www.alkemio.org-knowledge",
]


def retrieve(state):
    """Retrieve knowledge documents from 3 hardcoded Alkemio website collections."""
    last_msg = state.messages[0]
    last_message = state.rephrased_question or (
        last_msg["content"] if isinstance(last_msg, dict)
        else last_msg.content
    )

    result = {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    for collection_name in COLLECTIONS:
        try:
            tmp_result = query_documents(
                last_message,
                collection_name,
                num_docs=3,
                include=["documents", "metadatas", "distances"],
            )
            if (
                tmp_result
                and tmp_result.get("documents")
                and tmp_result.get("distances")
                and tmp_result.get("metadatas")
            ):
                documents = tmp_result["documents"]
                distances = tmp_result["distances"]
                metadatas = tmp_result["metadatas"]
                if documents and documents[0]:
                    result["documents"][0] += documents[0]
                if distances and distances[0]:
                    result["distances"][0] += distances[0]
                if metadatas and metadatas[0]:
                    result["metadatas"][0] += metadatas[0]
        except Exception as e:
            logger.warning(
                f"Failed to query collection '{collection_name}': {e}"
            )

    combined_knowledge_docs = combine_query_results(result)
    return {
        "knowledge_docs": result,
        "combined_knowledge_docs": combined_knowledge_docs,
    }


async def invoke(input: Input) -> Response:
    try:
        if not input.prompt_graph:
            raise Exception("promptGraph is required in Input.")

        prompt_graph = PromptGraph.from_dict(input.prompt_graph)

        logger.info(
            f"Invoking graph "
            f"history_messages={len(input.history)}"
        )
        logger.debug(
            f"Full conversation history: "
            f"{history_as_dict(input.history)}"
        )

        graph = prompt_graph.compile(
            llm=mistral_small,
            special_nodes={"retrieve": retrieve},
        )
        start_time = time.time()
        messages = history_as_dict(input.history)
        input_state = {
            "messages": messages,
            "current_question": (
                messages[0]["content"] if messages else ""
            ),
            "conversation": history_as_conversation(input.history),
            "description": input.description,
            "display_name": input.display_name,
        }
        result = {}
        for step in graph.stream(
            input_state, stream_mode="updates"
        ):
            for node_name, node_output in step.items():
                logger.info(f"Step '{node_name}' completed")
                logger.debug(
                    f"Step '{node_name}' output: {node_output}"
                )
                result.update(node_output)
        duration = time.time() - start_time
        logger.info(
            f"Graph invocation completed in {duration:.2f}s"
        )

        json_result = {
            "result": result.get("final_answer", ""),
            "original_result": result.get("knowledge_answer", ""),
            "human_language": result.get("human_language", "en"),
            "result_language": result.get(
                "knowledge_language", "en"
            ),
            "knowledge_language": result.get(
                "knowledge_language", "en"
            ),
            "source_scores": {},
        }
        knowledge_docs = result.get("knowledge_docs", {})
        source_scores = result.get("source_scores", {})
        sources = []
        if len(source_scores) > 0:
            for index, doc in enumerate(
                knowledge_docs.get("metadatas", [[]])[0]
            ):
                str_index = str(index)
                if (
                    str_index in source_scores
                    and source_scores[str_index] > 0
                ):
                    doc_type = doc.get("type", "unknown")
                    doc_title = doc.get("title", "")
                    sources.append(
                        dict(doc) | {
                            "score": source_scores[str_index],
                            "uri": doc.get("source", ""),
                            "title": "[{}] {}".format(
                                re.sub(
                                    r'(?<=[a-z])(?=[A-Z])|_',
                                    ' ',
                                    str(doc_type),
                                ).capitalize(),
                                doc_title,
                            ),
                        }
                    )
            json_result["sources"] = list(
                {doc["source"]: doc for doc in sources
                 if "source" in doc}.values()
            )

        logger.debug(f"Full result: {json_result}")

        return Response(**json_result)

    except Exception as inst:
        logger.exception(inst)
        result = (
            f"{input.display_name} - the Alkemio's "
            f"VirtualContributor is currently unavailable."
        )

        return Response(
            **{
                "result": result,
                "original_result": result,
                "sources": [],
            }
        )
