import json
from azure.ai.inference.models import SystemMessage, UserMessage
from alkemio_virtual_contributor_engine.events.response import Response
from alkemio_virtual_contributor_engine.events.input import Input
from create_context import create_context
from logger import setup_logger
from models import invoke_model
from alkemio_virtual_contributor_engine.utils import (
    history_as_text,
)
from prompts import (
    condense_prompt,
    bok_system_prompt,
    response_system_prompt,
)
from config import env

logger = setup_logger(__name__)


async def invoke(input: Input) -> Response:
    logger.info("Ai Adapter invoked.")
    logger.debug(f"Input is: {input}")
    try:
        # important to await the result before returning
        return await query_chain(input)
    except Exception as inst:
        logger.exception(inst)
        result = f"{input.display_name} - the Alkemio's VirtualContributor is currently unavailable."

        return Response(
            {
                "result": result,
                "original_result": result,
                "human_language": input.language,
                "result_language": input.language,
                "knowledge_language": "en",
                "sources": [],
            }
        )


async def query_chain(input: Input) -> Response:

    message = input.message
    logger.debug(f"User message is: {message}")

    history = input.history[(env.history_length + 1) * -1 : -1]
    if len(history) > 0:
        logger.info(f"We have history. Let's rephrase. Length is: {len(history)}.")
        messages = [
            SystemMessage(
                content=condense_prompt.format(
                    chat_history=history_as_text(history), message=message
                )
            )
        ]
        result = invoke_model(messages, 0)
        logger.info(
            f"Original message is: '{message}'; Rephrased message is: '{result}'"
        )
        message = result
    else:
        logger.info("No history to handle, initial interaction")

    documents, context = create_context(message)

    messages = [
        SystemMessage(content=bok_system_prompt.format(knowledge=context)),
        SystemMessage(content=response_system_prompt.format(context=context)),
        UserMessage(content=message),
    ]

    logger.info("Invoking LLM.")
    response = json.loads(invoke_model(messages))
    logger.info("LLM invocation completed.")
    logger.info(f"LLM message is: {response}")

    sources = []
    for index, metadata in enumerate(documents["metadatas"][0]):
        index = str(index)
        if (
            {"uri": metadata["source"]} not in sources
            and index in response["source_scores"]
            and response["source_scores"][index] > 0
        ):
            sources.append({"uri": metadata["source"]})

    return Response(
        {
            "result": response["result"],
            "original_result": response["result"],
            "human_language": input.language,
            "result_language": input.language,
            "knowledge_language": "en",
            "sources": sources,
        }
    )
