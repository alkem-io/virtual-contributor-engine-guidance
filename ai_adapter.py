from azure.ai.inference.models import SystemMessage, UserMessage
from alkemio_virtual_contributor_engine.events.response import Response
from alkemio_virtual_contributor_engine.events.input import Input
from logger import setup_logger
from logger import setup_logger
from models import get_retriever, invoke_model
from alkemio_virtual_contributor_engine.utils import (
    combine_documents,
    get_language_by_code,
    history_as_text,
)
from prompts import chat_system_prompt, condense_prompt
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

    documents = get_retriever().invoke(message)
    logger.info("Context retrieved.")
    logger.debug(f"Context is {documents}")
    context = combine_documents(documents)

    messages = [
        SystemMessage(
            content=chat_system_prompt.format(
                context=context, language=get_language_by_code(input.language)
            )
        ),
        UserMessage(content=message),
    ]

    logger.info("Invoking LLM.")
    result = invoke_model(messages)
    logger.info("LLM invocation completed.")
    logger.info(f"LLM message is: {result}")

    return Response(
        {
            "result": result,
            "original_result": result,
            "human_language": input.language,
            "result_language": input.language,
            "knowledge_language": "en",
            "sources": [{"uri": document.metadata["source"]} for document in documents],
        }
    )
