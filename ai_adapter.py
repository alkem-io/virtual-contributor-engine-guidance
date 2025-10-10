import json
from config import env
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from alkemio_virtual_contributor_engine import (
    Response,
    Input,
    history_as_text,
    setup_logger,
    mistral_medium as llm,
)
from create_context import create_context

from prompts import (
    condense_prompt,
    bok_system_prompt,
    response_system_prompt,
)

logger = setup_logger(__name__)


async def invoke(input: Input) -> Response:
    logger.info("Ai Adapter invoked.")
    logger.debug(f"Input is: {input}")
    try:
        # important to await the result before returning
        return await query_chain(input)
    except Exception as inst:
        logger.exception(inst)
        result = (
            f"{input.display_name} - the Alkemio's VirtualContributor is currently "
            "unavailable."
        )

        return Response(
            result=result,
            original_result=result,
            human_language=input.language,
            result_language=input.language,
            knowledge_language="en",
            sources=[],
        )


async def query_chain(input: Input) -> Response:

    message = input.message
    logger.debug(f"User message is: {message}")

    history = input.history[(env.history_length + 1) * -1: -1]
    if len(history) > 0:
        logger.info(f"We have history. Let's rephrase. Length is: {len(history)}.")
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(
                content=condense_prompt.format(
                    chat_history=history_as_text(history), message=message
                )
            )
        ])
        chain = prompt | llm
        result = chain.invoke({})
        logger.info(
            f"Original message is: '{message}'; Rephrased message is: '{result}'"
        )
        message = result.content
    else:
        logger.info("No history to handle, initial interaction")

    documents, context = create_context(message)

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=bok_system_prompt.format(knowledge=context)),
        SystemMessage(content=response_system_prompt),
        HumanMessage(content=message),
    ])

    logger.info("Invoking LLM.")
    chain = prompt | llm
    result = chain.invoke({})

    response = json.loads(result.content)
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
        result=response["result"],
        original_result=response["result"],
        human_language=input.language,
        result_language=input.language,
        knowledge_language="en",
        sources=sources,
    )
