from langchain.callbacks import get_openai_callback
from langchain.memory import ConversationBufferWindowMemory

# import pika
import json
import ai_adapter
import logging
import sys
import io
import asyncio
import os
import aio_pika
import aiormq
from aio_pika import connect, RobustConnection, ExchangeType
from config import (
    config,
    website_source_path,
    website_generated_path,
    website_source_path2,
    website_generated_path2,
    vectordb_path,
    local_path,
    LOG_LEVEL,
)
import ai_adapter
import asyncio
import os
from alkemio_virtual_contributor_engine.alkemio_vc_engine import (
    AlkemioVirtualContributorEngine,
)
from alkemio_virtual_contributor_engine.events.input import Input
from alkemio_virtual_contributor_engine.events.result import Response
from config import LOG_LEVEL
from logger import setup_logger

logger = setup_logger(__name__)

logger.info(f"log level {os.path.basename(__file__)}: {LOG_LEVEL}")


# define variables
user_data = {}
user_chain = {}
# dictionary to keep track of the locks for each user
user_locks = {}
# Dictionary to keep track of the tasks for each user
user_tasks = {}
# Lock to prevent multiple ingestions from happening at the same time
ingestion_lock = asyncio.Lock()


async def query(input: Input) -> Response:
    user_id = input.user_id
    language_code = input.language
    message = input.message

    async with ingestion_lock:
        logger.info(f"\nQuery from user {input.user_id}: {message}\n")

        if user_id not in user_data:
            reset(user_id)

        user_data[user_id]["language"] = ai_adapter.get_language_by_code(language_code)

        logger.debug(f"\nlanguage: {user_data[user_id]['language']}\n")
        # chat_history = user_data[user_id]['chat_history']

        with get_openai_callback() as cb:
            llm_result = await ai_adapter.invoke(
                input,
                user_data[user_id]["chat_history"],
            )
            answer = llm_result.result

        # clean up the document sources to avoid sending too much information over.
        sources = [doc.uri for doc in llm_result.sources]
        logger.debug(f"\n\nsources: {sources}\n\n")

        logger.debug(f"\nTotal Tokens: {cb.total_tokens}")
        logger.debug(f"\nPrompt Tokens: {cb.prompt_tokens}")
        logger.debug(f"\nCompletion Tokens: {cb.completion_tokens}")
        logger.debug(f"\nTotal Cost (USD): ${cb.total_cost}")

        logger.debug(f"\n\nLLM result: {llm_result}\n\n")
        logger.info(f"\n\nanswer: {answer}\n\n")
        logger.debug(f"\n\nsources: {sources}\n\\ n")

        # user_data[user_id]["chat_history"].save_context(
        #     {"question": message}, {"answer": answer.content}
        # )
        # logger.debug(f"new chat history {user_data[user_id]['chat_history']}\n")
        # response = json.dumps(
        #     {
        #         "question": message,
        #         "answer": str(answer.content),
        #         "sources": sources,
        #         "prompt_tokens": cb.prompt_tokens,
        #         "completion_tokens": cb.completion_tokens,
        #         "total_tokens": cb.total_tokens,
        #         "total_cost": cb.total_cost,
        #     }
        # )

        return llm_result  # response


def reset(user_id):
    if user_id not in user_data:
        user_data[user_id] = {}
        user_data[user_id]["chat_history"] = ConversationBufferWindowMemory(
            k=3, return_messages=True, output_key="answer", input_key="question"
        )
    user_data[user_id]["chat_history"].clear()
    return "Reset function executed"


async def on_request(message: Input) -> Response:
    # Get the user ID from the message body
    user_id = message.user_id

    logger.info(f"\nrequest arriving for user id: {user_id}, deciding what to do\n\n")

    # If there's no lock for this user, create one
    if user_id not in user_locks:
        user_locks[user_id] = asyncio.Lock()

    # Check if the lock is locked
    if user_locks[user_id].locked():
        logger.info(
            f"existing task running for user id: {user_id}, waiting for it to finish first\n\n"
        )
    else:
        logger.info(f"no task running for user id: {user_id}, let's move!\n\n")

    # Acquire the lock for this user
    async with user_locks[user_id]:
        # Process the message
        return await process_message(message)


async def process_message(message: Input) -> Response:
    user_id = message.user_id

    # hardcoded temporary
    operation = "query"

    response = Response(
        {
            "result": "result",
            "original_result": "result",
            "human_language": message.language,
            "result_language": message.language,
            "knowledge_language": "en",
            "sources": [],
        }
    )

    # if operation == "ingest":
    #     try:
    #         logger.info("Attempting to acquire lock and run ingest operation")
    #         async with ingestion_lock:
    #             logger.info("Lock acquired, running ingest operation")
    #             logger.info("Ingest operation completed")
    #         response = "Ingest successful"
    #     except Exception as e:
    #         logger.error(f"Ingest failed: Exception: {e}")
    #         response = "Ingest failed"
    # else:
    #     if user_id is None:
    #         response = "userId not provided"
    #     else:
    #         if operation == "query":
    return await query(message)
    # elif operation == "reset":
    #     logger.info(f"reset user id: {user_id}\n\n")
    #     response = reset(user_id)


# async def main():
#     logger.info(f"main fucntion (re)starting\n")
#     # rabbitmq is an instance of the RabbitMQ class defined earlier
#     await rabbitmq.connect()

#     await rabbitmq.channel.set_qos(prefetch_count=20)
#     queue = await rabbitmq.channel.declare_queue(
#         rabbitmq.queue, auto_delete=False, durable=True
#     )

#     # Start consuming messages
#     asyncio.create_task(queue.consume(on_request))

#     logger.info("Waiting for RPC requests")

#     # Create an Event that is never set, and wait for it forever
#     # This will keep the program running indefinitely
#     stop_event = asyncio.Event()
#     await stop_event.wait()


# loop = asyncio.get_event_loop()
# loop.run_until_complete(main())

# async def on_request(input: Input) -> Response:
#     logger.info(f"Expert engine invoked; Input is {input.to_dict()}")
#     logger.info(
#         f"AiPersonaServiceID={input.persona_service_id} with VC name `{input.display_name}` invoked."
#     )
#     result = await ai_adapter.invoke(input)
#     logger.info(f"LLM result: {result.to_dict()}")
#     return result


engine = AlkemioVirtualContributorEngine()
engine.register_handler(on_request)
asyncio.run(engine.start())
