from langchain.callbacks import get_openai_callback
from langchain.memory import ConversationBufferMemory
# import pika
import json
import ai_adapter
import logging
import sys
import io
import asyncio
import os
import def_ingest
import aio_pika
import aiormq
from aio_pika import connect, RobustConnection, ExchangeType
from config import config, website_source_path, website_generated_path, website_source_path2, website_generated_path2, vectordb_path, local_path, LOG_LEVEL

# configure logging
logger = logging.getLogger(__name__)
assert LOG_LEVEL in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
logger.setLevel(getattr(logging, LOG_LEVEL))  # Set logger level


# Create handlers
c_handler = logging.StreamHandler(io.TextIOWrapper(sys.stdout.buffer, line_buffering=True))
f_handler = logging.FileHandler(os.path.join(os.path.expanduser(local_path), 'app.log'))

c_handler.setLevel(level=getattr(logging, LOG_LEVEL))
f_handler.setLevel(logging.WARNING)

# Create formatters and add them to handlers
c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', '%m-%d %H:%M:%S')
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', '%m-%d %H:%M:%S')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)

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

class RabbitMQ:
    def __init__(self, host, login, password, queue):
        self.host = host
        self.login = login
        self.password = password
        self.queue = queue
        self.connection = None
        self.channel = None

    async def connect(self):
        self.connection: RobustConnection = await connect(
            host=self.host,
            login=self.login,
            password=self.password
        )
        self.channel = await self.connection.channel()
        await self.channel.declare_queue(self.queue, auto_delete=False)


rabbitmq = RabbitMQ(
    host=config['rabbitmq_host'],
    login=config['rabbitmq_user'],
    password=config['rabbitmq_password'],
    queue=config['rabbitmqrequestqueue']
)

async def query(user_id, query, language_code):
    async with ingestion_lock:
        logger.info(f"\nQuery from user {user_id}: {query}\n")

        if user_id not in user_data:
            user_data[user_id] = {}
            user_data[user_id]['chat_history'] = ConversationBufferMemory(return_messages=True, output_key="answer", input_key="question")
            # user_chain[user_id]=ai_utils.setup_chain()
            reset(user_id)
            # chat_history=[]

        user_data[user_id]['language'] = ai_adapter.get_language_by_code(language_code)

        logger.debug(f"\nlanguage: {user_data[user_id]['language']}\n")
        # chat_history = user_data[user_id]['chat_history']

        with get_openai_callback() as cb:
            llm_result = await ai_adapter.query_chain({"question": query}, {"language": user_data[user_id]['language']}, user_data[user_id]['chat_history'])
            answer = llm_result['answer']

        # clean up the document sources to avoid sending too much information over.
        sources = [doc.metadata['source'] for doc in llm_result['source_documents']]
        logger.debug(f"\n\nsources: {sources}\n\n")

        logger.debug(f"\nTotal Tokens: {cb.total_tokens}")
        logger.debug(f"\nPrompt Tokens: {cb.prompt_tokens}")
        logger.debug(f"\nCompletion Tokens: {cb.completion_tokens}")
        logger.debug(f"\nTotal Cost (USD): ${cb.total_cost}")

        logger.debug(f"\n\nLLM result: {llm_result}\n\n")
        logger.info(f"\n\nanswer: {answer}\n\n")
        logger.debug(f"\n\nsources: {sources}\n\\ n")

        user_data[user_id]['chat_history'].save_context({"question": query}, {"answer": answer.content})
        logger.debug(f"new chat history {user_data[user_id]['chat_history']}\n")
        response = json.dumps({"question": query, "answer": str(answer.content), "sources": sources, "prompt_tokens": cb.prompt_tokens,
                              "completion_tokens": cb.completion_tokens, "total_tokens": cb.total_tokens, "total_cost": cb.total_cost})

        return response

def reset(user_id):
    user_data[user_id]['chat_history'].clear()

    return "Reset function executed"

async def ingest(source_url, website_repo, destination_path, source_path, source_url2, website_repo2, destination_path2, source_path2):
    def_ingest.clone_and_generate(website_repo, destination_path, source_path)
    def_ingest.clone_and_generate(website_repo2, destination_path2, source_path2)
    def_ingest.create_vector_db(source_url, source_url2)

    return "Ingest function executed"


async def on_request(message: aio_pika.IncomingMessage):
    async with message.process():
        # Parse the message body as JSON
        body = json.loads(message.body)

        # Get the user ID from the message body
        user_id = body['data']['userId']

        logger.info(f"\nrequest arriving for user id: {user_id}, deciding what to do\n\n")

        # If there's no lock for this user, create one
        if user_id not in user_locks:
            user_locks[user_id] = asyncio.Lock()

        # Check if the lock is locked
        if user_locks[user_id].locked():
            logger.info(f"existing task running for user id: {user_id}, waiting for it to finish first\n\n")
        else:
            logger.info(f"no task running for user id: {user_id}, let's move!\n\n")

        # Acquire the lock for this user
        async with user_locks[user_id]:
            # Process the message
            await process_message(message)


async def process_message(message: aio_pika.IncomingMessage):
    body = json.loads(message.body.decode())
    user_id = body['data'].get('userId')

    operation = body['pattern']['cmd']

    if operation == 'ingest':
        try:
            logger.info("Attempting to acquire lock and run ingest operation")
            async with ingestion_lock:
                logger.info("Lock acquired, running ingest operation")
                await ingest(config['source_website'], config['website_repo'], website_generated_path, website_source_path, config['source_website2'], config['website_repo2'], website_generated_path2, website_source_path2)
                logger.info("Ingest operation completed")
            response = "Ingest successful"
        except Exception as e:
            logger.error(f"Ingest failed: Exception: {e}")
            response = "Ingest failed"
    else:
        if user_id is None:
            response = "userId not provided"
        else:
            if operation == 'query':
                if ('question' in body['data']) and ('language' in body['data']):
                    logger.info(f"query time for user id: {user_id}, let's call the query() function!\n\n")
                    response = await query(user_id, body['data']['question'], body['data']['language'])
                else:
                    response = "Query parameter(s) not provided"
            elif operation == 'reset':
                response = reset(user_id)
            else:
                response = "Unknown function"

    try:
        if rabbitmq.connection.is_closed or rabbitmq.channel.is_closed:
            logger.error("Connection or channel is not open. Cannot publish message.")
            return

        await rabbitmq.channel.default_exchange.publish(
            aio_pika.Message(
                body=json.dumps({"operation": "feedback", "result": response}).encode(),
                correlation_id=message.correlation_id,
                reply_to=message.reply_to
            ),
            routing_key=message.reply_to
        )
        logger.info(f"Response sent for correlation_id: {message.correlation_id}")
        logger.info(f"Response sent to: {message.reply_to}")
        logger.debug(f"response: {response}")
    except (aio_pika.exceptions.AMQPError, asyncio.exceptions.CancelledError, aiormq.exceptions.ChannelInvalidStateError) as e:
        logger.error(f"Failed to publish message due to a RabbitMQ error: {e}")


async def main():
    logger.info(f"main fucntion (re)starting\n")
    # rabbitmq is an instance of the RabbitMQ class defined earlier
    await rabbitmq.connect()

    await rabbitmq.channel.set_qos(prefetch_count=20)
    queue = await rabbitmq.channel.declare_queue(rabbitmq.queue, auto_delete=False)

    # Start consuming messages
    asyncio.create_task(queue.consume(on_request))

    logger.info("Waiting for RPC requests")

    # Create an Event that is never set, and wait for it forever
    # This will keep the program running indefinitely
    stop_event = asyncio.Event()
    await stop_event.wait()

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
