from langchain.callbacks import get_openai_callback
from langchain.memory import ConversationBufferMemory
import pika
import json
import ai_utils
import logging
import sys
import io
import os
import def_ingest
from config import config, website_source_path, website_generated_path, website_source_path2, website_generated_path2, vectordb_path, generate_website, local_path, LOG_LEVEL

# configure logging
logger = logging.getLogger(__name__)
assert LOG_LEVEL in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
logger.setLevel(getattr(logging, LOG_LEVEL))  # Set logger level


# Create handlers
c_handler = logging.StreamHandler(io.TextIOWrapper(sys.stdout.buffer, line_buffering=True))
f_handler = logging.FileHandler(os.path.join(os.path.expanduser(local_path),'app.log'))

c_handler.setLevel(level=getattr(logging, LOG_LEVEL))
f_handler.setLevel(logging.WARNING)

# Create formatters and add them to handlers
c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)

logger.info(f"log level app: {LOG_LEVEL}")

user_data = {}
user_chain = {}

credentials = pika.PlainCredentials(config['rabbitmq_user'],
                                    config['rabbitmq_password'])
parameters = pika.ConnectionParameters(host=config['rabbitmq_host'],
                                       credentials=credentials)
logger.info(f"About to connect to RabbitMQ with params {config['rabbitmq_user']}: {config['rabbitmq_host']}\n")
connection = pika.BlockingConnection(parameters)
channel = connection.channel()

channel.queue_declare(queue=config['rabbitmqrequestqueue'])

def query(user_id, query, language_code):
    logger.info(f"\nQuery from user {user_id}: {query}\n")

    if user_id not in user_data:
        user_data[user_id] = {}
        user_data[user_id]['chat_history'] = ConversationBufferMemory(return_messages=True, output_key="answer", input_key="question")
        #user_chain[user_id]=ai_utils.setup_chain()
        reset(user_id)
        #chat_history=[]

    user_data[user_id]['language'] = ai_utils.get_language_by_code(language_code)

    logger.debug(f"\nlanguage: {user_data[user_id]['language']}\n")
    #chat_history = user_data[user_id]['chat_history']



    with get_openai_callback() as cb:
        llm_result = ai_utils.query_chain({"question": query}, {"language": user_data[user_id]['language']}, user_data[user_id]['chat_history'])
        answer = llm_result['answer']


    # clean up the document sources to avoid sending too much information over.
    sources = [doc.metadata['source'] for doc in llm_result['source_documents']]
    logger.debug(f"\n\nsources: {sources}\n\n")

    logger.info(f"\nTotal Tokens: {cb.total_tokens}")
    logger.info(f"\nPrompt Tokens: {cb.prompt_tokens}")
    logger.info(f"\nCompletion Tokens: {cb.completion_tokens}")
    logger.info(f"\nTotal Cost (USD): ${cb.total_cost}")

    logger.debug(f"\n\nLLM result: {llm_result}\n\n")
    logger.info(f"\n\nanswer: {answer}\n\n")
    logger.debug(f"\n\nsources: {sources}\n\ n")

    user_data[user_id]['chat_history'].save_context({"question": query}, {"answer": answer.content})
    logger.debug(f"new chat history {user_data[user_id]['chat_history']}\n")
    response = json.dumps({
        "question": query, "answer": str(answer), "sources": sources, "prompt_tokens": cb.prompt_tokens, "completion_tokens": cb.completion_tokens, "total_tokens": cb.total_tokens, "total_cost": cb.total_cost
    }
    )

    return response

def reset(user_id):
    user_data[user_id]['chat_history'].clear()

    return "Reset function executed"

def ingest(source_url, website_repo, destination_path, source_path, source_url2, website_repo2, destination_path2, source_path2):
    def_ingest.clone_and_generate(website_repo, destination_path, source_path)
    def_ingest.clone_and_generate(website_repo2, destination_path2, source_path2)
    def_ingest.mainapp(source_url, source_url2)

    return "Ingest function executed"

def on_request(ch, method, props, body):
    message = json.loads(body)
    user_id = message['data'].get('userId')

    operation = message['pattern']['cmd']

    if operation == 'ingest':
        response = ingest(config['source_website'], config['website_repo'], website_generated_path, website_source_path, config['source_website2'], config['website_repo2'], website_generated_path2, website_source_path2)
    else:
        if user_id is None:
            response = "userId not provided"
        else:
            if operation == 'query':
                if ('question' in message['data']) and ('language' in message['data']):
                    response = query(user_id, message['data']['question'], message['data']['language'])
                else:
                    response = "Query parameter(s) not provided"
            elif operation == 'reset':
                response = reset(user_id)
            else:
                response = "Unknown function"

    ch.basic_publish(
        exchange='',
        routing_key=props.reply_to,
        properties=pika.BasicProperties(correlation_id=props.correlation_id),
        body=json.dumps({"operation": "feedback", "result": response})
    )

    ch.basic_ack(delivery_tag=method.delivery_tag)
    logger.info(f"Response sent for correlation_id: {props.correlation_id}")
    logger.info(f"Response sent to: {props.reply_to}")
    logger.info(f"response: {response}")


channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue=config['rabbitmqrequestqueue'], on_message_callback=on_request)

logger.info("Waiting for RPC requests")
channel.start_consuming()
