from langchain.callbacks import get_openai_callback
import os
import pika
import json
import ai_utils
import def_ingest
import time
from dotenv import load_dotenv
load_dotenv()

config = {
    "rabbitmq_host": os.getenv('RABBITMQ_HOST'),
    "rabbitmq_user": os.getenv('RABBITMQ_USER'),
    "rabbitmq_password": os.getenv('RABBITMQ_PASSWORD'),
    "rabbitmqrequestqueue": "alkemio-chat-guidance",
    "source_website": os.getenv('AI_SOURCE_WEBSITE'),
    "website_repo": os.getenv('AI_WEBSITE_REPO'),
    "local_path": os.getenv('AI_LOCAL_PATH')
}

local_path = config['local_path']
website_source_path = local_path+'/website/source'
website_generated_path = local_path+'/website/generated'
vectordb_path = local_path+"/local_index"

user_data = {}

credentials = pika.PlainCredentials(config['rabbitmq_user'],
                                    config['rabbitmq_password'])
parameters = pika.ConnectionParameters(host=config['rabbitmq_host'],
                                       credentials=credentials)
print(f"\About to connect to RabbitMQ with params {config['rabbitmq_user']}: {config['rabbitmq_host']}\n")
connection = pika.BlockingConnection(parameters)
channel = connection.channel()

channel.queue_declare(queue=config['rabbitmqrequestqueue'])

def query(user_id, query, language_code):
    print(f"\nQuery from user {user_id}: {query}\n")

    if user_id not in user_data:
        reset(user_id)

    user_data[user_id]['language'] = ai_utils.get_language_by_code(language_code)

    print(f"\nlanguage: {user_data[user_id]['language']}\n")
    chat_history = user_data[user_id]['chat_history']

    # llm_result =ai_utils.qa_chain(
    #    query,
    #    chat_history,
    #    user_data[user_id]['language']
    # )
    with get_openai_callback() as cb:
        llm_result = qa_chain({"question": query, "chat_history": chat_history})
        translation=ai_utils.translate_answer(llm_result['answer'],user_data[user_id]['language'],chat_history)

    print(f"\nTotal Tokens: {cb.total_tokens}")
    print(f"\nPrompt Tokens: {cb.prompt_tokens}")
    print(f"\nCompletion Tokens: {cb.completion_tokens}")
    print(f"\nTotal Cost (USD): ${cb.total_cost}")

    print(f"\n\nLLM result: {llm_result}\n\n")
    print(f"\n\ntranslation result: {translation}\n\n")

    formatted_messages = (
        f"Human:'{llm_result['question']}'",
        f"AI:'{llm_result['answer']}'"
    )
    user_data[user_id]['chat_history'].append(formatted_messages)

    # only keep the last 3 entires of that chat history to avoid exceeding the token limit.
    user_data[user_id]['chat_history'] = user_data[user_id]['chat_history'][-3:]

    print(f"new chat history {user_data[user_id]['chat_history']}")

    response = json.dumps({
        "question": str(llm_result["question"]), "answer": str(translation), "sources": str(llm_result["source_documents"]), "prompt_tokens": cb.prompt_tokens, "completion_token": cb.completion_tokens, "total_tokens": cb.total_tokens, "total_cost": cb.total_cost
    }
    )

    return response

def reset(user_id):
    user_data[user_id] = {
        'chat_history': []
    }

    return "Reset function executed"

def ingest(source_url, website_repo, destination_path, source_path):
    def_ingest.clone_and_generate(website_repo, destination_path, source_path)
    def_ingest.mainapp(source_url)

    return "Ingest function executed"

def on_request(ch, method, props, body):
    message = json.loads(body)
    user_id = message['data'].get('userId')

    operation = message['pattern']['cmd']

    if operation == 'ingest':
        response = ingest(config['source_website'], config['website_repo'], website_generated_path, website_source_path)
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
    print(f"Response sent for correlation_id: {props.correlation_id}")
    print(f"Response sent to: {props.reply_to}")
    print(f"response: {response}")


# Check if the vector database exists
if os.path.exists(vectordb_path+"/index.pkl"):
    print(f"The file vector database is present")
else:
    # ingest data
    def_ingest.clone_and_generate(config['website_repo'], website_generated_path, website_source_path)
    def_ingest.mainapp(config['source_website'])

qa_chain = ai_utils.setup_chain(vectordb_path)

channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue=config['rabbitmqrequestqueue'], on_message_callback=on_request)

print("Waiting for RPC requests")
channel.start_consuming()
