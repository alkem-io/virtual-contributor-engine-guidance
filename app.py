import os
import pika
import json
import ai_utils
import def_ingest
from dotenv import load_dotenv

load_dotenv()

config = {
    "rabbitmq_host": os.getenv('RABBITMQ_HOST'),
    "rabbitmq_user": os.getenv('RABBITMQ_USER'),
    "rabbitmq_password": os.getenv('RABBITMQ_PASSWORD'),
    "rabbitmqrequestqueue": "alkemio-chat-guidance",
}

user_data = {}

credentials = pika.PlainCredentials(config['rabbitmq_user'],
                                    config['rabbitmq_password'])
parameters = pika.ConnectionParameters(host=config['rabbitmq_host'],
                                       credentials=credentials)
connection = pika.BlockingConnection(parameters)
channel = connection.channel()

channel.queue_declare(queue=config['rabbitmqrequestqueue'])

def query(user_id, query):
    print(f"Query from user {user_id}: {query}")

    if user_id not in user_data:
        reset(user_id)

    chat_history = user_data[user_id]['chat_history']

    llm_result = ai_utils.setup_chain()(
        {"question": query, "chat_history": chat_history}
    )

    chat_history.append((llm_result["question"], llm_result["answer"]))

    response = json.dumps({
        "question": str(llm_result["question"]),
        "answer": str(llm_result["answer"]),
        "sources": str(llm_result["source_documents"])
    }
    )

    return response

def reset(user_id):
    user_data[user_id] = {
        'chat_history': [],
        'docs': []
    }

    return "Reset function executed"

def ingest():
    def_ingest.mainapp()
    return "Ingest function executed"

def on_request(ch, method, props, body):
    message = json.loads(body)
    user_id = message['data'].get('userId')

    operation = message['pattern']['cmd']

    if operation == 'ingest':
        response = ingest()
    else:
        if user_id is None:
            response = "userId not provided"
        else:
            if operation == 'query':
                if 'question' in message['data']:
                    response = query(user_id, message['data']['question'])
                else:
                    response = "Query parameter not provided"
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


def_ingest.mainapp()

channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue=config['rabbitmqrequestqueue'], on_message_callback=on_request)

print("Waiting for RPC requests")
channel.start_consuming()
