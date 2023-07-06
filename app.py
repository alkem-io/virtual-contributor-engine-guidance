import os
import pika
import json
import ai_utils
import def_ingest

rabbitmqhost = os.environ['RABBITMQ_HOST']
rabbitmqrequestqueue = "alkemio-chatbot-request"
rabbitmqresponsequeue = "alkemio-chatbot-response"

# Dictionary to store chat history and documents for each user
user_data = {}

# Establish a connection to RabbitMQ
connection = pika.BlockingConnection(pika.ConnectionParameters(rabbitmqhost))
channel = connection.channel()

# Declare the queues
channel.queue_declare(queue=rabbitmqrequestqueue)
channel.queue_declare(queue=rabbitmqresponsequeue)

# Define the functions
def query(user_id, query):
    print("Query from user", user_id, ": ", query)
    
    # Retrieve or initialize user data
    if user_id not in user_data:
        user_data[user_id] = {
            'chat_history': [],
            'docs': []
        }
    
    chat_history = user_data[user_id]['chat_history']
    
    llm_result = ai_utils.setup_chain()(
        {"question": query, "chat_history": chat_history}
    )
    
    chat_history.append((llm_result["question"], llm_result["answer"]))
    
    
    response = ("[{'question':'" + str(llm_result["question"]) 
            + "'}, {'answer':'" + str(llm_result["answer"]) 
            + "'}, {'sources':'" + str(llm_result["source_documents"])
            + "'}]")

    return response

def reset(user_id):
    if user_id in user_data:
        user_data[user_id]['chat_history'] = []
        user_data[user_id]['docs'] = []
    
    return "Reset function executed"

def ingest():
    def_ingest.mainapp()
    return "Ingest function executed"

def on_request(ch, method, props, body):
    message = json.loads(body)
    user_id = props.correlation_id
    
    if user_id is None:
        response = "Correlation ID not provided"
    else:
        if message['operation'] == 'query':
            if 'param' in message:
                response = query(user_id, message['param'])
            else:
                response = "Query parameter not provided"
        elif message['operation'] == 'reset':
            response = reset(user_id)
        elif message['operation'] == 'ingest':
            response = ingest()
        else:
            response = "Unknown function"

    ch.basic_publish(
        exchange='',
        routing_key=props.reply_to,
        properties=pika.BasicProperties(correlation_id=user_id),
        body=json.dumps({"operation": "feedback", "result": response})
    )
    
    ch.basic_ack(delivery_tag=method.delivery_tag)
    print("Response sent for correlation_id:", user_id)

# Ensure the data is ingested at least once
def_ingest.mainapp()

channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue=rabbitmqrequestqueue, on_message_callback=on_request)

print("Waiting for RPC requests")
channel.start_consuming()
