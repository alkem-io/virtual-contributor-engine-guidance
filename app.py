import os
import pika
import json
import ai_utils
import def_ingest

rabbitmqhost = os.environ['RABBITMQ_HOST']
rabbitmqrequestqueue = "alkemio-chatbot-request"
rabbitmqresponsequeue = "alkemio-chatbot-response"

chat_history = []
docs = []
chain = ai_utils.setup_chain()

# Establish a connection to RabbitMQ
connection = pika.BlockingConnection(pika.ConnectionParameters(rabbitmqhost))
channel = connection.channel()

# Declare the queue
channel.queue_declare(queue=rabbitmqrequestqueue)
channel.queue_declare(queue=rabbitmqresponsequeue)

# Define the functions
def query(query):
    print("query was: ", query)
    llm_result = chain(
       {"question": query, "chat_history": chat_history}
    )
    chat_history.append(
            (llm_result["question"], llm_result["answer"])
        )
    return ("[{'question':'" + str(llm_result["question"]) 
            + "'}, {'answer':'" + str(llm_result["answer"]) 
            + "'}, {'sources':'" + str(llm_result["source_documents"])
            + "'}]")

def reset():
     chat_history = []
     docs = []
     return "Reset function executed"

def ingest():
    def_ingest.mainapp()
    return "Ingest function executed"

def on_request(ch, method, props, body):
    message = json.loads(body)

    if message['operation'] == 'query':
        response = query(message['param'])
    elif message['operation'] == 'reset':
        response = reset()
    elif message['operation'] == 'ingest':
        response = ingest()
    else:
        response = "Unknown function"

    ch.basic_publish(exchange='',
                     routing_key=props.reply_to,
                     properties=pika.BasicProperties(correlation_id = props.correlation_id),
                     body=json.dumps({"operation": "feedback", "result": str(response)}))
    ch.basic_ack(delivery_tag=method.delivery_tag)
    print("body: ",json.dumps({"operation": "feedback", "result": str(response)}))

# ensure the data is ingested at least once
def_ingest.mainapp()

print("setup chain")
ai_utils.setup_chain()

channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue=rabbitmqrequestqueue, on_message_callback=on_request)

print("Waiting for RPC requests")
channel.start_consuming()
