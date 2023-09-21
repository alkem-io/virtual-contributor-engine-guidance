from langchain.callbacks import get_openai_callback
import os
import pika
import json
import ai_utils
from config import config, website_source_path, website_generated_path, vectordb_path, generate_website
from langchain.memory import ConversationSummaryMemory, ChatMessageHistory, ConversationSummaryBufferMemory
from langchain.llms import AzureOpenAI

user_data = {}
user_history = {}
user_chain = {}

credentials = pika.PlainCredentials(config['rabbitmq_user'],
                                    config['rabbitmq_password'])
parameters = pika.ConnectionParameters(host=config['rabbitmq_host'],
                                       credentials=credentials)
print(f"\About to connect to RabbitMQ with params {config['rabbitmq_user']}: {config['rabbitmq_host']}\n")
connection = pika.BlockingConnection(parameters)
channel = connection.channel()

channel.queue_declare(queue=config['rabbitmqrequestqueue'])


# define memory
memory_llm=AzureOpenAI(deployment_name=os.environ["AI_DEPLOYMENT_NAME"], model_name=os.environ["AI_MODEL_NAME"],
                                temperature=0, verbose=True)


def query(user_id, query, language_code):
    print(f"\nQuery from user {user_id}: {query}\n")

    if user_id not in user_data:
        reset(user_id)
        chat_history=[]
        summary=""
    else:
        summary=user_data[user_id]['memory'].predict_new_summary(user_data[user_id]['memory'].chat_memory.messages, existing_summary = user_data[user_id]['summary'])
        print(f"\nnew summary: {summary}\n\n")
        user_data[user_id]['summary'] = summary

    user_data[user_id]['language'] = ai_utils.get_language_by_code(language_code)

    print(f"\nlanguage: {user_data[user_id]['language']}\n")
    chat_history = user_data[user_id]['chat_history']

    with get_openai_callback() as cb:
        llm_result = user_chain[user_id]({"question": query, "chat_history": chat_history, "history": summary})
        translation = llm_result['answer']

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
    
    #user_history[user_id].add_user_message(query)
    #user_history[user_id].add_ai_message(llm_result['answer'])
    user_data[user_id]['memory'].save_context({"input": query}, {"output": llm_result['answer']})

    # only keep the last 3 entries of that chat history to avoid exceeding the token limit.
    user_data[user_id]['chat_history'] = user_data[user_id]['chat_history'][-3:]

    print(f"new chat history {user_data[user_id]['chat_history']}")
    response = json.dumps({
        "question": str(llm_result["question"]), "answer": str(translation), "sources": str(llm_result["source_documents"]), "prompt_tokens": cb.prompt_tokens, "completion_tokens": cb.completion_tokens, "total_tokens": cb.total_tokens, "total_cost": cb.total_cost
    }
    )

    return response

def reset(user_id):
    #user_history[user_id] = ChatMessageHistory()
    #user_history[user_id].clear()
    user_data[user_id] = {
        'chat_history': [],
        'memory': ConversationSummaryMemory(llm=memory_llm, memory_key="chat_history", return_messages = False),
        'summary': ""
    }
    user_chain[user_id]=ai_utils.setup_chain(user_data[user_id]['memory'])
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


channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue=config['rabbitmqrequestqueue'], on_message_callback=on_request)

print("Waiting for RPC requests")
channel.start_consuming()
