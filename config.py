import os
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
website_source_path = local_path + '/website/source'
website_generated_path = local_path + '/website/generated'
vectordb_path = local_path + "/vectordb"
generate_website = True
