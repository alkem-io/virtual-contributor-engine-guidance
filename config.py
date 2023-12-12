import os
from dotenv import load_dotenv
load_dotenv()

config = {
    "llm_deployment_name": os.getenv('LLM_DEPLOYMENT_NAME'),
    "embeddings_deployment_name": os.getenv('EMBEDDINGS_DEPLOYMENT_NAME'),
    "openai_api_version": os.getenv('OPENAI_API_VERSION'),
    "rabbitmq_host": os.getenv('RABBITMQ_HOST'),
    "rabbitmq_user": os.getenv('RABBITMQ_USER'),
    "rabbitmq_password": os.getenv('RABBITMQ_PASSWORD'),
    "rabbitmqrequestqueue": "alkemio-chat-guidance",
    "source_website": os.getenv('AI_SOURCE_WEBSITE'),
    "website_repo": os.getenv('AI_WEBSITE_REPO'),
    "source_website2": os.getenv('AI_SOURCE_WEBSITE2'),
    "website_repo2": os.getenv('AI_WEBSITE_REPO2'),
    "github_user": os.getenv('AI_GITHUB_USER'),
    "github_pat": os.getenv('AI_GITHUB_PAT'),
    "local_path": os.getenv('AI_LOCAL_PATH')
}

local_path = config['local_path']
github_user = config['github_user']
github_pat = config['github_pat']
website_source_path = local_path + '/website/source'
website_source_path2 = local_path + '/website2/source'
website_generated_path = local_path + '/website/generated'
website_generated_path2 = local_path + '/website2/generated'
vectordb_path = local_path + "/vectordb"
generate_website = True

LOG_LEVEL = 'DEBUG'  # Possible values: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
