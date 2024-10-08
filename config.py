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
    "rabbitmqrequestqueue": "virtual-contributor-engine-guidance",
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
website_source_path = local_path + os.sep + 'website' + os.sep + 'source'
website_source_path2 = local_path + os.sep + 'website2' + os.sep + 'source'
website_generated_path = local_path + os.sep + 'website' + os.sep + 'generated'
website_generated_path2 = local_path + os.sep + 'website2' + os.sep + 'generated'
vectordb_path = local_path + os.sep + 'vectordb'

chunk_size = 3000
# token limit for for the completion of the chat model, this does not include the overall context length
max_token_limit = 2000

LOG_LEVEL = 'INFO'  # Possible values: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
