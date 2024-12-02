# import os
# from dotenv import load_dotenv

# load_dotenv()

# config = {
#     "llm_deployment_name": os.getenv("LLM_DEPLOYMENT_NAME"),
#     "embeddings_deployment_name": os.getenv("EMBEDDINGS_DEPLOYMENT_NAME"),
#     "openai_api_version": os.getenv("OPENAI_API_VERSION"),
#     "source_website": os.getenv("AI_SOURCE_WEBSITE"),
#     "website_repo": os.getenv("AI_WEBSITE_REPO"),
#     "source_website2": os.getenv("AI_SOURCE_WEBSITE2"),
#     "website_repo2": os.getenv("AI_WEBSITE_REPO2"),
#     "github_user": os.getenv("AI_GITHUB_USER"),
#     "github_pat": os.getenv("AI_GITHUB_PAT"),
#     "local_path": os.getenv("AI_LOCAL_PATH"),
# }

# local_path = config["local_path"]
# github_user = config["github_user"]
# github_pat = config["github_pat"]
# website_source_path = local_path + os.sep + "website" + os.sep + "source"
# website_source_path2 = local_path + os.sep + "website2" + os.sep + "source"
# website_generated_path = local_path + os.sep + "website" + os.sep + "generated"
# website_generated_path2 = local_path + os.sep + "website2" + os.sep + "generated"
# vectordb_path = local_path + os.sep + "vectordb"

# chunk_size = 3000
# # token limit for for the completion of the chat model, this does not include the overall context length
# max_token_limit = 2000

# LOG_LEVEL = "INFO"  # Possible values: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'


import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Env:
    model_name: str
    embeddings_model_name: str
    openai_api_version: str
    site_url: str
    welcome_site_url: str
    site_repo: str
    welcome_site_repo: str
    github_user: str
    github_pat: str
    local_path: str
    vector_db_path: str
    chunk_size: int
    token_limit: int
    repos_path: str
    model_temperature: float
    log_level: str
    verbose: bool
    history_length: int
    mistral_endpoint: str
    mistral_key: str

    def __init__(self):

        # Required configurations
        required_vars = []

        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )

        self.rabbitmq_host = os.getenv("RABBITMQ_HOST", "")
        self.rabbitmq_user = os.getenv("RABBITMQ_USER", "")
        self.rabbitmq_password = os.getenv("RABBITMQ_PASSWORD", "")
        self.rabbitmq_input_queue = os.getenv("RABBITMQ_QUEUE", "")
        self.rabbitmq_result_queue = os.getenv("RABBITMQ_RESULT_QUEUE", "")
        self.rabbitmq_exchange = os.getenv("RABBITMQ_EVENT_BUS_EXCHANGE", "")
        self.rabbitmq_result_routing_key = os.getenv("RABBITMQ_RESULT_ROUTING_KEY", "")

        self.model_name = os.getenv("LLM_DEPLOYMENT_NAME", "")
        self.embeddings_model_name = os.getenv("EMBEDDINGS_DEPLOYMENT_NAME", "")
        self.openai_api_version = os.getenv("OPENAI_API_VERSION", "")

        self.site_url = os.getenv("AI_SOURCE_WEBSITE", "")
        self.welcome_site_url = os.getenv("AI_SOURCE_WEBSITE2", "")

        self.site_repo = os.getenv("AI_WEBSITE_REPO", "")
        self.welcome_site_repo = os.getenv("AI_WEBSITE_REPO2", "")

        self.github_user = os.getenv("AI_GITHUB_USER", "")
        self.github_pat = os.getenv("AI_GITHUB_PAT", "")
        self.local_path = os.getenv("AI_LOCAL_PATH", "")
        self.vector_db_path = os.path.join(self.local_path, "vectordb")
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "3000"))
        self.token_limit = int(os.getenv("TOKEN_LIMIT", "2000"))
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.verbose = self.log_level == "DEBUG"

        self.model_temperature = float(os.getenv("AI_MODEL_TEMPERATURE", "0.4"))
        self.history_length = int(os.getenv("HISTORY_LEGNTH", "20"))

        self.repos_path = os.path.join(self.local_path, "repos", ".")

        self.mistral_endpoint = os.getenv("AZURE_MISTRAL_ENDPOINT", "")
        self.mistral_key = os.getenv("AZURE_MISTRAL_API_KEY", "")

        Path(self.repos_path).mkdir(parents=True, exist_ok=True)


env = Env()
