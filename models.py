from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from config import env
from chromadb.utils.embedding_functions.openai_embedding_function import (
    OpenAIEmbeddingFunction,
)


from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from logger import setup_logger

logger = setup_logger(__name__)

llm = ChatCompletionsClient(
    endpoint=env.mistral_endpoint,
    credential=AzureKeyCredential(env.mistral_key),
)


def invoke_model(messages, temperature=None):
    if temperature is None:
        temperature = env.model_temperature

    result = llm.complete(
        messages=messages,
        temperature=temperature,
        top_p=1,
        stream=False,
    )
    message = str(result["choices"][0]["message"]["content"])

    logger.debug(message)

    return message


embeddings = AzureOpenAIEmbeddings(
    azure_deployment=env.embeddings_model_name, chunk_size=1
)

embed_func = OpenAIEmbeddingFunction(
    api_key=env.openai_api_key,
    api_base=env.openai_endpoint,
    api_type="azure",
    api_version=env.openai_api_version,
    model_name=env.embeddings_model_name,
)


def get_retriever():
    vector_store = FAISS.load_local(
        env.vector_db_path, embeddings, allow_dangerous_deserialization=True
    )
    return vector_store.as_retriever(
        search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5}
    )
