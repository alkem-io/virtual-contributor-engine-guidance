from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from pydantic import SecretStr
from config import env
from langchain_mistralai.chat_models import ChatMistralAI

from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential

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
    return message


embeddings = AzureOpenAIEmbeddings(
    azure_deployment=env.embeddings_model_name, chunk_size=1
)


def get_retriever():
    vector_store = FAISS.load_local(
        env.vector_db_path, embeddings, allow_dangerous_deserialization=True
    )
    return vector_store.as_retriever(
        search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5}
    )
