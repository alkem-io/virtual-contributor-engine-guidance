from langchain.prompts.prompt import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
# Import Azure OpenAI
from langchain.llms import AzureOpenAI

import os


# Set Context for response
TEMPLATE = """Act as a product expert. Your role is to answer any Alkemio related questions. Return your response in markdown, so you can highlight important elements. If the answer cannot be found within the context, write 'I could not find an answer' 

Use the following context from the Alkemio website to answer the query. Make sure to read all the context before providing an answer.\nContext:\n{context}\nQuestion: {question}
"""

QA_PROMPT = PromptTemplate(template=TEMPLATE, input_variables=["question", "context"])


def setup_chain():
    llm = ChatOpenAI(
        temperature=0.5,
        openai_api_key=os.environ["OPENAI_API_KEY"],
        model_name="gpt-3.5-turbo",
    )
#    llm = AzureOpenAI(deployment_name="deploy-gpt-35-turbo",model_name="gpt-35-turbo", temperature=0.9)
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
#    embeddings = OpenAIEmbeddings(deployment="embedding", chunk_size=1)
    vectorstore = FAISS.load_local("local_index", embeddings)

    chain = ConversationalRetrievalChain.from_llm(
        llm, vectorstore.as_retriever(), return_source_documents=True
    )
    print("\n\nchain:\n",chain)

    return chain
