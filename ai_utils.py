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
TEMPLATE = """
- Act as a product and innovation expert.
- Your task is to answer user questions. 
- Return your response in markdown, and highlight important elements.
- If the answer cannot be found within the context, write 'I could not find an answer to your question'.
- Provide concise replies that are polite and professional. 
- Use the following context to answer the query. 

Context:
{context}

Question:
{question}
"""

QA_PROMPT = PromptTemplate(template=TEMPLATE, input_variables=["question", "context"])


def setup_chain():
    # llm = ChatOpenAI(
    #     temperature=0.6,
    #     openai_api_key=os.environ["OPENAI_API_KEY"],
    #     model_name="gpt-3.5-turbo",
    # )
    llm = AzureOpenAI(deployment_name="deploy-gpt-35-turbo",model_name="gpt-35-turbo", temperature=0.9)
    # embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    embeddings = OpenAIEmbeddings(deployment="embedding", chunk_size=1)
    vectorstore = FAISS.load_local("local_index", embeddings)

    chain = ConversationalRetrievalChain.from_llm(
        llm, vectorstore.as_retriever(), return_source_documents=True
    )
    print("\n\nchain:\n",chain)

    return chain
