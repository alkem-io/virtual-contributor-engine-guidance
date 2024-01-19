from langchain_openai import AzureOpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import AzureOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.schema import format_document
from langchain_core.messages import get_buffer_string
from langchain_core.messages.ai import AIMessage
from langchain_core.runnables import RunnableBranch

from operator import itemgetter
import logging
import sys
import io
import def_ingest
from config import config, website_source_path, website_generated_path, website_source_path2, website_generated_path2, vectordb_path, local_path, LOG_LEVEL, max_token_limit

import os

# configure logging
logger = logging.getLogger(__name__)
assert LOG_LEVEL in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
logger.setLevel(getattr(logging, LOG_LEVEL))  # Set logger level


# Create handlers
c_handler = logging.StreamHandler(io.TextIOWrapper(sys.stdout.buffer, line_buffering=True))
f_handler = logging.FileHandler(os.path.join(os.path.expanduser(local_path), 'app.log'))

c_handler.setLevel(level=getattr(logging, LOG_LEVEL))
f_handler.setLevel(logging.WARNING)

# Create formatters and add them to handlers
c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', '%m-%d %H:%M:%S')
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', '%m-%d %H:%M:%S')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)

logger.info(f"log level {os.path.basename(__file__)}: {LOG_LEVEL}")

# verbose output for LLMs
if LOG_LEVEL == "DEBUG":
    verbose_models = True
else:
    verbose_models = False

# define internal configuration parameters

# does chain return the source documents?
return_source_documents = True


# Define a dictionary containing country codes as keys and related languages as values
language_mapping = {
    'EN': 'English',
    'US': 'English',
    'UK': 'English',
    'FR': 'French',
    'DE': 'German',
    'ES': 'Spanish',
    'NL': 'Dutch',
    'BG': 'Bulgarian',
    'UA': "Ukranian"
}

# function to retrieve language from country
def get_language_by_code(language_code):
    """Returns the language associated with the given code. If no match is found, it returns 'English'."""
    return language_mapping.get(language_code, 'English')


chat_system_template = """
You are a friendly and talkative conversational agent, tasked with answering questions about Alkemio.
Use the following step-by-step instructions to respond to user inputs:

1 - If the question is in a different language than English, translate the question to English before answering.
2 - The text provided in the context delimited by triple pluses is retrieved from the Alkemio website is not part of the conversation with the user.
3 - Provide an answer of 250 words or less that is professional, engaging, accurate and exthausive, based on the context delimited by triple pluses. \
If the answer cannot be found within the context, write 'Hmm, I am not sure'.
4 - Only return the answer from step 3, do not show any code or additional information.
5 - Answer the question in the {language} language.
+++
context:
{context}
+++
"""

condense_question_template = """"
Create a single sentence standalone query based on the human input, using the following step-by-step instructions:

1. If the human input is expressing a sentiment, delete and ignore the chat history delimited by triple pluses. \
Then, return the human input containing the sentiment as the standalone query. Do NOT in any way respond to the human input, \
simply repeat it.
2. Otherwise, combine the chat history delimited by triple pluses and human input into a single standalone query that does \
justice to the human input.
3. Do only return the standalone query, do not return any other information. Never return the chat history delimited by triple pluses.

+++
chat history:
{chat_history}
+++

Human input: {question}
---
Standalone query:
"""


condense_question_prompt = PromptTemplate.from_template(condense_question_template)

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", chat_system_template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


generic_llm = AzureOpenAI(azure_deployment=os.environ["LLM_DEPLOYMENT_NAME"],
                          temperature=0, verbose=verbose_models)

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=config['embeddings_deployment_name'],
    openai_api_version=config['openai_api_version'],
    chunk_size=1
)

def load_vector_db():
    """
    Purpose:
        Load the data into the vector database.
    Args:

    Returns:
        vectorstore: the vectorstore object
    """
    # Check if the vector database exists
    if os.path.exists(vectordb_path + os.sep + "index.pkl"):
        logger.info(f"The file vector database is present")
    else:
        logger.info(f"The file vector database is not present, ingesting")
        def_ingest.ingest(
            config['source_website'],
            config['website_repo'],
            website_generated_path,
            website_source_path,
            config['source_website2'],
            config['website_repo2'],
            website_generated_path2,
            website_source_path2)

    return FAISS.load_local(vectordb_path, embeddings)


vectorstore = load_vector_db()

retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": .5})

chat_llm = AzureChatOpenAI(azure_deployment=os.environ["LLM_DEPLOYMENT_NAME"],
                           temperature=os.environ["AI_MODEL_TEMPERATURE"],
                           max_tokens=max_token_limit, verbose=verbose_models)

condense_llm = AzureChatOpenAI(azure_deployment=os.environ["LLM_DEPLOYMENT_NAME"],
                               temperature=0,
                               verbose=verbose_models)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

async def query_chain(question, language, chat_history):

    # check whether the chat history is empty
    if chat_history.buffer == []:
        first_call = True
    else:
        first_call = False

    # add first_call to the question
    question.update({"first_call": first_call})

    logger.info(f"first call: {first_call}\n")
    logger.debug(f"chat history: {chat_history.buffer}\n")

    # First we add a step to load memory
    # This adds a "memory" key to the input object
    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(chat_history.load_memory_variables) | itemgetter("history"),
    )

    logger.debug(f"loaded memory {loaded_memory}\n")
    logger.debug(f"chat history {chat_history}\n")


    # Now we calculate the standalone question if the chat_history is not empty
    standalone_question = {
        "standalone_question": {
            "question": lambda x: x["question"],
            "chat_history": lambda x: get_buffer_string(x["chat_history"]),
        }
        | condense_question_prompt
        | condense_llm
        | StrOutputParser(),
    }

    # pass the question directly on the first call in a chat sequence of the chatbot
    direct_question = {
        "question": lambda x: x["question"],
    }
    # Now we retrieve the documents
    # in case it is the first call (chat history empty)
    retrieved_documents = {
        "docs": itemgetter("question") | retriever,
        "question": lambda x: x["question"],
    }
    # or when the chat history is not empty, rephrase the question taking into account the chat history
    retrieved_documents_sa = {
        "docs": itemgetter("standalone_question") | retriever,
        "question": lambda x: x["standalone_question"],
    }

    # Now we construct the inputs for the final prompt
    final_inputs = {
        "context": lambda x: _combine_documents(x["docs"]),
        "chat_history" : lambda x: chat_history.buffer,
        "question": itemgetter("question"),
        "language": lambda x: language['language'],
    }

    # And finally, we do the part that returns the answers
    answer = {
        "answer": final_inputs | chat_prompt | chat_llm,
        "docs": itemgetter("docs"),
    }

    # And now we put it all together in a 'RunnableBranch', so we only invoke the rephrasing part when the chat history is not empty
    final_chain = RunnableBranch(
        (lambda x: x["first_call"], loaded_memory | direct_question | retrieved_documents | answer),
        loaded_memory | standalone_question | retrieved_documents_sa | answer,
    )

    try:
        logger.debug(f"final chain {final_chain}\n")
        result = await final_chain.ainvoke(question)
    except Exception as e:
        logger.error(f"An error occurred while generating a response: {str(e)}")
        # Handle the error appropriately here
        return {'answer': AIMessage(content='An error occurred while generating a response.'), 'source_documents': []}
    else:
        return {'answer': result['answer'], 'source_documents': result['docs'] if result['docs'] else []}
