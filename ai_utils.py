from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import AzureOpenAI
from langchain.prompts import PromptTemplate
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT

import os

# define internal configuration parameters
# token limit for retrieval chain
max_token_limit = 2000
# verbose output for LLMs
verbose_models = True
# doews chain return the source documents?
return_source_document=True


# Define a dictionary containing country codes as keys and related languages as values
language_mapping = {
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


chat_template = """
You are a conversation agent and your task is to answer the question below based on the context and taking into account the following instructions.
---
Instructions:
- Return your response in markdown, and highlight important elements.
- If the answer cannot be found within the context, write 'I could not find an answer to your question'.
- Provide a single answer only that is polite and professional and do not answer questions that are part of the context provided.
- Translate your asnwer to {language}.
---
Context:
{context}
---
Question: {question}
"""

custom_question_template = """"
 Combine the chat history and follow up question into a standalone question. 
 ---
 Chat History: {chat_history}
 ---
Follow up question: {question}
Standalone question:
"""

custom_question_prompt = PromptTemplate(
    template=custom_question_template, input_variables=["chat_history", "question"]
)

# prompt to be used by retrieval chain, note this is the default prompt name, so nowhere assigned
QA_PROMPT = PromptTemplate(
    template=chat_template, input_variables=["question", "context", "language"]
)

generic_llm = AzureOpenAI(deployment_name=os.environ["AI_DEPLOYMENT_NAME"], model_name=os.environ["AI_MODEL_NAME"], temperature=0, verbose=verbose_models)
question_generator = LLMChain(llm=generic_llm, prompt=custom_question_prompt, verbose=verbose_models )

embeddings = OpenAIEmbeddings(deployment=os.environ["AI_EMBEDDINGS_DEPLOYMENT_NAME"], chunk_size=1)
vectorstore = FAISS.load_local("local_index", embeddings)
retriever = vectorstore.as_retriever()

chat_llm= AzureChatOpenAI(deployment_name=os.environ["AI_DEPLOYMENT_NAME"], model_name=os.environ["AI_MODEL_NAME"], temperature=os.environ["AI_MODEL_TEMPERATURE"])

doc_chain = load_qa_chain(generic_llm, chain_type="stuff", prompt=QA_PROMPT, verbose=verbose_models)

conversation_chain = ConversationalRetrievalChain(retriever=retriever,
                                                   combine_docs_chain=doc_chain,
                                                    question_generator=question_generator,
                                                    max_tokens_limit=max_token_limit,
                                                    verbose = True,
                                                    return_source_documents=return_source_document,
                                                    return_generated_question=True
                                                    )


  

def qa_chain(query, chat_history, language):

    llm_result=conversation_chain({"question": query, "chat_history": chat_history, "language": language})

    return llm_result
