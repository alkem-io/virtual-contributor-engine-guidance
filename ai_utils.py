from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import AzureOpenAI
from langchain.prompts import PromptTemplate
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
import def_ingest
from config import config, website_source_path, website_generated_path, vectordb_path, generate_website

import os

# define internal configuration parameters
# token limit for retrieval chain
max_token_limit = 2000
# verbose output for LLMs
verbose_models = True
# doews chain return the source documents?
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


chat_template = """
You are a conversational agent. Use the following step-by-step instructions to respond to user inputs.
1 - The text provided in the context delimited by triple pluses may contain questions. Remove those questions from the context. 
2 - Provide a single paragragh answer that is polite and professional taking into account the context and the chat history, both delimited by triple pluses. If the answer cannot be found within the context, write 'I could not find an answer to your question'.
+++
Context:
{context}
+++
Chat history:
{chat_history}
+++
Question: {question}
"""

custom_question_template = """"
Combine the chat history and follow up question into a standalone question. 
+++
Chat History: {chat_history}
+++
Follow up question: {question}
+++
Standalone question:
"""

translate_template = """"
Act as a professional translator. Use the following step-by-step instructions:
1: assess in what language input below delimited by triple pluses is written.
2. carry out one of tasks A or B below:
A: if the input language is different from {language} then translate the input below delimited by triple pluses to natural {language} language, maintaining tone of voice and length
B: if the input language is the same as {language} there is no need for translation, simply return the original input below delimited by triple pluses as the answer.
3. Only return the answer from step 2, do not show any code or additional information.
+++
input:
{answer}
+++
Translated input:
"""

custom_question_prompt = PromptTemplate(
    template=custom_question_template, input_variables=["chat_history", "question"]
)

translation_prompt = PromptTemplate(
    template=translate_template, input_variables=["language", "answer"]
)

# prompt to be used by retrieval chain, note this is the default prompt name, so nowhere assigned
QA_PROMPT = PromptTemplate(
    template=chat_template, input_variables=["question", "context", "chat_history"]
)

generic_llm = AzureOpenAI(deployment_name=os.environ["AI_DEPLOYMENT_NAME"], model_name=os.environ["AI_MODEL_NAME"],
                            temperature=0, verbose=verbose_models)

question_generator = LLMChain(llm=generic_llm, prompt=custom_question_prompt, verbose=verbose_models)

embeddings = OpenAIEmbeddings(deployment=os.environ["AI_EMBEDDINGS_DEPLOYMENT_NAME"], chunk_size=1)

# Check if the vector database exists
if os.path.exists(vectordb_path+"/index.pkl"):
    print(f"The file vector database is present")
else:
    # ingest data
    if generate_website:
        def_ingest.clone_and_generate(config['website_repo'], website_generated_path, website_source_path)
    def_ingest.mainapp(config['source_website'])

vectorstore = FAISS.load_local(vectordb_path, embeddings)
retriever = vectorstore.as_retriever()

chat_llm = AzureChatOpenAI(deployment_name=os.environ["AI_DEPLOYMENT_NAME"],
                            model_name=os.environ["AI_MODEL_NAME"], temperature=os.environ["AI_MODEL_TEMPERATURE"],
                            max_tokens=max_token_limit)

doc_chain = load_qa_chain(generic_llm, chain_type="stuff", prompt=QA_PROMPT, verbose=verbose_models)

def translate_answer(answer, language):
    translate_llm = AzureOpenAI(deployment_name=os.environ["AI_DEPLOYMENT_NAME"], model_name=os.environ["AI_MODEL_NAME"],
                                temperature=0, verbose=verbose_models)
    prompt = translation_prompt.format(answer=answer, language=language)
    return translate_llm(prompt)


def setup_chain(user_memory):

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=chat_llm,
        retriever=retriever,
        condense_question_prompt=custom_question_prompt,
        chain_type="stuff",
        verbose=verbose_models,
        condense_question_llm=generic_llm,
        return_source_documents=return_source_documents,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT}
    )


    #conversation_chain = ConversationalRetrievalChain(retriever=retriever,
    #                                                combine_docs_chain=doc_chain,
    #                                                    question_generator=question_generator,
    #                                                    max_tokens_limit=max_token_limit,
    #                                                    verbose = True,
    #                                                    memory=user_memory,
    #                                                    return_source_documents=return_source_documents,
    #                                                    return_generated_question=False,
    #                                                    )

    return conversation_chain
