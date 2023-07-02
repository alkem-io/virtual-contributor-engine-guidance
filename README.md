# Proof of Concept Alkemio Gen AI driven Chatbot

## Introduction
The purpose of this proof of concept is to assess what is required to create a versatile, reliable and intuitive chatbot for users to engage on Alkemio related topics. The project is not deployable as is, but should serve as valuable input for showing generative AI capabilities and help assessing what is required to embed this functionality in the platform.

## Approach
Large Language Models (LLMs), have significantly improved over the recent period and are not ubiquitous and performant. This opens a lot of possibilities for their usage in different areas. [OpenAI](https://openai.com) is the best known commercial provider of LLMs, but there are ample choices for LLM models, either commercial or open source. Whilst this provides options, it also creates the risk of provider lock-in. 


LLMs are just one component required for the practical implementation off generative AI solutions, and many other 'building blocks' are necessary too. [Langchain](https://langchain.com/) is a popular open source library that provides these building blocks and creates an abstraction layer, creating provider independance.


Training a LLM is prohibitatively expensive for most organisations, but for most practical implementations there is a need to incorporate organisation specific data. A common approach is to add specific context to a user question to the prompt that is submitted to the LLM. This poses a challenge, as LLMs generally only allow prompts with a finite size (typically around 4k tokens). Therefore it is important that the relevant contextual information is provided and for the that following needs to be done:

 - Data Collection
 - Creating Text Embeddings
 - Prompt Engineering
 - Creating the Chat Interface

 This project has been inspired by many articles, but theoretical and practical. A significant part of the code base comes from the [Building an AWS Well-Architected Chatbot with LangChain](https://dev.to/aws/building-an-aws-well-architected-chatbot-with-langchain-13cd) project.

## Implementation

The projects has been implemented as a container based micro-service with a RESTful API. The API has three endpoints:
- ingest: data collection from the Alkemio foundation website and embedding using the [OpenAI Ada text model](https://openai.com/blog/new-and-improved-embedding-model).
- reset: reset the chat history for the ongoing chat
- query: post the next question in a chat sequence..

The first two endpoints are implemented as GET HTML request and the query endpoint is implemeted as a POST HTML request.

### example query
An example query looks like this, 
`curl --header "Content-Type: application/json"  --request POST  --data '{"query":"Who are the co-founders of alkemio?"}'  http://localhost:5000/query`

### Docker 
The following command can be used to build the container from the Docker CLI:
`docker build -t genai-api . `

The following command can be used to start the container from the Docker CLI:
`docker run --name genai-api -v /dev/shm:/dev/shm -p 5000:5000 -e "OPENAI_API_KEY=$OPENAI_API_KEY" genai-api `

### Python
The required Python packages are listed in the `requirements.txt` file.

### Linux
The project required Python 3.10 as a minimum and the chromium driver is required for scraing of the Alkemio website:
install Chromium-driver: `sudo apt-get install chromium-driver`

### Docker
`docker run --name genai-api -v /dev/shm:/dev/shm -p 5000:5000 -e "OPENAI_API_KEY=$OPENAI_API_KEY" genai-api`

## Outstanding tasks
This Proof of Concept is functional, but morework is required to make it production ready, including:
- allow for multiple users at the same time, each having their own chat history.
- run on a production [WSGI server](https://flask.palletsprojects.com/en/2.2.x/deploying/).
- make it deployable on Kubernetes.
- improve the LLM performance (e.g. chunck sizes, LLM parameters, prompt template).
- improve security and error handling.
- improve the comments in the code and code optimsation.
- add additional configuration options (e.g. eas yswitch between OpenAI and Azure OpenAI, target website)