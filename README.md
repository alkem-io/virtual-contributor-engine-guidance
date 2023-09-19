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

The projects has been implemented as a container based micro-service with a RabbitMQ RPC. There is one RabbitMQ queue:
- `alkemio-chat-guidance` - queue for submitting requests to the microservice

The request payload consists of json with the following structure (example for a query):
```
{
    "data": {
        "userId": "userID",
        "question": "What are the key Alkemio concepts?",
        "language": "UK"
    },
    "pattern": {
        "cmd": "query"
    }
}
```

The operation types are:
- `ingest`: data collection from the Alkemio foundation website (through the Github source) and embedding using the [OpenAI Ada text model](https://openai.com/blog/new-and-improved-embedding-model), no *addition request data*.
- `reset`: reset the chat history for the ongoing chat, needs userId.
- `query`: post the next question in a chat sequence, see exmaple

The response is published in an auto-generated, exclusive, unnamed queue.

There is a draft implementation for the interaction language of the model (this needs significant improvement). If no language code is specified, English will be assumed. Choices are:
    'EN': 'English',
    'US': 'English',
    'UK': 'English',
    'FR': 'French',
    'DE': 'German',
    'ES': 'Spanish',
    'NL': 'Dutch',
    'BG': 'Bulgarian',
    'UA': "Ukranian"

*note: there is an earlier (outdated) RESTful implementation available at https://github.com/alkem-io/guidance-engine/tree/http-api

### Docker 
The following command can be used to build the container from the Docker CLI (default architecture is amd64, so `--build-arg ARCHITECTURE=arm64` for amd64 builds):
`docker build --build-arg ARCHITECTURE=arm64 --no-cache -t alkemio/guidance-engine:v0.2.0 .`
`docker build--no-cache -t alkemio/guidance-engine:v0.2.0 .`
The Dockerfile has some self-explanatory configuration arguments.

The following command can be used to start the container from the Docker CLI:
`docker run --name guidance-engine -v /dev/shm:/dev/shm --env-file .env guidance-engine`
where `.env` based on `.azure-template.env`
Alternatively use `docker-compose up -d`.

with:
- `OPENAI_API_KEY`: a valid OpenAI API key
- `OPENAI_API_TYPE`: a valid OpenAI API type. For Azure, the value is `azure`
- `OPENAI_API_VERSION`: a valid Azure OpenAI version. At the moment of writing, latest is `2023-05-15`
- `OPENAI_API_BASE`: a valid Azure OpenAI base URL, e.g. `https://{your-azure-resource-name}.openai.azure.com/`
- `RABBITMQ_HOST`: the RabbitMQ host name
- `RABBITMQ_USER`: the RabbitMQ user
- `RABBITMQ_PASSWORD`: the RabbitMQ password
- `AI_MODEL_TEMPERATURE`: the `temperature` of the model, use value between 0 and 1. 1 means more randomized answer, closer to 0 - a stricter one
- `AI_MODEL_NAME`: the model name in Azure
- `AI_DEPLOYMENT_NAME`: the AI gpt model deployment name in Azure
- `AI_EMBEDDINGS_DEPLOYMENT_NAME`: the AI embeddings model deployment name in Azure
- `AI_SOURCE_WEBSITE`: the URL of the website that contains the source data (for references only)
- `AI_LOCAL_PATH`: local file path for storing data
- `AI_WEBSITE_REPO`: url of the Git repository containing the website source data, based on Hugo

You can find sample values in `.azure-template.env` and `.openai-template.env`. Configure them and create `.env` file with the updated settings.

### Python & Poetry
The project requires Python & Poetry installed. The minimum version dependencies can be found at `pyproject.toml`.
After installing Python & Poetry, you simply need to run `poetry run python app.py`

### Linux
The project requires Python 3.11 as a minimum and needs Go and Hugo installed for creating a local version of the website. See Go and Hugo documentation for installation instructions (only when running outside container)


## Outstanding
The following tasks are still outstanding:
- clean up code and add more comments.
- improve interaction language.
- assess overall quality and performance of the model and make improvements as and when required.
- assess the need to summarize the chat history to avoid exceeding the prompt token limit.
- update the yaml manifest.
- add error handling.
- perform extensive testing, in particular in multi-user scenarios.
- look at improvements of the ingestion. As a minimum the service engine should not consume queries whilst the ingestion is ongoing, as thatwill lead to errors.
- look at the use of `temperature` for the `QARetrievalChain`. It is not so obvious how this is handled.
- look at the possibility to implement reinforcement learning.
- return the actual LLM costs and token usage for queries.
