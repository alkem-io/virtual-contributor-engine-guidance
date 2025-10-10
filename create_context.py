from alkemio_virtual_contributor_engine import chromadb_client, openai_embeddings, setup_logger


logger = setup_logger(__name__)


def combine_documents(docs, document_separator="\n\n"):
    chunks_array = []
    for index, document in enumerate(docs["documents"][0]):
        chunks_array.append(f"[source:{index}] {document}")

    return document_separator.join(chunks_array)


def get_documents(message: str):

    collections = [
        "alkem.io-knowledge",
        "welcome.alkem.io-knowledge",
        "www.alkemio.org-knowledge",
    ]
    result = {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    for collection in collections:
        try:
            collection = chromadb_client.get_collection(
                collection
            )
            embeddings = openai_embeddings.embed_documents([message])

            tmp_result = collection.query(
                query_embeddings=list(embeddings),
                include=[
                    'documents',
                    'metadatas',
                    'distances',
                ],
                n_results=3,
            )
            if (
                tmp_result
                and tmp_result["documents"]
                and tmp_result["distances"]
                and tmp_result["metadatas"]
            ):
                result["distances"][0] += tmp_result["distances"][0]
                result["documents"][0] += tmp_result["documents"][0]
                result["metadatas"][0] += tmp_result["metadatas"][0]
        except Exception as e:
            logger.error(f"Failed to retrieve documents from collection: {collection}")
            logger.exception(e)

    return result


def create_context(message):
    documents = get_documents(message)
    logger.info("Context retrieved.")
    logger.debug(f"Context is {documents}")
    return documents, combine_documents(documents)
