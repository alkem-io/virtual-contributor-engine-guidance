from alkemio_virtual_contributor_engine import chromadb_client, openai_embeddings, setup_logger


logger = setup_logger(__name__)


def combine_documents(docs, document_separator="\n\n"):
    chunks_array = []
    documents = docs.get("documents")
    if not documents or not documents[0]:
        return ""
    for index, document in enumerate(documents[0]):
        if document:
            chunks_array.append(f"[source:{index}] {document}")

    return document_separator.join(chunks_array)


def get_documents(message: str):

    collections = [
        "alkem.io-knowledge",
        "welcome.alkem.io-knowledge",
        "www.alkemio.org-knowledge",
    ]
    result = {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    for collection_name in collections:
        try:
            collection = chromadb_client.get_collection(
                collection_name
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
                and tmp_result.get("documents")
                and tmp_result.get("distances")
                and tmp_result.get("metadatas")
            ):
                documents = tmp_result["documents"]
                distances = tmp_result["distances"]
                metadatas = tmp_result["metadatas"]
                if documents and documents[0]:
                    result["documents"][0] += documents[0]
                if distances and distances[0]:
                    result["distances"][0] += distances[0]
                if metadatas and metadatas[0]:
                    result["metadatas"][0] += metadatas[0]
        except Exception as e:
            logger.error(
                f"Failed to retrieve documents from collection: {collection_name}"
            )
            logger.exception(e)

    return result


def create_context(message):
    documents = get_documents(message)
    logger.info("Context retrieved.")
    logger.debug(f"Context is {documents}")
    return documents, combine_documents(documents)
