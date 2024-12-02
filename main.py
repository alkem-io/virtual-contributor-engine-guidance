import ai_adapter
import asyncio
import ai_adapter
import asyncio
from alkemio_virtual_contributor_engine.alkemio_vc_engine import (
    AlkemioVirtualContributorEngine,
)
from alkemio_virtual_contributor_engine.events.input import Input, InvocationOperation
from alkemio_virtual_contributor_engine.events.response import Response

from ingest import ensure_ingested
from logger import setup_logger

logger = setup_logger(__name__)

# Lock to prevent multiple ingestions from happening at the same time
ingestion_lock = asyncio.Lock()


async def query(input: Input) -> Response:
    logger.info("Query method invoked.")
    if input.operation is InvocationOperation.INGEST:
        logger.info("Operation is INGEST.")
        async with ingestion_lock:
            await ensure_ingested(True)
            # return empty response - we can extend this to give valid feedback
            return Response()

    logger.info("Operation is QUERY.")
    result = await ai_adapter.invoke(input)
    logger.info("Query method completed.")
    return result


engine = AlkemioVirtualContributorEngine()
engine.register_handler(query)
asyncio.run(engine.start())
