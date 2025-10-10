import asyncio
import ai_adapter
from alkemio_virtual_contributor_engine.alkemio_vc_engine import (
    Input,
    AlkemioVirtualContributorEngine,
    Response,
    setup_logger
)


logger = setup_logger(__name__)

engine = AlkemioVirtualContributorEngine()


async def query(input: Input) -> Response:
    logger.info("Query method invoked.")
    result = await ai_adapter.invoke(input)
    logger.info("Query method completed.")
    return result


engine.register_handler(query)
asyncio.run(engine.start())
