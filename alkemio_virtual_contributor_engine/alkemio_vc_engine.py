import asyncio
from collections.abc import Callable, Coroutine
import json
from typing import Any

from aio_pika.abc import AbstractIncomingMessage

from alkemio_virtual_contributor_engine.config import env
from alkemio_virtual_contributor_engine.events.input import Input
from alkemio_virtual_contributor_engine.events.result import Response
from alkemio_virtual_contributor_engine.rabbitmq import RabbitMQ
from logger import setup_logger


logger = setup_logger(__name__)


class AlkemioVirtualContributorEngine:

    def __init__(self):
        self.rabbitmq = RabbitMQ()
        self.handler = None

    async def start(self):
        await self.rabbitmq.connect()
        if self.handler is None:
            raise ValueError(
                "Message handler not defined. Ensure `engine.register_handler` with argument signature handler: `Callable[[Input], Coroutine[Any, Any, Response]]`  is called"
            )
        await self.rabbitmq.consume(self.invoke_handler)
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            logger.info("Engine shutdown initiated")
        except Exception as e:
            logger.error(f"Unexpected error in engine: {e}", exc_info=True)
            raise

    async def invoke_handler(self, message: AbstractIncomingMessage):
        logger.info("New message received.")
        if self.handler is None:
            raise ValueError(
                "Message handler not defined. Ensure `engine.register_handler` with argument signature handler: `Callable[[Input], Coroutine[Any, Any, Response]]`  is called"
            )

        async with message.process():

            try:
                body = json.loads(json.loads(message.body.decode()))
                input = Input(body["input"])
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse message: {e}")
                return
            except KeyError as e:
                logger.exception(f"Missing required field: {e}")
                return

            response = await self.handler(input)

            await self.rabbitmq.publish(
                {"response": response.to_dict(), "original": input.to_dict()}
            )

            logger.info("Response for published.")

    def register_handler(
        self, handler: Callable[[Input], Coroutine[Any, Any, Response]]
    ):
        self.handler = handler
