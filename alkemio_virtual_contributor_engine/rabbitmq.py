import asyncio
import json
from aio_pika import Message, ExchangeType, connect_robust
import aio_pika
import aiormq

from alkemio_virtual_contributor_engine.config import env
from logger import setup_logger

logger = setup_logger(__name__)


class RabbitMQ:
    def __init__(self):
        self.host = env.rabbitmq_host
        self.username = env.rabbitmq_user
        self.password = env.rabbitmq_password
        self.input_queue_name = env.rabbitmq_input_queue
        self.result_queue_name = env.rabbitmq_result_queue

        self.connection = None
        self.channel = None
        self.input_queue = None
        self.result_queue = None

    async def connect(self):
        try:
            self.connection = await connect_robust(
                host=self.host, login=self.username, password=self.password
            )
            self.channel = await self.connection.channel()

            self.input_queue = await self.channel.declare_queue(
                self.input_queue_name, auto_delete=False, durable=True
            )
            self.result_queue = await self.channel.declare_queue(
                self.result_queue_name, auto_delete=False, durable=True
            )
            self.exchange = await self.channel.declare_exchange(
                env.rabbitmq_exchange, ExchangeType.DIRECT, durable=True
            )
            await self.result_queue.bind(self.exchange, env.rabbitmq_result_routing_key)
        except (aio_pika.exceptions.AMQPError, Exception) as e:
            logger.error(f"Failed to establish RabbitMQ connection: {e}")
            raise

    async def consume(self, callback):
        if self.input_queue is not None:
            try:
                await self.input_queue.consume(callback)
            except Exception as e:
                logger.error(f"Error during message consumption: {e}")
                raise

    async def publish(self, message):
        try:
            await self.exchange.publish(
                Message(
                    body=json.dumps(message).encode(),
                ),
                routing_key=env.rabbitmq_result_routing_key,
            )
        except (
            aio_pika.exceptions.AMQPError,
            asyncio.exceptions.CancelledError,
            aiormq.exceptions.ChannelInvalidStateError,
            Exception,
        ) as e:
            logger.error(f"Failed to publish message due to a RabbitMQ error: {e}")
