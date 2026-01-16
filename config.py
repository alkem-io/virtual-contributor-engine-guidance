import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Env:
    history_length: int

    def __init__(self):
        self.history_length = int(os.getenv("HISTORY_LENGTH", "20"))


env = Env()
