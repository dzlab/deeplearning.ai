import logging

import giskard

from .byte_chapters import ByteChaptersBot
from .zb_app import ZephyrApp

logging.getLogger("httpx").setLevel("CRITICAL")
logging.getLogger("giskard").setLevel("ERROR")
logging.getLogger().setLevel("ERROR")

__all__ = ["ZephyrApp", "ByteChaptersBot"]
