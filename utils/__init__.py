# Set Windows event loop policy FIRST before any async imports
import asyncio
import sys
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from utils.database import __DatabaseManager as DatabaseManager
from utils.utils import BM25Reranker