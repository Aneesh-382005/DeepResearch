import asyncio
from tavily import AsyncTavilyClient
from dotenv import load_dotenv
from datetime import timedelta, datetime
from typing import List, Dict, Any
load_dotenv()

class CachedTavilyClient:
    def __init__(self, APIkey: str, cacheDurationMinutes: int = 10, maxRetries: int = 3):
        self.client = AsyncTavilyClient(APIkey)
        self.cache = {}
        self.cacheDuration = timedelta(minutes = cacheDurationMinutes)
        self.maxRetries = maxRetries

    async def _retryOperation(self, operation, *args, **kwargs):
        for attempt in range(self.maxRetries):
            try: 
                return await operation(*args, **kwargs)
            except Exception as e:
                if attempt == self.maxRetries - 1:
                    raise
                await asyncio.sleep(2 ** attempt) # Exponential backoff

    async def search(self, query: str, **kwargs) -> Dict[str, Any]:
        cacheKEY = f"search:{query}"
        if cacheKEY in self.cache:
            cachedResult, timestamp = self.cache[cacheKEY]
            if datetime.now() - timestamp < self.cacheDuration:
                return cachedResult

        result = await self._retryOperation(self.client.search, query, **kwargs)
        self.cache[cacheKEY] = (result, datetime.now())
        return result 

    async def extract(self, url: str) -> Dict[str, Any]:
        cacheKEY = f"extract:{url}"
        if cacheKEY in self.cache:
            cachedResult, timestamp = self.cache[cacheKEY]
            if datetime.now() - timestamp < self.cacheDuration:
                return cachedResult
    
        result = await self._retryOperation(self.client.extract, url)
        self.cache[cacheKEY] = (result, datetime.now())
        return result
    