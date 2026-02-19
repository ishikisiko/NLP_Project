"""LangChain-compatible search tools for web search providers."""

from __future__ import annotations

import os
import sys
import json
from typing import Any, Dict, List, Optional, Type, Union

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search.search import (
    CombinedSearchClient,
    GoogleSearchClient,
    SearchClient,
    SearchHit,
    SerpAPISearchClient,
    YouSearchClient,
)


class WebSearchInput(BaseModel):
    """Input schema for web search tools."""
    
    query: str = Field(description="The search query to execute")
    num_results: int = Field(default=5, description="Number of results to return")


class WebSearchTool(BaseTool):
    """LangChain tool wrapper for web search.
    
    Wraps any SearchClient implementation (SerpAPI, You.com, Google, MCP, Combined)
    as a LangChain tool that can be used with agents.
    """
    
    name: str = "web_search"
    description: str = (
        "Useful for searching the web for current information. "
        "Input should be a search query string. "
        "Returns a list of search results with titles, URLs, and snippets."
    )
    args_schema: Type[BaseModel] = WebSearchInput
    
    search_client: SearchClient = Field(exclude=True)
    return_direct: bool = False
    
    class Config:
        arbitrary_types_allowed = True

    def __init__(self, search_client: SearchClient, **kwargs: Any) -> None:
        super().__init__(search_client=search_client, **kwargs)

    def _run(
        self,
        query: str,
        num_results: int = 5,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the search and return formatted results."""
        try:
            hits = self.search_client.search(query, num_results=num_results)
            return self._format_results(hits)
        except Exception as exc:
            return f"Search failed: {exc}"

    def _format_results(self, hits: List[SearchHit]) -> str:
        """Format search results as a readable string."""
        if not hits:
            return "No search results found."
        
        results = []
        for i, hit in enumerate(hits, 1):
            result = f"{i}. {hit.title or 'Untitled'}\n"
            result += f"   URL: {hit.url or 'N/A'}\n"
            result += f"   {hit.snippet or 'No description available.'}"
            results.append(result)
        
        return "\n\n".join(results)
    
    def search_raw(
        self,
        query: str,
        num_results: int = 5,
        **kwargs: Any,
    ) -> List[SearchHit]:
        """Execute search and return raw SearchHit objects."""
        return self.search_client.search(query, num_results=num_results, **kwargs)


class SerpAPITool(WebSearchTool):
    """LangChain tool for SerpAPI search."""
    
    name: str = "serpapi_search"
    description: str = (
        "Search the web using SerpAPI (Google search). "
        "Best for general web searches with comprehensive results."
    )

    def __init__(self, api_key: str, **kwargs: Any) -> None:
        client = SerpAPISearchClient(api_key=api_key)
        super().__init__(search_client=client, **kwargs)


class YouSearchTool(WebSearchTool):
    """LangChain tool for You.com search."""
    
    name: str = "you_search"
    description: str = (
        "Search the web using You.com API. "
        "Provides web results and news with AI-friendly snippets."
    )

    def __init__(self, api_key: str, **kwargs: Any) -> None:
        client = YouSearchClient(api_key=api_key)
        super().__init__(search_client=client, **kwargs)


class GoogleSearchTool(WebSearchTool):
    """LangChain tool for Google Custom Search."""
    
    name: str = "google_search"
    description: str = (
        "Search the web using Google Custom Search API. "
        "Provides authoritative search results from Google."
    )

    def __init__(self, api_key: str, cx: str, **kwargs: Any) -> None:
        client = GoogleSearchClient(api_key=api_key, cx=cx)
        super().__init__(search_client=client, **kwargs)





class CombinedSearchTool(WebSearchTool):
    """LangChain tool that combines multiple search providers."""
    
    name: str = "combined_search"
    description: str = (
        "Search the web using multiple search providers simultaneously. "
        "Aggregates and deduplicates results from all available sources."
    )

    def __init__(self, clients: List[SearchClient], **kwargs: Any) -> None:
        combined_client = CombinedSearchClient(clients)
        super().__init__(search_client=combined_client, **kwargs)


# Factory function to create search tool from config
def create_search_tool_from_config(config: Dict[str, Any]) -> Optional[WebSearchTool]:
    """Create a search tool from configuration dictionary.
    
    Args:
        config: Configuration dictionary (from config.json)
    
    Returns:
        WebSearchTool instance or None if no search providers configured
    """
    clients: List[SearchClient] = []
    
    # SerpAPI
    serp_key = (config.get("SERPAPI_API_KEY") or "").strip()
    if serp_key:
        try:
            clients.append(SerpAPISearchClient(api_key=serp_key))
        except Exception as exc:
            print(f"[search tool] SerpAPI disabled: {exc}")
    
    # You.com
    you_cfg = config.get("youSearch") or {}
    you_key = (you_cfg.get("api_key") or config.get("YOU_API_KEY") or "").strip()
    if you_key:
        try:
            you_kwargs: Dict[str, Any] = {}
            if you_cfg.get("base_url"):
                you_kwargs["base_url"] = you_cfg["base_url"]
            if you_cfg.get("timeout"):
                you_kwargs["timeout"] = int(you_cfg["timeout"])
            clients.append(YouSearchClient(api_key=you_key, **you_kwargs))
        except Exception as exc:
            print(f"[search tool] You.com disabled: {exc}")
    
    # Google Custom Search
    google_cfg = config.get("googleSearch") or {}
    google_key = (google_cfg.get("api_key") or config.get("GOOGLE_API_KEY") or "").strip()
    google_cx = (google_cfg.get("cx") or config.get("GOOGLE_CX") or "").strip()
    if google_key and google_cx:
        try:
            google_kwargs: Dict[str, Any] = {}
            if google_cfg.get("gl"):
                google_kwargs["gl"] = google_cfg["gl"]
            if google_cfg.get("lr"):
                google_kwargs["lr"] = google_cfg["lr"]
            clients.append(GoogleSearchClient(api_key=google_key, cx=google_cx, **google_kwargs))
        except Exception as exc:
            print(f"[search tool] Google Search disabled: {exc}")
    
    if not clients:
        return None
    
    if len(clients) == 1:
        return WebSearchTool(search_client=clients[0])
    
    return CombinedSearchTool(clients=clients)


# Retriever wrapper for RAG use
class SearchRetriever:
    """Wrapper that converts search results to LangChain Documents for RAG.
    
    This allows using web search as a retriever in LangChain chains.
    """
    
    def __init__(
        self,
        search_tool: WebSearchTool,
        k: int = 5,
        include_metadata: bool = True,
    ) -> None:
        self.search_tool = search_tool
        self.k = k
        self.include_metadata = include_metadata
    
    def get_relevant_documents(self, query: str) -> List:
        """Retrieve documents relevant to the query."""
        from langchain_core.documents import Document as LCDocument
        
        hits = self.search_tool.search_raw(query, num_results=self.k)
        
        documents = []
        for hit in hits:
            content = f"{hit.title or ''}\n\n{hit.snippet or ''}"
            metadata = {
                "source": hit.url or "",
                "title": hit.title or "",
            } if self.include_metadata else {}
            
            documents.append(LCDocument(page_content=content, metadata=metadata))
        
        return documents
    
    async def aget_relevant_documents(self, query: str) -> List:
        """Async version of get_relevant_documents."""
        # For now, just call sync version
        return self.get_relevant_documents(query)
