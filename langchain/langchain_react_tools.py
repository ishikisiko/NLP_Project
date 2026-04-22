"""ReAct tools for LangChain agents.

This module wraps existing search, domain API, and local doc tools as LangChain BaseTool
implementations suitable for use with ReAct agents.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search.search import SearchClient, SearchHit
from search.source_selector import IntelligentSourceSelector
from rag.local_rag import LocalRAG
from utils.timing_utils import TimingRecorder


class WebSearchInput(BaseModel):
    """Input schema for web search tool."""
    query: str = Field(description="The search query to execute")


class DomainApiInput(BaseModel):
    """Input schema for domain API tool."""
    query: str = Field(description="The domain query to execute")


class LocalDocInput(BaseModel):
    """Input schema for local document tool."""
    query: str = Field(description="The query for local documents")


class ReActSearchTool(BaseTool):
    """LangChain Tool wrapping search_client.search() for ReAct agents."""

    name: str = "web_search"
    description: str = (
        "Search the web for current information. "
        "Input should be a search query string. "
        "Returns a list of search results with titles, URLs, and snippets."
    )
    args_schema: Type[BaseModel] = WebSearchInput
    return_direct: bool = False

    search_client: SearchClient = Field(exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, search_client: SearchClient, **kwargs: Any) -> None:
        super().__init__(search_client=search_client, **kwargs)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the search and return formatted results."""
        try:
            hits = self.search_client.search(query, num_results=5)
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


class ReActDomainTool(BaseTool):
    """LangChain Tool wrapping IntelligentSourceSelector for ReAct agents."""

    name: str = "domain_api"
    description: str = (
        "Get professional domain-specific data including weather, finance, sports, and transportation. "
        "Input should be a domain query string. "
        "Returns structured data or natural language answer from specialized APIs."
    )
    args_schema: Type[BaseModel] = DomainApiInput
    return_direct: bool = False

    source_selector: Any = Field(exclude=True)
    llm: Optional[BaseChatModel] = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        source_selector: IntelligentSourceSelector,
        llm: Optional[BaseChatModel] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(source_selector=source_selector, **kwargs)
        self.llm = llm

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute domain API query and return formatted results."""
        try:
            timing_recorder = TimingRecorder(enabled=False)

            # Select domain
            domain, sources = self.source_selector.select_sources(
                query, timing_recorder=timing_recorder
            )

            # Fetch domain data
            domain_api_result = self.source_selector.fetch_domain_data(
                query, domain, timing_recorder=timing_recorder
            )

            if not domain_api_result:
                return "No domain-specific data found for this query."

            # Check if handled
            if domain_api_result.get("handled") and domain_api_result.get("answer"):
                answer = domain_api_result.get("answer", "")
                data = domain_api_result.get("data")

                # Enhance with LLM if data available and we have an LLM
                if data and self.llm and not domain_api_result.get("continue_search"):
                    enhanced = self._enhance_answer(query, domain, data)
                    if enhanced:
                        return enhanced

                return answer

            return "No domain-specific data found for this query."

        except Exception as exc:
            return f"Domain API error: {exc}"

    def _enhance_answer(
        self,
        query: str,
        domain: str,
        data: Any,
    ) -> Optional[str]:
        """Enhance domain answer with LLM."""
        if not self.llm:
            return None

        from langchain_core.messages import HumanMessage, SystemMessage

        prompts = {}

        if domain == "weather" and isinstance(data, dict):
            prompts["weather"] = (
                "你是天气助手。根据实时数据，给出自然、丰富的回复，包括当前状况、穿衣/出行建议。",
                f"位置：{data.get('location', {}).get('formatted_address', '未知')}\n"
                f"概况：{data.get('weatherCondition', {}).get('description', {}).get('text', '未知')}\n"
                f"温度：{data.get('temperature', {}).get('degrees', '未知')}°C"
            )
        elif domain == "finance":
            prompts["finance"] = (
                "你是金融助手。根据提供的股票数据，分析价格走势和表现。如果包含多只股票，请进行对比。",
                json.dumps(data, ensure_ascii=False, indent=2)
            )

        if domain not in prompts:
            return None

        system_prompt, data_summary = prompts[domain]
        user_prompt = f"查询：{query}\n数据：\n{data_summary}\n生成回复："

        try:
            from langchain_core.language_models.chat_models import BaseChatModel
            if isinstance(self.llm, BaseChatModel):
                response = self.llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ])
                content = response.content if hasattr(response, 'content') else str(response)
                return content
        except Exception:
            pass

        return None


class ReActLocalDocTool(BaseTool):
    """LangChain Tool wrapping LocalRAG for ReAct agents."""

    name: str = "local_docs"
    description: str = (
        "Query the local knowledge base for information from documents. "
        "Input should be a query string. "
        "Returns relevant document snippets."
    )
    args_schema: Type[BaseModel] = LocalDocInput
    return_direct: bool = False

    data_path: str = Field(exclude=True)
    llm: Optional[BaseChatModel] = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        data_path: str,
        llm: Optional[BaseChatModel] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(data_path=data_path, **kwargs)
        self.llm = llm
        self._rag: Optional[LocalRAG] = None

    def _get_rag(self) -> Optional[LocalRAG]:
        """Get or create LocalRAG instance."""
        if not self.data_path or not os.path.isdir(self.data_path):
            return None

        if self._rag is None:
            if self.llm:
                try:
                    # Use LangChainLLMWrapper to wrap the LangChain model
                    from langchain.langchain_llm import LangChainLLMWrapper
                    llm_client = LangChainLLMWrapper(self.llm)
                    self._rag = LocalRAG(
                        llm_client=llm_client,
                        data_path=self.data_path,
                    )
                except Exception:
                    return None
        return self._rag

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Query local documents and return formatted results."""
        if not self.data_path:
            return "Local knowledge base is not available (no data path configured)."

        rag = self._get_rag()
        if not rag:
            return "Local knowledge base is not available (no data path configured)."

        try:
            result = rag.answer(query, num_retrieved_docs=3)
            if not result or not result.get("answer"):
                return "No relevant documents found."

            answer = result.get("answer", "")

            # Add source references if available
            retrieved_docs = result.get("retrieved_docs", [])
            if retrieved_docs:
                answer += "\n\n**本地文档来源：**\n"
                for idx, doc in enumerate(retrieved_docs, 1):
                    source = doc.get("source", f"文档 {idx}")
                    answer += f"{idx}. {source}\n"

            return answer

        except Exception as exc:
            return f"Local documents query failed: {exc}"


def create_react_tools_from_config(
    config: Dict[str, Any],
    llm: Optional[BaseChatModel] = None,
    search_client: Optional[SearchClient] = None,
    data_path: Optional[str] = None,
) -> List[BaseTool]:
    """Create ReAct tools from configuration.

    Args:
        config: Configuration dictionary
        llm: Optional LangChain chat model for domain API enhancement
        search_client: Search client for web search
        data_path: Optional path to local documents

    Returns:
        List of LangChain BaseTool instances
    """
    tools: List[BaseTool] = []

    # Create search tool if search client is available
    if search_client:
        tools.append(ReActSearchTool(search_client=search_client))

    # Create domain API tool
    try:
        source_selector = IntelligentSourceSelector(
            llm_client=None,  # We'll use the provided llm for enhancement
            use_llm=False,
            google_api_key=config.get("googleSearch", {}).get("api_key") or config.get("GOOGLE_API_KEY"),
            finnhub_api_key=config.get("FINNHUB_API_KEY"),
            sportsdb_api_key=config.get("SPORTSDB_API_KEY"),
            apisports_api_key=config.get("APISPORTS_KEY"),
            config=config,
        )
        tools.append(ReActDomainTool(
            source_selector=source_selector,
            llm=llm,
        ))
    except Exception:
        pass  # Domain API tool is optional

    # Create local docs tool if data path is available
    if data_path and os.path.isdir(data_path):
        tools.append(ReActLocalDocTool(
            data_path=data_path,
            llm=llm,
        ))

    return tools
