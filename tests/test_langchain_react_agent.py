from __future__ import annotations

from typing import Any

import server
from langchain.langchain_react_tools import (
    ReActDomainTool,
    ReActLocalDocTool,
    ReActSearchTool,
)
from orchestrators.react_agent_orchestrator import ReactAgentOrchestrator
from search.search import SearchClient, SearchHit


class StubSearchClient(SearchClient):
    def __init__(self, hits: list[SearchHit]) -> None:
        super().__init__()
        self._hits = hits

    def search(
        self,
        query: str,
        num_results: int = 5,
        *,
        per_source_limit: int | None = None,
        freshness: str | None = None,
        date_restrict: str | None = None,
    ) -> list[SearchHit]:
        return list(self._hits)


class StubSourceSelector:
    def select_sources(self, query: str, timing_recorder: Any = None):
        return "weather", ["weather"]

    def fetch_domain_data(self, query: str, domain: str, timing_recorder: Any = None):
        return {
            "handled": True,
            "answer": "Weather is sunny.",
            "data": {"temperature": {"degrees": 25}},
            "continue_search": True,
        }


def test_react_tools_wrap_search_domain_and_local_docs(monkeypatch, tmp_path):
    search_tool = ReActSearchTool(
        search_client=StubSearchClient(
            [
                SearchHit(
                    title="OpenAI",
                    url="https://openai.com/",
                    snippet="AI research and products.",
                )
            ]
        )
    )
    assert "OpenAI" in search_tool._run("openai")
    assert "https://openai.com/" in search_tool._run("openai")

    domain_tool = ReActDomainTool(source_selector=StubSourceSelector(), llm=None)
    assert domain_tool._run("weather in sf") == "Weather is sunny."

    local_tool = ReActLocalDocTool(data_path=str(tmp_path), llm=None)

    class StubLocalRAG:
        def answer(self, query: str, num_retrieved_docs: int = 3):
            return {
                "answer": "Local answer",
                "retrieved_docs": [{"source": "notes.md"}],
            }

    monkeypatch.setattr(local_tool, "_get_rag", lambda: StubLocalRAG())
    result = local_tool._run("local question")
    assert "Local answer" in result
    assert "notes.md" in result


def test_react_agent_orchestrator_answer_returns_compatible_payload(monkeypatch):
    class StubExecutor:
        def invoke(self, agent_input: dict[str, Any]) -> dict[str, str]:
            assert agent_input == {"input": "compare apple and microsoft"}
            return {"output": "Final synthesized answer"}

    monkeypatch.setattr(
        "orchestrators.react_agent_orchestrator.LangChainOrchestrator.create_react_agent",
        lambda **kwargs: StubExecutor(),
    )

    tool = ReActLocalDocTool(data_path="/tmp", llm=None)
    orchestrator = ReactAgentOrchestrator(
        llm=object(),
        tools=[tool],
        max_iterations=3,
    )

    result = orchestrator.answer("compare apple and microsoft")

    assert result["answer"] == "Final synthesized answer"
    assert result["search_hits"] == []
    assert result["control"]["search_mode"] == "react_agent"
    assert result["control"]["max_iterations"] == 3
    assert result["control"]["local_docs_present"] is True


def test_server_build_pipeline_switches_with_orchestrator_mode(monkeypatch):
    base_config = {
        "providers": {
            "minimax": {
                "api_key": "valid-key",
                "model": "MiniMax-M2.7-highspeed",
            }
        },
        "braveSearch": {},
        "brightDataSearch": {},
        "youSearch": {},
        "googleSearch": {},
        "rerank": {},
        "displayResponseTimes": False,
    }

    monkeypatch.setattr(server, "create_chat_model", lambda config=None: object())
    monkeypatch.setattr(server, "build_search_client", lambda config, sources=None: None)
    monkeypatch.setattr(server, "build_reranker", lambda config: (None, config.get("rerank", {})))

    react_calls: list[dict[str, Any]] = []
    default_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(
        "orchestrators.react_agent_orchestrator.ReactAgentOrchestrator.create_from_config",
        lambda **kwargs: react_calls.append(kwargs) or "react-orchestrator",
    )
    monkeypatch.setattr(
        server,
        "create_langchain_orchestrator",
        lambda **kwargs: default_calls.append(kwargs) or "default-orchestrator",
    )

    monkeypatch.setattr(server, "load_base_config", lambda: {**base_config, "orchestrator_mode": "react"})
    assert server.build_pipeline() == "react-orchestrator"
    assert len(react_calls) == 1

    monkeypatch.setattr(server, "load_base_config", lambda: {**base_config, "orchestrator_mode": "langchain"})
    assert server.build_pipeline() == "default-orchestrator"
    assert len(default_calls) == 1
