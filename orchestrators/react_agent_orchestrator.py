"""React Agent Orchestrator - LangChain ReAct-based query handling.

This module provides an alternative orchestrator implementation using
LangChain's ReAct agent for iterative reasoning and tool calling.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.langchain_react_tools import (
    create_react_tools_from_config,
    ReActSearchTool,
    ReActSearchRecoveryTool,
    ReActDomainTool,
    ReActLocalDocTool,
)
from langchain.langchain_orchestrator import LangChainOrchestrator
from search.search import SearchClient
from utils.timing_utils import TimingRecorder


class ReactAgentOrchestrator:
    """ReAct-based orchestrator using LangChain agent for iterative reasoning.

    This orchestrator uses LangChain's ReAct agent to iteratively reason about
    queries and call tools (web search, domain API, local docs) as needed.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        tools: Optional[List[Any]] = None,
        max_iterations: int = 5,
        *,
        search_client: Optional[SearchClient] = None,
        data_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        show_timings: bool = False,
    ) -> None:
        """Initialize the ReactAgentOrchestrator.

        Args:
            llm: LangChain chat model
            tools: Optional list of tools; if not provided, created from config
            max_iterations: Maximum number of ReAct iterations
            search_client: Optional search client
            data_path: Optional path to local documents
            config: Optional configuration dictionary
            show_timings: Whether to record and return timing information
        """
        self.llm = llm
        self.max_iterations = max_iterations
        self.show_timings = show_timings
        self.config = config or {}

        # Use provided tools or create from config
        if tools:
            self.tools = tools
        else:
            self.tools = create_react_tools_from_config(
                config=self.config,
                llm=llm,
                search_client=search_client,
                data_path=data_path,
            )

        # Create the ReAct agent executor
        self._agent_executor = LangChainOrchestrator.create_react_agent(
            llm=llm,
            tools=self.tools,
            max_iterations=max_iterations,
        )

    def answer(
        self,
        query: str,
        *,
        num_search_results: int = 10,
        per_source_search_results: Optional[int] = None,
        num_retrieved_docs: int = 5,
        max_tokens: int = 8000,
        temperature: float = 0.3,
        allow_search: bool = True,
        reference_limit: Optional[int] = None,
        force_search: bool = False,
        images: Optional[List[Dict[str, str]]] = None,
        fallback_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Answer a query using ReAct agent.

        Args:
            query: User query
            num_search_results: Number of search results (used if search tool called)
            per_source_search_results: Results per source
            num_retrieved_docs: Number of local docs to retrieve
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            allow_search: Whether to allow search
            reference_limit: Limit on reference sources in response
            force_search: Force search even if agent decides not to
            images: Optional image data (not currently used in ReAct mode)

        Returns:
            Dictionary with answer, control metadata, and search_hits
        """
        timing_recorder = TimingRecorder(enabled=self.show_timings)
        timing_recorder.start()

        try:
            # Build input for the agent
            agent_input = {"input": self._build_agent_input(query, fallback_context)}

            # Execute the agent
            result = self._agent_executor.invoke(agent_input)

            # Parse the output
            output = result.get("output", "")

            # Build response structure compatible with other orchestrators
            search_hits = self._extract_search_hits(output)

            response: Dict[str, Any] = {
                "query": query,
                "answer": output,
                "search_hits": search_hits or list(fallback_context.get("search_hits") or []) if fallback_context else search_hits,
                "llm_raw": None,
                "llm_warning": None,
                "llm_error": None,
            }

            # Add control metadata
            control: Dict[str, Any] = {
                "search_performed": len(search_hits) > 0,
                "decision": {
                    "needs_search": len(search_hits) > 0,
                    "reason": "react_agent_iteration" if not fallback_context else "react_fallback_iteration",
                },
                "search_mode": "react_agent" if not fallback_context else "react_fallback",
                "keywords": [],
                "hybrid_mode": False,
                "local_docs_present": self._has_local_docs_tool(),
                "search_allowed": allow_search,
                "max_iterations": self.max_iterations,
                "final_executor": "react_fallback" if fallback_context else "react_agent",
                "fallback_triggered": bool(fallback_context),
            }
            if fallback_context:
                control["fallback_context"] = self._fallback_context_meta(fallback_context)

            response["control"] = control

            # Add timing information if enabled
            if timing_recorder.enabled:
                timing_recorder.stop()
                timing_payload = timing_recorder.to_dict()
                if timing_payload:
                    timing_payload["领域智能类型"] = "ReAct Agent"
                    response["response_times"] = timing_payload

            return response

        except Exception as exc:
            return {
                "query": query,
                "answer": f"Agent execution failed: {exc}",
                "search_hits": [],
                "llm_raw": None,
                "llm_warning": None,
                "llm_error": str(exc),
                "control": {
                    "search_performed": False,
                    "decision": {
                        "needs_search": False,
                        "reason": f"react_agent_error: {exc}",
                    },
                    "search_mode": "react_agent_error",
                    "keywords": [],
                    "hybrid_mode": False,
                    "local_docs_present": self._has_local_docs_tool(),
                    "search_allowed": allow_search,
                },
            }

    def _extract_search_hits(self, output: str) -> List[Dict[str, Any]]:
        """Extract search hits from agent output if present."""
        # ReAct agent output doesn't typically contain structured search_hits
        # This is a placeholder that can be enhanced if needed
        return []

    def _build_agent_input(
        self,
        query: str,
        fallback_context: Optional[Dict[str, Any]],
    ) -> str:
        """Build the ReAct agent input, optionally with fallback context."""
        if not fallback_context:
            return query

        lines = [
            f"Original Query:\n{query}",
            "",
            "You are acting as a recovery agent. Improve the previous answer using tools only when needed.",
        ]

        previous_answer = str(fallback_context.get("previous_answer") or "").strip()
        if previous_answer:
            lines.extend(["", f"Previous Answer:\n{previous_answer}"])

        failure_types = fallback_context.get("failure_types") or []
        if failure_types:
            lines.extend(["", f"Post-check Failure Types: {', '.join(map(str, failure_types))}"])

        missing_constraints = fallback_context.get("missing_constraints") or []
        if missing_constraints:
            lines.extend(["", f"Missing Constraints: {', '.join(map(str, missing_constraints))}"])

        evidence_summary = str(fallback_context.get("evidence_summary") or "").strip()
        if evidence_summary:
            lines.extend(["", f"Available Evidence Summary:\n{evidence_summary}"])

        recovery_goal = str(fallback_context.get("recovery_goal") or "").strip()
        if recovery_goal:
            lines.extend(["", f"Recovery Goal:\n{recovery_goal}"])

        return "\n".join(lines)

    def _fallback_context_meta(self, fallback_context: Dict[str, Any]) -> Dict[str, Any]:
        """Return safe metadata about the fallback context."""
        return {
            "failure_types": list(fallback_context.get("failure_types") or []),
            "missing_constraints": list(fallback_context.get("missing_constraints") or []),
            "has_previous_answer": bool(fallback_context.get("previous_answer")),
            "has_evidence_summary": bool(fallback_context.get("evidence_summary")),
        }

    def _has_local_docs_tool(self) -> bool:
        """Check if local docs tool is available."""
        return any(
            isinstance(t, ReActLocalDocTool) for t in self.tools
        )

    def _has_search_recovery_tool(self) -> bool:
        """Check if high-level search recovery tool is available."""
        return any(isinstance(t, ReActSearchRecoveryTool) for t in self.tools)

    @classmethod
    def create_from_config(
        cls,
        config: Optional[Dict[str, Any]] = None,
        llm: Optional[BaseChatModel] = None,
        search_client: Optional[SearchClient] = None,
        **kwargs: Any,
    ) -> "ReactAgentOrchestrator":
        """Create a ReactAgentOrchestrator from configuration.

        Args:
            config: Configuration dictionary
            llm: Optional LangChain chat model
            search_client: Optional search client
            **kwargs: Additional arguments passed to ReactAgentOrchestrator

        Returns:
            Configured ReactAgentOrchestrator instance
        """
        if config is None:
            import json as json_module
            import os
            config_path = os.getenv("NLP_CONFIG_PATH", "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json_module.load(f)
            else:
                config = {}

        if llm is None:
            from langchain.langchain_llm import create_chat_model
            llm = create_chat_model(config=config)

        # Extract relevant config sections
        data_path = config.get("dataPath") or config.get("data_path")

        # Get max iterations from config if not in kwargs
        max_iterations = kwargs.pop("max_iterations", None)
        if max_iterations is None:
            max_iterations = config.get("reactAgent", {}).get("max_iterations", 5)

        # Create tools from config
        tools = create_react_tools_from_config(
            config=config,
            llm=llm,
            search_client=search_client,
            data_path=data_path,
        )

        show_timings = kwargs.pop("show_timings", config.get("displayResponseTimes", False))

        return cls(
            llm=llm,
            tools=tools,
            max_iterations=max_iterations,
            search_client=search_client,
            data_path=data_path,
            config=config,
            show_timings=show_timings,
            **kwargs,
        )
