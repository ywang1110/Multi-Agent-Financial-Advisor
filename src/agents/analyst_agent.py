from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from src.agents.base_agent import BaseAgent
from src.models.research_task import ResearchReport, ResearchTask
from src.models.state import ConversationState


class AnalystAgent(BaseAgent):
    """
    Research analyst agent with access to web search and knowledge base.
    Uses a ReAct (Reason + Act) loop internally to call tools iteratively
    until sufficient information is gathered.
    """

    def __init__(self, tools: list[BaseTool]) -> None:
        super().__init__()
        self._tools = tools
        self._react_agent = create_react_agent(
            model=self._llm,
            tools=self._tools,
        )

    def run(self, state: ConversationState) -> dict:
        task: ResearchTask = state["latest_research"]
        prompt = self._build_research_prompt(task)

        result = self._react_agent.invoke(
            {"messages": [HumanMessage(content=prompt)]}
        )

        findings = result["messages"][-1].content
        sources = self._extract_sources(result["messages"])

        report = ResearchReport(
            task=task,
            findings=findings,
            sources=sources,
            confidence=0.9,
        )
        return {"latest_research": report}

    def _build_research_prompt(self, task: ResearchTask) -> str:
        return f"""You are a financial research analyst. Complete the following research task thoroughly.

Research Query: {task.query}

Context (why this matters for the client): {task.context}

Tool guidance: {task.tool_hint.value}
- Use 'web_search' for current market data, recent news, interest rates, ETF performance.
- Use 'knowledge_base_search' for investment principles, strategies, and financial frameworks.
- Use both if needed for a complete answer.

Deliver a structured research report with:
1. Key findings
2. Specific data points or statistics (with sources where possible)
3. Actionable insights for the financial advisor"""

    def _extract_sources(self, messages: list) -> list[str]:
        sources: list[str] = []
        for msg in messages:
            content = str(getattr(msg, "content", ""))
            for line in content.split("\n"):
                stripped = line.strip()
                # Web search: lines like "URL: https://..."
                if stripped.startswith("URL:"):
                    url = stripped[len("URL:"):].strip()
                    if url and url not in sources:
                        sources.append(url)
                # Knowledge base: lines like "[Source 1: asset_allocation.txt]"
                elif stripped.startswith("[Source") and ":" in stripped and stripped.endswith("]"):
                    kb_ref = stripped.strip("[]")
                    if kb_ref and kb_ref not in sources:
                        sources.append(kb_ref)
        return sources
