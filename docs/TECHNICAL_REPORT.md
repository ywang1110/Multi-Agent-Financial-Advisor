# Multi-Agent Financial Investment Advisor — Technical Report

> **JPMC Take-Home Assignment** · April 2026

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Agent Design & Conversation Flow](#3-agent-design--conversation-flow)
4. [Guardrail Pipeline](#4-guardrail-pipeline)
5. [Knowledge Infrastructure](#5-knowledge-infrastructure)
6. [Investment Strategy Layer](#6-investment-strategy-layer)
7. [State Management](#7-state-management)
8. [Design Patterns Reference](#8-design-patterns-reference)
9. [Test Coverage](#9-test-coverage)
10. [Key Engineering Decisions](#10-key-engineering-decisions)

---

## 1. Executive Summary

This project implements a **production-grade multi-agent financial advisory system** built on LangGraph. Three specialised AI agents collaborate to simulate a complete advisory session: a **Client Agent** that authentically role-plays the investor, an **Advisor Agent** that drives discovery and recommendation, and an **Analyst Agent** that fetches real-time market data and retrieves internal financial knowledge on demand.

The system is designed around four core principles:

| Principle | Realisation |
|-----------|-------------|
| **Safety first** | Multi-layer guardrail pipeline (PII scrubbing, off-topic filtering, disclaimer injection, turn-limit enforcement) |
| **Grounded advice** | Recommendations are never invented — the Analyst always runs web search + RAG before the Advisor commits to specifics |
| **Clean architecture** | Every component maps to a classic design pattern (Strategy, Factory, Chain of Responsibility, Repository, ReAct) |
| **Production readiness** | LangSmith tracing, LangGraph Studio support, human-handoff memo, configurable turn limit, full unit test suite |

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        LangGraph StateGraph                     │
│                                                                 │
│   START ──► init ──► Advisor ──────────────────────► Client     │
│                         │                               │       │
│                         │ needs_research=True           │ satisfied=True ──► END
│                         ▼                               │       │
│                      Analyst                            │ turn_limit ──► Handoff ──► END
│                         │                               │       │
│                         └──────────► Advisor ◄──────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

### Component Map

```
src/
├── agents/
│   ├── base_agent.py        # Abstract base — LLM initialisation
│   ├── advisor_agent.py     # Hub agent: discovery → research → recommendation
│   ├── analyst_agent.py     # ReAct tool-using research agent
│   ├── client_agent.py      # LLM-simulated investor persona
│   └── factory.py           # AgentFactory (Factory Pattern)
├── guardrails/
│   └── validators.py        # Chain of Responsibility guardrail pipeline
├── models/
│   ├── state.py             # Typed ConversationState (LangGraph MessagesState)
│   ├── client_profile.py    # Pydantic ClientProfile + enums
│   └── research_task.py     # ResearchTask / ResearchReport data models
├── orchestration/
│   ├── graph.py             # StateGraph builder + routing functions
│   └── studio_entry.py      # LangGraph Studio entry point
├── strategies/
│   └── investment_strategy.py  # Strategy Pattern: Conservative / Moderate / Aggressive
└── tools/
    ├── web_search.py        # Tavily real-time search tool
    ├── knowledge_store.py   # ChromaDB RAG repository
    └── analyst_tools.py     # Tool registry for the Analyst
```

---

## 3. Agent Design & Conversation Flow

### 3.1 The Three-Phase Advisory Protocol

The Advisor is explicitly prompted to operate in **three strict phases**, preventing premature recommendations and ensuring every suggestion is evidence-backed:

```
Phase 1 — DISCOVERY  (turns 1-2)
  ┌──────────────────────────────────────────────────────────────┐
  │  Warm personalised greeting → ONE clarifying question/turn   │
  │  No recommendations. No analyst trigger.                     │
  └──────────────────────────────────────────────────────────────┘
                              ▼
Phase 2 — RESEARCH  (exactly once, triggered when context is sufficient)
  ┌──────────────────────────────────────────────────────────────┐
  │  Advisor sets needs_research=True + research_query           │
  │  → Analyst runs ReAct loop (web_search + knowledge_base)     │
  │  → ResearchReport injected back into state                   │
  └──────────────────────────────────────────────────────────────┘
                              ▼
Phase 3 — RECOMMENDATION  (research report available)
  ┌──────────────────────────────────────────────────────────────┐
  │  Concrete ETF tickers, allocation %, rebalancing schedule    │
  │  Unicode bar chart + Markdown table for visual clarity       │
  │  Continues until client is_satisfied=True                    │
  │  Re-triggers Analyst only for genuinely new topics           │
  └──────────────────────────────────────────────────────────────┘
```

### 3.2 Agent Responsibilities

| Agent | LLM Output Type | Key Responsibilities |
|-------|----------------|----------------------|
| **ClientAgent** | `ClientOutput` (Pydantic) | Personas the investor; sets `is_satisfied`; guardrailed on input |
| **AdvisorAgent** | `AdvisorDecision` (Pydantic) | Drives all three phases; sets `needs_research`, `is_done`, `final_summary`; generates handoff memo |
| **AnalystAgent** | ReAct tool loop | Calls `web_search` + `knowledge_base_search`; returns `ResearchReport` with sources + confidence |

### 3.3 Conversation Sequence Diagram

```
Client        Advisor       Analyst        Guardrails
  │              │              │               │
  │◄─── greeting + Q  ──────────┤               │
  │              │              │    ← output guardrail (PII, disclaimer)
  │── answer ───►│              │    ← input guardrail (off-topic, PII)
  │              │── research_query ──►│
  │              │              │── web_search()
  │              │              │── knowledge_base_search()
  │              │◄─ ResearchReport  ──┤
  │◄── concrete recommendation ┤    ← output guardrail
  │── follow-up ►│             │    ← input guardrail
  │◄── final plan ─────────────┤
  │ (is_satisfied=True)        │
  │              │             │
  [END]
```

### 3.4 Human Handoff

When `turn_count ≥ MAX_CONVERSATION_TURNS`, the graph routes to a **handoff node** instead of ending. The Advisor generates a structured briefing memo (Priority Concerns / Recommendations Made / Unresolved Questions / Immediate Actions) for the human agent taking over — ensuring no context is lost at the AI/human boundary.

---

## 4. Guardrail Pipeline

The guardrail system uses the **Chain of Responsibility** pattern. Each node independently checks content and passes it down the chain only if it passes:

```
Advisor Output Chain:
  TurnLimitGuardrail → OffTopicGuardrail → PIIScrubbingGuardrail → DisclaimerGuardrail
        ▲ short-circuits here if limit hit

Client Input Chain:
  OffTopicGuardrail → PIIScrubbingGuardrail
```

| Guardrail | Mechanism | Action on Fail |
|-----------|-----------|----------------|
| `TurnLimitGuardrail` | Compares `current_turn` vs `MAX_CONVERSATION_TURNS` | Blocks message, triggers handoff routing |
| `OffTopicGuardrail` | LLM semantic classifier (temperature=0, max_tokens=10) | Rejects with explanation |
| `PIIScrubbingGuardrail` | Regex patterns (SSN, card #, account #, email) | Redacts in-place, always passes |
| `DisclaimerGuardrail` | Checks for existing disclaimer keyword | Appends regulatory disclaimer |

**Design note:** `OffTopicGuardrail` uses LLM classification instead of keyword matching to handle paraphrasing (e.g., "liquidating a position" would falsely trigger a keyword filter on "liquid").

---

## 5. Knowledge Infrastructure

The Analyst has two complementary knowledge sources, each suited to a different information type:

```
                    ┌────────────────────┐
  Research Query ──►│  Analyst (ReAct)   │
                    └────────┬───────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
   ┌──────────────────┐         ┌─────────────────────┐
   │  Tavily Web API  │         │  ChromaDB (local)   │
   │  (real-time)     │         │  (RAG, persistent)  │
   ├──────────────────┤         ├─────────────────────┤
   │ • ETF prices     │         │ asset_allocation.txt│
   │ • Market news    │         │ retirement_plan.txt │
   │ • Interest rates │         │ risk_management.txt │
   │ • Recent returns │         │ tax_optimization.txt│
   └──────────────────┘         └─────────────────────┘
              │                             │
              └──────────────┬──────────────┘
                             ▼
                    ResearchReport
                    (findings + sources + confidence)
```

**ChromaDB details:**
- Embedding model: `text-embedding-3-small` (OpenAI)
- Chunking: `RecursiveCharacterTextSplitter` (800 chars, 100 overlap)
- Smart initialisation: loads existing collection on restart; rebuilds only if empty or corrupted
- Singleton via `@lru_cache(maxsize=1)` — one warm instance per process

**Query length guard:** If the Advisor sends a verbose `research_query > 400 chars`, the web search tool automatically compresses it via LLM before hitting Tavily.

---

## 6. Investment Strategy Layer

The **Strategy Pattern** decouples the advisor's recommendation logic from the client's risk profile:

```
ClientProfile.risk_tolerance
        │
        ▼
StrategyFactory.get_strategy()
        │
        ├── CONSERVATIVE → ConservativeStrategy  (25/55/10/10 eq/bond/re/cash)
        ├── MODERATE     → ModerateStrategy      (60/30/7/3)
        └── AGGRESSIVE   → AggressiveStrategy    (85/10/5/0)
                │
                ▼
        StrategyRecommendation
        (strategy_name, allocation, rationale,
         suggested_instruments, rebalancing_frequency, key_risks)
                │
                ▼
        Injected into Advisor system prompt as structured context
```

The strategy recommendation grounds every advisor conversation from turn 1 — the LLM never invents an allocation from scratch.

---

## 7. State Management

`ConversationState` extends LangGraph's `MessagesState` with typed fields:

```python
class ConversationState(MessagesState):
    client_profile: ClientProfile        # investor persona
    turn_count: int                      # conversation progress
    is_satisfied: bool                   # client satisfaction signal
    needs_research: bool                 # routing signal → analyst
    latest_research: ResearchReport | None  # analyst output
    research_history: list[str]          # dedup guard (prevents re-research)
    final_summary: str                   # advisor's closing statement
    handoff_summary: str                 # briefing memo for human agent
```

Key state design decisions:
- **`research_history`** prevents the Advisor from triggering redundant analyst calls on already-covered topics
- **`needs_research`** is a clean routing signal that decouples the Advisor's decision from graph topology
- **`handoff_summary`** is populated only in the `handoff` node, keeping the happy path clean
- The `init` node coerces `client_profile: dict → ClientProfile` for LangGraph Studio compatibility

---

## 8. Design Patterns Reference

| Pattern | Location | Purpose |
|---------|----------|---------|
| **Strategy** | `src/strategies/investment_strategy.py` | Swap allocation logic per risk profile |
| **Factory** | `src/agents/factory.py` | Centralise agent construction |
| **Chain of Responsibility** | `src/guardrails/validators.py` | Composable guardrail pipeline |
| **Repository** | `src/tools/knowledge_store.py` | Encapsulate ChromaDB access |
| **ReAct** | `src/agents/analyst_agent.py` | Tool-calling loop for grounded research |
| **Structured Output** | `AdvisorDecision`, `ClientOutput` | Type-safe LLM responses via Pydantic |

---

## 9. Test Coverage

The test suite covers all major components with **zero real API calls** (full mocking):

| Test File | What's Tested |
|-----------|--------------|
| `test_agents.py` | ClientAgent persona, AdvisorAgent routing signals, AnalystAgent ReAct result |
| `test_guardrails.py` | Each guardrail in isolation + full chain integration |
| `test_graph.py` | Routing logic for all branching conditions (research, satisfaction, turn limit, handoff) |
| `test_tools.py` | Tavily formatting, ChromaDB init/rebuild/skip logic, tool registration |

Notable test design choices:
- `OffTopicGuardrail` is tested via direct `_llm` injection (no `patch` site collision)
- `ChromaDB` resilience tests use `tmp_path` to simulate corruption and healthy store
- Turn limit is passed via `monkeypatch.setenv` to keep tests environment-independent

---

## 10. Key Engineering Decisions

### Why LangGraph over vanilla LangChain?
LangGraph's `StateGraph` gives explicit, inspectable control over agent routing — critical for a financial system where the sequence of operations (discovery before recommendation, always) must be enforced by the infrastructure, not just the prompt.

### Why structured output (Pydantic) for every agent?
`AdvisorDecision.needs_research` and `ClientOutput.is_satisfied` are **routing signals** consumed by graph logic. Using structured output makes these machine-readable with zero parsing fragility — a brittle string parse would be unacceptable for production routing.

### Why Chain of Responsibility for guardrails?
The pipeline is open for extension without modification. Adding a new guardrail (e.g., toxicity detection) requires one class and one line in the builder — no changes to existing nodes or agents.

### Why a human handoff node instead of a hard stop?
Real financial advisory systems cannot simply terminate when an AI session ends. The handoff memo preserves every actionable detail from the conversation for the human advisor, creating a seamless AI-to-human transition that a client would actually experience.

### Why separate Analyst from Advisor?
The Advisor is a conversationalist; the Analyst is a researcher. Mixing these roles leads to LLMs that either over-research (slow) or under-research (hallucinate). Separation enforces the discipline: the Advisor *must* ask for research before giving specifics, and the Analyst *only* does research — it never speaks to the client.

---

*Built with LangGraph · LangChain · OpenAI GPT-4.1-mini · ChromaDB · Tavily*
