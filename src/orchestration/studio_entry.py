"""
LangSmith Studio entry point.
Studio requires a module-level `graph` variable with no constructor arguments.
We use the DEMO_CLIENT profile so the graph is self-contained.
"""
from src.models.client_profile import DEMO_CLIENT
from src.orchestration.graph import build_graph

graph = build_graph(DEMO_CLIENT)
