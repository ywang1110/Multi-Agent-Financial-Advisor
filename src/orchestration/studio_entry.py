"""
LangSmith Studio entry point.
Studio requires a module-level `graph` variable with no constructor arguments.
client_profile is supplied at runtime via the Studio input panel.
"""
from src.orchestration.graph import build_graph_no_profile

graph = build_graph_no_profile()
