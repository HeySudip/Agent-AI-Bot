from .github import build_github_tools
from .search import build_search_tools
from .url import build_url_tools
from .utilities import build_utility_tools

__all__ = [
    "build_github_tools",
    "build_search_tools",
    "build_url_tools",
    "build_utility_tools",
]


def build_all_tools():
    tools = []
    tools.extend(build_github_tools())
    tools.extend(build_search_tools())
    tools.extend(build_url_tools())
    tools.extend(build_utility_tools())
    return tools
