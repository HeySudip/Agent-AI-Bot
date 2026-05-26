"""Tool registry exposed to the agent.

Each ``build_*_tools`` function returns the list of LangChain tools that
module wants to expose. ``build_all_tools`` aggregates them in a stable
order so the LLM's function-calling cache stays consistent across runs.
"""

from .github import build_github_tools
from .search import build_search_tools
from .url import build_url_tools
from .utilities import build_utility_tools
from .video import build_video_tools, video_qa, video_screenshots, video_to_pdf
from .youtube_pdf import (
    extract_youtube_to_pdf,
    generate_text_to_pdf,
    research_and_create_pdf,
    search_and_extract_youtube_to_pdf,
    youtube_video_to_pdf,
)

__all__ = [
    "build_all_tools",
    "build_github_tools",
    "build_search_tools",
    "build_url_tools",
    "build_utility_tools",
    "build_video_tools",
    "research_and_create_pdf",
    "generate_text_to_pdf",
    "video_to_pdf",
    "video_qa",
    "video_screenshots",
    # Legacy aliases — kept so old config / prompts keep working.
    "extract_youtube_to_pdf",
    "search_and_extract_youtube_to_pdf",
    "youtube_video_to_pdf",
]


def build_all_tools() -> list:
    """Return the flat list of tools registered with the agent."""
    tools: list = []
    tools.extend(build_github_tools())
    tools.extend(build_search_tools())
    tools.extend(build_url_tools())
    tools.extend(build_utility_tools())

    # New canonical video tools.
    tools.extend(build_video_tools())

    # Document/research PDF tools.
    tools.append(research_and_create_pdf)
    tools.append(generate_text_to_pdf)
    return tools
