from .github import build_github_tools
from .search import build_search_tools
from .url import build_url_tools
from .utilities import build_utility_tools
from .youtube_pdf import extract_youtube_to_pdf, generate_text_to_pdf

def build_all_tools():
    tools = []
    tools.extend(build_github_tools())
    tools.extend(build_search_tools())
    tools.extend(build_url_tools())
    tools.extend(build_utility_tools())
    tools.append(extract_youtube_to_pdf)
    tools.append(generate_text_to_pdf)
    return tools
