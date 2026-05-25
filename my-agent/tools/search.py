import logging
from langchain.tools import tool
from config import load_config, set_key

logger = logging.getLogger(__name__)


def build_search_tools() -> list:

    @tool
    def search_web(query: str) -> str:
        """
        Search the web for current, real-time information.
        Use for: news, events, prices, sports, weather, stocks, latest updates,
        anything that might have changed recently, or anything factual.
        Always cite sources. Summarize clearly.
        """
        cfg = load_config()
        tavily_key = cfg.get("tavily_api_key", "")

        if tavily_key:
            try:
                from tavily import TavilyClient
                client = TavilyClient(api_key=tavily_key)
                results = client.search(query, max_results=6, search_depth="advanced")
                formatted = []
                for r in results.get("results", []):
                    title = r.get("title", "")
                    url = r.get("url", "")
                    content = r.get("content", "")
                    score = r.get("score", 0)
                    formatted.append(
                        f"**{title}**\nURL: {url}\nRelevance: {score:.2f}\n{content}"
                    )
                answer = results.get("answer", "")
                output = ""
                if answer:
                    output += f"**Direct answer:** {answer}\n\n"
                output += "\n\n---\n\n".join(formatted)
                return output or "No results found."
            except Exception as e:
                logger.warning(f"Tavily search failed, falling back to DuckDuckGo: {e}")

        # Fallback: DuckDuckGo via ddgs
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=6))
            if not results:
                return "No results found."
            lines = []
            for r in results:
                lines.append(f"**{r.get('title', '')}**\n{r.get('href', '')}\n{r.get('body', '')}")
            return "\n\n---\n\n".join(lines)
        except Exception as e:
            logger.error(f"DuckDuckGo search also failed: {e}")
            return f"Search is temporarily unavailable. Error: {str(e)}"

    @tool
    def search_news(query: str) -> str:
        """
        Search for the latest news on a topic.
        Returns recent news articles with headlines and summaries.
        Use specifically when user asks about news or current events.
        """
        cfg = load_config()
        tavily_key = cfg.get("tavily_api_key", "")

        if tavily_key:
            try:
                from tavily import TavilyClient
                client = TavilyClient(api_key=tavily_key)
                results = client.search(
                    f"latest news {query}",
                    max_results=5,
                    search_depth="basic",
                    topic="news"
                )
                formatted = []
                for r in results.get("results", []):
                    published = r.get("published_date", "")
                    date_str = f" ({published})" if published else ""
                    formatted.append(
                        f"📰 **{r.get('title', '')}**{date_str}\n{r.get('content', '')[:300]}…\n{r.get('url', '')}"
                    )
                return "\n\n".join(formatted) or "No news found."
            except Exception as e:
                logger.warning(f"Tavily news search failed: {e}")

        # Fallback: DuckDuckGo news via ddgs
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.news(query, max_results=5))
            if not results:
                return "No news found."
            lines = []
            for r in results:
                date_str = f" ({r.get('date', '')})" if r.get("date") else ""
                lines.append(f"📰 **{r.get('title', '')}**{date_str}\n{r.get('body', '')[:300]}…\n{r.get('url', '')}")
            return "\n\n".join(lines)
        except Exception as e:
            return f"News search unavailable: {str(e)}"

    @tool
    def save_api_key(key_type: str, key_value: str) -> str:
        """
        Save an API key or token provided by the user.
        key_type values: 'gemini', 'anthropic', 'github', 'tavily'
        key_value: the actual key/token string
        Use this when the user shares any API key or token.
        """
        key_map = {
            "gemini": "gemini_api_key",
            "anthropic": "anthropic_api_key",
            "github": "github_token",
            "tavily": "tavily_api_key",
            "gemini_api_key": "gemini_api_key",
            "anthropic_api_key": "anthropic_api_key",
            "github_token": "github_token",
            "tavily_api_key": "tavily_api_key",
        }
        actual_key = key_map.get(key_type.lower().strip())
        if not actual_key:
            return f"Unknown key type '{key_type}'. Valid: gemini, anthropic, github, tavily"
        set_key(actual_key, key_value.strip())
        friendly = {
            "gemini_api_key": "Gemini API key",
            "anthropic_api_key": "Anthropic API key",
            "github_token": "GitHub token",
            "tavily_api_key": "Tavily API key",
        }
        return f"✅ {friendly.get(actual_key, actual_key)} saved successfully!"

    return [search_web, search_news, save_api_key]
