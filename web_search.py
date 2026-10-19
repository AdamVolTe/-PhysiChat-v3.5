"""
Live Web Search module using DuckDuckGo (Free & No API Key needed).
"""
from duckduckgo_search import DDGS

def search_web(query: str, max_results: int = 3) -> str:
    """Performs a live web search and returns formatted text with context and links."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "لم يتم العثور على نتائج مباشرة في الويب."
        
        output = []
        for r in results:
            title = r.get("title", "بدون عنوان")
            snippet = r.get("body", "")
            href = r.get("href", "")
            output.append(f"📌 **{title}**\n{snippet}\n🔗 [المصدر]({href})\n")
        return "\n".join(output)
    except Exception as e:
        return f"حدث خطأ أثناء البحث في الويب: {e}"
