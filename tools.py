from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain.agents import Tool

# External Tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=300)
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)

wiki_tool = Tool(
    name="Wikipedia",
    func=wiki_wrapper.run,
    description="Useful for searching Wikipedia articles"
)

arxiv_tool = Tool(
    name="Arxiv",
    func=arxiv_wrapper.run,
    description="Useful for searching scientific papers on Arxiv"
)

tools = [wiki_tool, arxiv_tool]