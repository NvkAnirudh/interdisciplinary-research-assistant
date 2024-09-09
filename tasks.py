from crew import Task
from agents import query_analyzer, llm_agent, wikipedia_agent, arxiv_agent, response_aggregator, query


analyze_task = Task(
        description=f"Analyze the query: '{query}' and determine which knowledge sources to use",
        agent=query_analyzer
    )
    
llm_task = Task(
    description=f"Generate a response to: '{query}' using the LLM",
    agent=llm_agent
)

wiki_task = Task(
    description=f"Search Wikipedia for information relevant to: '{query}'",
    agent=wikipedia_agent
)

arxiv_task = Task(
    description=f"Search Arxiv for scientific information relevant to: '{query}'",
    agent=arxiv_agent
)

aggregate_task = Task(
    description="Aggregate and synthesize the responses from different sources",
    agent=response_aggregator
)