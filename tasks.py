from crewai import Task
from agents import query_analyzer, llm_agent, wikipedia_agent, arxiv_agent, response_aggregator, follow_up_generator

query = "Tell me a bit about Liverpool football team's history"

def create_tasks(query):
    analyze_task = Task(
        description=f"Analyze the query: '{query}' and determine which knowledge sources to use",
        agent=query_analyzer,
        expected_output="JSON object containing knowledge sources and query understanding"
    )
    
    llm_task = Task(
        description=f"Generate a response to: '{query}' using the LLM",
        agent=llm_agent,
        expected_output="Text response with confidence score"
    )

    wiki_task = Task(
        description=f"Search Wikipedia for information relevant to: '{query}'",
        agent=wikipedia_agent,
        expected_output="List of relevant Wikipedia articles and sections"
    )

    arxiv_task = Task(
        description=f"Search Arxiv for scientific information relevant to: '{query}'",
        agent=arxiv_agent,
        expected_output="List of relevant Arxiv papers with abstracts and authors"
    )

    aggregate_task = Task(
        description="Aggregate and synthesize the responses from different sources. Provide a comprehensive summary, list of key concepts, and estimate source contributions.",
        agent=response_aggregator,
        expected_output="JSON object containing summary, sources, contributions, and key concepts"
    )

    follow_up_task = Task(
        description="Generate follow-up questions and further reading suggestions based on the aggregated response",
        agent=follow_up_generator,
        expected_output="JSON object containing follow-up questions and further reading suggestions"
    )

    return [analyze_task, llm_task, wiki_task, arxiv_task, aggregate_task, follow_up_task]