from crewai import Crew
from agents import query_analyzer, llm_agent, wikipedia_agent, arxiv_agent, response_aggregator
from tasks import create_tasks

# Chatbot function
def chatbot(query):
    tasks = create_tasks(query)

    crew = Crew(
        agents=[query_analyzer, llm_agent, wikipedia_agent, arxiv_agent, response_aggregator],
        tasks=tasks,
        # tasks=[analyze_task, llm_task, wiki_task, arxiv_task, aggregate_task],
        verbose=True
    )
    
    result = crew.kickoff()
    return result

query = "Tell me a bit about Liverpool football team's history"
response = chatbot(query)
print(response)