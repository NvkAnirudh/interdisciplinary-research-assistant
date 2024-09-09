import os
from dotenv import load_dotenv
from tools import tools, wiki_tool, arxiv_tool
from tasks import analyze_task, llm_task, wiki_task, arxiv_task, aggregate_task

from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq

load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
groq_api_key = os.environ['GROQ_API_KEY']

# LLM Setup
llm = ChatGroq(api_key=groq_api_key, model_name='Gemma2-9b-It')
llm_with_tools = llm.bind_tools(tools=tools)

# Agents
query_analyzer = Agent(
    role='Quer Analyzer',
    goal='Analyze user queries to determine relevant knowledge sources',
    backstory='Expert in natural language processing and query intent classification',
    allow_delegation=True,
    llm=llm,
)

llm_agent = Agent(
    role='LLM Responder',
    goal="Generate responses based on the LLM's knowledge",
    backstory='AI language model with broad general knowledge',
    allow_delegation=False,
    llm=llm_with_tools
)

wikipedia_agent = Agent(
    role="Wikipedia Researcher",
    goal="Find relevant information from Wikipedia",
    backstory="Expert in efficiently searching and summarizing Wikipedia articles",
    allow_delegation=False,
    llm=llm,
    tools=[wiki_tool]
)

arxiv_agent = Agent(
    role="Arxiv Researcher",
    goal="Find relevant scientific information from Arxiv",
    backstory="Expert in searching and summarizing scientific papers",
    allow_delegation=False,
    llm=llm,
    tools=[arxiv_tool]
)

response_aggregator = Agent(
    role="Response Aggregator",
    goal="Combine and synthesize responses from different sources",
    backstory="Expert in information synthesis and summary",
    allow_delegation=True,
    llm=llm
)

# Chatbot function
def chatbot(query):
    crew = Crew(
        agents=[query_analyzer, llm_agent, wikipedia_agent, arxiv_agent, response_aggregator],
        tasks=[analyze_task, llm_task, wiki_task, arxiv_task, aggregate_task],
        verbose=True
    )
    
    result = crew.kickoff()
    return result

query = "Tell me a bit about Liverpool football team's history"
response = chatbot(query)
print(response)