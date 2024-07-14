from crewai import Agent, Task, Crew, Process
from crewai_tools import FileReadTool, DirectoryReadTool

from langchain_aws import ChatBedrock
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.agent_toolkits import FileManagementToolkit
import sys


search_tool = DuckDuckGoSearchRun()
docs_tool = DirectoryReadTool(directory='./blogs')
file_tool = FileReadTool()

llm = ChatBedrock(model_id="anthropic.claude-3-sonnet-20240229-v1:0", model_kwargs=dict(temperature=0, max_tokens=8000))

def kickoffTheCrew(topic):
    researcher = Agent(
        role = "Internet Researcher",
        goal = f"Perform research on the {topic}",
        verbose = True,
        llm=llm,
        backstory = f"""
        You are an export Internet Researcher, 
        Who knows how to search the internet for detailed content on {topic} .
        Include code examples with documentation
    """
    )

    blogger = Agent(
        role = "Blogger",
        goal = f"Write engaging and interesting blog in a maximum of 4000 words.{topic}",
        verbose = True,
        llm=llm,
        backstory = f"""
        You are an Expert Blogger, 
        Who knows how to write a blog post on {topic} .
        Any information will be written in markdown format.
    """,
    tools=[docs_tool, file_tool]
    )


    task_search = Task(
        description = f"""Search for all the details about {topic}
        Your final answer MUST be a consolidated content that can be used for blogging.
        The content should be well organized and easy to understand.        
        """,
        expected_output = F"A comprehensive 5000 words about {topic}",
        max_iter=3,
        tools=[search_tool],
        agent=researcher
    )

    task_post = Task(
        description = f"""Write a well structred blog post about {topic}
        Your final answer MUST be a consolidated content that can be used for blogging.
        The content should be well organized and easy to understand.
        Once the blog post is ready, create a new file {topic}.md, and sage the blog in that file in the current directory.        
        """,
        expected_output = F"A comprehensive 5000 words about {topic} in markdown format.",
        max_iter=2,
        agent=blogger,
        output_file=f"blogs/{topic}.md"
    )


    crew = Crew(
        agents=[researcher, blogger],
        tasks=[task_search, task_post],
        verbose=2,
        process=Process.sequential
    )

    result = crew.kickoff()
    return result

n = len(sys.argv)

if n == 2:
    topic = sys.argv[1]
    result = kickoffTheCrew(topic)
    print(result)
else:
    print("Please provide a topic to research and blog about.")
    print("Example: python agentX.py 'How to create a chatbot'")