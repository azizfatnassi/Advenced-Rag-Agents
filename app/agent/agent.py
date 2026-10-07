
import os

from dotenv import load_dotenv

from app.agent.memory import get_preferences_as_text
load_dotenv()

from langfuse import get_client

# langfuse pinned to 2.60.0 — newer versions removed .trace() method
# which breaks Phase 2 /chat/memory endpoint
# do not upgrade until Phase 2 LangFuse code is migrated to new SDK API

from app.agent.tools import calculate, get_stock_price, search_documents
#create_react_agent is depricated in new versions of langraph
#migrate to create_agent from langchain.agents when upgrading/updating langchain
from langgraph.prebuilt import create_react_agent 
from app.agent.llm import groq_llm
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3


langfuse = get_client()

SYSTEM_PROMPT = """You are a financial expert and assistant with access to 
company financial documents and real-time market data.

Rules:
1. Always search for data before trying to answer from memory.
2. if the question asks for  a margin , ratio, or growth rate and the
document contains the raw numbers, calculate it yourself using the calculate tool.
Example: gross profit margin = (gross profit / revenue) * 100.
3. Be specific — cite the source company and year when referencing numbers.
4. If data is not found, say so clearly — never hallucinate numbers.
5. when using the calculate tool always use actual numbers not variables when calculating,
6. If your first search does not return enough data to answer, search again 
   with different keywords before giving up.
    """
def build_agent():
    tools=[search_documents, calculate, get_stock_price]
    conn= sqlite3.connect("agent-memory.db",check_same_thread=False)
    checkpointer=SqliteSaver(conn)
    agent=create_react_agent(
        model=groq_llm,
        tools=tools,
        prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer

    )
    return agent

def run_agent(agent, question: str, session_id: str = "dev",user_id:str="default", question_type: str = "unknown") -> str:
    handler = CallbackHandler()
    preferences = get_preferences_as_text(user_id)
    config={"callbacks":[handler],"configurable":{"thread_id": session_id}}
    
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config=config
        )

       
        answer = result["messages"][-1].content
        return answer

    except Exception as e:
        raise
    finally:
        langfuse.flush()



if __name__ == "__main__":
    agent = build_agent()

    test_questions = [
        "What was Tesla's revenue in 2023?",
        "What is Tesla's current stock price?",
        "Calculate Tesla's revenue growth from 2021 to 2023",
    ]

    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"QUESTION: {q}")
        answer = run_agent(agent, q)
        print(f"ANSWER: {answer}")