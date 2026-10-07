

from typing import Optional, TypedDict, Literal
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from app.agent.llm import groq_llm
from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.tools import calculate, get_last_chunks, get_stock_price, search_documents
from app.monitoring.metrics import AGENT_ITERATIONS, TOKEN_USAGE


def _record_tokens(response, model_name: str, task: str) -> None:
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        total_tokens = response.usage_metadata.get("total_tokens")
        if total_tokens:
            TOKEN_USAGE.labels(model=model_name, task=task).inc(total_tokens)



class SupervisorState(TypedDict):
    question: str
    plan: Optional[str]
    agents_to_call: Optional[list[str]]
    retrieval_output:Optional[str]
    market_output: Optional[str]
    calculation_output: Optional[str]
    final_answer: Optional[str]
    history:Optional[list[dict]]
    retrieval_chunks:Optional[str]
    iterations:int
    max_iterations: int 

fast_llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

SUPERVISOR_PROMPT= """you are a supervisor of financial analysis system.
  you have three specialist agents available:
  -retrieval: searches financial documents ( annual reports, 10-k fillings ...)
  -market: gets live stock prices and market data
  -calculation: performs mathematical calculations on numbers

  Given the question and what agents have already returned, decide what to do next.
   Respond in this exact format and nothing else:
   NEXT: <agent_name or FINISH>
   REASON: <one sentence why>

   Rules: 
   -If you have enough information to answer, respond with NEXT: FINISH
   -If an agent returned " not found" or empty , try a different agent or FINISH
   -Never call the same agent twice
   -If iterations is 3, always respond with NEXT:FINISH

"""

def supervisor_node(state: SupervisorState)-> dict:
    context= f"question{state['question']}\n\n"
    context+= f"iterations so far:{state['iterations']}\n\n"

    history=state.get("history") or []

    if history :
         context+="Previous conversations"
         for turn in history[-3:]:
              context+=f"User asked: {turn['question']}\n"
              context+=f"Answer was: {turn['final_answer']}\n"

    if state.get('retrieval_output'):
        context += f"Retrieval agent returned : \n {state['retrieval_output']} \n\n"

    if state.get('market_output'):
            context += f"market agent returned : \n {state['market_output']} \n\n"

    if state.get('calculation_output'):
            context += f"Calculation agent returned : \n {state['calculation_output']} \n\n"

    messages=[
         SystemMessage(content=SUPERVISOR_PROMPT),
         HumanMessage(content=context)
    ]

    response = groq_llm.invoke(messages)
    _record_tokens(response, "openai/gpt-oss-120b", "supervisor_routing")

    raw = response.content.strip()
 
    next_agent= None
    for line in raw.split("\n"):
          if line.startswith("NEXT:"):
            next_agent=line.replace("NEXT:","").strip().lower()

    return {
         "plan":raw,
         "agents_to_call": [next_agent] if next_agent else ["finish"],
         "iterations": state['iterations'] + 1
    }


def retrieval_agent(state:SupervisorState)->dict:
     result= search_documents.invoke(state["question"])
     chunks= get_last_chunks()
     return{"retrieval_output":result,"retrieval_chunks":chunks}

def market_agent(state:SupervisorState)->dict:

     messages=[
          SystemMessage(content="Extract only the stock ticker symbol from this question . respond only with the ticker , nothing else, example : TSLA"),
          HumanMessage(content=state["question"])
     ]

     ticker_response = fast_llm.invoke(messages)
     _record_tokens(ticker_response, "openai/gpt-oss-20b", "market_ticker_extraction")
     ticker = ticker_response.content.strip()
     result= get_stock_price.invoke(ticker)
     return {"market_output": result}

def calculation_agent(state:SupervisorState)->dict:

     context = state.get("retrieval_output", "") or ""
     messages = [
        SystemMessage(content="You are a financial calculator. Extract the numbers from the context and write a math expression to answer the question. Reply with ONLY the math expression using actual numbers, no variables."),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {state['question']}")
    ]

     expr_response = groq_llm.invoke(messages)
     _record_tokens(expr_response, "openai/gpt-oss-120b", "calculation_expression")
     expression = expr_response.content.strip()
     result= calculate.invoke(expression)
     return{"calculation_output":result}


def synthesizer_node(state: SupervisorState) -> dict:
    context = f"Question: {state['question']}\n\n"
    
    if state.get("retrieval_output"):
        context += f"Financial documents data:\n{state['retrieval_output']}\n\n"
    if state.get("market_output"):
        context += f"Market data:\n{state['market_output']}\n\n"
    if state.get("calculation_output"):
        context += f"Calculation result:\n{state['calculation_output']}\n\n"

    history = state.get("history") or []
    if history:
        context += "Previous conversation for context:\n"
        for turn in history[-3:]:
            context += f"Q: {turn['question']}\nA: {turn['final_answer']}\n\n"


    messages = [
        SystemMessage(content="You are a senior financial analyst. Using all the data provided, write a clear, precise answer. Cite specific numbers. Never hallucinate."),
        HumanMessage(content=context)
    ]
    
    answer_response = fast_llm.invoke(messages)
    _record_tokens(answer_response, "openai/gpt-oss-20b", "synthesizer")
    answer = answer_response.content.strip()

    AGENT_ITERATIONS.observe(state["iterations"])

    updated_history = history + [{"question": state["question"], "final_answer": answer}]
    return {"final_answer": answer,
             "history": updated_history}

def route_after_supervisor(state:SupervisorState)->str:


     next_agent=state["agents_to_call"][0]

     if next_agent == "finish" or state["iterations"] >= state["max_iterations"]:
          return "synthesizer"

     if next_agent in ["retrieval","calculation","market"]:
          return next_agent

     return "synthesizer"
          
    
     

def build_supervisor_graph():
     
     graph=StateGraph(SupervisorState)

     graph.add_node("supervisor",supervisor_node)
     graph.add_node("retrieval",retrieval_agent)
     graph.add_node("calculation",calculation_agent)
     graph.add_node("market",market_agent)
     graph.add_node("synthesizer",synthesizer_node)

     graph.set_entry_point("supervisor")



     graph.add_conditional_edges( 
        "supervisor",
        route_after_supervisor,
         {
           "retrieval":"retrieval",
           "market":"market",
           "calculation":"calculation",
           "synthesizer": "synthesizer"
             
         }

         )
     graph.add_edge("retrieval","supervisor")
     graph.add_edge("calculation","supervisor")
     graph.add_edge("market","supervisor")
     graph.add_edge("synthesizer",END)

     return graph.compile()

