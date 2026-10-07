

from typing import Any, Optional, TypedDict
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from loguru import logger

from app.agent.classifier import classify_question
from app.agent.generator import generate_answer
from app.dependencies import get_vectordb
from app.rag.reranker import rerank
from app.rag.retriever import advanced_retrieval

load_dotenv()



class RouterState(TypedDict):
    question: str
    question_type : Optional[str]
    answer: Optional[str]
    chunks: Optional[list]




def build_router_graph(vectordb):

 def classify_node(state:RouterState)->dict:
    logger.info(f"[classify_node] classifying: {state['question']}")
    question_type=classify_question(state["question"])
    logger.info(f"[classify_node] type: {question_type}")

    return {"question_type": question_type}

 def route_question(state:RouterState)->str:
    destination=f"{state['question_type']}_node"
    logger.info(f"[router] routing to : {destination}")
    return destination

 def factual_node(state:RouterState)->dict:
     logger.info(f"[factual_node] running simple retrieval")
     chunks= vectordb.similarity_search(state["question"])
     return {"chunks": chunks}


 def comparative_node(state: RouterState) -> dict:
    logger.info(f"[comparative_node] running multi-query retrieval")
    chunks= advanced_retrieval(state["question"],vectordb,k=3)
    reranked=rerank(state["question"], chunks, top_k=3)
    return {"chunks":reranked}

 def analytical_node(state: RouterState) -> dict:
    chunks = advanced_retrieval(state["question"], vectordb, k=7)
    reranked = rerank(state["question"], chunks, top_k=6)
    return {"chunks": reranked}


 def generate_node(state: RouterState) -> dict:
    logger.info(f"[generate_node] generating answer for type: {state['question_type']}")
    answer = generate_answer(
        question=state["question"],
        chunks=state["chunks"],
        question_type=state["question_type"]
    )
    return {"answer": answer}
 

 graph= StateGraph(RouterState)

 graph.add_node("classify_node",classify_node)
 graph.add_node("factual_node",factual_node)
 graph.add_node("comparative_node",comparative_node)
 graph.add_node("analytical_node",analytical_node)
 graph.add_node("generate_node", generate_node)

 graph.set_entry_point("classify_node")

 graph.add_conditional_edges(
        "classify_node",route_question,  {
            "factual_node": "factual_node",
            "comparative_node": "comparative_node",
            "analytical_node": "analytical_node",
        }
    )

 graph.add_edge("factual_node", "generate_node")
 graph.add_edge("comparative_node", "generate_node")
 graph.add_edge("analytical_node", "generate_node")

 graph.add_edge("generate_node", END)
 return graph.compile()

if __name__ == "__main__":
    vectordb=get_vectordb()
    router = build_router_graph(vectordb)
    
    test_questions = [
        "What was Tesla's revenue in 2023?",
        "compare Tesla's revenue change from 2021 to 2023?",
        "Why did Tesla's profit margins decline despite growing revenue?",
    ]
    
    for q in test_questions:
        print(f"\n{'='*60}")
        result = router.invoke({"question": q})
        print(f"RESULT: type={result['question_type']} | answer={result['answer']}")
