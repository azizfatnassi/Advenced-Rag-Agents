import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.rag.eval_dataset import EVAL_QUESTIONS
from app.rag.retriever import advanced_retrieval
from app.rag.reranker import rerank
from app.rag.evaluate import evaluate_rag
from langchain_cohere import CohereEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import SentenceTransformerEmbeddings
import json

VECTORSTORE_DIR = "./app/vectorstore"

def get_vectorstore():
    embedding_fn = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=embedding_fn)

def generate_answer(question, chunks):
    context = "\n\n".join([c.page_content for c in chunks])
    llm = ChatGroq(model="openai/gpt-oss-120b")
    prompt = ChatPromptTemplate.from_template("""You are a helpful assistant. Use the context to answer.

Context:
{context}

Question: {question}

Answer:""")
    chain = prompt | llm
    result = chain.invoke({"context": context, "question": question})
    return result.content if hasattr(result, 'content') else str(result)

def run_full_evaluation():
    print("Loading vectorstore...")
    vectorstore = get_vectorstore()
    results = []

    for i, item in enumerate(EVAL_QUESTIONS):
        question = item["question"]
        print(f"\n[{i+1}/10] Evaluating: {question}")

        chunks = advanced_retrieval(question, vectorstore)
        reranked = rerank(question, chunks, top_k=3)
        answer = generate_answer(question, reranked)
        scores = evaluate_rag(question, answer, reranked)

        result = {
            "question": question,
            "answer": answer,
            "faithfulness": scores.get("faithfulness"),
            "answer_relevancy": scores.get("answer_relevancy")
        }
        results.append(result)
        print(f"Faithfulness: {scores.get('faithfulness')} | Relevancy: {scores.get('answer_relevancy')}")

    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    faith_scores = [r["faithfulness"] for r in results if r["faithfulness"] is not None]
    rel_scores = [r["answer_relevancy"] for r in results if r["answer_relevancy"] is not None]

    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Average Faithfulness:     {sum(faith_scores)/len(faith_scores):.3f}")
    print(f"Average Answer Relevancy: {sum(rel_scores)/len(rel_scores):.3f}")
    print("Results saved to evaluation_results.json")

if __name__ == "__main__":
    run_full_evaluation()