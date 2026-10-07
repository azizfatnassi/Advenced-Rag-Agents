import os
import math
from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings
from loguru import logger
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_community.embeddings import SentenceTransformerEmbeddings

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

answer_relevancy = AnswerRelevancy(strictness=1)

def evaluate_rag(question: str, answer: str, contexts: list) -> dict:
    llm = LangchainLLMWrapper(ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=os.getenv("GROQ_API_KEY"),
        max_tokens=2000
    ))
    embeddings = LangchainEmbeddingsWrapper(SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"))

    samples = [
        SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=[c.page_content[:500] for c in contexts],
        )
    ]

    dataset = EvaluationDataset(samples=samples)

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=llm,
        embeddings=embeddings,
    )

    logger.info(f"RAW RESULT: {result}")
    logger.info(f"RAW SCORES:{result.scores}")

    def safe_score(val):
        try:
            v = float(val[0]) if isinstance(val, list) else float(val)
            return round(v, 3) if not math.isnan(v) and not math.isinf(v) else None
        except:
            return None

    return {
        "faithfulness": safe_score(result["faithfulness"]),
        "answer_relevancy": safe_score(result["answer_relevancy"]),
    }