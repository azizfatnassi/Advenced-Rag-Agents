# advanced/multi_query.py
from langchain_core.prompts import ChatPromptTemplate 
from langchain_groq import  ChatGroq
from loguru import logger

from app.monitoring.metrics import TOKEN_USAGE

llm = ChatGroq(model="openai/gpt-oss-120b",temperature=0)

MULTI_QUERY_PROMPT = ChatPromptTemplate.from_template("""
You are an AI assistant. Your task is to generate 2 different 
versions of the user's question to improve document retrieval.

Generate 3 variations that:
- Use different words but same meaning
- Approach the question from different angles
- Are specific and searchable

Original question: {question}

Output ONLY the 3 questions, one per line, no numbering.
""")

def generate_queries(question: str) -> list[str]:
    chain = MULTI_QUERY_PROMPT | llm

    try:
        result = chain.invoke({"question": question})

        if hasattr(result, "usage_metadata") and result.usage_metadata:
            total_tokens = result.usage_metadata.get("total_tokens")
            if total_tokens:
                TOKEN_USAGE.labels(model="openai/gpt-oss-120b", task="multi_query").inc(total_tokens)

        # Parse the 3 questions
        queries = [q.strip() for q in result.content.strip().split("\n") if q.strip()]

        # Always include original
        queries.append(question)

        return queries[:4]  # max 4 queries including original

    except Exception as e:
        logger.warning(f"Multiquery generation failed, falling back to original question only. Error: {e}")
        return [question]
     
     