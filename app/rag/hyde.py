
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from loguru import logger

from app.monitoring.metrics import TOKEN_USAGE

llm=ChatGroq(model="openai/gpt-oss-20b",temperature=0)

HYDE_PROMPT=ChatPromptTemplate.from_template(""" Write a short factual
       paragraph that would answer the question asked .
       write it as if you found it in a financial document .
      Do not say " i think " or "maybe" write it as a fact 

    Question: {question}
    Paragraph: """)


def hyde_answer(question: str) -> str:
    chain = HYDE_PROMPT | llm

    try:
        result = chain.invoke({"question": question})

        if hasattr(result, "usage_metadata") and result.usage_metadata:
            total_tokens = result.usage_metadata.get("total_tokens")
            if total_tokens:
                TOKEN_USAGE.labels(model="openai/gpt-oss-20b", task="hyde").inc(total_tokens)

        return result.content

    except Exception as e:
        logger.warning(f"HyDE generation failed, falling back to raw question. Error: {e}")
        return question