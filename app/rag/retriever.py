from app.rag.hybrid_search import bm25_search
from app.rag.hyde import hyde_answer
from app.rag.multiquery import generate_queries

def advanced_retrieval(question: str, vectordb, k: int = 3, filters: dict = None) -> list:
    queries = generate_queries(question)
    
    all_chunks = []
    seen_ids = set()
    
    for query in queries:  
        if filters:
            results = vectordb.similarity_search(query, k=k, filter=filters)
        else:   
            results = vectordb.similarity_search(query, k=k)  
        
        for chunk in results:
            chunk_id = hash(chunk.page_content)  
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                all_chunks.append(chunk)

    # HyDE
    hypothetical_answer = hyde_answer(question)

    if filters:
        hyde_results = vectordb.similarity_search(hypothetical_answer, k=k, filter=filters)
    else:
        hyde_results = vectordb.similarity_search(hypothetical_answer, k=k)

    for chunk in hyde_results:
        chunk_id = hash(chunk.page_content)
        if chunk_id not in seen_ids:
            seen_ids.add(chunk_id)
            all_chunks.append(chunk)

    # BM25 re-score
    bm25_results = bm25_search(question, all_chunks, top_k=10)

    final_chunks = []
    final_seen = set()

    for chunk in all_chunks + bm25_results:
        chunk_id = hash(chunk.page_content)
        if chunk_id not in final_seen:
            final_seen.add(chunk_id)
            final_chunks.append(chunk)
    
    return final_chunks[:10]  