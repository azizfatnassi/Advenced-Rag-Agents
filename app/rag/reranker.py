
#from sentence_transformers import CrossEncoder
#import os

#os.environ["TRANSFORMERS_OFFLINE"]="1"
#os.environ["HF_DATASETS_OFFLINE"]="1"
#_model=None

#def get_model():
 # global _model
  #if _model is None:
  #  _model =CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
  #return _model

#model=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

#def rerank(question:str, chunks: list, top_k=3)->list:
 # model= get_model()
 # pairs=[[question,chunk.page_content] for chunk in chunks]
 # scores=model.predict(pairs)

 # scored_chunks=sorted(zip(scores,chunks),key=lambda x:x[0],reverse=True)
  #return [chunk for _, chunk in scored_chunks[:top_k]]



import logging
import time

import cohere
import os

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.monitoring.metrics import RERANK_LATENCY


logger= logging.getLogger(__name__)
_client = None

def get_client():
    global _client
    if _client is None:
        api_key=os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY environment variable is not set ")
        _client = cohere.ClientV2(api_key)
    return _client

@retry(stop=stop_after_attempt(3),
       wait= wait_exponential(multiplier=1,min=2,max=10),
       retry= retry_if_exception_type((Exception,)),
       reraise= False)

def _rerank_with_retry(client, question:str, texts:list,top_k:int ):

    return client.rerank(
        model="rerank-v3.5" ,
        query=question ,
        documents=texts,
        top_n=top_k
   )



def rerank(question: str, chunks: list, top_k=3) -> list:
    

    if not chunks :
        return chunks 

    # Extract text from chunks (same as before, chunks are LangChain Documents)
    texts = [chunk.page_content for chunk in chunks]
    start_time= time.time()
    outcome="failure"
    
    try :

     client = get_client()
     results = _rerank_with_retry(client,question,texts,top_k)
     outcome= "success"
     return [chunks[r.index] for r in results.results]
     
    
    except Exception as e:
        logger.warning(
            f"Reranking failed after retries, falling back to original order. Error {e}"
        )
        return chunks[:top_k]
    
    finally:
     duration= time.time() - start_time
     RERANK_LATENCY.labels(outcome=outcome).observe(duration)

