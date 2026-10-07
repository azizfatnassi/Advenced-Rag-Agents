from contextlib import asynccontextmanager
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from loguru import logger
from app.api.routes import router
from app.monitoring.logging_config import setup_logging
from app.monitoring.metrics import ERROR_COUNT, REQUEST_COUNT, REQUEST_LATENCY

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from app.rag.reranker import rerank
        from langchain_core.documents import Document
        dummy = [Document(page_content="warmup", metadata={})]
        rerank("warmup", dummy, top_k=1)
        logger.info("reranker warmed up")
    except Exception as e:
        logger.warning(f"Reranker warmup skipped: {e}")
    yield


setup_logging()

app = FastAPI(docs_url=None, lifespan=lifespan)



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def track_metrics(request:Request,call_next):
    start_time= time.time()
    response=None
    status="500"
    error_type=None

    try:
        response= await call_next(request)
        status= str(response.status_code)
        return response
    except Exception as e:
        error_type= type(e).__name__
        raise 
    finally:
     duration= time.time() - start_time
     route= request.scope.get("route")
     endpoint= route.path if route else "unmatched"

     if endpoint== "unmatched":
         logger.warning(f"unmatched route hit : {request.url.path}")



     REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)
     REQUEST_COUNT.labels(endpoint=endpoint,status=status).inc()
     if error_type:
         ERROR_COUNT.labels(endpoint=endpoint, error_type=error_type).inc()




    



app.include_router(router)

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Advanced RAG-Agent API",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
    )