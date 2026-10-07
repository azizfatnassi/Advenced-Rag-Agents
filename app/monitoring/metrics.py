from prometheus_client import Counter, Histogram , Gauge

REQUEST_COUNT=Counter("rag_requests_total", "Total requests recently",["endpoint","status"])

REQUEST_LATENCY=Histogram("rag_request_latency_seconds",
                          "Requests latency in seconds",
                          ["endpoint"],
                          buckets=(0.5,1,2,5,10,20,30,40,50,60,80,95))


SESSIONS_CREATED=Counter("rag_sessions_created_total","Total sessions created")

TOKEN_USAGE = Counter(
    "rag_tokens_total", "Total tokens consumed", ["model","task"]
)

AGENT_ITERATIONS = Histogram(
    "rag_agent_iterations", "Number of iterations per agent run",
    buckets=(1, 2, 3, 4, 5, 6, 8, 10)
)

RERANK_LATENCY = Histogram(
    "rag_rerank_latency_seconds",
    "Reranking latency in seconds",
    ["outcome"],
    buckets=(0.1, 0.25, 0.5, 1, 2, 3, 5, 10)
)

ERROR_COUNT=Counter(
   "rag_errors_total","Total errors by type ",["endpoint","error_type"])
