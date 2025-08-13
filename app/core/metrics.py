from prometheus_client import Counter, Gauge, Histogram

# --- Request Metrics ---

REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total number of HTTP requests.",
    ["method", "path", "status_code"],
)

REQUESTS_LATENCY_SECONDS = Histogram(
    "http_requests_latency_seconds",
    "HTTP request latency in seconds.",
    ["method", "path"],
)

REQUESTS_IN_PROGRESS = Gauge(
    "http_requests_in_progress",
    "Number of HTTP requests currently in progress.",
    ["method", "path"],
)

# --- Model Loading Metrics ---

MODEL_LOAD_SUCCESS_TOTAL = Counter(
    "model_load_success_total",
    "Total number of successful model loads.",
    ["model_name", "model_version", "model_stage"],
)

MODEL_LOAD_FAILED_TOTAL = Counter(
    "model_load_failed_total",
    "Total number of failed model loads.",
    ["model_name"],
)

MODEL_WARMUP_LATENCY_SECONDS = Histogram(
    "model_warmup_latency_seconds",
    "Model warmup latency in seconds.",
    ["model_name", "model_version"],
)

MODEL_LOADED_INFO = Gauge(
    "model_loaded_info",
    "Information about the loaded models.",
    ["model_name", "model_version", "model_stage", "run_id"],
)

# --- Prediction Metrics ---

PREDICTIONS_TOTAL = Counter(
    "predictions_total",
    "Total number of predictions made.",
    ["model_name", "model_version"],
)

PREDICTION_LATENCY_SECONDS = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds.",
    ["model_name", "model_version"],
)
