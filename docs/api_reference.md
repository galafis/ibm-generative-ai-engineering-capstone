# Referência da API - Generative AI Platform

## 🤖 Model Management

### Load Model
```http
POST /api/models/load
Content-Type: application/json

{
    "model_name": "gpt-3.5-turbo",
    "device": "cuda",
    "precision": "fp16"
}
```

### List Models
```http
GET /api/models
```

Response:
```json
{
    "models": [
        {
            "name": "gpt-3.5-turbo",
            "status": "loaded",
            "memory_usage": "4.2GB"
        }
    ]
}
```

## 🎯 Text Generation

### Generate Text
```http
POST /api/generate
Content-Type: application/json

{
    "prompt": "Write a story about AI",
    "max_tokens": 100,
    "temperature": 0.7,
    "top_p": 0.9
}
```

Response:
```json
{
    "generated_text": "Once upon a time...",
    "tokens_used": 95,
    "latency_ms": 150
}
```

### Batch Generation
```http
POST /api/generate/batch
Content-Type: application/json

{
    "prompts": ["Prompt 1", "Prompt 2"],
    "max_tokens": 50
}
```

## 🎨 Prompt Engineering

### Optimize Prompt
```http
POST /api/prompts/optimize
Content-Type: application/json

{
    "task": "sentiment_analysis",
    "examples": [
        {"input": "Great!", "output": "positive"},
        {"input": "Terrible", "output": "negative"}
    ],
    "iterations": 50
}
```

### Test Prompt
```http
POST /api/prompts/test
Content-Type: application/json

{
    "prompt_template": "Classify: {text}",
    "test_cases": [
        {"text": "Amazing product!"},
        {"text": "Poor service"}
    ]
}
```

## 🔧 Fine-tuning

### Start Training
```http
POST /api/training/start
Content-Type: application/json

{
    "base_model": "gpt-3.5-turbo",
    "dataset": "custom_data.json",
    "config": {
        "learning_rate": 1e-4,
        "epochs": 3,
        "batch_size": 16
    }
}
```

### Training Status
```http
GET /api/training/status/{job_id}
```

Response:
```json
{
    "status": "training",
    "progress": 0.65,
    "current_loss": 0.234,
    "eta_minutes": 45
}
```

## 📊 Evaluation

### Run Benchmark
```http
POST /api/evaluation/benchmark
Content-Type: application/json

{
    "model": "custom_model",
    "benchmark": "glue",
    "tasks": ["sst2", "mrpc"]
}
```

### Custom Evaluation
```http
POST /api/evaluation/custom
Content-Type: application/json

{
    "model": "custom_model",
    "test_data": "test_dataset.json",
    "metrics": ["accuracy", "f1_score"]
}
```

## 🛡️ Safety & Ethics

### Content Filter
```http
POST /api/safety/filter
Content-Type: application/json

{
    "text": "Content to check",
    "filters": ["toxicity", "bias", "harmful"]
}
```

### Bias Detection
```http
POST /api/ethics/bias
Content-Type: application/json

{
    "model": "custom_model",
    "test_data": "bias_test.json",
    "protected_attributes": ["gender", "race"]
}
```

## 📈 Monitoring

### Model Metrics
```http
GET /api/monitoring/metrics/{model_name}
```

Response:
```json
{
    "latency_p95": 200,
    "throughput_rps": 50,
    "error_rate": 0.01,
    "memory_usage": "4.2GB"
}
```

### System Health
```http
GET /api/monitoring/health
```

## 🔐 Authentication

All endpoints require authentication:

```http
Authorization: Bearer YOUR_API_KEY
```

### Rate Limits
- Free tier: 100 requests/hour
- Pro tier: 1000 requests/hour
- Enterprise: Unlimited

## 📝 Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request |
| 401 | Unauthorized |
| 429 | Rate Limited |
| 500 | Server Error |
| 503 | Model Unavailable |

## 🔄 Webhooks

Configure webhooks for training completion:

```http
POST /api/webhooks/configure
Content-Type: application/json

{
    "url": "https://your-app.com/webhook",
    "events": ["training_complete", "evaluation_done"]
}
```
