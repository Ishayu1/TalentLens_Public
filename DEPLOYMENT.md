# TalentLens Cloud Run Deployment

The Docker image pre-caches `all-MiniLM-L6-v2` at build time and points
`TALENTLENS_EMBEDDING_MODEL` at that local path. This lets the API load semantic
FAISS retrieval in Cloud Run without downloading the embedding model at runtime.

## Build

```sh
gcloud builds submit --tag gcr.io/PROJECT_ID/talentlens-api
```

## Deploy

```sh
gcloud run deploy talentlens-api \
  --image gcr.io/PROJECT_ID/talentlens-api \
  --region us-west1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300
```

## Verify

After deployment, call `/health`. The health endpoint warms the engine and should
report semantic chunk retrieval when the model, FAISS index, and chunk metadata
load successfully.

```sh
curl https://SERVICE_URL/health
```

Expected fields:

```json
{
  "engine_loaded": true,
  "mode_label": "Live",
  "retrieval_backend": "semantic-chunk"
}
```

If `retrieval_backend` is `lexical-chunk`, inspect `startup_issues` in the
health response.
