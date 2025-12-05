# GCP Deployment Guide for YRSN-IARS

This guide covers deploying the Intelligent Approval Routing System on Google Cloud Platform.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           YRSN-IARS on GCP                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│  │ Cloud    │───>│ Cloud    │───>│ Workflows│───>│ Cloud    │             │
│  │ Endpoints│    │ Functions│    │          │    │ Functions│             │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘             │
│                        │              │               │                    │
│                        ▼              ▼               ▼                    │
│                  ┌──────────┐   ┌──────────┐   ┌──────────┐               │
│                  │ Vertex   │   │ Vertex   │   │ Firestore│               │
│                  │AI Embed  │   │AI Predict│   │ Database │               │
│                  └──────────┘   └──────────┘   └──────────┘               │
│                        │              │               │                    │
│                        ▼              ▼               ▼                    │
│                  ┌──────────┐   ┌──────────┐   ┌──────────┐               │
│                  │  Cloud   │   │  Cloud   │   │  Pub/Sub │               │
│                  │ Storage  │   │Monitoring│   │  Alerts  │               │
│                  └──────────┘   └──────────┘   └──────────┘               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- GCP Account with billing enabled
- gcloud CLI configured (`gcloud auth login`)
- Python 3.10+
- Docker (for Cloud Run)

## Quick Start

```bash
# Clone repository
git clone https://github.com/your-org/yrsn-iars.git
cd yrsn-iars

# Install dependencies
pip install -e ".[full,dev]"

# Configure GCP
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable \
    cloudfunctions.googleapis.com \
    run.googleapis.com \
    aiplatform.googleapis.com \
    firestore.googleapis.com \
    storage.googleapis.com \
    workflows.googleapis.com \
    monitoring.googleapis.com

# Deploy infrastructure
cd infrastructure/gcp
./deploy.sh
```

---

## Step 1: Cloud Storage (Data Storage)

### Create Buckets

```bash
# Set project
PROJECT_ID=$(gcloud config get-value project)
REGION="us-central1"

# Create buckets
gsutil mb -l $REGION gs://${PROJECT_ID}-yrsn-data
gsutil mb -l $REGION gs://${PROJECT_ID}-yrsn-models
gsutil mb -l $REGION gs://${PROJECT_ID}-yrsn-artifacts

# Enable versioning
gsutil versioning set on gs://${PROJECT_ID}-yrsn-data
```

### Upload Data

```bash
# Upload training data
gsutil cp data/approval_history.csv gs://${PROJECT_ID}-yrsn-data/training/

# Upload policy documents
gsutil -m cp -r data/policies/ gs://${PROJECT_ID}-yrsn-data/policies/
```

---

## Step 2: Vertex AI (Embeddings & Predictions)

### Enable Vertex AI

```bash
gcloud services enable aiplatform.googleapis.com
```

### Generate Embeddings with Vertex AI

```python
# scripts/vertex_embeddings.py
from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel

# Initialize
aiplatform.init(project="YOUR_PROJECT_ID", location="us-central1")

def get_embedding(text: str) -> list:
    """Get embedding from Vertex AI."""
    model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
    embeddings = model.get_embeddings([text])
    return embeddings[0].values

# Test
embedding = get_embedding("Test approval request")
print(f"Embedding dimension: {len(embedding)}")  # 768
```

### Deploy Classifier to Vertex AI

```python
# scripts/deploy_vertex_model.py
from google.cloud import aiplatform

aiplatform.init(project="YOUR_PROJECT_ID", location="us-central1")

# Upload model
model = aiplatform.Model.upload(
    display_name="yrsn-iars-classifier",
    artifact_uri="gs://YOUR_PROJECT_ID-yrsn-models/classifier/",
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-2:latest"
)

# Deploy to endpoint
endpoint = model.deploy(
    machine_type="n1-standard-2",
    min_replica_count=1,
    max_replica_count=5,
    traffic_split={"0": 100},
    deployed_model_display_name="yrsn-classifier-v1"
)

print(f"Endpoint: {endpoint.resource_name}")
```

### Serverless Prediction (Alternative)

For variable workloads, use Vertex AI Prediction with auto-scaling:

```python
# Deploy with auto-scaling
endpoint = model.deploy(
    machine_type="n1-standard-2",
    min_replica_count=0,  # Scale to zero
    max_replica_count=10,
    accelerator_type=None,
    traffic_split={"0": 100}
)
```

---

## Step 3: Firestore (Decision Storage)

### Create Firestore Database

```bash
# Create Firestore in Native mode
gcloud firestore databases create --location=us-central1
```

### Collection Schema

```python
# Example decision document
decision_doc = {
    'request_id': 'REQ-00001',
    'timestamp': firestore.SERVER_TIMESTAMP,
    'stream': 'green',
    'confidence_score': 0.92,
    'R': 0.75,
    'S': 0.15,
    'N': 0.10,
    'collapse_type': 'none',
    'temperature': 1.33,
    'urgency': 'standard',
    'urgency_score': 0.45,
    'category': 'software_license',
    'amount': 5000,
    'p_green': 0.72,
    'p_yellow': 0.21,
    'p_red': 0.07,
    'processing_time_ms': 245
}
```

### Firestore Client Code

```python
from google.cloud import firestore

db = firestore.Client()

def store_decision(decision: dict):
    """Store routing decision in Firestore."""
    doc_ref = db.collection('decisions').document(decision['request_id'])
    doc_ref.set(decision)

def get_decisions_by_stream(stream: str, limit: int = 100):
    """Query decisions by stream."""
    decisions_ref = db.collection('decisions')
    query = decisions_ref.where('stream', '==', stream).limit(limit)
    return [doc.to_dict() for doc in query.stream()]
```

---

## Step 4: Cloud Functions (Serverless Compute)

### Main Router Function

```python
# cloud_functions/router/main.py
import json
import functions_framework
from google.cloud import aiplatform, firestore
from vertexai.language_models import TextEmbeddingModel
from datetime import datetime

from yrsn_iars.adapters.cleanlab_adapter import CleanlabAdapter
from yrsn_iars.adapters.temperature import TemperatureConfig, TemperatureMode
from yrsn_iars.pipelines.approval_router import ApprovalRouter, ApprovalRequest

# Initialize clients
aiplatform.init(project="YOUR_PROJECT_ID", location="us-central1")
db = firestore.Client()
embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")

# Initialize router
adapter = CleanlabAdapter()
temp_config = TemperatureConfig(
    mode=TemperatureMode.ADAPTIVE,
    tau_min=0.3,
    tau_max=3.0
)
router = ApprovalRouter(
    cleanlab_adapter=adapter,
    temperature_config=temp_config
)

# Vertex AI endpoint
ENDPOINT_ID = "YOUR_ENDPOINT_ID"
endpoint = aiplatform.Endpoint(ENDPOINT_ID)

@functions_framework.http
def route_request(request):
    """HTTP Cloud Function for routing approval requests."""
    start_time = datetime.now()

    # Parse request
    request_json = request.get_json(silent=True)
    if not request_json:
        return json.dumps({'error': 'Invalid JSON'}), 400

    # Create ApprovalRequest
    approval_request = ApprovalRequest(
        request_id=request_json['request_id'],
        text=request_json['text'],
        category=request_json['category'],
        amount=float(request_json['amount']),
        requestor_id=request_json['requestor_id'],
        deadline=request_json.get('deadline'),
        requestor_level=request_json.get('requestor_level')
    )

    # Get embedding
    embeddings = embedding_model.get_embeddings([approval_request.text])
    embedding = embeddings[0].values

    # Get prediction from Vertex AI
    prediction = endpoint.predict(instances=[{"embedding": embedding}])
    pred_probs = prediction.predictions[0]['pred_probs']
    classifier_confidence = max(pred_probs)

    # Route request
    decision = router.route(
        request=approval_request,
        classifier_confidence=classifier_confidence,
        label_quality=0.85,
        pred_probs=pred_probs
    )

    # Calculate processing time
    processing_time = (datetime.now() - start_time).total_seconds() * 1000

    # Store in Firestore
    decision_doc = {
        **decision.to_dict(),
        'processing_time_ms': processing_time
    }
    db.collection('decisions').document(decision.request_id).set(decision_doc)

    # Return response
    return json.dumps({
        'request_id': decision.request_id,
        'stream': decision.stream.value,
        'confidence_score': decision.confidence_score,
        'urgency': decision.urgency.value,
        'temperature': decision.temperature,
        'YRSN': {'R': decision.R, 'S': decision.S, 'N': decision.N},
        'soft_probabilities': {
            'green': decision.p_green,
            'yellow': decision.p_yellow,
            'red': decision.p_red
        },
        'processing_time_ms': processing_time
    }), 200, {'Content-Type': 'application/json'}
```

### Requirements.txt

```
# cloud_functions/router/requirements.txt
functions-framework==3.*
google-cloud-aiplatform>=1.36.0
google-cloud-firestore>=2.13.0
vertexai>=1.36.0
yrsn-iars>=0.1.0
cleanlab>=2.5.0
numpy>=1.22.0
pandas>=1.4.0
scikit-learn>=1.1.0
```

### Deploy Function

```bash
cd cloud_functions/router

gcloud functions deploy yrsn-iars-router \
    --gen2 \
    --runtime python311 \
    --trigger-http \
    --allow-unauthenticated \
    --region us-central1 \
    --memory 1024MB \
    --timeout 60s \
    --entry-point route_request \
    --set-env-vars PROJECT_ID=YOUR_PROJECT_ID
```

---

## Step 5: Cloud Run (Alternative to Cloud Functions)

For more control and longer execution times:

### Dockerfile

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Run with gunicorn
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app
```

### Flask Application

```python
# main.py
import os
from flask import Flask, request, jsonify
from google.cloud import aiplatform, firestore
from vertexai.language_models import TextEmbeddingModel

from yrsn_iars.adapters.cleanlab_adapter import CleanlabAdapter
from yrsn_iars.adapters.temperature import TemperatureConfig, TemperatureMode
from yrsn_iars.pipelines.approval_router import ApprovalRouter, ApprovalRequest

app = Flask(__name__)

# Initialize (same as Cloud Function)
# ...

@app.route('/route', methods=['POST'])
def route():
    """Route approval request."""
    # Same logic as Cloud Function
    pass

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
```

### Deploy Cloud Run

```bash
# Build and deploy
gcloud run deploy yrsn-iars-router \
    --source . \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 1Gi \
    --cpu 1 \
    --min-instances 0 \
    --max-instances 10 \
    --timeout 60
```

---

## Step 6: Cloud Workflows (Orchestration)

### Workflow Definition

```yaml
# workflows/approval_workflow.yaml
main:
  params: [request]
  steps:
    - route_request:
        call: http.post
        args:
          url: ${sys.get_env("ROUTER_URL")}
          body: ${request}
        result: routing_result

    - check_stream:
        switch:
          - condition: ${routing_result.body.stream == "green"}
            next: auto_process
          - condition: ${routing_result.body.stream == "yellow"}
            next: ai_assisted_review
          - condition: ${routing_result.body.stream == "red"}
            next: expert_review

    - auto_process:
        call: http.post
        args:
          url: ${sys.get_env("AUTO_PROCESS_URL")}
          body: ${routing_result.body}
        result: process_result
        next: end

    - ai_assisted_review:
        call: http.post
        args:
          url: ${sys.get_env("AI_ASSIST_URL")}
          body: ${routing_result.body}
        result: ai_result
        next: wait_human_decision

    - expert_review:
        steps:
          - notify_expert:
              call: googleapis.pubsub.v1.projects.topics.publish
              args:
                topic: ${"projects/" + sys.get_env("PROJECT_ID") + "/topics/expert-review"}
                body:
                  messages:
                    - data: ${base64.encode(json.encode(routing_result.body))}
          - await_decision:
              call: events.await_callback
              args:
                callback_id: ${routing_result.body.request_id}
                timeout: 86400  # 24 hours
              result: expert_decision
        next: process_decision

    - wait_human_decision:
        call: events.await_callback
        args:
          callback_id: ${routing_result.body.request_id}
          timeout: 3600  # 1 hour
        result: human_decision
        next: process_decision

    - process_decision:
        call: http.post
        args:
          url: ${sys.get_env("PROCESS_DECISION_URL")}
          body:
            request_id: ${routing_result.body.request_id}
            final_decision: ${human_decision}
        result: final_result

    - end:
        return: ${final_result}
```

### Deploy Workflow

```bash
gcloud workflows deploy yrsn-iars-approval \
    --location us-central1 \
    --source workflows/approval_workflow.yaml \
    --set-env-vars ROUTER_URL=https://yrsn-iars-router-xxx.run.app,PROJECT_ID=YOUR_PROJECT_ID
```

---

## Step 7: Vertex AI Search (RAG for Policies)

### Create Search App

```bash
# Create data store
gcloud alpha discovery-engine data-stores create yrsn-policies \
    --location global \
    --content-type document

# Import documents from GCS
gcloud alpha discovery-engine documents import \
    --data-store yrsn-policies \
    --location global \
    --input-uris "gs://${PROJECT_ID}-yrsn-data/policies/**"

# Create search app
gcloud alpha discovery-engine engines create yrsn-policy-search \
    --location global \
    --data-store yrsn-policies \
    --solution-type search
```

### Query Policies in Router

```python
from google.cloud import discoveryengine_v1 as discoveryengine

def search_policies(query: str, project_id: str) -> list:
    """Search policy documents."""
    client = discoveryengine.SearchServiceClient()

    serving_config = f"projects/{project_id}/locations/global/dataStores/yrsn-policies/servingConfigs/default_search"

    request = discoveryengine.SearchRequest(
        serving_config=serving_config,
        query=query,
        page_size=3
    )

    response = client.search(request)

    results = []
    for result in response.results:
        results.append({
            'title': result.document.derived_struct_data.get('title'),
            'snippet': result.document.derived_struct_data.get('snippets', [{}])[0].get('snippet'),
            'link': result.document.derived_struct_data.get('link')
        })

    return results
```

---

## Step 8: Cloud Monitoring

### Create Dashboard

```python
# scripts/create_gcp_dashboard.py
from google.cloud import monitoring_dashboard_v1

client = monitoring_dashboard_v1.DashboardsServiceClient()
project_name = f"projects/YOUR_PROJECT_ID"

dashboard = monitoring_dashboard_v1.Dashboard(
    display_name="YRSN-IARS Production",
    grid_layout=monitoring_dashboard_v1.GridLayout(
        columns=2,
        widgets=[
            monitoring_dashboard_v1.Widget(
                title="Routing Distribution",
                xy_chart=monitoring_dashboard_v1.XyChart(
                    data_sets=[
                        monitoring_dashboard_v1.XyChart.DataSet(
                            time_series_query=monitoring_dashboard_v1.TimeSeriesQuery(
                                time_series_filter=monitoring_dashboard_v1.TimeSeriesFilter(
                                    filter='metric.type="custom.googleapis.com/yrsn/stream_count"'
                                )
                            )
                        )
                    ]
                )
            ),
            monitoring_dashboard_v1.Widget(
                title="Average Temperature",
                xy_chart=monitoring_dashboard_v1.XyChart(
                    data_sets=[
                        monitoring_dashboard_v1.XyChart.DataSet(
                            time_series_query=monitoring_dashboard_v1.TimeSeriesQuery(
                                time_series_filter=monitoring_dashboard_v1.TimeSeriesFilter(
                                    filter='metric.type="custom.googleapis.com/yrsn/temperature"'
                                )
                            )
                        )
                    ]
                )
            )
        ]
    )
)

client.create_dashboard(parent=project_name, dashboard=dashboard)
```

### Custom Metrics

```python
from google.cloud import monitoring_v3
from google.protobuf import timestamp_pb2
import time

def publish_metrics(decision, processing_time, project_id):
    """Publish custom metrics to Cloud Monitoring."""
    client = monitoring_v3.MetricServiceClient()
    project_name = f"projects/{project_id}"

    now = time.time()
    seconds = int(now)
    nanos = int((now - seconds) * 10**9)

    interval = monitoring_v3.TimeInterval(
        end_time={"seconds": seconds, "nanos": nanos}
    )

    # Stream count metric
    stream_series = monitoring_v3.TimeSeries(
        metric=monitoring_v3.Metric(
            type="custom.googleapis.com/yrsn/stream_count",
            labels={"stream": decision.stream.value}
        ),
        resource=monitoring_v3.MonitoredResource(
            type="global",
            labels={"project_id": project_id}
        ),
        points=[
            monitoring_v3.Point(
                interval=interval,
                value=monitoring_v3.TypedValue(int64_value=1)
            )
        ]
    )

    # Temperature metric
    temp_series = monitoring_v3.TimeSeries(
        metric=monitoring_v3.Metric(
            type="custom.googleapis.com/yrsn/temperature"
        ),
        resource=monitoring_v3.MonitoredResource(
            type="global",
            labels={"project_id": project_id}
        ),
        points=[
            monitoring_v3.Point(
                interval=interval,
                value=monitoring_v3.TypedValue(double_value=decision.temperature)
            )
        ]
    )

    client.create_time_series(
        name=project_name,
        time_series=[stream_series, temp_series]
    )
```

### Alerting Policy

```bash
# Create alert for high collapse rate
gcloud alpha monitoring policies create \
    --display-name="YRSN High Collapse Rate" \
    --condition-filter='metric.type="custom.googleapis.com/yrsn/collapse_count" AND metric.labels.type!="none"' \
    --condition-threshold-value=50 \
    --condition-threshold-comparison=COMPARISON_GT \
    --condition-threshold-duration=300s \
    --notification-channels="projects/YOUR_PROJECT/notificationChannels/CHANNEL_ID"
```

---

## Step 9: Pub/Sub (Event Messaging)

### Create Topics

```bash
# Create topics
gcloud pubsub topics create yrsn-routing-events
gcloud pubsub topics create yrsn-expert-review
gcloud pubsub topics create yrsn-alerts

# Create subscriptions
gcloud pubsub subscriptions create yrsn-routing-sub \
    --topic yrsn-routing-events \
    --push-endpoint https://your-handler-url.run.app

gcloud pubsub subscriptions create yrsn-expert-review-sub \
    --topic yrsn-expert-review \
    --ack-deadline 600
```

### Publish Events

```python
from google.cloud import pubsub_v1
import json

publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path("YOUR_PROJECT_ID", "yrsn-routing-events")

def publish_routing_event(decision):
    """Publish routing decision event."""
    data = json.dumps(decision.to_dict()).encode('utf-8')

    future = publisher.publish(
        topic_path,
        data,
        stream=decision.stream.value,
        category=decision.category
    )

    return future.result()
```

---

## Step 10: IAM & Security

### Service Account

```bash
# Create service account
gcloud iam service-accounts create yrsn-iars-sa \
    --display-name="YRSN-IARS Service Account"

# Grant roles
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:yrsn-iars-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:yrsn-iars-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/datastore.user"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:yrsn-iars-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.objectViewer"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:yrsn-iars-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/monitoring.metricWriter"
```

### VPC Service Controls (Optional)

```bash
# Create access policy
gcloud access-context-manager policies create \
    --organization YOUR_ORG_ID \
    --title "YRSN-IARS Policy"

# Create service perimeter
gcloud access-context-manager perimeters create yrsn-perimeter \
    --policy POLICY_ID \
    --title "YRSN Service Perimeter" \
    --resources "projects/YOUR_PROJECT_NUMBER" \
    --restricted-services "storage.googleapis.com,aiplatform.googleapis.com"
```

---

## Cost Estimation

| Service | Usage | Estimated Monthly Cost |
|---------|-------|----------------------|
| Cloud Functions | 1M invocations | $0.40 |
| Cloud Run | 1M requests, 1 vCPU | $20 |
| Vertex AI Embeddings | 1M requests | $25 |
| Vertex AI Prediction | 1M predictions | $50-100 |
| Firestore | 1M writes, 5M reads | $15 |
| Cloud Storage | 10GB | $0.20 |
| Cloud Monitoring | Metrics + Logs | $10 |
| **Total** | | **~$120-170/month** |

For cost optimization:
- Use committed use discounts for Vertex AI
- Enable Cloud Functions min instances (cold start vs cost tradeoff)
- Use Firestore TTL to auto-delete old decisions

---

## Terraform Deployment (IaC)

```hcl
# terraform/main.tf
provider "google" {
  project = var.project_id
  region  = var.region
}

# Cloud Storage
resource "google_storage_bucket" "data" {
  name     = "${var.project_id}-yrsn-data"
  location = var.region
}

# Firestore
resource "google_firestore_database" "default" {
  name        = "(default)"
  location_id = var.region
  type        = "FIRESTORE_NATIVE"
}

# Cloud Run Service
resource "google_cloud_run_service" "router" {
  name     = "yrsn-iars-router"
  location = var.region

  template {
    spec {
      containers {
        image = "gcr.io/${var.project_id}/yrsn-iars-router:latest"
        resources {
          limits = {
            memory = "1Gi"
            cpu    = "1"
          }
        }
      }
      service_account_name = google_service_account.yrsn.email
    }
  }
}

# Service Account
resource "google_service_account" "yrsn" {
  account_id   = "yrsn-iars-sa"
  display_name = "YRSN-IARS Service Account"
}

# IAM bindings
resource "google_project_iam_member" "vertex_ai" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.yrsn.email}"
}
```

### Deploy with Terraform

```bash
cd terraform
terraform init
terraform plan
terraform apply
```

---

## Production Checklist

- [ ] Enable Cloud Armor for DDoS protection
- [ ] Configure Identity-Aware Proxy (IAP)
- [ ] Set up Secret Manager for API keys
- [ ] Enable Cloud Trace for debugging
- [ ] Configure Cloud Backup for Firestore
- [ ] Set up Cloud Build for CI/CD
- [ ] Configure multi-region deployment (if required)
- [ ] Implement request quotas
- [ ] Set up budget alerts
- [ ] Review IAM policies (least privilege)

---

## Troubleshooting

### Cold Start Latency

```bash
# Set min instances for Cloud Run
gcloud run services update yrsn-iars-router \
    --min-instances 1 \
    --region us-central1
```

### Vertex AI Quota

```bash
# Request quota increase
gcloud alpha services quota update \
    --service aiplatform.googleapis.com \
    --consumer projects/YOUR_PROJECT_ID \
    --metric aiplatform.googleapis.com/online_prediction_requests_per_minute \
    --value 1000
```

### Firestore Performance

```python
# Use batch writes
batch = db.batch()
for decision in decisions:
    doc_ref = db.collection('decisions').document(decision['request_id'])
    batch.set(doc_ref, decision)
batch.commit()
```

---

## GCP vs AWS Service Mapping

| AWS Service | GCP Equivalent | Notes |
|-------------|---------------|-------|
| Lambda | Cloud Functions / Cloud Run | Cloud Run for longer timeouts |
| SageMaker | Vertex AI | Similar ML workflow |
| Bedrock | Vertex AI (Gemini, Palm) | Different model selection |
| DynamoDB | Firestore | Firestore has richer queries |
| API Gateway | Cloud Endpoints / API Gateway | API Gateway is newer |
| Step Functions | Workflows | Similar orchestration |
| S3 | Cloud Storage | Nearly identical |
| CloudWatch | Cloud Monitoring | Similar metrics/logs |
| SNS/SQS | Pub/Sub | Unified messaging service |
| Kendra | Vertex AI Search | Enterprise search |

---

For questions or support, see the [GitHub Issues](https://github.com/your-org/yrsn-iars/issues).
