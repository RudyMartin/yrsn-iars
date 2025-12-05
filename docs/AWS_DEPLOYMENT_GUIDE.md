# AWS Deployment Guide for YRSN-IARS

This guide covers deploying the Intelligent Approval Routing System on AWS infrastructure.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           YRSN-IARS on AWS                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│  │ API GW   │───>│ Lambda   │───>│ Step     │───>│ Lambda   │             │
│  │ /request │    │ Ingest   │    │ Functions│    │ Router   │             │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘             │
│                        │              │               │                    │
│                        ▼              ▼               ▼                    │
│                  ┌──────────┐   ┌──────────┐   ┌──────────┐               │
│                  │ Bedrock  │   │SageMaker │   │ DynamoDB │               │
│                  │Embeddings│   │Classifier│   │ Decisions│               │
│                  └──────────┘   └──────────┘   └──────────┘               │
│                        │              │               │                    │
│                        ▼              ▼               ▼                    │
│                  ┌──────────┐   ┌──────────┐   ┌──────────┐               │
│                  │    S3    │   │CloudWatch│   │   SNS    │               │
│                  │  Data    │   │ Metrics  │   │ Alerts   │               │
│                  └──────────┘   └──────────┘   └──────────┘               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- AWS Account with appropriate permissions
- AWS CLI configured (`aws configure`)
- Python 3.10+
- Docker (for Lambda layers)

## Quick Start

```bash
# Clone repository
git clone https://github.com/your-org/yrsn-iars.git
cd yrsn-iars

# Install dependencies
pip install -e ".[full,dev]"

# Configure AWS
aws configure

# Deploy infrastructure
cd infrastructure
./deploy.sh
```

---

## Step 1: S3 Setup (Data Storage)

### Create Buckets

```bash
# Set your bucket prefix (must be globally unique)
BUCKET_PREFIX="yrsn-iars-$(aws sts get-caller-identity --query Account --output text)"

# Create buckets
aws s3 mb s3://${BUCKET_PREFIX}-data
aws s3 mb s3://${BUCKET_PREFIX}-models
aws s3 mb s3://${BUCKET_PREFIX}-artifacts

# Enable versioning
aws s3api put-bucket-versioning \
    --bucket ${BUCKET_PREFIX}-data \
    --versioning-configuration Status=Enabled
```

### Upload Training Data

```bash
# Upload historical approval data
aws s3 cp data/approval_history.csv s3://${BUCKET_PREFIX}-data/training/

# Upload policy documents (for RAG)
aws s3 sync data/policies/ s3://${BUCKET_PREFIX}-data/policies/
```

### Bucket Policy (restrict access)

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": [
                    "sagemaker.amazonaws.com",
                    "lambda.amazonaws.com",
                    "bedrock.amazonaws.com"
                ]
            },
            "Action": [
                "s3:GetObject",
                "s3:PutObject"
            ],
            "Resource": "arn:aws:s3:::${BUCKET_PREFIX}-*/*"
        }
    ]
}
```

---

## Step 2: Amazon Bedrock (Embeddings)

### Enable Bedrock Access

1. Go to AWS Console → Amazon Bedrock
2. Navigate to "Model access"
3. Request access to:
   - **Amazon Titan Embeddings** (for text embeddings)
   - **Claude** (optional, for AI-assisted review)

### Test Bedrock Access

```python
import boto3
import json

bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')

def get_embedding(text: str) -> list:
    """Get embedding from Amazon Titan."""
    response = bedrock.invoke_model(
        modelId='amazon.titan-embed-text-v1',
        body=json.dumps({"inputText": text})
    )
    result = json.loads(response['body'].read())
    return result['embedding']

# Test
embedding = get_embedding("Test approval request")
print(f"Embedding dimension: {len(embedding)}")  # Should be 1536
```

### Bedrock Embedding Lambda

```python
# lambda_functions/embedding_handler.py
import boto3
import json

bedrock = boto3.client('bedrock-runtime')

def handler(event, context):
    """Generate embeddings for approval requests."""
    texts = event.get('texts', [])

    embeddings = []
    for text in texts:
        response = bedrock.invoke_model(
            modelId='amazon.titan-embed-text-v1',
            body=json.dumps({"inputText": text})
        )
        result = json.loads(response['body'].read())
        embeddings.append(result['embedding'])

    return {
        'statusCode': 200,
        'embeddings': embeddings
    }
```

---

## Step 3: SageMaker (Classifier Endpoint)

### Option A: SageMaker Serverless Inference (Recommended for Variable Load)

```python
# scripts/deploy_sagemaker_serverless.py
import boto3
import sagemaker
from sagemaker.sklearn import SKLearnModel

# Initialize
session = sagemaker.Session()
role = "arn:aws:iam::ACCOUNT_ID:role/SageMakerExecutionRole"

# Create model
model = SKLearnModel(
    model_data=f"s3://{BUCKET_PREFIX}-models/classifier/model.tar.gz",
    role=role,
    framework_version="1.2-1",
    py_version="py3",
    entry_point="inference.py"
)

# Deploy serverless endpoint
predictor = model.deploy(
    serverless_inference_config=sagemaker.serverless.ServerlessInferenceConfig(
        memory_size_in_mb=2048,
        max_concurrency=10
    ),
    endpoint_name="yrsn-iars-classifier"
)

print(f"Endpoint deployed: {predictor.endpoint_name}")
```

### Option B: Real-time Endpoint (For High-Volume)

```python
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name="yrsn-iars-classifier-realtime"
)
```

### Inference Script

```python
# inference/inference.py
import joblib
import numpy as np

def model_fn(model_dir):
    """Load model."""
    return joblib.load(f"{model_dir}/classifier.joblib")

def predict_fn(input_data, model):
    """Make predictions."""
    embeddings = np.array(input_data['embeddings'])

    # Get prediction probabilities
    pred_probs = model.predict_proba(embeddings)
    predictions = model.predict(embeddings)

    return {
        'predictions': predictions.tolist(),
        'pred_probs': pred_probs.tolist(),
        'confidence': pred_probs.max(axis=1).tolist()
    }
```

### Train and Upload Model

```python
# scripts/train_and_upload.py
import joblib
import tarfile
from sklearn.linear_model import LogisticRegression
import boto3

# Train model (using your training data)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Save model
joblib.dump(clf, 'classifier.joblib')

# Create tarball
with tarfile.open('model.tar.gz', 'w:gz') as tar:
    tar.add('classifier.joblib')
    tar.add('inference.py')

# Upload to S3
s3 = boto3.client('s3')
s3.upload_file('model.tar.gz', f'{BUCKET_PREFIX}-models', 'classifier/model.tar.gz')
```

---

## Step 4: DynamoDB (Decision Storage)

### Create Tables

```bash
# Decisions table
aws dynamodb create-table \
    --table-name yrsn-iars-decisions \
    --attribute-definitions \
        AttributeName=request_id,AttributeType=S \
        AttributeName=timestamp,AttributeType=S \
    --key-schema \
        AttributeName=request_id,KeyType=HASH \
        AttributeName=timestamp,KeyType=RANGE \
    --billing-mode PAY_PER_REQUEST

# Metrics table
aws dynamodb create-table \
    --table-name yrsn-iars-metrics \
    --attribute-definitions \
        AttributeName=metric_type,AttributeType=S \
        AttributeName=date,AttributeType=S \
    --key-schema \
        AttributeName=metric_type,KeyType=HASH \
        AttributeName=date,KeyType=RANGE \
    --billing-mode PAY_PER_REQUEST
```

### Decision Schema

```python
# Example decision record
decision_record = {
    'request_id': 'REQ-00001',
    'timestamp': '2024-01-15T10:30:00Z',
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
    'knockout_rule': None,
    'processing_time_ms': 245
}
```

---

## Step 5: Lambda Functions

### Create Lambda Layer (Dependencies)

```bash
# Create layer with yrsn-iars and dependencies
mkdir -p layer/python
pip install yrsn-iars cleanlab numpy pandas scikit-learn -t layer/python/

cd layer
zip -r ../yrsn-iars-layer.zip python/

aws lambda publish-layer-version \
    --layer-name yrsn-iars-deps \
    --zip-file fileb://../yrsn-iars-layer.zip \
    --compatible-runtimes python3.10 python3.11
```

### Main Router Lambda

```python
# lambda_functions/router_handler.py
import json
import boto3
from datetime import datetime
from decimal import Decimal

from yrsn_iars.adapters.cleanlab_adapter import CleanlabAdapter
from yrsn_iars.adapters.temperature import TemperatureConfig, TemperatureMode
from yrsn_iars.pipelines.approval_router import ApprovalRouter, ApprovalRequest

# Initialize clients
bedrock = boto3.client('bedrock-runtime')
sagemaker = boto3.client('sagemaker-runtime')
dynamodb = boto3.resource('dynamodb')

# Initialize router (cold start)
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

# Tables
decisions_table = dynamodb.Table('yrsn-iars-decisions')

def get_embedding(text: str) -> list:
    """Get embedding from Bedrock."""
    response = bedrock.invoke_model(
        modelId='amazon.titan-embed-text-v1',
        body=json.dumps({"inputText": text})
    )
    return json.loads(response['body'].read())['embedding']

def get_prediction(embedding: list) -> dict:
    """Get prediction from SageMaker."""
    response = sagemaker.invoke_endpoint(
        EndpointName='yrsn-iars-classifier',
        ContentType='application/json',
        Body=json.dumps({'embeddings': [embedding]})
    )
    return json.loads(response['Body'].read())

def handler(event, context):
    """Process approval request and route."""
    start_time = datetime.now()

    # Parse request
    body = json.loads(event.get('body', '{}'))

    request = ApprovalRequest(
        request_id=body['request_id'],
        text=body['text'],
        category=body['category'],
        amount=float(body['amount']),
        requestor_id=body['requestor_id'],
        deadline=body.get('deadline'),
        requestor_level=body.get('requestor_level')
    )

    # Get embedding
    embedding = get_embedding(request.text)

    # Get classifier prediction
    prediction = get_prediction(embedding)
    classifier_confidence = prediction['confidence'][0]
    pred_probs = prediction['pred_probs'][0]

    # Route request
    decision = router.route(
        request=request,
        classifier_confidence=classifier_confidence,
        label_quality=0.85,  # Default or from Cleanlab batch analysis
        pred_probs=pred_probs
    )

    # Calculate processing time
    processing_time = (datetime.now() - start_time).total_seconds() * 1000

    # Store decision
    decision_record = {
        **decision.to_dict(),
        'processing_time_ms': Decimal(str(processing_time))
    }
    decisions_table.put_item(Item=decision_record)

    # Return response
    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/json'},
        'body': json.dumps({
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
        })
    }
```

### Deploy Lambda

```bash
# Package Lambda
cd lambda_functions
zip -r router_handler.zip router_handler.py

# Create function
aws lambda create-function \
    --function-name yrsn-iars-router \
    --runtime python3.11 \
    --handler router_handler.handler \
    --role arn:aws:iam::ACCOUNT_ID:role/LambdaExecutionRole \
    --zip-file fileb://router_handler.zip \
    --layers arn:aws:lambda:REGION:ACCOUNT_ID:layer:yrsn-iars-deps:1 \
    --timeout 30 \
    --memory-size 1024 \
    --environment Variables="{SAGEMAKER_ENDPOINT=yrsn-iars-classifier}"
```

---

## Step 6: API Gateway

### Create REST API

```bash
# Create API
API_ID=$(aws apigateway create-rest-api \
    --name "YRSN-IARS-API" \
    --query 'id' --output text)

# Get root resource
ROOT_ID=$(aws apigateway get-resources \
    --rest-api-id $API_ID \
    --query 'items[0].id' --output text)

# Create /route resource
ROUTE_ID=$(aws apigateway create-resource \
    --rest-api-id $API_ID \
    --parent-id $ROOT_ID \
    --path-part "route" \
    --query 'id' --output text)

# Create POST method
aws apigateway put-method \
    --rest-api-id $API_ID \
    --resource-id $ROUTE_ID \
    --http-method POST \
    --authorization-type AWS_IAM

# Integrate with Lambda
aws apigateway put-integration \
    --rest-api-id $API_ID \
    --resource-id $ROUTE_ID \
    --http-method POST \
    --type AWS_PROXY \
    --integration-http-method POST \
    --uri "arn:aws:apigateway:REGION:lambda:path/2015-03-31/functions/arn:aws:lambda:REGION:ACCOUNT_ID:function:yrsn-iars-router/invocations"

# Deploy
aws apigateway create-deployment \
    --rest-api-id $API_ID \
    --stage-name prod
```

### API Usage

```bash
# Test API
curl -X POST \
    "https://${API_ID}.execute-api.us-east-1.amazonaws.com/prod/route" \
    -H "Content-Type: application/json" \
    -d '{
        "request_id": "REQ-TEST-001",
        "text": "Request for AWS license renewal - $5,000",
        "category": "software_license",
        "amount": 5000,
        "requestor_id": "EMP-1234"
    }'
```

---

## Step 7: Step Functions (Workflow Orchestration)

For complex workflows with human-in-the-loop:

```json
{
  "Comment": "YRSN-IARS Approval Workflow",
  "StartAt": "RouteRequest",
  "States": {
    "RouteRequest": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:REGION:ACCOUNT:function:yrsn-iars-router",
      "Next": "CheckStream"
    },
    "CheckStream": {
      "Type": "Choice",
      "Choices": [
        {
          "Variable": "$.stream",
          "StringEquals": "green",
          "Next": "AutoProcess"
        },
        {
          "Variable": "$.stream",
          "StringEquals": "yellow",
          "Next": "AIAssistedReview"
        },
        {
          "Variable": "$.stream",
          "StringEquals": "red",
          "Next": "ExpertReview"
        }
      ]
    },
    "AutoProcess": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:REGION:ACCOUNT:function:auto-process",
      "End": true
    },
    "AIAssistedReview": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:REGION:ACCOUNT:function:ai-assist",
      "Next": "WaitForHumanDecision"
    },
    "ExpertReview": {
      "Type": "Task",
      "Resource": "arn:aws:states:::sns:publish.waitForTaskToken",
      "Parameters": {
        "TopicArn": "arn:aws:sns:REGION:ACCOUNT:expert-review-queue",
        "Message.$": "$"
      },
      "Next": "ProcessExpertDecision"
    },
    "WaitForHumanDecision": {
      "Type": "Task",
      "Resource": "arn:aws:states:::sqs:sendMessage.waitForTaskToken",
      "Parameters": {
        "QueueUrl": "https://sqs.REGION.amazonaws.com/ACCOUNT/human-review-queue",
        "MessageBody.$": "$"
      },
      "Next": "ProcessHumanDecision"
    },
    "ProcessHumanDecision": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:REGION:ACCOUNT:function:process-decision",
      "End": true
    },
    "ProcessExpertDecision": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:REGION:ACCOUNT:function:process-decision",
      "End": true
    }
  }
}
```

---

## Step 8: CloudWatch (Monitoring)

### Create Dashboard

```python
# scripts/create_dashboard.py
import boto3
import json

cloudwatch = boto3.client('cloudwatch')

dashboard_body = {
    "widgets": [
        {
            "type": "metric",
            "properties": {
                "title": "Routing Distribution",
                "metrics": [
                    ["YRSN-IARS", "GreenStreamCount"],
                    ["YRSN-IARS", "YellowStreamCount"],
                    ["YRSN-IARS", "RedStreamCount"]
                ],
                "period": 300,
                "stat": "Sum"
            }
        },
        {
            "type": "metric",
            "properties": {
                "title": "Average Temperature",
                "metrics": [
                    ["YRSN-IARS", "AvgTemperature"]
                ],
                "period": 300,
                "stat": "Average"
            }
        },
        {
            "type": "metric",
            "properties": {
                "title": "Collapse Events",
                "metrics": [
                    ["YRSN-IARS", "CollapseCount", "Type", "poisoning"],
                    ["YRSN-IARS", "CollapseCount", "Type", "confusion"],
                    ["YRSN-IARS", "CollapseCount", "Type", "distraction"],
                    ["YRSN-IARS", "CollapseCount", "Type", "clash"]
                ],
                "period": 300,
                "stat": "Sum"
            }
        },
        {
            "type": "metric",
            "properties": {
                "title": "Processing Latency",
                "metrics": [
                    ["YRSN-IARS", "ProcessingTime"]
                ],
                "period": 300,
                "stat": "p99"
            }
        }
    ]
}

cloudwatch.put_dashboard(
    DashboardName='YRSN-IARS-Production',
    DashboardBody=json.dumps(dashboard_body)
)
```

### Custom Metrics in Lambda

```python
# Add to router_handler.py
import boto3

cloudwatch = boto3.client('cloudwatch')

def publish_metrics(decision, processing_time):
    """Publish custom metrics."""
    cloudwatch.put_metric_data(
        Namespace='YRSN-IARS',
        MetricData=[
            {
                'MetricName': f'{decision.stream.value.capitalize()}StreamCount',
                'Value': 1,
                'Unit': 'Count'
            },
            {
                'MetricName': 'AvgTemperature',
                'Value': decision.temperature,
                'Unit': 'None'
            },
            {
                'MetricName': 'ProcessingTime',
                'Value': processing_time,
                'Unit': 'Milliseconds'
            },
            {
                'MetricName': 'CollapseCount',
                'Value': 1 if decision.collapse_type.value != 'none' else 0,
                'Unit': 'Count',
                'Dimensions': [
                    {'Name': 'Type', 'Value': decision.collapse_type.value}
                ]
            }
        ]
    )
```

### Alarms

```bash
# High collapse rate alarm
aws cloudwatch put-metric-alarm \
    --alarm-name "YRSN-IARS-HighCollapseRate" \
    --metric-name "CollapseCount" \
    --namespace "YRSN-IARS" \
    --statistic Sum \
    --period 300 \
    --threshold 50 \
    --comparison-operator GreaterThanThreshold \
    --evaluation-periods 2 \
    --alarm-actions arn:aws:sns:REGION:ACCOUNT:alerts

# High temperature alarm (data quality degrading)
aws cloudwatch put-metric-alarm \
    --alarm-name "YRSN-IARS-HighTemperature" \
    --metric-name "AvgTemperature" \
    --namespace "YRSN-IARS" \
    --statistic Average \
    --period 300 \
    --threshold 2.5 \
    --comparison-operator GreaterThanThreshold \
    --evaluation-periods 3 \
    --alarm-actions arn:aws:sns:REGION:ACCOUNT:alerts
```

---

## Step 9: Bedrock Knowledge Base (RAG for Policies)

### Create Knowledge Base

```python
# scripts/create_knowledge_base.py
import boto3

bedrock_agent = boto3.client('bedrock-agent')

# Create knowledge base
response = bedrock_agent.create_knowledge_base(
    name='yrsn-iars-policies',
    roleArn='arn:aws:iam::ACCOUNT:role/BedrockKBRole',
    knowledgeBaseConfiguration={
        'type': 'VECTOR',
        'vectorKnowledgeBaseConfiguration': {
            'embeddingModelArn': 'arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v1'
        }
    },
    storageConfiguration={
        'type': 'OPENSEARCH_SERVERLESS',
        'opensearchServerlessConfiguration': {
            'collectionArn': 'arn:aws:aoss:REGION:ACCOUNT:collection/COLLECTION_ID',
            'vectorIndexName': 'policy-index',
            'fieldMapping': {
                'vectorField': 'embedding',
                'textField': 'text',
                'metadataField': 'metadata'
            }
        }
    }
)

kb_id = response['knowledgeBase']['knowledgeBaseId']

# Add S3 data source
bedrock_agent.create_data_source(
    knowledgeBaseId=kb_id,
    name='policy-documents',
    dataSourceConfiguration={
        'type': 'S3',
        's3Configuration': {
            'bucketArn': f'arn:aws:s3:::{BUCKET_PREFIX}-data',
            'inclusionPrefixes': ['policies/']
        }
    }
)
```

### Query Knowledge Base in Router

```python
def get_policy_context(query: str, kb_id: str) -> dict:
    """Retrieve relevant policy context."""
    bedrock_agent_runtime = boto3.client('bedrock-agent-runtime')

    response = bedrock_agent_runtime.retrieve(
        knowledgeBaseId=kb_id,
        retrievalQuery={'text': query},
        retrievalConfiguration={
            'vectorSearchConfiguration': {
                'numberOfResults': 3
            }
        }
    )

    contexts = []
    for result in response['retrievalResults']:
        contexts.append({
            'text': result['content']['text'],
            'score': result['score'],
            'source': result['location']['s3Location']['uri']
        })

    return {
        'contexts': contexts,
        'avg_score': sum(c['score'] for c in contexts) / len(contexts) if contexts else 0
    }
```

---

## Step 10: IAM Roles and Policies

### Lambda Execution Role

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel"
            ],
            "Resource": "arn:aws:bedrock:*::foundation-model/*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "sagemaker:InvokeEndpoint"
            ],
            "Resource": "arn:aws:sagemaker:*:*:endpoint/yrsn-iars-*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "dynamodb:PutItem",
                "dynamodb:GetItem",
                "dynamodb:Query"
            ],
            "Resource": "arn:aws:dynamodb:*:*:table/yrsn-iars-*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "cloudwatch:PutMetricData"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents"
            ],
            "Resource": "arn:aws:logs:*:*:*"
        }
    ]
}
```

---

## Cost Estimation

| Service | Usage | Estimated Monthly Cost |
|---------|-------|----------------------|
| Lambda | 1M invocations, 1s avg | $20 |
| SageMaker Serverless | 1M inferences | $50-100 |
| Bedrock Titan Embeddings | 1M requests | $10 |
| DynamoDB | 1M writes, 5M reads | $15 |
| API Gateway | 1M requests | $3.50 |
| S3 | 10GB storage | $0.23 |
| CloudWatch | Metrics + Logs | $10 |
| **Total** | | **~$110-160/month** |

For high-volume (10M+ requests/month), consider:
- SageMaker real-time endpoint with auto-scaling
- DynamoDB provisioned capacity
- Reserved Bedrock throughput

---

## Production Checklist

- [ ] Enable AWS WAF on API Gateway
- [ ] Configure VPC for Lambda (if accessing internal resources)
- [ ] Set up AWS Secrets Manager for API keys
- [ ] Enable X-Ray tracing for debugging
- [ ] Configure backup for DynamoDB
- [ ] Set up CI/CD with CodePipeline
- [ ] Configure multi-region failover (if required)
- [ ] Implement request throttling
- [ ] Set up cost alerts in AWS Budgets
- [ ] Review IAM policies (least privilege)

---

## Troubleshooting

### Lambda Cold Starts

```python
# Use provisioned concurrency for consistent latency
aws lambda put-provisioned-concurrency-config \
    --function-name yrsn-iars-router \
    --qualifier prod \
    --provisioned-concurrent-executions 5
```

### SageMaker Timeout

```python
# Increase endpoint timeout
sagemaker.invoke_endpoint(
    EndpointName='yrsn-iars-classifier',
    ContentType='application/json',
    Body=payload,
    CustomAttributes='timeout=60'  # 60 seconds
)
```

### Bedrock Rate Limits

```python
# Implement exponential backoff
import time
from botocore.exceptions import ClientError

def invoke_with_retry(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return func()
        except ClientError as e:
            if e.response['Error']['Code'] == 'ThrottlingException':
                time.sleep(2 ** attempt)
            else:
                raise
    raise Exception("Max retries exceeded")
```

---

## Next Steps

1. **A/B Testing**: Deploy multiple router versions with API Gateway canary deployments
2. **Model Retraining**: Set up SageMaker Pipelines for periodic retraining
3. **Feedback Loop**: Implement human decision capture for continuous improvement
4. **Multi-Region**: Deploy to multiple regions for disaster recovery

For questions or support, see the [GitHub Issues](https://github.com/your-org/yrsn-iars/issues).
