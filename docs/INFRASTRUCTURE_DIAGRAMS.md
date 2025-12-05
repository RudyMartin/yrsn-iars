# YRSN-IARS Infrastructure Diagrams

GitHub-compatible Mermaid diagrams for AWS and GCP deployments.

---

## AWS Architecture

```mermaid
flowchart TB
    subgraph Client["Client Layer"]
        A[Web App / Mobile]
        B[Internal Systems]
    end

    subgraph Gateway["API Layer"]
        C[API Gateway]
        D[WAF]
    end

    subgraph Compute["Compute Layer"]
        E[Lambda: Ingest]
        F[Lambda: Router]
        G[Lambda: Process]
    end

    subgraph ML["ML Layer"]
        H[Bedrock Embeddings]
        I[SageMaker Endpoint]
        J[Bedrock KB - RAG]
    end

    subgraph Orchestration["Orchestration"]
        K[Step Functions]
        L[EventBridge]
    end

    subgraph Storage["Storage Layer"]
        M[(DynamoDB: Decisions)]
        N[(S3: Documents)]
        O[(S3: Models)]
    end

    subgraph Messaging["Messaging"]
        P[SNS: Alerts]
        Q[SQS: Review Queue]
    end

    subgraph Monitoring["Observability"]
        R[CloudWatch Metrics]
        S[CloudWatch Logs]
        T[X-Ray Traces]
    end

    A --> D
    B --> D
    D --> C
    C --> E
    E --> H
    E --> K
    K --> F
    F --> I
    F --> J
    F --> M
    K --> G
    G --> M
    G --> P
    G --> Q
    I --> O
    J --> N
    E --> S
    F --> S
    G --> S
    F --> R
    F --> T
    L --> K
```

---

## AWS Data Flow

```mermaid
flowchart LR
    subgraph Input["1 - Request Input"]
        A1[Approval Request]
        A2[Text + Metadata]
    end

    subgraph Embedding["2 - Embedding"]
        B1[Bedrock Titan]
        B2[768-dim Vector]
    end

    subgraph Classification["3 - Classification"]
        C1[SageMaker Endpoint]
        C2[Prediction Probs]
        C3[Confidence Score]
    end

    subgraph YRSN["4 - YRSN Analysis"]
        D1[CleanlabAdapter]
        D2[R Component]
        D3[S Component]
        D4[N Component]
    end

    subgraph Temperature["5 - Temperature"]
        E1[Compute Tau]
        E2[Adjust Thresholds]
    end

    subgraph Routing["6 - Routing Decision"]
        F1{Stream?}
        F2[GREEN: Auto]
        F3[YELLOW: Assist]
        F4[RED: Expert]
    end

    subgraph Output["7 - Output"]
        G1[DynamoDB Store]
        G2[SNS Notify]
        G3[API Response]
    end

    A1 --> A2
    A2 --> B1
    B1 --> B2
    B2 --> C1
    C1 --> C2
    C2 --> C3
    C3 --> D1
    D1 --> D2
    D1 --> D3
    D1 --> D4
    D2 --> E1
    E1 --> E2
    E2 --> F1
    F1 -->|High Cs Low Tau| F2
    F1 -->|Medium| F3
    F1 -->|Low Cs High Tau| F4
    F2 --> G1
    F3 --> G1
    F4 --> G1
    F3 --> G2
    F4 --> G2
    G1 --> G3
```

---

## AWS Step Functions Workflow

```mermaid
stateDiagram-v2
    [*] --> RouteRequest

    RouteRequest --> CheckKnockout

    CheckKnockout --> KnockoutAction: Knockout Rule Match
    CheckKnockout --> CheckCollapse: No Knockout

    KnockoutAction --> StoreDecision

    CheckCollapse --> ForceRed: POISONING or CONFUSION
    CheckCollapse --> CheckConfidence: No Critical Collapse

    ForceRed --> ExpertReview

    CheckConfidence --> GreenStream: High Cs + Low Tau
    CheckConfidence --> YellowStream: Medium Cs
    CheckConfidence --> RedStream: Low Cs + High Tau

    GreenStream --> AutoProcess
    AutoProcess --> StoreDecision

    YellowStream --> AIAssist
    AIAssist --> WaitHumanYellow
    WaitHumanYellow --> ProcessHumanDecision

    RedStream --> ExpertReview
    ExpertReview --> WaitHumanRed
    WaitHumanRed --> ProcessHumanDecision

    ProcessHumanDecision --> StoreDecision

    StoreDecision --> PublishMetrics
    PublishMetrics --> [*]
```

---

## GCP Architecture

```mermaid
flowchart TB
    subgraph Client["Client Layer"]
        A[Web App / Mobile]
        B[Internal Systems]
    end

    subgraph Gateway["API Layer"]
        C[Cloud Endpoints]
        D[Cloud Armor]
    end

    subgraph Compute["Compute Layer"]
        E[Cloud Run: Router]
        F[Cloud Functions: Process]
        G[Cloud Functions: Notify]
    end

    subgraph ML["ML Layer"]
        H[Vertex AI Embeddings]
        I[Vertex AI Prediction]
        J[Vertex AI Search]
    end

    subgraph Orchestration["Orchestration"]
        K[Workflows]
        L[Cloud Scheduler]
    end

    subgraph Storage["Storage Layer"]
        M[(Firestore: Decisions)]
        N[(Cloud Storage: Docs)]
        O[(Cloud Storage: Models)]
    end

    subgraph Messaging["Messaging"]
        P[Pub/Sub: Events]
        Q[Pub/Sub: Reviews]
    end

    subgraph Monitoring["Observability"]
        R[Cloud Monitoring]
        S[Cloud Logging]
        T[Cloud Trace]
    end

    A --> D
    B --> D
    D --> C
    C --> E
    E --> H
    E --> K
    K --> E
    E --> I
    E --> J
    E --> M
    K --> F
    F --> M
    F --> P
    G --> Q
    I --> O
    J --> N
    E --> S
    F --> S
    E --> R
    E --> T
    L --> K
    P --> G
```

---

## GCP Data Flow

```mermaid
flowchart LR
    subgraph Input["1 - Request Input"]
        A1[Approval Request]
        A2[JSON Payload]
    end

    subgraph Embedding["2 - Embedding"]
        B1[Vertex AI Gecko]
        B2[768-dim Vector]
    end

    subgraph Classification["3 - Classification"]
        C1[Vertex AI Endpoint]
        C2[Prediction Probs]
        C3[Confidence Score]
    end

    subgraph YRSN["4 - YRSN Analysis"]
        D1[CleanlabAdapter]
        D2[R - Relevant]
        D3[S - Superfluous]
        D4[N - Noise]
    end

    subgraph Temperature["5 - Temperature"]
        E1[Tau = 1 div Alpha]
        E2[Threshold Adjust]
    end

    subgraph Routing["6 - Stream Decision"]
        F1{Route?}
        F2[GREEN]
        F3[YELLOW]
        F4[RED]
    end

    subgraph Output["7 - Output"]
        G1[Firestore]
        G2[Pub/Sub]
        G3[Response]
    end

    A1 --> A2
    A2 --> B1
    B1 --> B2
    B2 --> C1
    C1 --> C2
    C2 --> C3
    C3 --> D1
    D1 --> D2
    D1 --> D3
    D1 --> D4
    D2 --> E1
    E1 --> E2
    E2 --> F1
    F1 -->|Auto| F2
    F1 -->|Assist| F3
    F1 -->|Expert| F4
    F2 --> G1
    F3 --> G1
    F4 --> G1
    F3 --> G2
    F4 --> G2
    G1 --> G3
```

---

## GCP Workflows State Machine

```mermaid
stateDiagram-v2
    [*] --> ReceiveRequest

    ReceiveRequest --> CallRouter

    CallRouter --> EvaluateStream

    EvaluateStream --> AutoProcess: stream = green
    EvaluateStream --> AIAssist: stream = yellow
    EvaluateStream --> ExpertQueue: stream = red

    AutoProcess --> ExecuteDecision

    AIAssist --> GenerateSummary
    GenerateSummary --> PublishToQueue
    PublishToQueue --> AwaitCallback
    AwaitCallback --> ProcessCallback
    ProcessCallback --> ExecuteDecision

    ExpertQueue --> NotifyExpert
    NotifyExpert --> AwaitExpertCallback
    AwaitExpertCallback --> ProcessCallback

    ExecuteDecision --> WriteToFirestore
    WriteToFirestore --> PublishMetrics
    PublishMetrics --> ReturnResponse
    ReturnResponse --> [*]
```

---

## Comparison: AWS vs GCP Services

```mermaid
flowchart LR
    subgraph AWS["AWS Services"]
        A1[API Gateway]
        A2[Lambda]
        A3[Bedrock]
        A4[SageMaker]
        A5[DynamoDB]
        A6[S3]
        A7[Step Functions]
        A8[CloudWatch]
        A9[SNS/SQS]
    end

    subgraph GCP["GCP Services"]
        G1[Cloud Endpoints]
        G2[Cloud Run]
        G3[Vertex AI]
        G4[Vertex AI]
        G5[Firestore]
        G6[Cloud Storage]
        G7[Workflows]
        G8[Cloud Monitoring]
        G9[Pub/Sub]
    end

    A1 <-.-> G1
    A2 <-.-> G2
    A3 <-.-> G3
    A4 <-.-> G4
    A5 <-.-> G5
    A6 <-.-> G6
    A7 <-.-> G7
    A8 <-.-> G8
    A9 <-.-> G9
```

---

## Temperature-Quality Duality Flow

```mermaid
flowchart TD
    subgraph Quality["Quality Measurement"]
        A[Cleanlab Signals]
        B[Label Quality]
        C[OOD Score]
        D[Duplicate Check]
    end

    subgraph YRSN["YRSN Decomposition"]
        E[Compute R]
        F[Compute S]
        G[Compute N]
        H[R + S + N = 1]
    end

    subgraph Alpha["Alpha Calculation"]
        I[Alpha = R]
        J[Quality Score]
    end

    subgraph Tau["Temperature"]
        K[Tau = 1 / Alpha]
        L{Tau Range}
        M[Tau less than 1: Tight]
        N[Tau = 1: Balanced]
        O[Tau greater than 1: Loose]
    end

    subgraph Thresholds["Threshold Adjustment"]
        P[Base Thresholds]
        Q[Adjusted Thresholds]
        R[Green: 0.95 + adj]
        S[Yellow: 0.70 + adj]
    end

    subgraph Decision["Routing"]
        T{Cs vs Thresholds}
        U[GREEN Stream]
        V[YELLOW Stream]
        W[RED Stream]
    end

    A --> B
    A --> C
    A --> D
    B --> E
    C --> G
    D --> F
    E --> H
    F --> H
    G --> H
    H --> I
    I --> J
    J --> K
    K --> L
    L --> M
    L --> N
    L --> O
    M --> Q
    N --> Q
    O --> Q
    P --> Q
    Q --> R
    Q --> S
    R --> T
    S --> T
    T -->|Cs gte Green| U
    T -->|Cs gte Yellow| V
    T -->|Cs lt Yellow| W
```

---

## Collapse Type Detection

```mermaid
flowchart TD
    subgraph Input["YRSN Values"]
        A[R Value]
        B[S Value]
        C[N Value]
    end

    subgraph Check["Collapse Checks"]
        D{N gt 0.3 AND S gt 0.25?}
        E{N gt 0.3?}
        F{S gt 0.4?}
        G{Risk gt 0.6?}
    end

    subgraph Types["Collapse Types"]
        H[CONFUSION]
        I[POISONING]
        J[DISTRACTION]
        K[CLASH]
        L[NONE]
    end

    subgraph Action["Routing Action"]
        M[Force RED]
        N[Force RED]
        O[Prefer YELLOW]
        P[Check Temperature]
        Q[Normal Routing]
    end

    A --> D
    B --> D
    C --> D
    D -->|Yes| H
    D -->|No| E
    E -->|Yes| I
    E -->|No| F
    F -->|Yes| J
    F -->|No| G
    G -->|Yes| K
    G -->|No| L
    H --> M
    I --> N
    J --> O
    K --> P
    L --> Q
```

---

## Urgency Scoring Formula

```mermaid
flowchart LR
    subgraph Inputs["Input Components"]
        A[T_expiry: Deadline]
        B[P_business: Amount + Level]
        C[S_sentiment: NLP Score]
        D[C_clarity: R - N]
    end

    subgraph Weights["Weights"]
        E[w1 = 0.4]
        F[w2 = 0.3]
        G[w3 = 0.2]
        H[w4 = 0.1]
    end

    subgraph Formula["Weighted Sum"]
        I[U = w1*T + w2*P + w3*S + w4*C]
    end

    subgraph Levels["Urgency Levels"]
        J{U Score}
        K[EMERGENCY: gte 0.8]
        L[CRITICAL: gte 0.6]
        M[EXPEDITED: gte 0.4]
        N[STANDARD: lt 0.4]
    end

    A --> E
    B --> F
    C --> G
    D --> H
    E --> I
    F --> I
    G --> I
    H --> I
    I --> J
    J --> K
    J --> L
    J --> M
    J --> N
```

---

## Multi-Region Deployment (AWS)

```mermaid
flowchart TB
    subgraph Global["Global Layer"]
        A[Route 53]
        B[CloudFront]
    end

    subgraph Primary["us-east-1 - Primary"]
        C1[API Gateway]
        D1[Lambda]
        E1[DynamoDB Global Table]
        F1[SageMaker]
    end

    subgraph Secondary["us-west-2 - DR"]
        C2[API Gateway]
        D2[Lambda]
        E2[DynamoDB Global Table]
        F2[SageMaker]
    end

    subgraph Shared["Shared Services"]
        G[S3 Cross-Region Replication]
        H[Bedrock - Multi-Region]
    end

    A --> B
    B --> C1
    B --> C2
    C1 --> D1
    C2 --> D2
    D1 --> E1
    D2 --> E2
    E1 <--> E2
    D1 --> F1
    D2 --> F2
    D1 --> H
    D2 --> H
    F1 --> G
    F2 --> G
```

---

## Multi-Region Deployment (GCP)

```mermaid
flowchart TB
    subgraph Global["Global Layer"]
        A[Cloud DNS]
        B[Cloud CDN]
        C[Global Load Balancer]
    end

    subgraph Primary["us-central1 - Primary"]
        D1[Cloud Run]
        E1[Firestore Multi-Region]
        F1[Vertex AI]
    end

    subgraph Secondary["us-east1 - DR"]
        D2[Cloud Run]
        E2[Firestore Multi-Region]
        F2[Vertex AI]
    end

    subgraph Shared["Shared Services"]
        G[Cloud Storage Multi-Region]
        H[Vertex AI - Regional]
    end

    A --> B
    B --> C
    C --> D1
    C --> D2
    D1 --> E1
    D2 --> E2
    E1 <--> E2
    D1 --> F1
    D2 --> F2
    D1 --> G
    D2 --> G
```

---

## Security Architecture

```mermaid
flowchart TB
    subgraph External["External"]
        A[Users]
        B[Applications]
    end

    subgraph Edge["Edge Security"]
        C[WAF / Cloud Armor]
        D[DDoS Protection]
        E[Rate Limiting]
    end

    subgraph Auth["Authentication"]
        F[IAM / Identity Platform]
        G[API Keys]
        H[OAuth 2.0]
    end

    subgraph Network["Network Security"]
        I[VPC]
        J[Private Subnets]
        K[NAT Gateway]
    end

    subgraph Data["Data Security"]
        L[Encryption at Rest]
        M[Encryption in Transit]
        N[KMS / Secret Manager]
    end

    subgraph Audit["Audit & Compliance"]
        O[CloudTrail / Audit Logs]
        P[Config Rules]
        Q[Security Hub / SCC]
    end

    A --> C
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    F --> H
    G --> I
    H --> I
    I --> J
    J --> K
    J --> L
    L --> M
    M --> N
    I --> O
    O --> P
    P --> Q
```

---

## CI/CD Pipeline

```mermaid
flowchart LR
    subgraph Source["Source"]
        A[GitHub Repo]
    end

    subgraph Build["Build"]
        B[Run Tests]
        C[Build Container]
        D[Security Scan]
    end

    subgraph Stage["Staging"]
        E[Deploy to Staging]
        F[Integration Tests]
        G[Load Tests]
    end

    subgraph Approve["Approval"]
        H{Manual Approval}
    end

    subgraph Prod["Production"]
        I[Deploy Canary 10%]
        J[Monitor Metrics]
        K{Healthy?}
        L[Full Rollout]
        M[Rollback]
    end

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H -->|Approved| I
    I --> J
    J --> K
    K -->|Yes| L
    K -->|No| M
    M --> A
```

---

## Monitoring Dashboard Layout

```mermaid
flowchart TB
    subgraph Row1["Row 1: Traffic"]
        A[Request Rate]
        B[Error Rate]
        C[Latency P99]
    end

    subgraph Row2["Row 2: Routing"]
        D[Stream Distribution]
        E[Automation Rate]
        F[Knockout Rate]
    end

    subgraph Row3["Row 3: Quality"]
        G[Avg Temperature]
        H[Avg Alpha]
        I[Collapse Events]
    end

    subgraph Row4["Row 4: YRSN"]
        J[R Distribution]
        K[S Distribution]
        L[N Distribution]
    end

    subgraph Row5["Row 5: Business"]
        M[By Category]
        N[By Amount Range]
        O[Urgency Distribution]
    end
```

---

*These diagrams are GitHub-compatible and will render correctly in GitHub markdown files.*
