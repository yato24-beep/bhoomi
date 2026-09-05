# Document Processing Platform

An end-to-end, production-ready document processing platform with asynchronous background execution, S3-compatible object storage, relational metadata persistence, and a modern frontend dashboard.

---

## 🏗️ Architecture Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **API Server** | **FastAPI + Pydantic v2** | High-performance async REST API, validation, routing, OpenAPI docs |
| **Database** | **PostgreSQL 16** | Relational metadata persistence (Users, Documents, Extraction Results) |
| **Object Storage**| **MinIO** | High-performance, S3-compatible raw file storage |
| **Message Broker**| **Redis 7** | Fast in-memory queue & task broker for Celery |
| **Worker Queue** | **Celery** | Asynchronous background processing worker (OCR, Parsing, Extraction) |
| **Frontend** | **Next.js 14 / TypeScript / Tailwind** | Responsive web dashboard, drag-and-drop upload, result viewer |
| **Orchestration**| **Docker Compose** | Single-command containerized local and production deployment |

---

## 🔄 End-to-End Ingestion & Processing Flow

```text
User uploads file (PDF/Image)
  │
  ▼
[FastAPI Backend]
  ├── 1. Compute SHA-256 Hash (Check for duplicates)
  ├── 2. Upload raw file stream to MinIO Object Storage
  ├── 3. Save initial record in PostgreSQL (Status: PENDING)
  └── 4. Dispatch Celery Task via Redis Broker
        │
        ▼
[Celery Background Worker]
  ├── 1. Fetch task from Redis & update status to PROCESSING
  ├── 2. Retrieve raw document from MinIO
  ├── 3. Execute extraction/parsing pipeline
  └── 4. Store structured JSON results in PostgreSQL (Status: COMPLETED)
        │
        ▼
[Next.js Frontend]
  └── Polls/Fetches document details & renders structured key-value extraction data
```

---

## 🚀 Quick Start (Docker Compose)

### 1. Configure Environment
Copy the example environment file:
```bash
cp .env.example .env
```

### 2. Start All Services
```bash
docker compose up -d --build
```

### 3. Service Endpoints
- **Frontend Dashboard**: [http://localhost:3000](http://localhost:3000)
- **Backend API & Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **MinIO Web Console**: [http://localhost:9001](http://localhost:9001) (User: `minioadmin`, Password: `minioadmin_password`)
- **PostgreSQL**: `localhost:5432` (User: `postgres`, Password: `postgres_password`, DB: `doc_platform`)
- **Redis**: `localhost:6379`

---

## 📁 Directory Structure

```text
doc-processing-platform/
├── docker-compose.yml
├── .env.example
├── .env
├── backend/                  # FastAPI + Celery backend
│   ├── Dockerfile
│   ├── pyproject.toml
│   ├── alembic/              # Database schema migrations
│   └── app/
│       ├── main.py           # FastAPI entrypoint
│       ├── config.py         # App configuration & settings
│       ├── core/             # Security (JWT, bcrypt) & SHA-256 hashing
│       ├── db/               # Database engine & session
│       ├── models/           # SQLAlchemy DB models (User, Document, ExtractionResult)
│       ├── schemas/          # Pydantic request/response schemas
│       ├── api/              # API router endpoints (/auth, /documents)
│       ├── services/         # Storage service (MinIO)
│       └── workers/          # Celery worker & processing tasks
└── frontend/                 # Next.js 14 + Tailwind frontend
    ├── Dockerfile
    ├── package.json
    └── src/                  # App router, UI components, API client
```
