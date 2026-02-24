# Graphwise

Graphwise is a web app that turns a codebase into a searchable, queryable knowledge graph. You can upload a repository (as a ZIP), let Graphwise ingest and structure the code, and then ask questions to explore architecture, components, relationships, and relevant context across the project.

## What Graphwise does

- Upload a repository ZIP for ingestion
- Builds a code graph and stores it in Neo4j
- Stores job/run metadata in Postgres
- Uses a queue + worker pipeline for long-running ingestion work
- Provides a web UI to query and explore results
- Exposes a single clean HTTPS endpoint in production (no port numbers)

## High-level architecture

Graphwise runs as multiple Docker services:

- frontend: React + Vite UI
- api_gateway: FastAPI gateway that serves the UI’s API endpoints (ingest, jobs, query, health)
- graph / graph_service: internal services used for graph construction and querying
- worker: background job runner (queue-based)
- neo4j: graph database
- postgres: relational database for job metadata
- redis: queue broker
- caddy: reverse proxy + automatic TLS certificates

In production, Caddy is the only public entry point:
- `/api/*` routes to `api_gateway`
- everything else routes to `frontend`

## Tech stack

Frontend
- React
- Vite
- TypeScript
- Tailwind CSS

Backend
- FastAPI (Python)
- SQLAlchemy (Postgres)
- Redis queue + worker pipeline

Datastores
- Neo4j (graph)
- Postgres (jobs/metadata)
- Redis (queue/broker)

Infra
- Docker + Docker Compose
- Caddy (reverse proxy + HTTPS)

CI/CD
- GitHub Actions for CI and VM deployment (rsync + docker compose on the VM)

## Repository layout

- `frontend/` — React + Vite application
- `backend/` — backend services (gateway, graph services, shared code)
- `infra/` — Docker Compose + Caddy configuration
- `data/` — local data volume mount (uploads, intermediate artifacts)

## Configuration (.env)

Graphwise uses a `.env` file that must **not** be committed to GitHub. In production, keep it on the VM (for example: `/opt/graphwise/.env`).

Minimum required configuration includes:
- `OPENAI_API_KEY` (set this yourself)
- `CORS_ORIGINS` (your production domain)
- `VITE_API_BASE` (recommended `/api` for production behind Caddy)

Example:
```env
OPENAI_API_KEY=REPLACE_ME
CORS_ORIGINS=https://graphwise-satish.norwayeast.cloudapp.azure.com,http://graphwise-satish.norwayeast.cloudapp.azure.com
VITE_API_BASE=/api
