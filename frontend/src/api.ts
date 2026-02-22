export type JobStatus = "queued" | "running" | "completed" | "failed";

export interface Job {
  job_id: string;
  repo_id: string;
  job_type: string;
  status: JobStatus;
  progress: number;
  current_step: string;
  attempts: number;
  error: string | null;
  created_at: string;
  updated_at: string;
}

export type RetrievalNodeType = "file" | "class" | "function" | "module" | string;

export interface RetrievalNode {
  id: string;
  repo_id?: string;
  type: RetrievalNodeType;
  name: string;
  path: string;
  code_snippet: string;
}

export interface RetrievalEdge {
  source: string;
  target: string;
  type: "contains" | "imports" | "calls" | string;
}

export interface RetrievalSnippet {
  id: string;
  path: string;
  name: string;
  type: RetrievalNodeType;
  code_snippet: string;
  score?: number;
  semantic_score?: number | null;
  keyword_score?: number | null;
}

export interface QueryResult {
  answer: string;
  citations: string[];
  warning?: string | null;
  retrieval_pack?: {
    snippets?: RetrievalSnippet[];
    nodes?: RetrievalNode[];
    edges?: RetrievalEdge[];
    scores?: Record<string, { semantic?: number | null; keyword?: number | null; combined?: number | null }>;
  };
}

export interface KgLinkedEntity {
  name: string;
  type: string;
  score: number;
}

export interface KgSubgraphNode {
  id: string;
  kind?: string;
  name?: string;
  type?: string;
  repo_id?: string;
  path?: string;
  text?: string;
  [key: string]: unknown;
}

export interface KgSubgraphEdge {
  source: string;
  target: string;
  type?: string;
  relation_type?: string;
  confidence?: number;
  evidence_chunk_id?: string;
  [key: string]: unknown;
}

export interface KgEvidence {
  chunk_id: string;
  doc_path: string;
  text: string;
  score: number;
}

export interface KgRetrievalTrace {
  step: string;
  detail: string;
}

export interface KgQueryResult {
  linked_entities: KgLinkedEntity[];
  subgraph: {
    nodes: KgSubgraphNode[];
    edges: KgSubgraphEdge[];
  };
  evidence: KgEvidence[];
  retrieval_trace: KgRetrievalTrace[];
}

function resolveApiBase(): string {
  const configured = import.meta.env.VITE_API_BASE as string | undefined;
  if (configured && configured.trim()) return configured.trim();
  if (typeof window !== "undefined") {
    return `${window.location.protocol}//${window.location.hostname}:8000`;
  }
  return "http://localhost:8000";
}

const API_BASE = resolveApiBase();

function normalize(path: string): string {
  const base = API_BASE.endsWith("/") ? API_BASE.slice(0, -1) : API_BASE;
  const suffix = path.startsWith("/") ? path : `/${path}`;
  return `${base}${suffix}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(normalize(path), {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `${res.status} ${res.statusText}`);
  }

  return (await res.json()) as T;
}

export function health() {
  return request<{ ok: boolean }>("/health");
}

export async function ingestZip(file: File) {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(normalize("/ingest/zip"), {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `${res.status} ${res.statusText}`);
  }

  return (await res.json()) as { job_id: string; repo_id: string };
}

export function getJob(jobId: string) {
  return request<Job>(`/jobs/${jobId}`);
}

export function listJobs(repoId: string) {
  return request<Job[]>(`/jobs?repo_id=${encodeURIComponent(repoId)}`);
}

export function repoStatus(repoId: string) {
  return request<{
    repo_id: string;
    indexed_node_count: number;
    indexed_edge_count: number;
    embedded_nodes: number;
    embeddings_exist: boolean;
  }>(`/repos/${repoId}/status`);
}

export function queryRepo(repoId: string, question: string) {
  return request<QueryResult>("/query", {
    method: "POST",
    body: JSON.stringify({ repo_id: repoId, question }),
  });
}

export function queryKg(repoId: string, question: string) {
  return request<KgQueryResult>("/kg/query", {
    method: "POST",
    body: JSON.stringify({
      repo_id: repoId,
      question,
      top_k_chunks: 10,
      hops: 2,
    }),
  });
}
