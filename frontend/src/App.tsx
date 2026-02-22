import { useEffect, useMemo, useRef, useState, type Dispatch, type ReactNode, type SetStateAction } from "react";
import { Link, Navigate, Route, Routes, useLocation } from "react-router-dom";
import { motion } from "framer-motion";
import { useMutation, useQuery } from "@tanstack/react-query";
import ReactFlow, {
  applyEdgeChanges,
  applyNodeChanges,
  Background,
  Controls,
  type EdgeChange,
  type NodeChange,
  MiniMap,
  MarkerType,
  type Edge as RFEdge,
  type Node as RFNode,
} from "reactflow";
import "reactflow/dist/style.css";
import {
  getJob,
  health,
  ingestZip,
  type KgEvidence,
  type KgLinkedEntity,
  type KgQueryResult,
  type KgRetrievalTrace,
  type KgSubgraphEdge,
  type KgSubgraphNode,
  listJobs,
  queryRepo,
  type QueryResult,
  queryKg,
  type RetrievalNode,
  type RetrievalSnippet,
  repoStatus,
  type Job,
} from "./api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";

function useLocalStorageState<T>(key: string, initialValue: T) {
  const [value, setValue] = useState<T>(() => {
    if (typeof window === "undefined") return initialValue;
    try {
      const raw = window.localStorage.getItem(key);
      if (!raw) return initialValue;
      return JSON.parse(raw) as T;
    } catch {
      return initialValue;
    }
  });

  const setAndStore: Dispatch<SetStateAction<T>> = (next) => {
    setValue((prev) => {
      const resolved = typeof next === "function" ? (next as (old: T) => T)(prev) : next;
      if (typeof window !== "undefined") {
        window.localStorage.setItem(key, JSON.stringify(resolved));
      }
      return resolved;
    });
  };

  return [value, setAndStore] as const;
}

function formatTime(iso?: string): string {
  if (!iso) return "-";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
}

function truncate(text: string, max = 72): string {
  if (text.length <= max) return text;
  return `${text.slice(0, max - 1)}…`;
}

function Shell({ children }: { children: ReactNode }) {
  const location = useLocation();
  return (
    <div className="app-bg">
      <div className="app-grid" />
      <header className="topbar">
        <div>
          <p className="eyebrow">CodeGraph Control Plane</p>
          <h1>Repository Intelligence Console</h1>
        </div>
        <nav>
          <Link className={location.pathname === "/dashboard" ? "active" : ""} to="/dashboard">
            Dashboard
          </Link>
          <Link className={location.pathname === "/jobs" ? "active" : ""} to="/jobs">
            Jobs
          </Link>
        </nav>
      </header>
      <main className="content">{children}</main>
    </div>
  );
}

function Dashboard() {
  const [repoId, setRepoId] = useLocalStorageState("cg.repo_id", "");
  const [jobId, setJobId] = useLocalStorageState("cg.job_id", "");
  const [question, setQuestion] = useLocalStorageState("cg.question", "What does this repository do?");
  const [kgResult, setKgResult] = useState<KgQueryResult | null>(null);
  const [codeResult, setCodeResult] = useState<QueryResult | null>(null);
  const [kgResultError, setKgResultError] = useState<string | null>(null);
  const [codeResultError, setCodeResultError] = useState<string | null>(null);
  const [zipHintName, setZipHintName] = useLocalStorageState("cg.zip_hint_name", "");
  const [zipFile, setZipFile] = useState<File | null>(null);
  const zipInputRef = useRef<HTMLInputElement | null>(null);

  const healthQ = useQuery({ queryKey: ["health"], queryFn: health, refetchInterval: 5000 });

  const jobQ = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => getJob(jobId),
    enabled: Boolean(jobId),
    refetchInterval: (q) => {
      const data = q.state.data as Job | undefined;
      if (!data) return 3000;
      return data.status === "completed" || data.status === "failed" ? false : 3000;
    },
  });

  const jobsQ = useQuery({
    queryKey: ["jobs", repoId],
    queryFn: () => listJobs(repoId),
    enabled: Boolean(repoId),
    refetchInterval: 5000,
  });

  const repoStatusQ = useQuery({
    queryKey: ["repo-status", repoId],
    queryFn: () => repoStatus(repoId),
    enabled: Boolean(repoId),
    refetchInterval: 8000,
  });

  const ingestZipM = useMutation({
    mutationFn: () => {
      const current = zipInputRef.current?.files?.[0] ?? zipFile;
      if (!current) throw new Error("Choose a .zip file first");
      return ingestZip(current);
    },
    onSuccess: (result) => {
      setRepoId(result.repo_id);
      setJobId(result.job_id);
      setZipFile(null);
      if (zipInputRef.current) {
        zipInputRef.current.value = "";
      }
    },
  });

  const queryM = useMutation({
    mutationFn: async () => {
      const normalizedRepo = repoId.trim();
      const normalizedQuestion = question.trim();
      if (!normalizedRepo) throw new Error("Provide a repo id");
      if (!normalizedQuestion) throw new Error("Provide a question");
      const [kgSettled, codeSettled] = await Promise.allSettled([
        queryKg(normalizedRepo, normalizedQuestion),
        queryRepo(normalizedRepo, normalizedQuestion),
      ]);

      const nextKgResult = kgSettled.status === "fulfilled" ? kgSettled.value : null;
      const nextCodeResult = codeSettled.status === "fulfilled" ? codeSettled.value : null;
      const nextKgError =
        kgSettled.status === "rejected" ? (kgSettled.reason instanceof Error ? kgSettled.reason.message : String(kgSettled.reason)) : null;
      const nextCodeError =
        codeSettled.status === "rejected" ? (codeSettled.reason instanceof Error ? codeSettled.reason.message : String(codeSettled.reason)) : null;

      if (!nextKgResult && !nextCodeResult) {
        throw new Error(`KG query failed: ${nextKgError ?? "unknown"} | Code query failed: ${nextCodeError ?? "unknown"}`);
      }

      return {
        kgResult: nextKgResult,
        codeResult: nextCodeResult,
        kgError: nextKgError,
        codeError: nextCodeError,
      };
    },
    onMutate: () => {
      setKgResultError(null);
      setCodeResultError(null);
    },
    onSuccess: (result) => {
      setKgResult(result.kgResult);
      setCodeResult(result.codeResult);
      setKgResultError(result.kgError);
      setCodeResultError(result.codeError);
    },
  });

  const running = useMemo(() => {
    const jobs = jobsQ.data ?? [];
    return jobs.filter((j) => j.status === "running").length;
  }, [jobsQ.data]);

  return (
    <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="stack">
      <section className="card">
        <h2>Summary</h2>
        <div className="stack">
          <section className="cards cards-3">
            <article className="card stat">
              <h3>Gateway</h3>
              <p className={healthQ.data?.ok ? "ok" : "bad"}>{healthQ.data?.ok ? "Healthy" : "Down"}</p>
            </article>
            <article className="card stat">
              <h3>Running Jobs</h3>
              <p>{running}</p>
            </article>
            <article className="card stat">
              <h3>Current Repo</h3>
              <p>{repoId || "-"}</p>
            </article>
          </section>

          <section className="cards cards-2">
            <article className="card">
              <h3>Current Job</h3>
              <label>
                Job ID
                <Input value={jobId} onChange={(e) => setJobId(e.target.value)} placeholder="job UUID" />
              </label>
              {jobQ.data && (
                <div className="job-details">
                  <p>
                    Status: <strong>{jobQ.data.status}</strong>
                  </p>
                  <p>
                    Progress: <strong>{jobQ.data.progress}%</strong>
                  </p>
                  <p>
                    Step: <strong>{jobQ.data.current_step}</strong>
                  </p>
                  <p>Updated: {formatTime(jobQ.data.updated_at)}</p>
                  {jobQ.data.error && <p className="error">{jobQ.data.error}</p>}
                </div>
              )}
            </article>

            <article className="card">
              <h3>Repo Status</h3>
              <label>
                Repo ID
                <Input value={repoId} onChange={(e) => setRepoId(e.target.value)} placeholder="repo UUID" />
              </label>
              {repoStatusQ.data && (
                <div className="job-details">
                  <p>
                    Indexed Nodes: <strong>{repoStatusQ.data.indexed_node_count}</strong>
                  </p>
                  <p>
                    Indexed Edges: <strong>{repoStatusQ.data.indexed_edge_count}</strong>
                  </p>
                  <p>
                    Embedded Nodes: <strong>{repoStatusQ.data.embedded_nodes}</strong>
                  </p>
                  <p>
                    Embeddings: <strong>{repoStatusQ.data.embeddings_exist ? "Yes" : "No"}</strong>
                  </p>
                </div>
              )}
            </article>
          </section>
        </div>
      </section>

      <section className="card">
        <h2>Ingestion</h2>
        <div className="dashboard-ingest-layout">
          <form
            className="stack compact"
            onSubmit={(e) => {
              e.preventDefault();
              ingestZipM.mutate();
            }}
          >
            <label>
              Upload .zip
              <input
                ref={zipInputRef}
                type="file"
                accept=".zip,application/zip"
                onChange={(e) => {
                  const next = e.target.files?.[0] ?? null;
                  setZipFile(next);
                  setZipHintName(next?.name || "");
                }}
                required
              />
            </label>
            {!zipFile && zipHintName && (
              <p className="muted small">
                Previously selected file: {zipHintName}. Please choose it again before uploading.
              </p>
            )}
            <Button disabled={ingestZipM.isPending}>{ingestZipM.isPending ? "Uploading..." : "Start ZIP Ingest"}</Button>
            {ingestZipM.error && <p className="error">{(ingestZipM.error as Error).message}</p>}
          </form>
        </div>
      </section>

      <section className="card">
        <h2>Ask (Level-2 Merged)</h2>
        <label>
          Repo ID
          <Input value={repoId} onChange={(e) => setRepoId(e.target.value)} placeholder="repo UUID" />
        </label>
        <label>
          Question
          <Textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            rows={4}
            placeholder="Ask about architecture, dependencies, key classes, or where a function is used..."
          />
        </label>
        <Button onClick={() => queryM.mutate()} disabled={queryM.isPending}>
          {queryM.isPending ? "Asking..." : "Ask"}
        </Button>
        {queryM.error && <p className="error">{(queryM.error as Error).message}</p>}
      </section>

      <section className="card">
        <h2>Answer</h2>
        <Level2AnswerPanel
          question={question}
          kgResult={kgResult}
          codeResult={codeResult}
          kgError={kgResultError}
          codeError={codeResultError}
        />
      </section>

      {(kgResult || codeResult) && (
        <Level2QuestionWorkspace
          question={question}
          kgResult={kgResult}
          codeResult={codeResult}
          kgError={kgResultError}
          codeError={codeResultError}
        />
      )}

    </motion.div>
  );
}

function displayNodeName(node: KgSubgraphNode): string {
  if (typeof node.name === "string" && node.name.trim()) return node.name;
  return node.id;
}

function displayNodeType(node: KgSubgraphNode): string {
  if (typeof node.type === "string" && node.type.trim()) return node.type;
  if (typeof node.kind === "string" && node.kind.trim()) return node.kind;
  return "unknown";
}

const NUMERIC_ONLY_RE = /^[0-9]+$/;

function isLowSignalEntity(entity: KgLinkedEntity): boolean {
  const name = entity.name.trim();
  const type = entity.type.trim().toLowerCase();
  if (NUMERIC_ONLY_RE.test(name)) return true;
  return type === "unknown" && name.length <= 2;
}

function containsCaseInsensitive(value: string | undefined, query: string): boolean {
  if (!value) return false;
  return value.toLowerCase().includes(query.toLowerCase());
}

type SubgraphGroup = "class" | "function" | "module" | "other";

function subgraphGroupForNode(node: KgSubgraphNode): SubgraphGroup {
  const rawType = displayNodeType(node).toLowerCase();
  if (rawType.includes("class")) return "class";
  if (rawType.includes("function") || rawType.includes("method")) return "function";
  if (rawType.includes("module")) return "module";
  return "other";
}

function formatSubgraphEdge(edge: KgSubgraphEdge, nodeNameById: Map<string, string>): string {
  const source = nodeNameById.get(edge.source) ?? edge.source;
  const target = nodeNameById.get(edge.target) ?? edge.target;
  const relation = edge.relation_type || edge.type || "related";
  const confidence =
    typeof edge.confidence === "number" && Number.isFinite(edge.confidence) ? edge.confidence.toFixed(2) : null;
  const detail = confidence ? `${relation}/${confidence}` : relation;
  return `${source} --(${detail})--> ${target}`;
}

type GraphViewNodeKind = "question" | "entity" | "evidence";

type GraphViewNode = {
  id: string;
  kind: GraphViewNodeKind;
  label: string;
  subLabel?: string;
  x: number;
  y: number;
  entityName?: string;
  evidenceId?: string;
};

type GraphViewEdge = {
  id: string;
  source: string;
  target: string;
  label: "linked" | "supported_by" | "evidence";
};

type QuestionGraphNodeType = "question" | "entity" | "evidence" | "code";

interface QuestionGraphNode {
  id: string;
  type: QuestionGraphNodeType;
  label: string;
  subtitle?: string;
  ref_id?: string;
  source?: "kg" | "code" | "merged";
  meta?: Record<string, unknown>;
}

interface QuestionGraphEdge {
  id: string;
  source: string;
  target: string;
  label: string;
  meta?: Record<string, unknown>;
}

interface QuestionGraph {
  nodes: QuestionGraphNode[];
  edges: QuestionGraphEdge[];
}

const QG_MAX_NODES = 30;
const QG_MAX_EDGES = 50;
const QG_MAX_NEIGHBORS_PER_ANCHOR = 8;
const CODE_INTENT_RE = /\b(class|classes|function|functions|method|methods)\b|\bin\s+[\w./-]+\.[a-z0-9]+\b/i;
const FILENAME_RE = /\b([\w.-]+\.[a-z0-9]+)\b/i;

function _qgNodeId(type: QuestionGraphNodeType, key: string): string {
  return `${type}:${key}`;
}

function detectCodeIntent(question: string): boolean {
  return CODE_INTENT_RE.test(question);
}

function extractExplicitFilename(question: string): string | null {
  const inFileMatch = question.match(/\bin\s+([\w./-]+\.[a-z0-9]+)\b/i);
  if (inFileMatch?.[1]) return inFileMatch[1].split("/").pop() ?? inFileMatch[1];
  const generic = question.match(FILENAME_RE);
  if (!generic?.[1]) return null;
  return generic[1].split("/").pop() ?? generic[1];
}

function requestedClassCount(question: string): number | null {
  if (/\bthree classes?\b/i.test(question) || /\b3 classes?\b/i.test(question)) return 3;
  return null;
}

function snippetScoreValue(snippet: RetrievalSnippet | undefined): number {
  if (!snippet || typeof snippet !== "object") return 0;
  const s = snippet as { score?: number; semantic_score?: number | null; keyword_score?: number | null };
  if (typeof s.score === "number" && Number.isFinite(s.score)) return s.score;
  if (typeof s.semantic_score === "number" && Number.isFinite(s.semantic_score)) return s.semantic_score;
  if (typeof s.keyword_score === "number" && Number.isFinite(s.keyword_score)) return s.keyword_score;
  return 0;
}

function buildFallbackEvidenceFromCode(codeResult: QueryResult | null): KgEvidence[] {
  if (!codeResult?.retrieval_pack) return [];
  const snippets = codeResult.retrieval_pack.snippets ?? [];
  const citations = new Set(codeResult.citations ?? []);
  const codeNodes = codeResult.retrieval_pack.nodes ?? [];
  const snippetById = new Map(snippets.map((s) => [s.id, s]));
  const nodeById = new Map(codeNodes.map((n) => [n.id, n]));
  const orderedIds: string[] = [];

  for (const citation of citations) {
    if (snippetById.has(citation) || nodeById.has(citation)) orderedIds.push(citation);
  }
  for (const snippet of snippets) {
    if (!orderedIds.includes(snippet.id)) orderedIds.push(snippet.id);
  }
  for (const node of codeNodes) {
    if (!orderedIds.includes(node.id)) orderedIds.push(node.id);
  }

  return orderedIds
    .map((id, idx) => {
      const snip = snippetById.get(id);
      const node = nodeById.get(id);
      const text = snip?.code_snippet || node?.code_snippet || "";
      if (!text.trim()) return null;
      return {
        chunk_id: `code:${id}`,
        doc_path: snip?.path || node?.path || id,
        text,
        score: Math.max(0, snippetScoreValue(snip) || (citations.has(id) ? 1 - idx * 0.01 : 0.1)),
      } satisfies KgEvidence;
    })
    .filter((item): item is KgEvidence => Boolean(item));
}

type DefinitionHit = {
  name: string;
  kind: "class" | "function";
  path: string;
};

type EvidenceGate = {
  validPaths: Set<string>;
  validIdentifiers: Map<string, RetrievalNode[]>;
  evidenceSnippets: KgEvidence[];
  kgIsEmpty: boolean;
  codeIsEmpty: boolean;
  targetFilename: string | null;
  requestedClassCount: number | null;
  detectedDefinitions: DefinitionHit[];
};

function pathKey(path: string | undefined): string {
  return (path || "").trim().toLowerCase();
}

function pathBasename(path: string | undefined): string {
  const raw = (path || "").trim();
  if (!raw) return "";
  const parts = raw.split("/");
  return (parts[parts.length - 1] || raw).toLowerCase();
}

function pathAllowed(path: string | undefined, validPaths: Set<string>): boolean {
  const key = pathKey(path);
  if (!key) return false;
  if (validPaths.has(key)) return true;
  const base = pathBasename(path);
  for (const known of validPaths) {
    if (known.endsWith(`/${base}`) || pathBasename(known) === base) return true;
  }
  return false;
}

function detectDefinitionsFromSnippet(text: string, path: string): DefinitionHit[] {
  const hits: DefinitionHit[] = [];
  const classRe = /^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\b/gm;
  const fnRe = /^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\b/gm;
  for (const match of text.matchAll(classRe)) {
    if (match[1]) hits.push({ name: match[1], kind: "class", path });
  }
  for (const match of text.matchAll(fnRe)) {
    if (match[1]) hits.push({ name: match[1], kind: "function", path });
  }
  return hits;
}

function identifierHasEvidence(
  identifier: string,
  candidates: RetrievalNode[],
  evidence: KgEvidence[],
): boolean {
  const name = identifier.trim();
  if (!name) return false;
  const needle = name.toLowerCase();
  return evidence.some((ev) => {
    if (containsCaseInsensitive(ev.text, needle)) return true;
    return candidates.some((node) => {
      const nodePath = node.path || "";
      return Boolean(nodePath) && containsCaseInsensitive(ev.doc_path, nodePath);
    });
  });
}

function buildEvidenceGate(question: string, kgResult: KgQueryResult | null, codeResult: QueryResult | null): EvidenceGate {
  const codeNodes = codeResult?.retrieval_pack?.nodes ?? [];
  const codeSnippets = codeResult?.retrieval_pack?.snippets ?? [];
  const validPaths = new Set<string>();
  for (const node of codeNodes) {
    if (node.path?.trim()) validPaths.add(pathKey(node.path));
  }
  for (const snippet of codeSnippets) {
    if (snippet.path?.trim()) validPaths.add(pathKey(snippet.path));
  }

  const validIdentifiers = new Map<string, RetrievalNode[]>();
  for (const node of codeNodes) {
    const type = String(node.type || "").toLowerCase();
    if (type !== "class" && type !== "function") continue;
    const key = node.name.trim().toLowerCase();
    if (!key) continue;
    if (!validIdentifiers.has(key)) validIdentifiers.set(key, []);
    validIdentifiers.get(key)?.push(node);
  }

  const mergedEvidence = [...(kgResult?.evidence ?? []), ...buildFallbackEvidenceFromCode(codeResult)];
  const seenEvidence = new Set<string>();
  const evidenceSnippets: KgEvidence[] = [];
  for (const ev of mergedEvidence) {
    if (!ev?.text?.trim()) continue;
    if (!pathAllowed(ev.doc_path, validPaths)) continue;
    if (seenEvidence.has(ev.chunk_id)) continue;
    seenEvidence.add(ev.chunk_id);
    evidenceSnippets.push(ev);
  }

  const detectedDefinitions = evidenceSnippets.flatMap((ev) => detectDefinitionsFromSnippet(ev.text, ev.doc_path));

  const kgIsEmpty = (kgResult?.linked_entities?.length ?? 0) === 0 && (kgResult?.evidence?.length ?? 0) === 0;
  const codeIsEmpty = codeNodes.length === 0 && codeSnippets.length === 0;

  return {
    validPaths,
    validIdentifiers,
    evidenceSnippets,
    kgIsEmpty,
    codeIsEmpty,
    targetFilename: extractExplicitFilename(question),
    requestedClassCount: requestedClassCount(question),
    detectedDefinitions,
  };
}

function buildQuestionGraph(params: {
  question: string;
  kgResult: KgQueryResult | null;
  codeResult: QueryResult | null;
  maxNodes?: number;
  maxEdges?: number;
  maxNeighborsPerAnchor?: number;
}): QuestionGraph {
  const {
    question,
    kgResult,
    codeResult,
    maxNodes = QG_MAX_NODES,
    maxEdges = QG_MAX_EDGES,
    maxNeighborsPerAnchor = QG_MAX_NEIGHBORS_PER_ANCHOR,
  } = params;
  const evidenceGate = buildEvidenceGate(question, kgResult, codeResult);

  const nodeMap = new Map<string, QuestionGraphNode>();
  const edges: QuestionGraphEdge[] = [];
  const edgeIds = new Set<string>();

  const addNode = (node: QuestionGraphNode) => {
    if (nodeMap.has(node.id)) return;
    if (nodeMap.size >= maxNodes) return;
    nodeMap.set(node.id, node);
  };
  const addEdge = (edge: QuestionGraphEdge) => {
    if (edges.length >= maxEdges) return;
    if (edgeIds.has(edge.id)) return;
    if (!nodeMap.has(edge.source) || !nodeMap.has(edge.target)) return;
    edgeIds.add(edge.id);
    edges.push(edge);
  };

  const questionId = _qgNodeId("question", "root");
  addNode({
    id: questionId,
    type: "question",
    label: truncate(question || "Question", 120),
    source: "merged",
  });

  const codeIntent = detectCodeIntent(question);
  const explicitFilename = extractExplicitFilename(question);
  const requestedTopClasses = requestedClassCount(question);

  const rawKgEntities = kgResult?.linked_entities ?? [];
  const kgSubgraphNodes = kgResult?.subgraph?.nodes ?? [];
  const kgEvidence = evidenceGate.evidenceSnippets;

  const entityNamesInGraph = new Set<string>();
  const retrievalPack = codeResult?.retrieval_pack;
  const snippets = retrievalPack?.snippets ?? [];
  const codeNodes = retrievalPack?.nodes ?? [];
  const codeEdges = retrievalPack?.edges ?? [];
  const citations = codeResult?.citations ?? [];
  const codeNodeById = new Map(codeNodes.map((node) => [node.id, node]));
  const snippetById = new Map(snippets.map((snippet) => [snippet.id, snippet]));
  const retrievalToGraphNodeId = new Map<string, string>();

  const kgEntities = rawKgEntities.filter((entity) => {
    if (!entity.name.trim()) return false;
    return kgEvidence.some(
      (ev) => containsCaseInsensitive(ev.text, entity.name) || containsCaseInsensitive(ev.doc_path, entity.name),
    );
  });
  const useCodeEntityFallback = codeIntent || kgEntities.length === 0;

  for (const entity of kgEntities) {
    const key = entity.name.trim().toLowerCase();
    if (!key) continue;
    const id = _qgNodeId("entity", key);
    entityNamesInGraph.add(key);
    addNode({
      id,
      type: "entity",
      label: entity.name,
      subtitle: entity.type,
      ref_id: entity.name,
      source: "kg",
      meta: { score: entity.score },
    });
  }
  for (const node of kgSubgraphNodes) {
    const kind = String(node.kind ?? node.type ?? "").toLowerCase();
    if (kind && kind !== "entity") continue;
    const name = String(node.name ?? "").trim();
    if (!name) continue;
    if (!kgEvidence.some((ev) => containsCaseInsensitive(ev.text, name) || containsCaseInsensitive(ev.doc_path, name))) {
      continue;
    }
    const key = name.toLowerCase();
    entityNamesInGraph.add(key);
    addNode({
      id: _qgNodeId("entity", key),
      type: "entity",
      label: name,
      subtitle: String(node.type ?? node.kind ?? "entity"),
      ref_id: String(node.id ?? key),
      source: "kg",
      meta: { repo_id: node.repo_id, original_id: node.id },
    });
  }

  if (useCodeEntityFallback && codeNodes.length > 0) {
    const normalizedFilename = explicitFilename?.toLowerCase() ?? "";
    const fileCandidates = codeNodes.filter((node) => {
      const path = (node.path || "").toLowerCase();
      return normalizedFilename ? path.endsWith(`/${normalizedFilename}`) || path === normalizedFilename : false;
    });

    const fileNodeSource =
      fileCandidates.find((n) => String(n.type).toLowerCase() === "file") ??
      fileCandidates[0] ??
      (normalizedFilename
        ? snippets.find((s) => (s.path || "").toLowerCase().endsWith(`/${normalizedFilename}`) || (s.path || "").toLowerCase() === normalizedFilename)
        : undefined);

    let fileGraphNodeId: string | null = null;
    let filePath: string | null = null;
    if (fileNodeSource) {
      const fileId = fileNodeSource.id;
      filePath = fileNodeSource.path || null;
      fileGraphNodeId = _qgNodeId("code", fileId);
      addNode({
        id: fileGraphNodeId,
        type: "code",
        label: filePath?.split("/").pop() || filePath || fileId,
        subtitle: filePath || "file",
        ref_id: fileNodeSource.id,
        source: "code",
        meta: {
          anchor: true,
          path: filePath || "",
          code_type: "file",
          code_snippet: fileNodeSource.code_snippet,
        },
      });
      addEdge({
        id: `qg:file_anchor:${fileId}`,
        source: questionId,
        target: fileGraphNodeId,
        label: "code_anchor",
      });
      retrievalToGraphNodeId.set(fileNodeSource.id, fileGraphNodeId);
    }

    const candidateNodes = [...codeNodes]
      .filter((node) => {
        const t = String(node.type || "").toLowerCase();
        return t === "class" || t === "function";
      })
      .sort((a, b) => {
        const aType = String(a.type || "").toLowerCase();
        const bType = String(b.type || "").toLowerCase();
        const aPath = (a.path || "").toLowerCase();
        const bPath = (b.path || "").toLowerCase();
        const aFileMatch = normalizedFilename ? Number(aPath.endsWith(`/${normalizedFilename}`) || aPath === normalizedFilename) : 0;
        const bFileMatch = normalizedFilename ? Number(bPath.endsWith(`/${normalizedFilename}`) || bPath === normalizedFilename) : 0;
        const aClass = Number(aType === "class");
        const bClass = Number(bType === "class");
        const aAnchor = Number(citations.includes(a.id));
        const bAnchor = Number(citations.includes(b.id));
        return (
          bFileMatch - aFileMatch ||
          bClass - aClass ||
          bAnchor - aAnchor ||
          (a.path || "").localeCompare(b.path || "") ||
          (a.name || "").localeCompare(b.name || "")
        );
      });

    let selectedCandidates = candidateNodes;
    if (requestedTopClasses) {
      const classOnly = candidateNodes.filter((node) => String(node.type || "").toLowerCase() === "class");
      selectedCandidates = (classOnly.length ? classOnly : candidateNodes).slice(0, requestedTopClasses);
    } else {
      selectedCandidates = candidateNodes.slice(0, 6);
    }

    for (const node of selectedCandidates) {
      const name = (node.name || "").trim();
      if (!name) continue;
      const entityKey = name.toLowerCase();
      const entityId = _qgNodeId("entity", entityKey);
      entityNamesInGraph.add(entityKey);
      addNode({
        id: entityId,
        type: "entity",
        label: node.name,
        subtitle: node.type,
        ref_id: node.id,
        source: "code",
        meta: {
          score: citations.includes(node.id) ? 1 : undefined,
          path: node.path,
          code_type: node.type,
          code_snippet: node.code_snippet,
        },
      });
      retrievalToGraphNodeId.set(node.id, entityId);
      addEdge({
        id: `qg:linked_code_entity:${node.id}`,
        source: questionId,
        target: entityId,
        label: "linked",
      });

      if (fileGraphNodeId && filePath && node.path && (node.path === filePath || node.path.endsWith(`/${filePath.split("/").pop()}`))) {
        addEdge({
          id: `qg:file_contains:${fileGraphNodeId}:${entityId}`,
          source: fileGraphNodeId,
          target: entityId,
          label: "contains",
        });
      }
    }
  }

  for (const entity of kgEntities) {
    const key = entity.name.trim().toLowerCase();
    if (!key) continue;
    addEdge({
      id: `qg:linked:${key}`,
      source: questionId,
      target: _qgNodeId("entity", key),
      label: "linked",
      meta: { score: entity.score },
    });
  }

  for (const ev of kgEvidence) {
    const evNodeId = _qgNodeId("evidence", ev.chunk_id);
    addNode({
      id: evNodeId,
      type: "evidence",
      label: ev.doc_path || ev.chunk_id,
      subtitle: `score ${ev.score.toFixed(3)}`,
      ref_id: ev.chunk_id,
      source: "kg",
      meta: { doc_path: ev.doc_path, text: ev.text, score: ev.score },
    });
  }

  for (const entityNameKey of entityNamesInGraph) {
    const entityNodeId = _qgNodeId("entity", entityNameKey);
    const entityNode = nodeMap.get(entityNodeId);
    if (!entityNode) continue;
    for (const ev of kgEvidence) {
      const text = (ev.text ?? "").toLowerCase();
      if (!text.includes(entityNameKey)) continue;
      addEdge({
        id: `qg:supported_by:${entityNameKey}:${ev.chunk_id}`,
        source: entityNodeId,
        target: _qgNodeId("evidence", ev.chunk_id),
        label: "supported_by",
      });
    }
  }
  const anchorIds: string[] = [];

  for (const citation of citations) {
    if (snippetById.has(citation) || codeNodeById.has(citation)) {
      anchorIds.push(citation);
    }
  }
  for (const snippet of snippets) {
    if (!anchorIds.includes(snippet.id)) anchorIds.push(snippet.id);
  }
  const limitedAnchors = anchorIds.slice(0, 3);

  for (const anchorId of limitedAnchors) {
    if (retrievalToGraphNodeId.has(anchorId)) {
      addEdge({
        id: `qg:code_anchor_existing:${anchorId}`,
        source: questionId,
        target: retrievalToGraphNodeId.get(anchorId)!,
        label: "code_anchor",
      });
      continue;
    }
    const snippet = snippetById.get(anchorId);
    const node = codeNodeById.get(anchorId);
    const label = snippet?.name || node?.name || anchorId;
    const subtitle = snippet?.path || node?.path || "code";
    addNode({
      id: _qgNodeId("code", anchorId),
      type: "code",
      label,
      subtitle,
      ref_id: anchorId,
      source: "code",
      meta: {
        anchor: true,
        path: snippet?.path || node?.path,
        code_snippet: snippet?.code_snippet || node?.code_snippet,
      },
    });
    retrievalToGraphNodeId.set(anchorId, _qgNodeId("code", anchorId));
    addEdge({
      id: `qg:code_anchor:${anchorId}`,
      source: questionId,
      target: _qgNodeId("code", anchorId),
      label: "code_anchor",
    });
  }

  const neighborsPerAnchor = new Map<string, number>();
  for (const edge of codeEdges) {
    const sourceMapped = retrievalToGraphNodeId.get(edge.source);
    const targetMapped = retrievalToGraphNodeId.get(edge.target);
    if (sourceMapped && targetMapped) {
      addEdge({
        id: `qg:selected_rel:${edge.source}:${edge.type}:${edge.target}`,
        source: sourceMapped,
        target: targetMapped,
        label: String(edge.type || "related").toLowerCase(),
      });
      continue;
    }

    for (const anchorId of limitedAnchors) {
      const isSource = edge.source === anchorId;
      const isTarget = edge.target === anchorId;
      if (!isSource && !isTarget) continue;
      const count = neighborsPerAnchor.get(anchorId) ?? 0;
      if (count >= maxNeighborsPerAnchor) continue;

      const neighborId = isSource ? edge.target : edge.source;
      const neighborNode = codeNodeById.get(neighborId);
      if (!neighborNode) continue;

      if (!retrievalToGraphNodeId.has(neighborId)) {
        addNode({
          id: _qgNodeId("code", neighborId),
          type: "code",
          label: neighborNode.name || neighborId,
          subtitle: neighborNode.path || neighborNode.type || "code",
          ref_id: neighborId,
          source: "code",
          meta: {
            anchor: false,
            path: neighborNode.path,
            code_snippet: neighborNode.code_snippet,
            code_type: neighborNode.type,
          },
        });
        retrievalToGraphNodeId.set(neighborId, _qgNodeId("code", neighborId));
      }

      const anchorGraphId = retrievalToGraphNodeId.get(anchorId);
      const neighborGraphId = retrievalToGraphNodeId.get(neighborId);
      if (!anchorGraphId || !neighborGraphId) continue;

      addEdge({
        id: `qg:code_rel:${anchorId}:${edge.type}:${neighborId}`,
        source: anchorGraphId,
        target: neighborGraphId,
        label: String(edge.type || "related").toLowerCase(),
      });
      neighborsPerAnchor.set(anchorId, count + 1);
    }
  }

  const codeNameToNodeIds = new Map<string, string[]>();
  for (const node of nodeMap.values()) {
    if (node.type !== "code") continue;
    const key = node.label.trim().toLowerCase();
    if (!key) continue;
    if (!codeNameToNodeIds.has(key)) codeNameToNodeIds.set(key, []);
    codeNameToNodeIds.get(key)?.push(node.id);
  }
  for (const entityNode of nodeMap.values()) {
    if (entityNode.type !== "entity") continue;
    const matches = codeNameToNodeIds.get(entityNode.label.trim().toLowerCase()) ?? [];
    for (const codeNodeId of matches) {
      addEdge({
        id: `qg:entity_code:${entityNode.id}:${codeNodeId}`,
        source: entityNode.id,
        target: codeNodeId,
        label: "name_match",
      });
    }
  }

  return { nodes: Array.from(nodeMap.values()), edges };
}

function spreadY(count: number, height: number, padding = 44): number[] {
  if (count <= 0) return [];
  if (count === 1) return [height / 2];
  const start = padding;
  const end = height - padding;
  const step = (end - start) / (count - 1);
  return Array.from({ length: count }, (_, index) => start + step * index);
}

function buildKgGraphView(
  question: string,
  entities: KgLinkedEntity[],
  evidence: KgEvidence[],
): { width: number; height: number; nodes: GraphViewNode[]; edges: GraphViewEdge[] } {
  const entityCount = Math.max(entities.length, 1);
  const evidenceCount = Math.max(evidence.length, 1);
  const height = Math.max(260, Math.max(entityCount, evidenceCount) * 76 + 48);
  const width = 1120;
  const questionNodeId = "graph:question";
  const nodes: GraphViewNode[] = [
    {
      id: questionNodeId,
      kind: "question",
      label: truncate(question || "Question", 70),
      subLabel: "Question",
      x: 170,
      y: height / 2,
    },
  ];
  const edges: GraphViewEdge[] = [];

  const entityY = spreadY(entities.length, height);
  const entityNodeIds = new Map<string, string[]>();
  entities.forEach((entity, index) => {
    const nodeId = `graph:entity:${index}`;
    const normalized = entity.name.toLowerCase();
    if (!entityNodeIds.has(normalized)) entityNodeIds.set(normalized, []);
    entityNodeIds.get(normalized)?.push(nodeId);

    nodes.push({
      id: nodeId,
      kind: "entity",
      label: truncate(entity.name, 36),
      subLabel: entity.type,
      x: 545,
      y: entityY[index] ?? height / 2,
      entityName: entity.name,
    });
    edges.push({
      id: `graph:edge:linked:${index}`,
      source: questionNodeId,
      target: nodeId,
      label: "linked",
    });
  });

  const evidenceY = spreadY(evidence.length, height);
  const evidenceNodeIds = new Map<string, string>();
  evidence.forEach((item, index) => {
    const nodeId = `graph:evidence:${index}`;
    evidenceNodeIds.set(item.chunk_id, nodeId);
    nodes.push({
      id: nodeId,
      kind: "evidence",
      label: truncate(item.doc_path || item.chunk_id, 36),
      subLabel: `score ${item.score.toFixed(3)}`,
      x: 920,
      y: evidenceY[index] ?? height / 2,
      evidenceId: item.chunk_id,
    });
  });

  const evidenceWithSupportedEdge = new Set<string>();
  evidence.forEach((item, evidenceIndex) => {
    const text = (item.text ?? "").toLowerCase();
    entities.forEach((entity, entityIndex) => {
      if (!text.includes(entity.name.toLowerCase())) return;
      evidenceWithSupportedEdge.add(item.chunk_id);
      edges.push({
        id: `graph:edge:supported_by:${entityIndex}:${evidenceIndex}`,
        source: `graph:entity:${entityIndex}`,
        target: `graph:evidence:${evidenceIndex}`,
        label: "supported_by",
      });
    });
  });

  evidence.forEach((item, evidenceIndex) => {
    if (evidenceWithSupportedEdge.has(item.chunk_id)) return;
    edges.push({
      id: `graph:edge:evidence:${evidenceIndex}`,
      source: questionNodeId,
      target: evidenceNodeIds.get(item.chunk_id) ?? `graph:evidence:${evidenceIndex}`,
      label: "evidence",
    });
  });

  return { width, height, nodes, edges };
}

export function KgQuerySection({
  repoId,
  question,
  onRepoIdChange,
  onQuestionChange,
  isPending,
  error,
  data,
  onRun,
}: {
  repoId: string;
  question: string;
  onRepoIdChange: (value: string) => void;
  onQuestionChange: (value: string) => void;
  isPending: boolean;
  error: Error | null;
  data: KgQueryResult | undefined;
  onRun: () => void;
}) {
  return (
    <section className="card">
      <h2>KG Query (GraphRAG)</h2>
      <label>
        Repo ID
        <Input value={repoId} onChange={(e) => onRepoIdChange(e.target.value)} placeholder="repo UUID" />
      </label>
      <label>
        Question
        <Textarea value={question} onChange={(e) => onQuestionChange(e.target.value)} rows={3} placeholder="Ask a KG question..." />
      </label>
      <Button onClick={onRun} disabled={isPending}>
        {isPending ? "Running..." : "Run KG Query"}
      </Button>
      {error && <p className="error">{error.message}</p>}
      {data && <KgQueryResultPanel data={data} question={question} />}
    </section>
  );
}

function Level2AnswerPanel({
  question,
  kgResult,
  codeResult,
  kgError,
  codeError,
}: {
  question: string;
  kgResult: KgQueryResult | null;
  codeResult: QueryResult | null;
  kgError: string | null;
  codeError: string | null;
}) {
  const evidenceGate = useMemo(() => buildEvidenceGate(question, kgResult, codeResult), [question, kgResult, codeResult]);
  const kgEntities = kgResult?.linked_entities ?? [];
  const kgEvidence = kgResult?.evidence ?? [];
  const kgTrace = kgResult?.retrieval_trace ?? [];
  const codeSnippets = codeResult?.retrieval_pack?.snippets ?? [];
  const codeNodes = codeResult?.retrieval_pack?.nodes ?? [];
  const codeEdges = codeResult?.retrieval_pack?.edges ?? [];
  const validCodeNodes = codeNodes.filter((node) => pathAllowed(node.path, evidenceGate.validPaths));
  const validPathsList = Array.from(evidenceGate.validPaths);
  const targetFilename = evidenceGate.targetFilename;
  const targetEvidence = targetFilename
    ? evidenceGate.evidenceSnippets.filter((ev) => pathBasename(ev.doc_path) === targetFilename.toLowerCase())
    : [];
  const targetDefinitions = targetFilename
    ? evidenceGate.detectedDefinitions.filter((d) => pathBasename(d.path) === targetFilename.toLowerCase())
    : [];
  const targetClasses = targetDefinitions.filter((d) => d.kind === "class");
  const targetFunctions = targetDefinitions.filter((d) => d.kind === "function");
  const evidenceBackedIdentifiers = Array.from(evidenceGate.validIdentifiers.entries())
    .filter(([name, nodes]) => identifierHasEvidence(name, nodes, evidenceGate.evidenceSnippets))
    .flatMap(([, nodes]) => nodes)
    .slice(0, 8);
  const safeKgEntities = kgEntities.filter((entity) =>
    evidenceGate.evidenceSnippets.some(
      (ev) => containsCaseInsensitive(ev.text, entity.name) || containsCaseInsensitive(ev.doc_path, entity.name),
    ),
  );
  const askThreeClassesInTarget = Boolean(evidenceGate.requestedClassCount === 3 && targetFilename && /\bclasses?\b/i.test(question));

  const hasAny = Boolean(kgResult || codeResult || kgError || codeError);
  if (!hasAny) {
    return <p className="muted">Ask a question to combine KG and code retrieval into a question-centric answer.</p>;
  }

  return (
    <div className="result-box">
      <h3>Question</h3>
      <p>{question}</p>

      <div className="job-details">
        <p>
          KG: <strong>{evidenceGate.kgIsEmpty ? "empty" : "ok"}</strong> · Code:{" "}
          <strong>{evidenceGate.codeIsEmpty ? "empty" : "ok"}</strong>
        </p>
        {evidenceGate.kgIsEmpty && <p className="muted small">Likely repo_id mismatch or KG not ingested.</p>}
      </div>

      <h3>KG Explanation</h3>
      {kgResult ? (
        <>
          {kgEntities.length === 0 && kgEvidence.length === 0 ? (
            <p className="muted">KG returned no entities/evidence.</p>
          ) : (
            <p>
              Retrieved <strong>{kgEntities.length}</strong> linked entities and <strong>{kgEvidence.length}</strong> KG evidence snippets
              {kgTrace.length ? ` across ${kgTrace.length} retrieval steps.` : "."}
            </p>
          )}
          {safeKgEntities.length > 0 ? (
            <ul>
              {safeKgEntities.slice(0, 5).map((entity) => (
                <li key={`${entity.name}-${entity.type}`}>
                  {entity.name} ({entity.type}) score={entity.score.toFixed(3)}
                </li>
              ))}
            </ul>
          ) : kgEntities.length > 0 && kgEvidence.length > 0 ? (
            <p className="muted">No evidence-backed KG entities passed the evidence gate.</p>
          ) : null}
        </>
      ) : (
        <p className="muted">{kgError ? `KG query unavailable: ${kgError}` : "KG query did not return data."}</p>
      )}

      <h3>Grounded in Code</h3>
      {codeResult ? (
        <>
          {evidenceGate.codeIsEmpty ? (
            <p className="muted">Code retrieval returned no nodes/snippets.</p>
          ) : askThreeClassesInTarget && targetClasses.length < 3 ? (
            <>
              <p>
                I could not find three class definitions in <strong>{targetFilename}</strong> from the retrieved context.
              </p>
              <p className="muted small">
                Retrieved snippets in target file: {targetEvidence.length}. Detected definitions:{" "}
                {targetDefinitions.length > 0
                  ? targetDefinitions.map((d) => `${d.kind} ${d.name}`).slice(0, 10).join(", ")
                  : "none"}
              </p>
              {targetEvidence.length > 0 ? (
                <div className="kg-evidence-list">
                  {targetEvidence.slice(0, 3).map((ev) => (
                    <article key={ev.chunk_id} className="kg-evidence-item">
                      <div className="kg-evidence-head">
                        <strong>{ev.doc_path}</strong>
                        <span>{ev.score.toFixed(3)}</span>
                      </div>
                      <pre className="kg-evidence-text">{ev.text}</pre>
                    </article>
                  ))}
                </div>
              ) : (
                <p className="muted">No target-file snippets were retrieved.</p>
              )}
            </>
          ) : (
            <>
              <p>
                Retrieved <strong>{evidenceGate.evidenceSnippets.length}</strong> evidence snippets across{" "}
                <strong>{validPathsList.length}</strong> paths and <strong>{validCodeNodes.length}</strong> code nodes in context.
              </p>
              {targetFilename && (
                <p className="muted small">
                  Target file: {targetFilename}. Matching snippets: {targetEvidence.length}. Classes detected: {targetClasses.length}. Functions
                  detected: {targetFunctions.length}.
                </p>
              )}
              {evidenceBackedIdentifiers.length > 0 ? (
                <ul>
                  {evidenceBackedIdentifiers.slice(0, 6).map((node) => (
                    <li key={`${node.id}:${node.path}`}>
                      {node.name} ({node.type}) in {node.path}
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="muted">No class/function identifiers passed the evidence gate.</p>
              )}
              {validPathsList.length > 0 && (
                <p className="muted small">Paths in retrieved context: {validPathsList.slice(0, 6).join(", ")}</p>
              )}
            </>
          )}
          <p className="muted small">
            citations={(codeResult.citations ?? []).length}, snippets={codeSnippets.length}, nodes={codeNodes.length}, edges={codeEdges.length}
          </p>
        </>
      ) : (
        <p className="muted">{codeError ? `Code query unavailable: ${codeError}` : "Code query did not return data."}</p>
      )}

      {kgError && kgResult && <p className="warning">KG partial warning: {kgError}</p>}
      {codeError && codeResult && <p className="warning">Code partial warning: {codeError}</p>}
    </div>
  );
}

type QuestionGraphLayoutNode = QuestionGraphNode & {
  x: number;
  y: number;
};

function buildQuestionGraphLayout(graph: QuestionGraph): { width: number; height: number; nodes: QuestionGraphLayoutNode[] } {
  const buckets: Record<QuestionGraphNodeType, QuestionGraphNode[]> = {
    question: [],
    entity: [],
    code: [],
    evidence: [],
  };
  for (const node of graph.nodes) buckets[node.type].push(node);

  const width = 1240;
  const maxBucket = Math.max(
    buckets.question.length || 1,
    buckets.entity.length || 1,
    buckets.code.length || 1,
    buckets.evidence.length || 1,
  );
  const height = Math.max(420, maxBucket * 86 + 90);

  const placeRange = (
    items: QuestionGraphNode[],
    x: number,
    startY: number,
    endY: number,
  ): QuestionGraphLayoutNode[] => {
    if (!items.length) return [];
    const safeStart = Math.max(40, startY);
    const safeEnd = Math.max(safeStart + 1, endY);
    const ys =
      items.length === 1
        ? [(safeStart + safeEnd) / 2]
        : Array.from({ length: items.length }, (_, index) => safeStart + ((safeEnd - safeStart) * index) / (items.length - 1));
    return items.map((item, index) => ({ ...item, x, y: ys[index] ?? height / 2 }));
  };

  const topBandEnd = height * 0.42;
  const bottomBandStart = height * 0.58;

  return {
    width,
    height,
    nodes: [
      ...placeRange(buckets.question, 140, height * 0.45, height * 0.55),
      ...placeRange(buckets.entity, 470, 60, topBandEnd),
      ...placeRange(buckets.code, 470, bottomBandStart, height - 60),
      ...placeRange(buckets.evidence, 1000, 60, height - 60),
    ],
  };
}

function Level2QuestionWorkspace({
  question,
  kgResult,
  codeResult,
  kgError,
  codeError,
}: {
  question: string;
  kgResult: KgQueryResult | null;
  codeResult: QueryResult | null;
  kgError: string | null;
  codeError: string | null;
}) {
  const evidenceGate = useMemo(() => buildEvidenceGate(question, kgResult, codeResult), [question, kgResult, codeResult]);
  const questionGraph = useMemo(
    () => buildQuestionGraph({ question, kgResult, codeResult }),
    [question, kgResult, codeResult],
  );
  const layout = useMemo(() => buildQuestionGraphLayout(questionGraph), [questionGraph]);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null);
  const [filterEvidenceBySelection, setFilterEvidenceBySelection] = useState(true);
  const [showAllEvidence, setShowAllEvidence] = useState(false);

  const nodeById = useMemo(() => new Map(questionGraph.nodes.map((node) => [node.id, node])), [questionGraph.nodes]);
  const selectedNode = useMemo(
    () => questionGraph.nodes.find((node) => node.id === selectedNodeId) ?? null,
    [questionGraph.nodes, selectedNodeId],
  );
  const selectedEdge = useMemo(
    () => questionGraph.edges.find((edge) => edge.id === selectedEdgeId) ?? null,
    [questionGraph.edges, selectedEdgeId],
  );

  const effectiveEvidence = evidenceGate.evidenceSnippets;
  const evidenceMap = useMemo(
    () => new Map(effectiveEvidence.map((ev) => [ev.chunk_id, ev])),
    [effectiveEvidence],
  );
  const codeSnippetMap = useMemo(
    () => new Map((codeResult?.retrieval_pack?.snippets ?? []).map((snip) => [snip.id, snip])),
    [codeResult],
  );
  const codeNodeMap = useMemo(
    () => new Map((codeResult?.retrieval_pack?.nodes ?? []).map((node) => [node.id, node])),
    [codeResult],
  );
  const questionGraphStats = useMemo(
    () => ({
      entities: questionGraph.nodes.filter((n) => n.type === "entity").length,
      evidence: questionGraph.nodes.filter((n) => n.type === "evidence").length,
      code: questionGraph.nodes.filter((n) => n.type === "code").length,
      edges: questionGraph.edges.length,
    }),
    [questionGraph],
  );
  const allEvidence = effectiveEvidence;
  const selectedEdgeEvidence = useMemo(() => {
    if (!selectedEdge) return null;
    const sourceNode = nodeById.get(selectedEdge.source) ?? null;
    const targetNode = nodeById.get(selectedEdge.target) ?? null;
    const evidenceNode =
      sourceNode?.type === "evidence" ? sourceNode : targetNode?.type === "evidence" ? targetNode : null;
    if (!evidenceNode?.ref_id) return null;
    return evidenceMap.get(evidenceNode.ref_id) ?? null;
  }, [selectedEdge, nodeById, evidenceMap]);
  const selectionMatchedEvidence = useMemo(() => {
    if (selectedNode) {
      if (selectedNode.type === "entity") {
        const selectedPath = typeof selectedNode.meta?.path === "string" ? selectedNode.meta.path : "";
        return allEvidence.filter(
          (ev) =>
            containsCaseInsensitive(ev.text, selectedNode.label) ||
            containsCaseInsensitive(ev.doc_path, selectedNode.label) ||
            (selectedPath ? containsCaseInsensitive(ev.doc_path, selectedPath) : false),
        );
      }
      if (selectedNode.type === "evidence" && selectedNode.ref_id) {
        const ev = evidenceMap.get(selectedNode.ref_id);
        return ev ? [ev] : [];
      }
      if (selectedNode.type === "question") {
        return allEvidence;
      }
      if (selectedNode.type === "code") {
        const exactName = selectedNode.label.trim().toLowerCase();
        const selectedPath = typeof selectedNode.meta?.path === "string" ? selectedNode.meta.path : "";
        return allEvidence.filter(
          (ev) =>
            containsCaseInsensitive(ev.text, exactName) ||
            (selectedPath ? containsCaseInsensitive(ev.doc_path, selectedPath) : false),
        );
      }
      return [];
    }
    if (selectedEdge) {
      if (selectedEdge.label === "supported_by" && selectedEdgeEvidence) return [selectedEdgeEvidence];
      const sourceNode = nodeById.get(selectedEdge.source);
      const targetNode = nodeById.get(selectedEdge.target);
      const relatedNames = [sourceNode, targetNode]
        .filter((n): n is QuestionGraphNode => Boolean(n))
        .filter((n) => n.type === "entity")
        .map((n) => n.label);
      if (!relatedNames.length) return selectedEdgeEvidence ? [selectedEdgeEvidence] : [];
      return allEvidence.filter((ev) =>
        relatedNames.some((name) => containsCaseInsensitive(ev.text, name) || containsCaseInsensitive(ev.doc_path, name)),
      );
    }
    return [];
  }, [selectedNode, selectedEdge, selectedEdgeEvidence, allEvidence, evidenceMap, nodeById]);
  const hasSelection = Boolean(selectedNode || selectedEdge);
  const effectiveEvidencePool = useMemo(() => {
    if (!filterEvidenceBySelection || !hasSelection) return allEvidence;
    return selectionMatchedEvidence;
  }, [filterEvidenceBySelection, hasSelection, allEvidence, selectionMatchedEvidence]);
  const visibleEvidence = useMemo(
    () => (showAllEvidence ? effectiveEvidencePool : effectiveEvidencePool.slice(0, 6)),
    [effectiveEvidencePool, showAllEvidence],
  );
  const selectionLabel = selectedNode
    ? `${selectedNode.type}: ${selectedNode.label}`
    : selectedEdge
      ? `edge: ${selectedEdge.label}`
      : null;

  return (
    <div className="kg-answer-columns">
      <div className="kg-answer-left">
        <section className="kg-panel">
          <h4>Graph View</h4>
          <QuestionGraphView
            graph={questionGraph}
            layout={layout}
            selectedNodeId={selectedNodeId}
            selectedEdgeId={selectedEdgeId}
            onSelectNode={(nodeId) => {
              setSelectedEdgeId(null);
              setSelectedNodeId((prev) => (prev === nodeId ? null : nodeId));
            }}
            onSelectEdge={(edgeId) => {
              setSelectedNodeId(null);
              setSelectedEdgeId((prev) => (prev === edgeId ? null : edgeId));
            }}
          />
        </section>
      </div>

      <div className="kg-answer-right">
            <section className="kg-panel">
              <h4>Details / Evidence</h4>
              {selectedEdge ? (
                <div className="kg-selected-edge-card">
                  <p className="muted small">Selected edge</p>
              <p>
                <strong>{nodeById.get(selectedEdge.source)?.label ?? selectedEdge.source}</strong> --({selectedEdge.label})--&gt;{" "}
                    <strong>{nodeById.get(selectedEdge.target)?.label ?? selectedEdge.target}</strong>
                  </p>
                  {selectedEdge.label === "supported_by" && selectedEdgeEvidence ? (
                    <article className="kg-evidence-item active">
                      <div className="kg-evidence-head">
                        <strong>{selectedEdgeEvidence.doc_path || selectedEdgeEvidence.chunk_id}</strong>
                        <span>{selectedEdgeEvidence.score.toFixed(3)}</span>
                      </div>
                      <pre className="kg-evidence-text">{selectedEdgeEvidence.text}</pre>
                    </article>
                  ) : null}
                </div>
              ) : null}

              {selectedNode ? (
                <div className="result-box">
              <h3>{selectedNode.label}</h3>
              <p className="muted small">
                type={selectedNode.type}
                {selectedNode.subtitle ? ` · ${selectedNode.subtitle}` : ""}
              </p>
              {selectedNode.type === "question" && (
                <>
                  <h4>Question Context</h4>
                  <p>{question}</p>
                  <h4>Combined Trace Summary</h4>
                  <ul>
                    <li>
                      KG: entities={(kgResult?.linked_entities ?? []).length}, evidence={(kgResult?.evidence ?? []).length}, trace_steps=
                      {(kgResult?.retrieval_trace ?? []).length}
                    </li>
                    <li>
                      Code: citations={(codeResult?.citations ?? []).length}, snippets=
                      {(codeResult?.retrieval_pack?.snippets ?? []).length}, nodes=
                      {(codeResult?.retrieval_pack?.nodes ?? []).length}, edges=
                      {(codeResult?.retrieval_pack?.edges ?? []).length}
                    </li>
                  </ul>
                  {(kgResult?.retrieval_trace ?? []).length > 0 && (
                    <>
                      <h4>KG Trace</h4>
                      <ol className="kg-trace-list">
                        {(kgResult?.retrieval_trace ?? []).map((step, idx) => (
                          <li key={`${step.step}-${idx}`}>
                            <strong>{step.step}</strong>
                            <p>{step.detail}</p>
                          </li>
                        ))}
                      </ol>
                    </>
                  )}
                </>
              )}
              {selectedNode.type === "entity" && (
                <>
                  <h4>Entity Details</h4>
                  <p className="muted small">
                    type={selectedNode.subtitle || "entity"} · kg_score=
                    {typeof selectedNode.meta?.score === "number" ? Number(selectedNode.meta.score).toFixed(3) : "n/a"}
                  </p>
                  <h4>Matching Evidence</h4>
                  <div className="kg-evidence-list">
                    {allEvidence
                      .filter((ev) => {
                        const selectedPath = typeof selectedNode.meta?.path === "string" ? selectedNode.meta.path : "";
                        return (
                          containsCaseInsensitive(ev.text, selectedNode.label) ||
                          containsCaseInsensitive(ev.doc_path, selectedNode.label) ||
                          (selectedPath ? containsCaseInsensitive(ev.doc_path, selectedPath) : false)
                        );
                      })
                      .slice(0, 8)
                      .map((ev) => (
                        <article key={ev.chunk_id} className="kg-evidence-item">
                          <div className="kg-evidence-head">
                            <strong>{ev.doc_path || ev.chunk_id}</strong>
                            <span>{ev.score.toFixed(3)}</span>
                          </div>
                          <pre className="kg-evidence-text">{ev.text}</pre>
                        </article>
                      ))}
                    {allEvidence.filter((ev) => {
                      const selectedPath = typeof selectedNode.meta?.path === "string" ? selectedNode.meta.path : "";
                      return (
                        containsCaseInsensitive(ev.text, selectedNode.label) ||
                        containsCaseInsensitive(ev.doc_path, selectedNode.label) ||
                        (selectedPath ? containsCaseInsensitive(ev.doc_path, selectedPath) : false)
                      );
                    }).length === 0 && (
                      <p className="muted">No matching snippet retrieved.</p>
                    )}
                  </div>
                </>
              )}
              {selectedNode.type === "evidence" && selectedNode.ref_id && (
                <>
                  <h4>Evidence Snippet</h4>
                  {evidenceMap.get(selectedNode.ref_id) ? (
                    <article className="kg-evidence-item active">
                      <div className="kg-evidence-head">
                        <strong>{evidenceMap.get(selectedNode.ref_id)?.doc_path || selectedNode.ref_id}</strong>
                        <span>{(evidenceMap.get(selectedNode.ref_id)?.score ?? 0).toFixed(3)}</span>
                      </div>
                      <pre className="kg-evidence-text">{evidenceMap.get(selectedNode.ref_id)?.text}</pre>
                    </article>
                  ) : (
                    <p className="muted">Evidence details unavailable.</p>
                  )}
                </>
              )}
              {selectedNode.type === "code" && selectedNode.ref_id && (
                <>
                  <h4>Code Context</h4>
                  <p className="muted small">
                    name={selectedNode.label}
                    {selectedNode.subtitle ? ` · ${selectedNode.subtitle}` : ""}
                  </p>
                  {typeof selectedNode.meta?.path === "string" && !pathAllowed(selectedNode.meta.path, evidenceGate.validPaths) ? (
                    <p className="muted">No matching snippet retrieved.</p>
                  ) : (
                    <>
                  {(codeResult?.citations ?? []).includes(selectedNode.ref_id) && (
                    <p className="muted small">Referenced by code citations.</p>
                  )}
                  {codeSnippetMap.get(selectedNode.ref_id) || codeNodeMap.get(selectedNode.ref_id) ? (
                    <article className="kg-evidence-item">
                      <div className="kg-evidence-head">
                        <strong>
                          {codeSnippetMap.get(selectedNode.ref_id)?.path ||
                            codeNodeMap.get(selectedNode.ref_id)?.path ||
                            selectedNode.ref_id}
                        </strong>
                        <span>{String(selectedNode.meta?.anchor ? "anchor" : "neighbor")}</span>
                      </div>
                      <pre className="kg-evidence-text">
                        {codeSnippetMap.get(selectedNode.ref_id)?.code_snippet ||
                          codeNodeMap.get(selectedNode.ref_id)?.code_snippet ||
                          "No code snippet available."}
                      </pre>
                    </article>
                  ) : (
                    <p className="muted">No matching snippet retrieved.</p>
                  )}
                    </>
                  )}
                </>
              )}
            </div>
          ) : (
            <div className="result-box">
              <p className="muted">Select a graph node or edge to inspect supporting evidence and code context.</p>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Metric</th>
                      <th>Count</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>Entity nodes</td>
                      <td>{questionGraphStats.entities}</td>
                    </tr>
                    <tr>
                      <td>Evidence nodes</td>
                      <td>{questionGraphStats.evidence}</td>
                    </tr>
                    <tr>
                      <td>Code nodes</td>
                      <td>{questionGraphStats.code}</td>
                    </tr>
                    <tr>
                      <td>Total edges</td>
                      <td>{questionGraphStats.edges}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          )}

          <div className="kg-divider" />
          <div className="kg-panel-head">
            <h4>Evidence</h4>
            <p className="muted small">
              {selectionLabel ? `Selection: ${selectionLabel}` : "Selection: none"} · {visibleEvidence.length}/{effectiveEvidencePool.length} shown
            </p>
          </div>
          <div className="kg-panel-actions">
            <label className="kg-toggle">
              <input
                type="checkbox"
                checked={filterEvidenceBySelection}
                onChange={(e) => setFilterEvidenceBySelection(e.target.checked)}
              />
              Filter evidence by selection
            </label>
            <label className="kg-toggle">
              <input
                type="checkbox"
                checked={showAllEvidence}
                onChange={(e) => setShowAllEvidence(e.target.checked)}
              />
              Show all evidence
            </label>
          </div>
          {!effectiveEvidencePool.length ? (
            <p className="muted">
              {hasSelection && filterEvidenceBySelection
                ? "No evidence matches the current selection."
                : "No evidence snippets available."}
            </p>
          ) : (
            <div className="kg-evidence-list">
              {visibleEvidence.map((ev) => (
                <article key={ev.chunk_id} className="kg-evidence-item">
                  <div className="kg-evidence-head">
                    <strong>{ev.doc_path || ev.chunk_id}</strong>
                    <span>{ev.score.toFixed(3)}</span>
                  </div>
                  <pre className="kg-evidence-text">{ev.text}</pre>
                </article>
              ))}
            </div>
          )}

          <div className="kg-divider" />
          <h4>Query Health</h4>
          <p className="muted small">
            {kgError ? `KG: ${kgError}` : `KG: ${evidenceGate.kgIsEmpty ? "empty" : "ok"}`}
          </p>
          <p className="muted small">
            {codeError ? `Code: ${codeError}` : `Code: ${evidenceGate.codeIsEmpty ? "empty" : "ok"}`}
          </p>
          {evidenceGate.kgIsEmpty && <p className="muted small">Likely repo_id mismatch or KG not ingested.</p>}
        </section>
      </div>
    </div>
  );
}

function QuestionGraphView({
  graph,
  layout,
  selectedNodeId,
  selectedEdgeId,
  onSelectNode,
  onSelectEdge,
}: {
  graph: QuestionGraph;
  layout: { width: number; height: number; nodes: QuestionGraphLayoutNode[] };
  selectedNodeId: string | null;
  selectedEdgeId: string | null;
  onSelectNode: (nodeId: string) => void;
  onSelectEdge: (edgeId: string) => void;
}) {
  const [focusSelection, setFocusSelection] = useState(false);

  const questionNodeId = useMemo(
    () => graph.nodes.find((node) => node.type === "question")?.id ?? null,
    [graph.nodes],
  );

  const visibleGraph = useMemo(() => {
    if (!focusSelection) return graph;
    const seedIds = new Set<string>();
    if (selectedNodeId) seedIds.add(selectedNodeId);
    if (selectedEdgeId) {
      const selected = graph.edges.find((edge) => edge.id === selectedEdgeId);
      if (selected) {
        seedIds.add(selected.source);
        seedIds.add(selected.target);
      }
    }
    if (!seedIds.size) return graph;

    const neighborIds = new Set(seedIds);
    for (const edge of graph.edges) {
      if (seedIds.has(edge.source) || seedIds.has(edge.target)) {
        neighborIds.add(edge.source);
        neighborIds.add(edge.target);
      }
    }
    const nodes = graph.nodes.filter((node) => neighborIds.has(node.id));
    const edges = graph.edges.filter((edge) => neighborIds.has(edge.source) && neighborIds.has(edge.target));
    return { nodes, edges };
  }, [focusSelection, graph, selectedEdgeId, selectedNodeId]);

  const visibleNodeIds = useMemo(() => new Set(visibleGraph.nodes.map((node) => node.id)), [visibleGraph.nodes]);
  const visibleLayoutNodes = useMemo(
    () => layout.nodes.filter((node) => visibleNodeIds.has(node.id)),
    [layout.nodes, visibleNodeIds],
  );

  const flowNodes = useMemo<RFNode[]>(
    () =>
      visibleLayoutNodes.map((node) => {
        const selected = selectedNodeId === node.id;
        const palette =
          node.type === "question"
            ? { fill: "#132646", stroke: "#68b6ff" }
            : node.type === "entity"
              ? { fill: "#102d2c", stroke: "#47d9b2" }
              : node.type === "code"
                ? { fill: "#2b1b10", stroke: "#f0a85c" }
                : { fill: "#1a1433", stroke: "#b18bf6" };
        return {
          id: node.id,
          position: { x: node.x - 88, y: node.y - 32 },
          draggable: true,
          selectable: true,
          data: {
            label: (
              <div className="qg-node-content">
                <div className="qg-node-title">{truncate(node.label, 26)}</div>
                {node.subtitle ? <div className="qg-node-subtitle">{truncate(node.subtitle, 26)}</div> : null}
              </div>
            ),
            nodeType: node.type,
          },
          style: {
            width: 176,
            minHeight: 64,
            borderRadius: 10,
            border: `2px solid ${selected ? "#f6fbff" : palette.stroke}`,
            background: palette.fill,
            color: "#eef5ff",
            boxShadow: selected ? "0 0 0 2px rgba(104,182,255,0.35)" : "none",
            padding: "8px 10px",
          },
        } satisfies RFNode;
      }),
    [visibleLayoutNodes, selectedNodeId],
  );

  const flowEdges = useMemo<RFEdge[]>(
    () =>
      visibleGraph.edges.map((edge) => {
        const active = selectedEdgeId === edge.id;
        const stroke =
          edge.label === "linked" ? "#68b6ff" : edge.label === "supported_by" ? "#47d9b2" : edge.label === "code_anchor" ? "#f0a85c" : "#b8c8df";
        return {
          id: edge.id,
          source: edge.source,
          target: edge.target,
          type: "smoothstep",
          label: edge.label,
          animated: edge.label === "linked" || edge.label === "supported_by",
          style: {
            stroke: active ? "#f6fbff" : stroke,
            strokeWidth: active ? 2.8 : 1.8,
          },
          labelStyle: {
            fill: "#cde0ff",
            fontSize: 10,
            fontWeight: 600,
          },
          markerEnd: {
            type: MarkerType.ArrowClosed,
            color: active ? "#f6fbff" : stroke,
          },
        } satisfies RFEdge;
      }),
    [visibleGraph.edges, selectedEdgeId],
  );

  const [rfNodes, setRfNodes] = useState<RFNode[]>([]);
  const [rfEdges, setRfEdges] = useState<RFEdge[]>([]);

  useEffect(() => {
    setRfNodes(flowNodes);
    setRfEdges(flowEdges);
  }, [flowNodes, flowEdges]);

  useEffect(() => {
    setFocusSelection(false);
  }, [graph.nodes, graph.edges]);

  if (!layout.nodes.length) {
    return <p className="muted">No merged question graph available.</p>;
  }

  return (
    <div className="kg-graph-view-wrap">
      <div className="qg-toolbar">
        <div className="qg-legend">
          <span className="qg-legend-item"><i className="qg-dot question" />Question</span>
          <span className="qg-legend-item"><i className="qg-dot entity" />Entity</span>
          <span className="qg-legend-item"><i className="qg-dot code" />Code</span>
          <span className="qg-legend-item"><i className="qg-dot evidence" />Evidence</span>
        </div>
        <div className="qg-actions">
          <Button
            type="button"
            variant="outline"
            className="chip"
            disabled={!selectedNodeId && !selectedEdgeId}
            onClick={() => setFocusSelection(true)}
          >
            Focus selection
          </Button>
          <Button
            type="button"
            variant="outline"
            className="chip"
            onClick={() => {
              setFocusSelection(false);
              if (questionNodeId) onSelectNode(questionNodeId);
            }}
          >
            Reset to question
          </Button>
        </div>
      </div>

      <div className="qg-reactflow-shell">
        <ReactFlow
          nodes={rfNodes}
          edges={rfEdges}
          onNodesChange={(changes: NodeChange[]) => setRfNodes((prev) => applyNodeChanges(changes, prev))}
          onEdgesChange={(changes: EdgeChange[]) => setRfEdges((prev) => applyEdgeChanges(changes, prev))}
          onNodeClick={(_, node) => onSelectNode(node.id)}
          onEdgeClick={(_, edge) => onSelectEdge(edge.id)}
          fitView
          panOnDrag
          zoomOnScroll
          nodesDraggable
          elementsSelectable
          className="qg-reactflow"
          defaultViewport={{ x: 0, y: 0, zoom: 0.9 }}
          proOptions={{ hideAttribution: true }}
        >
          <Background color="#1d3355" gap={24} />
          <MiniMap
            pannable
            zoomable
            nodeColor={(node) => {
              const nodeType = String(node.data?.nodeType ?? "");
              if (nodeType === "question") return "#68b6ff";
              if (nodeType === "entity") return "#47d9b2";
              if (nodeType === "code") return "#f0a85c";
              return "#b18bf6";
            }}
            maskColor="rgba(6, 8, 15, 0.55)"
          />
          <Controls showInteractive={false} />
        </ReactFlow>
      </div>
      {focusSelection && <p className="muted small">Showing selection and 1-hop neighbors.</p>}
    </div>
  );
}

function KgQueryResultPanel({ data, question }: { data: KgQueryResult; question: string }) {
  const linkedEntities = data.linked_entities ?? [];
  const nodes = data.subgraph?.nodes ?? [];
  const edges = data.subgraph?.edges ?? [];
  const evidence = data.evidence ?? [];
  const retrievalTrace = data.retrieval_trace ?? [];
  const [selectedEntityName, setSelectedEntityName] = useState<string | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedEvidenceId, setSelectedEvidenceId] = useState<string | null>(null);
  const [selectedGraphEdgeId, setSelectedGraphEdgeId] = useState<string | null>(null);
  const [subgraphQuery, setSubgraphQuery] = useState("");
  const [showAllEvidence, setShowAllEvidence] = useState(false);
  const [showLowSignalEntities, setShowLowSignalEntities] = useState(false);
  const evidenceItemRefs = useRef<Record<string, HTMLElement | null>>({});

  const visibleLinkedEntities = useMemo(() => {
    if (showLowSignalEntities) return linkedEntities;
    return linkedEntities.filter((entity) => !isLowSignalEntity(entity));
  }, [linkedEntities, showLowSignalEntities]);

  const visibleNodes = useMemo(() => {
    if (showLowSignalEntities) return nodes;
    return nodes.filter((node) => {
      const kind = (typeof node.kind === "string" ? node.kind : node.type ?? "").toString().trim().toLowerCase();
      if (kind !== "entity") return true;
      return !NUMERIC_ONLY_RE.test(displayNodeName(node).trim());
    });
  }, [nodes, showLowSignalEntities]);

  const filteredEvidence = useMemo(() => {
    if (!selectedEntityName) return evidence;
    const needle = selectedEntityName.toLowerCase();
    return evidence.filter((item) => {
      const text = (item.text ?? "").toLowerCase();
      const path = (item.doc_path ?? "").toLowerCase();
      return text.includes(needle) || path.includes(needle);
    });
  }, [evidence, selectedEntityName]);

  const visibleEvidence = useMemo(() => {
    if (!selectedEntityName) return evidence;
    if (filteredEvidence.length > 0) return filteredEvidence;
    return showAllEvidence ? evidence : [];
  }, [evidence, filteredEvidence, selectedEntityName, showAllEvidence]);

  const selectedEntityNeedle = selectedEntityName?.toLowerCase() ?? null;
  const linkedEntityByName = useMemo(
    () => new Map(linkedEntities.map((entity) => [entity.name.toLowerCase(), entity])),
    [linkedEntities],
  );
  const graphView = useMemo(
    () => buildKgGraphView(question, visibleLinkedEntities, evidence),
    [question, visibleLinkedEntities, evidence],
  );
  const graphNodeById = useMemo(() => new Map(graphView.nodes.map((node) => [node.id, node])), [graphView.nodes]);
  const selectedGraphEdge = useMemo(
    () => graphView.edges.find((edge) => edge.id === selectedGraphEdgeId) ?? null,
    [graphView.edges, selectedGraphEdgeId],
  );
  const searchedNodes = useMemo(() => {
    const query = subgraphQuery.trim().toLowerCase();
    if (!query) return visibleNodes;
    return visibleNodes.filter(
      (node) =>
        displayNodeName(node).toLowerCase().includes(query) || displayNodeType(node).toLowerCase().includes(query),
    );
  }, [visibleNodes, subgraphQuery]);
  const groupedNodes = useMemo(() => {
    const groups: Record<SubgraphGroup, KgSubgraphNode[]> = {
      class: [],
      function: [],
      module: [],
      other: [],
    };
    for (const node of searchedNodes) {
      groups[subgraphGroupForNode(node)].push(node);
    }
    return groups;
  }, [searchedNodes]);
  const searchedNodeIds = useMemo(() => new Set(searchedNodes.map((node) => node.id)), [searchedNodes]);
  const visibleEdges = useMemo(
    () => edges.filter((edge) => searchedNodeIds.has(edge.source) && searchedNodeIds.has(edge.target)),
    [edges, searchedNodeIds],
  );
  const nodeNameById = useMemo(() => new Map(nodes.map((node) => [node.id, displayNodeName(node)])), [nodes]);
  const selectedNode = useMemo(
    () => searchedNodes.find((node) => node.id === selectedNodeId) ?? null,
    [searchedNodes, selectedNodeId],
  );
  const selectedNodeLinkedEntity = useMemo(() => {
    if (!selectedNode) return null;
    return linkedEntityByName.get(displayNodeName(selectedNode).toLowerCase()) ?? null;
  }, [selectedNode, linkedEntityByName]);
  const selectedNodeEvidence = useMemo(() => {
    if (!selectedNode) return [];
    const name = displayNodeName(selectedNode);
    return evidence.filter((item) => containsCaseInsensitive(item.text, name));
  }, [evidence, selectedNode]);

  useEffect(() => {
    if (!selectedEntityName) return;
    const stillVisible = visibleLinkedEntities.some((entity) => entity.name.toLowerCase() === selectedEntityName.toLowerCase());
    if (!stillVisible) {
      setSelectedEntityName(null);
      setShowAllEvidence(false);
    }
  }, [selectedEntityName, visibleLinkedEntities]);

  useEffect(() => {
    if (!selectedNodeId) return;
    const stillVisible = searchedNodes.some((node) => node.id === selectedNodeId);
    if (!stillVisible) {
      setSelectedNodeId(null);
    }
  }, [selectedNodeId, searchedNodes]);

  useEffect(() => {
    if (!selectedEvidenceId) return;
    const stillVisible = visibleEvidence.some((item) => item.chunk_id === selectedEvidenceId);
    if (!stillVisible) {
      setSelectedEvidenceId(null);
    }
  }, [selectedEvidenceId, visibleEvidence]);

  useEffect(() => {
    if (!selectedGraphEdgeId) return;
    const stillVisible = graphView.edges.some((edge) => edge.id === selectedGraphEdgeId);
    if (!stillVisible) {
      setSelectedGraphEdgeId(null);
    }
  }, [selectedGraphEdgeId, graphView.edges]);

  useEffect(() => {
    if (!selectedEvidenceId) return;
    const element = evidenceItemRefs.current[selectedEvidenceId];
    if (!element) return;
    element.scrollIntoView({ behavior: "smooth", block: "center" });
  }, [selectedEvidenceId, visibleEvidence]);

  if (!linkedEntities.length) {
    return (
      <div className="result-box">
        <h3>KG Results</h3>
        <p className="muted">No KG data found for this repo_id.</p>
        <p className="muted small">Try re-ingesting this repository with KG ingest mode and ask again.</p>
      </div>
    );
  }

  return (
    <div className="result-box kg-result-box">
      <div className="kg-panels">
        <section className="kg-panel">
          <div className="kg-panel-head">
            <h4>Linked Entities</h4>
            <div className="kg-panel-actions">
              <label className="kg-toggle">
                <input
                  type="checkbox"
                  checked={showLowSignalEntities}
                  onChange={(e) => setShowLowSignalEntities(e.target.checked)}
                />
                Show low-signal entities
              </label>
              {selectedEntityName && (
                <Button
                  type="button"
                  variant="outline"
                  className="chip"
                  onClick={() => {
                    setSelectedEntityName(null);
                    setShowAllEvidence(false);
                  }}
                >
                  Clear filter
                </Button>
              )}
            </div>
          </div>
          {!visibleLinkedEntities.length ? (
            <p className="muted small">No linked entities after filtering. Enable "Show low-signal entities" to view all.</p>
          ) : (
          <div className="kg-chips">
            {visibleLinkedEntities.map((entity: KgLinkedEntity) => {
              const isSelected = selectedEntityName?.toLowerCase() === entity.name.toLowerCase();
              return (
              <button
                key={`${entity.name}-${entity.type}`}
                type="button"
                className={`kg-chip ${isSelected ? "active" : ""}`}
                onClick={() => {
                  setSelectedEntityName((prev) =>
                    prev?.toLowerCase() === entity.name.toLowerCase() ? null : entity.name,
                  );
                  setShowAllEvidence(false);
                }}
              >
                <span>{entity.name}</span>
                <small>{entity.type}</small>
                <em>{entity.score.toFixed(3)}</em>
              </button>
            );
            })}
          </div>
          )}
        </section>

        <div className="kg-answer-columns">
          <div className="kg-answer-left">
            <section className="kg-panel">
              <h4>Graph View</h4>
              <KgGraphView
                question={question}
                graph={graphView}
                selectedEntityName={selectedEntityName}
                selectedEvidenceId={selectedEvidenceId}
                selectedEdgeId={selectedGraphEdgeId}
                onSelectEntity={(entityName) => {
                  setSelectedGraphEdgeId(null);
                  setSelectedEntityName((prev) => (prev?.toLowerCase() === entityName.toLowerCase() ? null : entityName));
                  setShowAllEvidence(false);
                }}
                onSelectEvidence={(evidenceId) => {
                  setSelectedGraphEdgeId(null);
                  setSelectedEvidenceId(evidenceId);
                  setSelectedEntityName(null);
                  setShowAllEvidence(false);
                }}
                onSelectEdge={(edgeId) => {
                  setSelectedGraphEdgeId((prev) => (prev === edgeId ? null : edgeId));
                }}
              />
            </section>

            <section className="kg-panel">
              <div className="kg-panel-head">
                <h4>Subgraph</h4>
                <Input
                  value={subgraphQuery}
                  onChange={(event) => setSubgraphQuery(event.target.value)}
                  placeholder="Search subgraph nodes..."
                  className="kg-subgraph-search"
                />
              </div>
              {!searchedNodes.length ? (
                <p className="muted">No subgraph nodes returned.</p>
              ) : (
                <div className="kg-subgraph-layout">
                  <div className="kg-subgraph-groups">
                    {(["class", "function", "module", "other"] as const).map((group) => (
                      <section key={group} className="kg-subgraph-group">
                        <h5>
                          {group} <span>{groupedNodes[group].length}</span>
                        </h5>
                        {!groupedNodes[group].length ? (
                          <p className="muted small">No nodes.</p>
                        ) : (
                          <div className="kg-node-pill-grid">
                            {groupedNodes[group].map((node) => {
                              const selected = selectedNodeId === node.id;
                              const nodeName = displayNodeName(node);
                              const linked = linkedEntityByName.has(nodeName.toLowerCase());
                              const entityMatch = selectedEntityNeedle && nodeName.toLowerCase().includes(selectedEntityNeedle);
                              return (
                                <button
                                  key={node.id}
                                  type="button"
                                  className={`kg-node-pill ${linked ? "kg-node-pill-linked" : ""} ${entityMatch ? "kg-node-pill-active" : ""} ${selected ? "kg-node-pill-selected" : ""}`}
                                  onClick={() => {
                                    setSelectedGraphEdgeId(null);
                                    setSelectedNodeId(node.id);
                                  }}
                                >
                                  <span className="kg-node-pill-name">{truncate(nodeName, 56)}</span>
                                  <span className="kg-node-pill-meta">{displayNodeType(node)}</span>
                                  {linked && <span className="kg-node-pill-badge">Linked</span>}
                                </button>
                              );
                            })}
                          </div>
                        )}
                      </section>
                    ))}

                    {visibleEdges.length > 0 && (
                      <section className="kg-subgraph-group">
                        <h5>
                          edges <span>{visibleEdges.length}</span>
                        </h5>
                        <ul className="kg-edge-list">
                          {visibleEdges.map((edge, index) => (
                            <li key={`${edge.source}-${edge.target}-${index}`}>{formatSubgraphEdge(edge, nodeNameById)}</li>
                          ))}
                        </ul>
                      </section>
                    )}
                  </div>
                </div>
              )}
            </section>
          </div>

          <div className="kg-answer-right">
            <section className="kg-panel">
              <h4>Details / Evidence</h4>
              {selectedGraphEdge && (
                <div className="kg-selected-edge-card">
                  <p className="muted small">Selected Graph Edge</p>
                  <p>
                    <strong>{graphNodeById.get(selectedGraphEdge.source)?.label ?? selectedGraphEdge.source}</strong>{" "}
                    --({selectedGraphEdge.label})--&gt;{" "}
                    <strong>{graphNodeById.get(selectedGraphEdge.target)?.label ?? selectedGraphEdge.target}</strong>
                  </p>
                </div>
              )}
              <KgNodeDetailDrawer
                node={selectedNode}
                linkedEntity={selectedNodeLinkedEntity}
                evidence={selectedNodeEvidence}
                retrievalTrace={retrievalTrace}
                onClose={() => setSelectedNodeId(null)}
              />

              <div className="kg-divider" />

              <div className="kg-panel-head">
                <h4>Evidence</h4>
                <p className="muted small">
                  Evidence: {visibleEvidence.length}/{evidence.length} shown
                </p>
              </div>
              {!evidence.length ? (
                <p className="muted">No evidence snippets returned.</p>
              ) : selectedEntityName && filteredEvidence.length === 0 ? (
                <div>
                  <p className="muted">No evidence snippets mention {selectedEntityName}.</p>
                  <Button type="button" variant="outline" className="chip" onClick={() => setShowAllEvidence((prev) => !prev)}>
                    {showAllEvidence ? "Hide all evidence" : "Show all evidence"}
                  </Button>
                  {showAllEvidence && (
                    <div className="kg-evidence-list">
                      {evidence.map((item: KgEvidence) => (
                        <article
                          key={item.chunk_id}
                          ref={(element) => {
                            evidenceItemRefs.current[item.chunk_id] = element;
                          }}
                          className={`kg-evidence-item ${selectedEvidenceId === item.chunk_id ? "active" : ""}`}
                        >
                          <div className="kg-evidence-head">
                            <strong>{item.doc_path || item.chunk_id}</strong>
                            <span>{item.score.toFixed(3)}</span>
                          </div>
                          <pre className="kg-evidence-text">{item.text}</pre>
                        </article>
                      ))}
                    </div>
                  )}
                </div>
              ) : (
                <div className="kg-evidence-list">
                  {visibleEvidence.map((item: KgEvidence) => (
                    <article
                      key={item.chunk_id}
                      ref={(element) => {
                        evidenceItemRefs.current[item.chunk_id] = element;
                      }}
                      className={`kg-evidence-item ${selectedEvidenceId === item.chunk_id ? "active" : ""}`}
                    >
                      <div className="kg-evidence-head">
                        <strong>{item.doc_path || item.chunk_id}</strong>
                        <span>{item.score.toFixed(3)}</span>
                      </div>
                      <pre className="kg-evidence-text">{item.text}</pre>
                    </article>
                  ))}
                </div>
              )}
            </section>
          </div>
        </div>
      </div>
    </div>
  );
}

function KgGraphView({
  question,
  graph,
  selectedEntityName,
  selectedEvidenceId,
  selectedEdgeId,
  onSelectEntity,
  onSelectEvidence,
  onSelectEdge,
}: {
  question: string;
  graph: { width: number; height: number; nodes: GraphViewNode[]; edges: GraphViewEdge[] };
  selectedEntityName: string | null;
  selectedEvidenceId: string | null;
  selectedEdgeId: string | null;
  onSelectEntity: (entityName: string) => void;
  onSelectEvidence: (evidenceId: string) => void;
  onSelectEdge: (edgeId: string) => void;
}) {
  const nodeById = useMemo(() => new Map(graph.nodes.map((node) => [node.id, node])), [graph.nodes]);

  if (!graph.nodes.length) {
    return <p className="muted">No graph data returned.</p>;
  }

  const questionPreview = truncate(question, 90);

  return (
    <div className="kg-graph-view-wrap">
      <p className="muted small">Question: {questionPreview}</p>
      <svg viewBox={`0 0 ${graph.width} ${graph.height}`} className="kg-query-graph-svg" role="img" aria-label="kg-query-graph">
        <defs>
          <marker id="kg-graph-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#8eb5ea" />
          </marker>
        </defs>

        {graph.edges.map((edge) => {
          const source = nodeById.get(edge.source);
          const target = nodeById.get(edge.target);
          if (!source || !target) return null;
          const midX = (source.x + target.x) / 2;
          const midY = (source.y + target.y) / 2;
          const stroke = edge.label === "supported_by" ? "#47d9b2" : edge.label === "evidence" ? "#d6b3ff" : "#68b6ff";
          const active = selectedEdgeId === edge.id;
          return (
            <g key={edge.id} className="kg-query-edge clickable" onClick={() => onSelectEdge(edge.id)}>
              <line
                x1={source.x + 76}
                y1={source.y}
                x2={target.x - 76}
                y2={target.y}
                stroke={stroke}
                strokeWidth={active ? 2.8 : 1.8}
                opacity={active ? 1 : 0.9}
                markerEnd="url(#kg-graph-arrow)"
              />
              <text x={midX} y={midY - 4} className="kg-query-edge-label">
                {edge.label}
              </text>
            </g>
          );
        })}

        {graph.nodes.map((node) => {
          const isEntity = node.kind === "entity";
          const isEvidence = node.kind === "evidence";
          const isQuestion = node.kind === "question";
          const selectedEntity =
            isEntity && selectedEntityName && node.entityName?.toLowerCase() === selectedEntityName.toLowerCase();
          const selectedEvidence = isEvidence && selectedEvidenceId && node.evidenceId === selectedEvidenceId;
          const active = Boolean(selectedEntity || selectedEvidence);
          const fill = isQuestion ? "#132646" : isEntity ? "#102d2c" : "#1a1433";
          const stroke = isQuestion ? "#68b6ff" : isEntity ? "#47d9b2" : "#b18bf6";
          return (
            <g
              key={node.id}
              className={`kg-query-node ${isEntity || isEvidence ? "clickable" : ""}`}
              onClick={() => {
                if (node.kind === "entity" && node.entityName) onSelectEntity(node.entityName);
                if (node.kind === "evidence" && node.evidenceId) onSelectEvidence(node.evidenceId);
              }}
            >
              <rect
                x={node.x - 72}
                y={node.y - 28}
                rx={10}
                ry={10}
                width={144}
                height={56}
                fill={fill}
                stroke={stroke}
                strokeWidth={active ? 2.8 : 1.5}
              />
              <text x={node.x} y={node.y - 4} textAnchor="middle" className="kg-query-node-label">
                {node.label}
              </text>
              {node.subLabel && (
                <text x={node.x} y={node.y + 14} textAnchor="middle" className="kg-query-node-sub">
                  {truncate(node.subLabel, 22)}
                </text>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function KgNodeDetailDrawer({
  node,
  linkedEntity,
  evidence,
  retrievalTrace,
  onClose,
}: {
  node: KgSubgraphNode | null;
  linkedEntity: KgLinkedEntity | null;
  evidence: KgEvidence[];
  retrievalTrace: KgRetrievalTrace[];
  onClose: () => void;
}) {
  if (!node) {
    return (
      <aside className="kg-node-detail">
        <p className="muted small">Click a subgraph node to see why it appears and supporting evidence.</p>
      </aside>
    );
  }

  return (
    <aside className="kg-node-detail">
      <div className="kg-node-detail-head">
        <h5>{displayNodeName(node)}</h5>
        <Button type="button" variant="outline" className="chip" onClick={onClose}>
          Close
        </Button>
      </div>
      <p className="muted small">type={displayNodeType(node)}</p>
      <p className="muted small">repo_id={String(node.repo_id ?? "-")}</p>

      <h6>Why it appears</h6>
      {linkedEntity ? (
        <p className="kg-node-reason">
          Matched linked entity with score <strong>{linkedEntity.score.toFixed(3)}</strong>.
        </p>
      ) : (
        <p className="kg-node-reason">Retrieved via KG neighborhood expansion.</p>
      )}

      <h6>Evidence snippets</h6>
      {evidence.length ? (
        <div className="kg-drawer-evidence-list">
          {evidence.map((item) => (
            <article key={`${node.id}-${item.chunk_id}`} className="kg-drawer-evidence-item">
              <p className="muted small">{item.doc_path || item.chunk_id}</p>
              <pre className="kg-evidence-text">{item.text}</pre>
            </article>
          ))}
        </div>
      ) : (
        <p className="muted">No direct snippet match; retrieved via KG neighborhood.</p>
      )}

      <details className="kg-node-trace" open>
        <summary>Retrieval Trace</summary>
        {!retrievalTrace.length ? (
          <p className="muted small">No retrieval trace returned.</p>
        ) : (
          <ol className="kg-trace-list">
            {retrievalTrace.map((item, index) => (
              <li key={`${item.step}-${index}`}>
                <strong>{item.step}</strong>
                <p>{item.detail}</p>
              </li>
            ))}
          </ol>
        )}
      </details>
    </aside>
  );
}

function JobsPage() {
  const [repoId, setRepoId] = useLocalStorageState("cg.repo_id", "");
  const jobsQ = useQuery({
    queryKey: ["jobs-page", repoId],
    queryFn: () => listJobs(repoId),
    enabled: Boolean(repoId),
    refetchInterval: 4000,
  });

  return (
    <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="stack">
      <section className="card">
        <h2>Jobs</h2>
        <label>
          Repo ID
          <Input value={repoId} onChange={(e) => setRepoId(e.target.value)} placeholder="repo UUID" />
        </label>
        {jobsQ.error && <p className="error">{(jobsQ.error as Error).message}</p>}
        {!repoId && <p className="muted">Run ingest on Dashboard first, or paste a repo id to inspect jobs.</p>}
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Job</th>
                <th>Status</th>
                <th>Progress</th>
                <th>Step</th>
                <th>Updated</th>
              </tr>
            </thead>
            <tbody>
              {(jobsQ.data ?? []).map((job) => (
                <tr key={job.job_id}>
                  <td>{job.job_id}</td>
                  <td>{job.status}</td>
                  <td>{job.progress}%</td>
                  <td>{job.current_step}</td>
                  <td>{formatTime(job.updated_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </motion.div>
  );
}

export default function App() {
  return (
    <Shell>
      <Routes>
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/jobs" element={<JobsPage />} />
        <Route path="*" element={<Navigate to="/dashboard" replace />} />
      </Routes>
    </Shell>
  );
}
