import {
  useCallback,
  useMemo,
  useRef,
  useState,
  type Dispatch,
  type ReactNode,
  type SetStateAction,
} from "react";
import { Link, Navigate, Route, Routes, useLocation } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  QuestionGraph,
  QuestionGraphEdge,
  QuestionGraphNode,
  QuestionGraphNodeType,
} from "./types";
import "reactflow/dist/style.css";
import {
  getJob,
  health,
  ingestZip,
  listJobs,
  queryRepo,
  type UnifiedQueryResult,
  repoStatus,
  type Job,
} from "./api";
import { Input } from "@/components/ui/input";
import { ProgressBar } from "./components/ui/ProgressBar";
import { InteractiveQuestionGraph } from "./components/GraphView";

// ──────────────────────────────────────────────────────────
// localStorage hook
// ──────────────────────────────────────────────────────────
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

// ──────────────────────────────────────────────────────────
// Utilities
// ──────────────────────────────────────────────────────────
function formatTime(iso?: string): string {
  if (!iso) return "-";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
}

function truncate(text: string, max = 72): string {
  if (text.length <= max) return text;
  return `${text.slice(0, max - 1)}…`;
}

// ──────────────────────────────────────────────────────────
// Logo mark SVG
// ──────────────────────────────────────────────────────────
function LogoMark() {
  return (
    <svg className="topbar-logomark" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
      {/* Hexagon outline — white only, like Vercel */}
      <path
        d="M20 3 L34 11 L34 29 L20 37 L6 29 L6 11 Z"
        stroke="#ffffff"
        strokeWidth="1.5"
        fill="rgba(255,255,255,0.04)"
      />
      {/* Center node */}
      <circle cx="20" cy="20" r="3.5" fill="#ffffff" />
      {/* Outer nodes */}
      <circle cx="12" cy="14" r="2.5" fill="rgba(255,255,255,0.6)" />
      <circle cx="28" cy="14" r="2.5" fill="rgba(255,255,255,0.6)" />
      <circle cx="12" cy="26" r="2.5" fill="rgba(255,255,255,0.6)" />
      <circle cx="28" cy="26" r="2.5" fill="rgba(255,255,255,0.6)" />
      {/* Edges */}
      <line x1="20" y1="20" x2="12" y2="14" stroke="rgba(255,255,255,0.35)" strokeWidth="1.2" />
      <line x1="20" y1="20" x2="28" y2="14" stroke="rgba(255,255,255,0.35)" strokeWidth="1.2" />
      <line x1="20" y1="20" x2="12" y2="26" stroke="rgba(255,255,255,0.35)" strokeWidth="1.2" />
      <line x1="20" y1="20" x2="28" y2="26" stroke="rgba(255,255,255,0.35)" strokeWidth="1.2" />
      <line x1="12" y1="14" x2="12" y2="26" stroke="rgba(255,255,255,0.18)" strokeWidth="1" />
      <line x1="28" y1="14" x2="28" y2="26" stroke="rgba(255,255,255,0.18)" strokeWidth="1" />
    </svg>
  );
}

// ──────────────────────────────────────────────────────────
// Shell
// ──────────────────────────────────────────────────────────
function Shell({ children }: { children: ReactNode }) {
  const location = useLocation();
  return (
    <div className="app-bg">
      <div className="app-grid" />
      <header className="topbar">
        <div className="topbar-brand">
          <LogoMark />
          <div className="topbar-title">
            <p className="eyebrow">CodeGraph Navigator</p>
            <h1>Repository Intelligence</h1>
          </div>
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

// ──────────────────────────────────────────────────────────
// Build QuestionGraph from unified query result
// ──────────────────────────────────────────────────────────
const QG_MAX_NODES = 40;
const QG_MAX_EDGES = 50;

function _qgNodeId(type: QuestionGraphNodeType, key: string): string {
  return `${type}:${key}`;
}

function buildQuestionGraph(question: string, result: UnifiedQueryResult): QuestionGraph {
  const nodeMap = new Map<string, QuestionGraphNode>();
  const edges: QuestionGraphEdge[] = [];
  const edgeIds = new Set<string>();

  const addNode = (node: QuestionGraphNode) => {
    if (nodeMap.has(node.id)) return;
    if (nodeMap.size >= QG_MAX_NODES) return;
    nodeMap.set(node.id, node);
  };

  const addEdge = (edge: QuestionGraphEdge) => {
    if (edges.length >= QG_MAX_EDGES) return;
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
  });

  const serverIdToQgId = new Map<string, string>();
  for (const node of result.graph.nodes) {
    const nodeType = node.type || "code";
    const qgId = _qgNodeId(nodeType, node.id);
    serverIdToQgId.set(node.id, qgId);
    addNode({
      id: qgId,
      type: nodeType,
      label: node.label || node.id,
      subtitle: node.path || undefined,
      ref_id: node.id,
      meta: { path: node.path },
    });
  }

  const hasIncomingFromGraph = new Set<string>();
  for (const edge of result.graph.edges) {
    hasIncomingFromGraph.add(edge.target);
  }

  for (const node of result.graph.nodes) {
    const qgId = serverIdToQgId.get(node.id);
    if (!qgId) continue;
    if (!hasIncomingFromGraph.has(node.id)) {
      addEdge({ id: `qg:root:${node.id}`, source: questionId, target: qgId, label: "related" });
    }
  }

  for (let i = 0; i < result.graph.edges.length; i++) {
    const edge = result.graph.edges[i];
    const srcQgId = serverIdToQgId.get(edge.source);
    const tgtQgId = serverIdToQgId.get(edge.target);
    if (!srcQgId || !tgtQgId) continue;
    addEdge({ id: `qg:edge:${edge.id || i}`, source: srcQgId, target: tgtQgId, label: edge.label || "related" });
  }

  // BFS depth
  const depthMap = new Map<string, number>();
  depthMap.set(questionId, 0);
  const bfsQueue: string[] = [questionId];
  while (bfsQueue.length > 0) {
    const current = bfsQueue.shift()!;
    const currentDepth = depthMap.get(current) ?? 0;
    for (const edge of edges) {
      const neighbor =
        edge.source === current ? edge.target :
        edge.target === current ? edge.source : null;
      if (neighbor && !depthMap.has(neighbor)) {
        depthMap.set(neighbor, currentDepth + 1);
        bfsQueue.push(neighbor);
      }
    }
  }
  const nodesWithDepth = Array.from(nodeMap.values()).map((n) => ({
    ...n,
    meta: { ...(n.meta ?? {}), depth: depthMap.get(n.id) ?? 3 },
  }));
  return { nodes: nodesWithDepth, edges };
}

// ──────────────────────────────────────────────────────────
// NODE TYPE COLOR (matches GraphView palette)
// ──────────────────────────────────────────────────────────
const NODE_TYPE_COLOR: Record<string, string> = {
  question: "#3772FF",
  file:     "#307351",
  class:    "#307351",
  function: "#3d9167",
  code:     "#307351",
  concept:  "#FFCB47",
  evidence: "#DF2935",
  entity:   "#FFCB47",
};

// ──────────────────────────────────────────────────────────
// Chevron icon (inline)
// ──────────────────────────────────────────────────────────
function ChevronDown({ open }: { open: boolean }) {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 16 16"
      fill="none"
      className={`answer-card-chevron${open ? " open" : ""}`}
      style={{ transition: "transform 0.25s ease", transform: open ? "rotate(180deg)" : "rotate(0deg)" }}
    >
      <path d="M4 6L8 10L12 6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

// ──────────────────────────────────────────────────────────
// AnswerPanel
// ──────────────────────────────────────────────────────────
function AnswerPanel({
  question,
  result,
  error,
}: {
  question: string;
  result: UnifiedQueryResult | null;
  error: string | null;
}) {
  const [open, setOpen] = useState(true);

  if (!result && !error) return null;

  if (error && !result) {
    return (
      <div className="answer-card">
        <div className="answer-card-body">
          <p className="error">{error}</p>
        </div>
      </div>
    );
  }

  if (!result) return null;

  const graphStats = {
    nodes: result.graph.nodes.length,
    edges: result.graph.edges.length,
    concepts: result.graph.nodes.filter((n) => n.type === "concept").length,
    code: result.graph.nodes.filter((n) => ["file", "class", "function", "code"].includes(n.type)).length,
  };

  return (
    <motion.div
      className="answer-card"
      initial={{ opacity: 0, y: -12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: "easeOut" }}
    >
      <div className="answer-card-header" onClick={() => setOpen((v) => !v)}>
        <div className="answer-card-title">
          <span className="answer-card-pulse" />
          Answer
          <span style={{ fontSize: 12, color: "#8474a8", fontWeight: 400 }}>
            — {truncate(question, 60)}
          </span>
        </div>
        <ChevronDown open={open} />
      </div>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            key="answer-body"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25, ease: "easeInOut" }}
            style={{ overflow: "hidden" }}
          >
            <div className="answer-card-body">
              <div className="answer-text">
                {result.answer.split("\n").map((line, i) => (
                  <p key={i}>{line}</p>
                ))}
              </div>
              <div className="answer-meta">
                <span className="stat-badge green">
                  <strong>{graphStats.nodes}</strong> nodes
                </span>
                <span className="stat-badge violet">
                  <strong>{graphStats.edges}</strong> edges
                </span>
                {graphStats.concepts > 0 && (
                  <span className="stat-badge cyan">
                    <strong>{graphStats.concepts}</strong> concepts
                  </span>
                )}
                {graphStats.code > 0 && (
                  <span className="stat-badge">
                    <strong>{graphStats.code}</strong> code nodes
                  </span>
                )}
                {result.warning && (
                  <span className="warning" style={{ fontSize: 12 }}>⚠ {result.warning}</span>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

// ──────────────────────────────────────────────────────────
// Graph workspace — full width + slide-out details panel
// ──────────────────────────────────────────────────────────
function GraphWorkspace({
  question,
  result,
}: {
  question: string;
  result: UnifiedQueryResult;
}) {
  const questionGraph = useMemo(() => buildQuestionGraph(question, result), [question, result]);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null);

  const selectedNode = useMemo(
    () => questionGraph.nodes.find((n) => n.id === selectedNodeId) ?? null,
    [questionGraph.nodes, selectedNodeId],
  );
  const selectedEdge = useMemo(
    () => questionGraph.edges.find((e) => e.id === selectedEdgeId) ?? null,
    [questionGraph.edges, selectedEdgeId],
  );
  const nodeById = useMemo(() => new Map(questionGraph.nodes.map((n) => [n.id, n])), [questionGraph.nodes]);

  const graphStats = useMemo(() => ({
    total: questionGraph.nodes.length,
    code: questionGraph.nodes.filter((n) => ["file", "class", "function", "code"].includes(n.type)).length,
    concept: questionGraph.nodes.filter((n) => n.type === "concept").length,
    evidence: questionGraph.nodes.filter((n) => n.type === "evidence").length,
    edges: questionGraph.edges.length,
  }), [questionGraph]);

  const isPanelOpen = !!(selectedNode || selectedEdge);

  return (
    <div className="graph-hero">
      <div className="graph-hero-header">
        <div className="graph-hero-title">
          <span className="graph-hero-title-dot" />
          Knowledge Graph
          <span className="stat-badge green" style={{ marginLeft: 4 }}>
            <strong>{graphStats.total}</strong> nodes
          </span>
          <span className="stat-badge violet">
            <strong>{graphStats.edges}</strong> edges
          </span>
        </div>
        <span style={{ fontSize: 11, color: "#8474a8" }}>
          Click a node to inspect · Scroll to zoom · Drag to pan
        </span>
      </div>

      <div className="graph-hero-body">
        {/* Graph canvas */}
        <div className="graph-hero-canvas">
          <InteractiveQuestionGraph
            graph={questionGraph}
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
        </div>

        {/* Slide-out details panel — clip wrapper animates width, inner div is fixed 300px */}
        <AnimatePresence>
          {isPanelOpen && (
            <motion.div
              key="details-panel"
              initial={{ width: 0 }}
              animate={{ width: 300 }}
              exit={{ width: 0 }}
              transition={{ duration: 0.22, ease: "easeInOut" }}
              style={{ overflow: "hidden", flexShrink: 0 }}
            >
              <div className="slide-panel" style={{ width: 300 }}>
                <div className="slide-panel-inner">
                  {selectedEdge && (
                    <div>
                      <div className="slide-panel-node-type" style={{ color: "#71717a" }}>
                        Edge
                      </div>
                      <h3 style={{ fontSize: 13 }}>
                        {nodeById.get(selectedEdge.source)?.label ?? selectedEdge.source}
                        <span style={{ color: "#52525b", margin: "0 5px" }}>→</span>
                        {nodeById.get(selectedEdge.target)?.label ?? selectedEdge.target}
                      </h3>
                      <span className={`edge-pill ${selectedEdge.label}`}>
                        {selectedEdge.label}
                      </span>
                    </div>
                  )}

                  {selectedNode && (
                    <div>
                      <div
                        className="slide-panel-node-type"
                        style={{ color: NODE_TYPE_COLOR[selectedNode.type] ?? "#71717a" }}
                      >
                        {selectedNode.type}
                      </div>
                      <h3>{selectedNode.label}</h3>
                      {selectedNode.subtitle && (
                        <div className="slide-panel-sub">{selectedNode.subtitle}</div>
                      )}

                      {selectedNode.type === "question" && (
                        <div className="slide-panel-section">
                          <h4>Graph Summary</h4>
                          <div className="slide-panel-stat-row">
                            <div className="slide-panel-stat">
                              <span>Total nodes</span>
                              <span className="slide-panel-stat-val">{graphStats.total}</span>
                            </div>
                            <div className="slide-panel-stat">
                              <span>Code nodes</span>
                              <span className="slide-panel-stat-val" style={{ color: "#3d9167" }}>{graphStats.code}</span>
                            </div>
                            <div className="slide-panel-stat">
                              <span>Concept nodes</span>
                              <span className="slide-panel-stat-val" style={{ color: "#FFCB47" }}>{graphStats.concept}</span>
                            </div>
                            <div className="slide-panel-stat">
                              <span>Evidence</span>
                              <span className="slide-panel-stat-val" style={{ color: "#DF2935" }}>{graphStats.evidence}</span>
                            </div>
                            <div className="slide-panel-stat">
                              <span>Edges</span>
                              <span className="slide-panel-stat-val" style={{ color: "#3772FF" }}>{graphStats.edges}</span>
                            </div>
                          </div>
                        </div>
                      )}

                      {["file", "class", "function", "code"].includes(selectedNode.type) && selectedNode.subtitle && (
                        <div className="slide-panel-section">
                          <h4>File Path</h4>
                          <code style={{ fontSize: 11, color: "#3d9167", wordBreak: "break-all", fontFamily: '"Geist Mono", monospace' }}>
                            {selectedNode.subtitle}
                          </code>
                        </div>
                      )}

                      {(selectedNode.type === "concept" || selectedNode.type === "entity") && (
                        <div className="slide-panel-section">
                          <h4>Semantic Entity</h4>
                          <p style={{ fontSize: 12, color: "#71717a", margin: 0, lineHeight: 1.5 }}>
                            Extracted from code semantics by the knowledge graph pipeline.
                          </p>
                        </div>
                      )}

                      {selectedNode.type !== "question" && (() => {
                        const connEdges = questionGraph.edges.filter(
                          (e) => e.source === selectedNodeId || e.target === selectedNodeId
                        );
                        if (connEdges.length === 0) return null;
                        return (
                          <div className="slide-panel-section">
                            <h4>Connections ({connEdges.length})</h4>
                            {connEdges.slice(0, 8).map((e) => {
                              const other = e.source === selectedNodeId ? e.target : e.source;
                              const otherNode = nodeById.get(other);
                              return (
                                <div key={e.id} className="slide-panel-edge-item">
                                  <span className={`edge-pill ${e.label}`} style={{ marginRight: 6, display: "inline-block" }}>{e.label}</span>
                                  {otherNode?.label ?? other}
                                </div>
                              );
                            })}
                            {connEdges.length > 8 && (
                              <p style={{ fontSize: 11, color: "#52525b", marginTop: 4, marginBottom: 0 }}>
                                +{connEdges.length - 8} more
                              </p>
                            )}
                          </div>
                        );
                      })()}
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

// ──────────────────────────────────────────────────────────
// Stage: EMPTY — hero + drag-and-drop upload
// ──────────────────────────────────────────────────────────
function StageEmpty({
  existingRepoId,
  onResumeSession,
  onFileSelected,
  onUpload,
  isPending,
  uploadError,
}: {
  existingRepoId: string;
  onResumeSession: () => void;
  onFileSelected: (file: File) => void;
  onUpload: () => void;
  isPending: boolean;
  uploadError: string | null;
}) {
  const zipInputRef = useRef<HTMLInputElement | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file && (file.name.endsWith(".zip") || file.type === "application/zip")) {
        setSelectedFile(file);
        onFileSelected(file);
      }
    },
    [onFileSelected],
  );

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] ?? null;
    if (file) {
      setSelectedFile(file);
      onFileSelected(file);
    }
  };

  return (
    <motion.div
      className="stage-empty"
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -24 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
    >
      {/* Badge */}
      <div className="stage-hero-badge">
        <span className="stage-hero-badge-dot" />
        AI-powered code intelligence
      </div>

      {/* Headline */}
      <h2 className="stage-hero-headline">
        Understand any codebase<br /><em>in minutes</em>
      </h2>
      <p className="stage-hero-sub">
        Drop a ZIP archive of your repository and start asking questions about architecture, dependencies, and code structure.
      </p>

      {/* Dropzone */}
      <div
        className={`dropzone${isDragging ? " drag-over" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        onClick={() => zipInputRef.current?.click()}
      >
        <input
          ref={zipInputRef}
          type="file"
          accept=".zip,application/zip"
          onChange={handleFileChange}
          style={{ display: "none" }}
        />
        <div className="dropzone-icon">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
            <path d="M12 15V3M12 3L8 7M12 3L16 7" stroke="#3772FF" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
            <path d="M3 15V19C3 20.1 3.9 21 5 21H19C20.1 21 21 20.1 21 19V15" stroke="#3772FF" strokeWidth="1.8" strokeLinecap="round" />
          </svg>
        </div>
        <p className="dropzone-title">
          {selectedFile ? selectedFile.name : "Drop your ZIP here"}
        </p>
        <p className="dropzone-hint">
          {selectedFile ? `${(selectedFile.size / 1024 / 1024).toFixed(1)} MB — ready to upload` : "or click to browse files · .zip format"}
        </p>

        {selectedFile && (
          <div className="dropzone-selected" onClick={(e) => e.stopPropagation()}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
              <path d="M20 6L9 17L4 12" stroke="#39ff8f" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            {selectedFile.name} selected
          </div>
        )}
      </div>

      {/* Upload button */}
      {selectedFile && (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          style={{ marginTop: 20 }}
        >
          <button
            className="ask-btn"
            disabled={isPending}
            onClick={(e) => { e.stopPropagation(); onUpload(); }}
          >
            {isPending ? (
              <>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" style={{ animation: "spin-ring 1s linear infinite", transformOrigin: "center" }}>
                  <circle cx="12" cy="12" r="10" stroke="rgba(255,255,255,0.3)" strokeWidth="2" />
                  <path d="M12 2C6.48 2 2 6.48 2 12" stroke="white" strokeWidth="2" strokeLinecap="round" />
                </svg>
                Uploading…
              </>
            ) : (
              <>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                  <path d="M12 15V3M12 3L8 7M12 3L16 7" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                  <path d="M3 15V19C3 20.1 3.9 21 5 21H19C20.1 21 21 20.1 21 19V15" stroke="white" strokeWidth="2" strokeLinecap="round" />
                </svg>
                Start Analysis
              </>
            )}
          </button>
        </motion.div>
      )}

      {uploadError && <p className="error" style={{ marginTop: 12, fontSize: 13 }}>{uploadError}</p>}

      {/* Resume session */}
      {existingRepoId && !selectedFile && (
        <button className="resume-pill" onClick={onResumeSession}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
            <path d="M4 12L20 12M20 12L14 6M20 12L14 18" stroke="#a1a1aa" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          Resume previous session
        </button>
      )}

      {/* Steps explainer */}
      <div className="steps-row">
        <div className="step-item">
          <div className="step-num">1</div>
          <div>
            <div className="step-label">Upload ZIP</div>
            <div className="step-desc">Drop your repo archive</div>
          </div>
        </div>
        <div className="step-connector" />
        <div className="step-item">
          <div className="step-num">2</div>
          <div>
            <div className="step-label">Graph Analysis</div>
            <div className="step-desc">AI parses structure &amp; semantics</div>
          </div>
        </div>
        <div className="step-connector rev" />
        <div className="step-item">
          <div className="step-num">3</div>
          <div>
            <div className="step-label">Explore</div>
            <div className="step-desc">Ask questions, browse graph</div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}

// ──────────────────────────────────────────────────────────
// Stage: INGESTING — progress view
// ──────────────────────────────────────────────────────────
function StageIngesting({
  repoId,
  jobData,
  onUploadAnother,
}: {
  repoId: string;
  jobData: Job | undefined;
  onUploadAnother: () => void;
}) {
  return (
    <motion.div
      className="stage-ingesting"
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -24 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
    >
      <div className="ingesting-ring" />
      <h2 className="ingesting-title">Analysing repository…</h2>
      <p className="ingesting-sub">
        Building knowledge graph · extracting semantics · indexing code structure
      </p>

      {repoId && (
        <div className="ingesting-repo-id">{repoId}</div>
      )}

      {jobData ? (
        <div style={{ width: "100%", maxWidth: 540 }}>
          <ProgressBar
            status={jobData.status as import("./components/ui/ProgressBar").JobStatus}
            progress={jobData.progress ?? 0}
            currentStep={jobData.current_step ?? ""}
            error={jobData.error}
          />
          {jobData.updated_at && (
            <p style={{ fontSize: 11, color: "#8474a8", textAlign: "center", marginTop: 4 }}>
              Last update: {formatTime(jobData.updated_at)}
            </p>
          )}
        </div>
      ) : (
        <div style={{ width: "100%", maxWidth: 540 }}>
          <ProgressBar status="queued" progress={0} currentStep="" />
        </div>
      )}

      <button className="upload-another-btn" onClick={onUploadAnother}>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
          <path d="M19 12H5M5 12L12 5M5 12L12 19" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        Upload a different file
      </button>
    </motion.div>
  );
}

// ──────────────────────────────────────────────────────────
// Stage: READY — query + graph
// ──────────────────────────────────────────────────────────
function StageReady({
  repoId,
  repoStats,
  question,
  setQuestion,
  onAsk,
  isAsking,
  askError,
  queryResult,
  onNewRepo,
}: {
  repoId: string;
  repoStats: { indexed_node_count: number; indexed_edge_count: number; embedded_nodes: number; embeddings_exist: boolean } | undefined;
  question: string;
  setQuestion: (q: string) => void;
  onAsk: () => void;
  isAsking: boolean;
  askError: string | null;
  queryResult: UnifiedQueryResult | null;
  onNewRepo: () => void;
}) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      if (!isAsking) onAsk();
    }
  };

  return (
    <motion.div
      className="stage-ready"
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -24 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
    >
      {/* Status strip */}
      <div className="status-strip">
        <span className="status-strip-label">Active Repo</span>
        <span className="status-strip-id">{repoId}</span>

        {repoStats && (
          <>
            <span className="stat-badge green">
              <strong>{repoStats.indexed_node_count}</strong> nodes
            </span>
            <span className="stat-badge violet">
              <strong>{repoStats.indexed_edge_count}</strong> edges
            </span>
            <span className="stat-badge cyan">
              <strong>{repoStats.embedded_nodes}</strong> embedded
            </span>
          </>
        )}

        <div className="status-strip-actions">
          <button className="new-repo-btn" onClick={onNewRepo}>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
              <path d="M12 5V19M5 12H19" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
            </svg>
            New repo
          </button>
        </div>
      </div>

      {/* Query bar */}
      <div className="query-bar">
        <span className="query-bar-label">Ask anything about this codebase</span>
        <textarea
          ref={textareaRef}
          rows={3}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="What does this repository do? How is authentication handled? Which functions call the database?"
        />
        <div className="query-bar-footer">
          <span className="query-bar-hint">
            Press <kbd>⌘</kbd> <kbd>Enter</kbd> to ask
          </span>
          <button className="ask-btn" onClick={onAsk} disabled={isAsking || !question.trim()}>
            {isAsking ? (
              <>
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" style={{ animation: "spin-ring 1s linear infinite", transformOrigin: "center" }}>
                  <circle cx="12" cy="12" r="10" stroke="rgba(255,255,255,0.3)" strokeWidth="2" />
                  <path d="M12 2C6.48 2 2 6.48 2 12" stroke="white" strokeWidth="2" strokeLinecap="round" />
                </svg>
                Thinking…
              </>
            ) : (
              <>
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none">
                  <path d="M12 8V12L15 15M21 12C21 16.97 16.97 21 12 21C7.03 21 3 16.97 3 12C3 7.03 7.03 3 12 3C16.97 3 21 7.03 21 12Z" stroke="white" strokeWidth="2" strokeLinecap="round" />
                </svg>
                Ask
              </>
            )}
          </button>
        </div>
      </div>

      {askError && <p className="error" style={{ fontSize: 13, margin: 0 }}>{askError}</p>}

      {/* Answer card */}
      <AnimatePresence>
        {(queryResult || askError) && (
          <AnswerPanel question={question} result={queryResult} error={askError} />
        )}
      </AnimatePresence>

      {/* Graph workspace */}
      <AnimatePresence>
        {queryResult && (
          <motion.div
            key="graph-workspace"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.45, ease: "easeOut", delay: 0.1 }}
          >
            <GraphWorkspace question={question} result={queryResult} />
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

// ──────────────────────────────────────────────────────────
// Dashboard — stage machine orchestrator
// ──────────────────────────────────────────────────────────
function Dashboard() {
  const [repoId, setRepoId] = useLocalStorageState("cg.repo_id", "");
  const [jobId, setJobId] = useLocalStorageState("cg.job_id", "");
  const [question, setQuestion] = useLocalStorageState("cg.question", "What does this repository do?");
  const [queryResult, setQueryResult] = useState<UnifiedQueryResult | null>(null);
  const [queryError, setQueryError] = useState<string | null>(null);
  const [pendingFile, setPendingFile] = useState<File | null>(null);

  // Queries
  useQuery({ queryKey: ["health"], queryFn: health, refetchInterval: 10000 });

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

  const repoStatusQ = useQuery({
    queryKey: ["repo-status", repoId],
    queryFn: () => repoStatus(repoId),
    enabled: Boolean(repoId),
    refetchInterval: 8000,
  });

  // Ingest mutation
  const ingestZipM = useMutation({
    mutationFn: () => {
      if (!pendingFile) throw new Error("Choose a .zip file first");
      return ingestZip(pendingFile);
    },
    onSuccess: (result) => {
      setRepoId(result.repo_id);
      setJobId(result.job_id);
      setPendingFile(null);
    },
  });

  // Query mutation
  const queryM = useMutation({
    mutationFn: async () => {
      const normalizedRepo = repoId.trim();
      const normalizedQuestion = question.trim();
      if (!normalizedRepo) throw new Error("No repo loaded");
      if (!normalizedQuestion) throw new Error("Enter a question");
      return queryRepo(normalizedRepo, normalizedQuestion);
    },
    onMutate: () => {
      setQueryError(null);
      setQueryResult(null);
    },
    onSuccess: (result) => setQueryResult(result),
    onError: (err: Error) => setQueryError(err.message),
  });

  // Determine current stage
  const jobStatus = jobQ.data?.status;
  const stage: "empty" | "ingesting" | "ready" = useMemo(() => {
    if (!repoId) return "empty";
    if (jobId && jobStatus && jobStatus !== "completed" && jobStatus !== "failed") return "ingesting";
    return "ready";
  }, [repoId, jobId, jobStatus]);

  const handleNewRepo = () => {
    setRepoId("");
    setJobId("");
    setQueryResult(null);
    setQueryError(null);
    setPendingFile(null);
  };

  return (
    <AnimatePresence mode="wait">
      {stage === "empty" && (
        <StageEmpty
          key="stage-empty"
          existingRepoId={repoId}
          onResumeSession={() => {
            // repoId is set but jobId may be stale — just go to ready
            setJobId("");
          }}
          onFileSelected={(file) => setPendingFile(file)}
          onUpload={() => ingestZipM.mutate()}
          isPending={ingestZipM.isPending}
          uploadError={ingestZipM.error ? (ingestZipM.error as Error).message : null}
        />
      )}

      {stage === "ingesting" && (
        <StageIngesting
          key="stage-ingesting"
          repoId={repoId}
          jobData={jobQ.data}
          onUploadAnother={handleNewRepo}
        />
      )}

      {stage === "ready" && (
        <StageReady
          key="stage-ready"
          repoId={repoId}
          repoStats={repoStatusQ.data}
          question={question}
          setQuestion={setQuestion}
          onAsk={() => queryM.mutate()}
          isAsking={queryM.isPending}
          askError={queryError}
          queryResult={queryResult}
          onNewRepo={handleNewRepo}
        />
      )}
    </AnimatePresence>
  );
}

// ──────────────────────────────────────────────────────────
// Jobs page (cosmetic update only)
// ──────────────────────────────────────────────────────────
function Jobs() {
  const [repoId, setRepoId] = useLocalStorageState("cg.repo_id", "");

  const jobsQ = useQuery({
    queryKey: ["jobs", repoId],
    queryFn: () => listJobs(repoId),
    enabled: Boolean(repoId),
    refetchInterval: 5000,
  });

  return (
    <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="stack">
      <section className="card">
        <h2>Jobs</h2>
        <label>
          Repo ID
          <Input value={repoId} onChange={(e) => setRepoId(e.target.value)} placeholder="repo UUID" />
        </label>
        {jobsQ.isLoading && <p className="muted">Loading jobs…</p>}
        {jobsQ.error && <p className="error">{(jobsQ.error as Error).message}</p>}
        {jobsQ.data && (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Job ID</th>
                  <th>Type</th>
                  <th>Status</th>
                  <th>Progress</th>
                  <th>Step</th>
                  <th>Updated</th>
                </tr>
              </thead>
              <tbody>
                {jobsQ.data.map((job) => (
                  <tr key={job.job_id}>
                    <td className="muted small">{job.job_id.slice(0, 8)}…</td>
                    <td>{job.job_type}</td>
                    <td className={job.status === "completed" ? "ok" : job.status === "failed" ? "bad" : ""}>{job.status}</td>
                    <td>{job.progress}%</td>
                    <td>{job.current_step}</td>
                    <td className="muted small">{formatTime(job.updated_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </motion.div>
  );
}

// ──────────────────────────────────────────────────────────
// App router
// ──────────────────────────────────────────────────────────
export default function App() {
  return (
    <Shell>
      <Routes>
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/jobs" element={<Jobs />} />
      </Routes>
    </Shell>
  );
}
