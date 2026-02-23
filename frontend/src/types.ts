export type JobStatus = "queued" | "running" | "completed" | "failed";

export type GraphViewNodeKind = "question" | "entity" | "evidence";

export type GraphViewNode = {
    id: string;
    kind: GraphViewNodeKind;
    label: string;
    subLabel?: string;
    x: number;
    y: number;
    entityName?: string;
    evidenceId?: string;
};

export type GraphViewEdge = {
    id: string;
    source: string;
    target: string;
    label: "linked" | "supported_by" | "evidence";
};

export type QuestionGraphNodeType = "question" | "entity" | "evidence" | "code";

export interface QuestionGraphNode {
    id: string;
    type: QuestionGraphNodeType;
    label: string;
    subtitle?: string;
    ref_id?: string;
    source?: "kg" | "code" | "merged";
    meta?: Record<string, unknown>;
}

export interface QuestionGraphEdge {
    id: string;
    source: string;
    target: string;
    label: string;
    meta?: Record<string, unknown>;
}

export interface QuestionGraph {
    nodes: QuestionGraphNode[];
    edges: QuestionGraphEdge[];
}

export interface QuestionGraphLayoutNode extends QuestionGraphNode {
    x: number;
    y: number;
}
