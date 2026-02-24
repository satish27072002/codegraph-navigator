import { CheckCircle2, CircleDashed, Loader2, XCircle } from "lucide-react";
import { cn } from "@/lib/utils";

export type JobStatus = "queued" | "running" | "completed" | "failed";

const PIPELINE_STEPS = ["INGEST", "PARSE", "LOAD GRAPH", "EMBED", "KG LOAD"] as const;

function stepIndex(step: string): number {
    const normalized = step.toUpperCase().replace(/_/g, " ");
    const idx = PIPELINE_STEPS.findIndex((s) => s === normalized);
    return idx >= 0 ? idx : -1;
}

interface ProgressBarProps {
    status: JobStatus;
    progress: number;
    currentStep: string;
    error?: string | null;
}

/* ── Palette: only 3 colors ──
   Blue  #3772FF — running/active
   Green #307351 — completed steps
   Red   #DF2935 — failed
*/

export function ProgressBar({ status, progress, currentStep, error }: ProgressBarProps) {
    const isFailed    = status === "failed" || !!error;
    const isCompleted = status === "completed";
    const isRunning   = status === "running" || status === "queued";
    const activeStepIdx = stepIndex(currentStep);

    return (
        <div
            className="w-full rounded-xl p-4 mb-4"
            style={{
                background: "#111111",
                border: "1px solid #1f1f1f",
            }}
        >
            {/* Header */}
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                    {isCompleted && <CheckCircle2 className="w-4 h-4" style={{ color: "#307351" }} />}
                    {isFailed    && <XCircle      className="w-4 h-4" style={{ color: "#DF2935" }} />}
                    {isRunning && !isFailed && (
                        <Loader2 className="w-4 h-4 animate-spin" style={{ color: "#3772FF" }} />
                    )}
                    {status === "queued" && !isRunning && (
                        <CircleDashed className="w-4 h-4" style={{ color: "#444" }} />
                    )}
                    <span style={{
                        fontSize: 13,
                        fontWeight: 600,
                        color: isFailed ? "#DF2935" : isCompleted ? "#3d9167" : "#f0f0f0",
                        textTransform: "capitalize",
                        fontFamily: "Geist, sans-serif",
                        letterSpacing: "-0.01em",
                    }}>
                        {status}
                    </span>
                    {currentStep && !isFailed && (
                        <span style={{ fontSize: 12, color: "#555", fontFamily: "Geist Mono, monospace" }}>
                            — {currentStep.replace(/_/g, " ")}
                        </span>
                    )}
                </div>
                <span style={{
                    fontSize: 13,
                    fontWeight: 700,
                    fontFamily: "Geist Mono, monospace",
                    color: isFailed ? "#DF2935" : isCompleted ? "#3d9167" : "#3772FF",
                }}>
                    {Math.round(progress)}%
                </span>
            </div>

            {/* Pipeline steps — 5 segments, 2 colors only */}
            <div className="flex items-center gap-1.5 mb-4">
                {PIPELINE_STEPS.map((step, idx) => {
                    const isPast    = activeStepIdx > idx || isCompleted;
                    const isCurrent = activeStepIdx === idx && isRunning && !isFailed;
                    return (
                        <div key={step} className="flex-1 flex flex-col items-center gap-1.5">
                            <div
                                className={cn(
                                    "w-full h-0.5 rounded-full transition-all duration-700",
                                    isFailed && isCurrent ? "bg-[#DF2935]" :
                                    isPast               ? "bg-[#307351]" :
                                    isCurrent            ? "bg-[#3772FF]" :
                                                           "bg-[#1f1f1f]",
                                )}
                            />
                            <span style={{
                                fontSize: 9,
                                fontWeight: 600,
                                letterSpacing: "0.08em",
                                textTransform: "uppercase" as const,
                                fontFamily: "Geist Mono, monospace",
                                color: isFailed && isCurrent ? "#DF2935" :
                                       isPast               ? "#307351" :
                                       isCurrent            ? "#3772FF" :
                                                              "#333",
                                transition: "color 0.3s ease",
                            }}>
                                {step}
                            </span>
                        </div>
                    );
                })}
            </div>

            {/* Progress bar — single color, no rainbow */}
            <div
                className="w-full rounded-full overflow-hidden"
                style={{ height: 3, background: "#1f1f1f" }}
            >
                <div
                    className="h-full rounded-full transition-all duration-700 ease-out"
                    style={{
                        width: `${Math.max(0, Math.min(100, progress))}%`,
                        background: isFailed    ? "#DF2935" :
                                    isCompleted ? "#307351" :
                                                  "#3772FF",
                    }}
                />
            </div>

            {/* Error */}
            {error && (
                <div
                    className="mt-3 text-xs p-3 rounded-lg"
                    style={{
                        background: "rgba(223,41,53,0.07)",
                        border: "1px solid rgba(223,41,53,0.18)",
                        color: "#DF2935",
                        fontFamily: "Geist Mono, monospace",
                        fontSize: 11,
                    }}
                >
                    {error}
                </div>
            )}
        </div>
    );
}
