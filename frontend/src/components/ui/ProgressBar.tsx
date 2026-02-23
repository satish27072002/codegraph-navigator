import { CheckCircle2, CircleDashed, Loader2, XCircle } from "lucide-react";
import { cn } from "@/lib/utils";

export type JobStatus = "queued" | "running" | "completed" | "failed";

const PIPELINE_STEPS = ["INGEST", "PARSE", "LOAD_GRAPH", "EMBED", "KG_LOAD"] as const;

function stepIndex(step: string): number {
    const idx = PIPELINE_STEPS.findIndex((s) => s === step.toUpperCase());
    return idx >= 0 ? idx : -1;
}

interface ProgressBarProps {
    status: JobStatus;
    progress: number;
    currentStep: string;
    error?: string | null;
}

export function ProgressBar({ status, progress, currentStep, error }: ProgressBarProps) {
    const isFailed = status === "failed" || !!error;
    const isCompleted = status === "completed";
    const isRunning = status === "running" || status === "queued";
    const activeStepIdx = stepIndex(currentStep);

    return (
        <div
            className="w-full rounded-2xl p-5 shadow-lg mb-4"
            style={{
                background: "linear-gradient(135deg, rgba(15,23,42,0.95) 0%, rgba(2,6,23,0.98) 100%)",
                border: "1px solid rgba(148,163,184,0.12)",
            }}
        >
            {/* Header row */}
            <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2.5">
                    {isCompleted && <CheckCircle2 className="w-5 h-5 text-emerald-400" />}
                    {isFailed && <XCircle className="w-5 h-5 text-rose-400" />}
                    {isRunning && !isFailed && <Loader2 className="w-5 h-5 text-cyan-400 animate-spin" />}
                    {status === "queued" && !isRunning && <CircleDashed className="w-5 h-5 text-slate-500" />}

                    <span className="font-bold text-sm text-slate-100 capitalize tracking-wide">
                        {status}
                    </span>
                    {currentStep && !isFailed && (
                        <span className="text-xs text-slate-400 font-medium">
                            — {currentStep.replace(/_/g, " ")}
                        </span>
                    )}
                </div>
                <span
                    className="text-sm font-bold tabular-nums"
                    style={{
                        color: isFailed
                            ? "#fb7185"
                            : isCompleted
                                ? "#34d399"
                                : "#22d3ee",
                    }}
                >
                    {Math.round(progress)}%
                </span>
            </div>

            {/* Pipeline step indicators */}
            <div className="flex items-center gap-1 mb-3">
                {PIPELINE_STEPS.map((step, idx) => {
                    const isPast = activeStepIdx > idx || isCompleted;
                    const isCurrent = activeStepIdx === idx && isRunning && !isFailed;
                    return (
                        <div key={step} className="flex-1 flex flex-col items-center gap-1">
                            <div
                                className={cn(
                                    "w-full h-1 rounded-full transition-all duration-500",
                                    isPast && "bg-emerald-400",
                                    isCurrent && "bg-cyan-400",
                                    !isPast && !isCurrent && "bg-slate-700",
                                    isFailed && isCurrent && "bg-rose-400",
                                )}
                                style={isCurrent ? { animation: "pulse 1.5s ease-in-out infinite" } : undefined}
                            />
                            <span
                                className={cn(
                                    "text-[9px] font-semibold tracking-wider uppercase transition-colors",
                                    isPast && "text-emerald-400",
                                    isCurrent && "text-cyan-300",
                                    !isPast && !isCurrent && "text-slate-600",
                                    isFailed && isCurrent && "text-rose-400",
                                )}
                            >
                                {step.replace(/_/g, " ")}
                            </span>
                        </div>
                    );
                })}
            </div>

            {/* Progress bar with shimmer */}
            <div
                className="w-full rounded-full h-2.5 overflow-hidden"
                style={{
                    background: "rgba(30,41,59,0.8)",
                    border: "1px solid rgba(148,163,184,0.08)",
                }}
            >
                <div
                    className="h-2.5 rounded-full transition-all duration-500 ease-out relative overflow-hidden"
                    style={{
                        width: `${Math.max(0, Math.min(100, progress))}%`,
                        background: isFailed
                            ? "linear-gradient(90deg, #e11d48 0%, #fb7185 100%)"
                            : isCompleted
                                ? "linear-gradient(90deg, #059669 0%, #34d399 100%)"
                                : "linear-gradient(90deg, #0891b2 0%, #06b6d4 50%, #8b5cf6 100%)",
                    }}
                >
                    {/* Shimmer animation overlay for running state */}
                    {isRunning && !isFailed && (
                        <div
                            className="absolute inset-0"
                            style={{
                                background: "linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.3) 50%, transparent 100%)",
                                animation: "shimmer 1.8s ease-in-out infinite",
                            }}
                        />
                    )}
                </div>
            </div>

            {/* Error message */}
            {error && (
                <div
                    className="mt-3 text-xs text-rose-300 p-3 rounded-lg"
                    style={{
                        background: "rgba(225,29,72,0.08)",
                        border: "1px solid rgba(225,29,72,0.2)",
                    }}
                >
                    {error}
                </div>
            )}

            {/* Inline CSS keyframes */}
            <style>{`
                @keyframes shimmer {
                    0% { transform: translateX(-100%); }
                    100% { transform: translateX(200%); }
                }
            `}</style>
        </div>
    );
}
