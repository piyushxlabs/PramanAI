"use client";

import React, { useState } from "react";
import { CheckCircle2, Loader2, Circle, ChevronDown, ChevronUp, Layers } from "lucide-react";
import { GraphStepState } from "@/types";

interface SessionProgressProps {
  steps: GraphStepState[];
  isStreaming: boolean;
}

interface ProgressStepConfig {
  id: string;
  aliases: string[];
  label: string;
}

const PROGRESS_STEPS: ProgressStepConfig[] = [
  { id: "reading_question", aliases: ["query_interpretation", "reading_question"], label: "Reading your question" },
  { id: "scope_check", aliases: ["scope_screen", "scope_check"], label: "Checking this is something I can help with" },
  { id: "searching_records", aliases: ["retrieval_invocation", "searching_records"], label: "Searching official records" },
  { id: "supersession_check", aliases: ["confidence_supersession", "supersession_check"], label: "Checking for conflicts or updates" },
  { id: "drafting_answer", aliases: ["grounded_synthesis", "drafting_answer"], label: "Drafting a cited answer" },
  { id: "citation_verification", aliases: ["citation_integrity", "citation_verification"], label: "Double-checking every citation" },
  { id: "done", aliases: ["response_delivery", "done"], label: "Done" },
];

export const SessionProgress: React.FC<SessionProgressProps> = ({ steps, isStreaming }) => {
  const [isExpanded, setIsExpanded] = useState<boolean>(true);

  const getStepStatus = (stepConfig: ProgressStepConfig): "completed" | "started" | "retrying" | "pending" => {
    const found = steps.find(
      (s) => stepConfig.aliases.includes(s.node) || (s as { step?: string }).step === stepConfig.id
    );
    if (!found) return "pending";
    return found.status;
  };

  const activeStep = steps.find((s) => s.status === "started");
  const activeLabel = activeStep?.label || (isStreaming ? "Processing query..." : steps.length > 0 ? "Completed" : "Idle");

  return (
    <div className="glass-card rounded-2xl border border-slate-700/60 p-4 shadow-lg">
      {/* Stepper Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between text-left text-xs font-bold text-slate-200 uppercase tracking-wider"
      >
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-sky-400" />
          <span>Session Progress</span>
          {isStreaming && (
            <span className="flex h-2 w-2 relative">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-sky-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-sky-500"></span>
            </span>
          )}
        </div>
        <div className="flex items-center gap-1.5 text-slate-400">
          <span className="text-[10px] font-normal normal-case text-sky-400">
            {activeLabel}
          </span>
          {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </div>
      </button>

      {/* Stepper Body */}
      {isExpanded && (
        <div className="mt-4 space-y-3 relative before:absolute before:left-3 before:top-2 before:bottom-2 before:w-[1px] before:bg-slate-700/60">
          {PROGRESS_STEPS.map((item) => {
            const status = getStepStatus(item);

            return (
              <div key={item.id} className="flex items-center gap-3 relative z-10 text-xs">
                {status === "completed" ? (
                  <div className="w-6 h-6 rounded-full bg-emerald-500/20 border border-emerald-500/50 flex items-center justify-center text-emerald-400 shadow-sm shadow-emerald-500/20">
                    <CheckCircle2 className="w-3.5 h-3.5" />
                  </div>
                ) : status === "started" ? (
                  <div className="w-6 h-6 rounded-full bg-sky-500/20 border border-sky-500/50 flex items-center justify-center text-sky-400 shadow-sm shadow-sky-500/20 animate-pulse">
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  </div>
                ) : (
                  <div className="w-6 h-6 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center text-slate-500">
                    <Circle className="w-2.5 h-2.5 fill-current" />
                  </div>
                )}

                <span
                  className={`font-medium transition-colors duration-200 ${
                    status === "started"
                      ? "text-sky-300 font-semibold"
                      : status === "completed"
                      ? "text-slate-300"
                      : "text-slate-500"
                  }`}
                >
                  {item.label}
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
