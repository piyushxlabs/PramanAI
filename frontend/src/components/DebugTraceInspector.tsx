"use client";

import React, { useState } from "react";
import { Bug, Terminal, Database, Activity, ChevronRight, ChevronDown } from "lucide-react";
import { StateUpdateData, ToolExecutionLog } from "@/types";

interface DebugTraceInspectorProps {
  toolLogs: ToolExecutionLog[];
  stateUpdates: StateUpdateData[];
}

export const DebugTraceInspector: React.FC<DebugTraceInspectorProps> = ({
  toolLogs,
  stateUpdates,
}) => {
  const [activeTab, setActiveTab] = useState<"tools" | "state">("tools");
  const [expandedLogId, setExpandedLogId] = useState<string | null>(null);

  return (
    <div className="glass-card rounded-2xl border border-purple-500/40 p-4 shadow-xl text-xs">
      {/* Header */}
      <div className="flex items-center justify-between mb-3 pb-2 border-b border-purple-500/30">
        <div className="flex items-center gap-2 font-bold text-purple-300 uppercase tracking-wider">
          <Bug className="w-4 h-4 text-purple-400" />
          <span>Audit / Debug Inspector</span>
        </div>
        <span className="text-[10px] text-purple-300/70 font-mono">DPDP-Act-2023 Compliant</span>
      </div>

      {/* Tabs */}
      <div className="flex items-center gap-2 mb-3 bg-slate-900/60 p-1 rounded-xl border border-slate-800">
        <button
          onClick={() => setActiveTab("tools")}
          className={`flex-1 flex items-center justify-center gap-1.5 py-1.5 rounded-lg font-semibold transition-all ${
            activeTab === "tools"
              ? "bg-purple-600 text-white shadow-sm"
              : "text-slate-400 hover:text-slate-200"
          }`}
        >
          <Terminal className="w-3.5 h-3.5" />
          <span>Tool Calls ({toolLogs.length})</span>
        </button>
        <button
          onClick={() => setActiveTab("state")}
          className={`flex-1 flex items-center justify-center gap-1.5 py-1.5 rounded-lg font-semibold transition-all ${
            activeTab === "state"
              ? "bg-purple-600 text-white shadow-sm"
              : "text-slate-400 hover:text-slate-200"
          }`}
        >
          <Database className="w-3.5 h-3.5" />
          <span>State Updates ({stateUpdates.length})</span>
        </button>
      </div>

      {/* Tool Calls Tab */}
      {activeTab === "tools" && (
        <div className="space-y-2 max-h-[300px] overflow-y-auto pr-1">
          {toolLogs.length === 0 ? (
            <p className="text-slate-500 text-center py-6">No tool calls logged yet.</p>
          ) : (
            toolLogs.map((log, idx) => {
              const isExpanded = expandedLogId === `${log.toolCallId}_${idx}`;
              return (
                <div
                  key={`${log.toolCallId}_${idx}`}
                  className="p-2.5 rounded-xl bg-slate-950/80 border border-slate-800 text-[11px] font-mono"
                >
                  <button
                    onClick={() => setExpandedLogId(isExpanded ? null : `${log.toolCallId}_${idx}`)}
                    className="w-full flex items-center justify-between text-left text-purple-300 font-bold"
                  >
                    <div className="flex items-center gap-1.5">
                      <Activity className="w-3 h-3 text-purple-400" />
                      <span>{log.toolName}</span>
                    </div>
                    <div className="flex items-center gap-1 text-slate-500 text-[10px]">
                      <span>{log.timestamp}</span>
                      {isExpanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                    </div>
                  </button>

                  {isExpanded && (
                    <div className="mt-2 pt-2 border-t border-slate-800 space-y-2 text-[10px]">
                      {log.input && (
                        <div>
                          <span className="text-slate-400 font-bold block mb-0.5">Input:</span>
                          <pre className="p-2 bg-slate-900 rounded border border-slate-800 overflow-x-auto text-slate-300">
                            {JSON.stringify(log.input, null, 2)}
                          </pre>
                        </div>
                      )}
                      {log.output && (
                        <div>
                          <span className="text-slate-400 font-bold block mb-0.5">Output:</span>
                          <pre className="p-2 bg-slate-900 rounded border border-slate-800 overflow-x-auto text-slate-300">
                            {JSON.stringify(log.output, null, 2)}
                          </pre>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              );
            })
          )}
        </div>
      )}

      {/* State Updates Tab */}
      {activeTab === "state" && (
        <div className="space-y-2 max-h-[300px] overflow-y-auto pr-1">
          {stateUpdates.length === 0 ? (
            <p className="text-slate-500 text-center py-6">No state mutations recorded.</p>
          ) : (
            stateUpdates.map((upd, idx) => (
              <div
                key={idx}
                className="p-2.5 rounded-xl bg-slate-950/80 border border-slate-800 text-[11px] font-mono space-y-1"
              >
                <div className="flex items-center justify-between text-purple-300">
                  <span className="font-bold">{upd.field}</span>
                  <span className="text-[10px] text-slate-500">{upd.timestamp}</span>
                </div>
                <div className="flex items-center justify-between text-[10px] text-slate-400">
                  <span>Reducer: {upd.reducer}</span>
                  <span className="text-slate-300 font-semibold truncate max-w-[140px]">
                    {typeof upd.value === "object" ? JSON.stringify(upd.value) : String(upd.value)}
                  </span>
                </div>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
};
