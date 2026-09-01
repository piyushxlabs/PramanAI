"use client";

import React, { useState } from "react";
import { GitCommit, Clock, ArrowRight, FileText, CheckCircle2, AlertTriangle, ChevronRight, Eye } from "lucide-react";
import { Citation } from "@/types";

interface PolicyTimelineProps {
  citations: Citation[];
  supersessionStatus?: string;
  onSelectCitation?: (citation: Citation) => void;
}

export const PolicyTimeline: React.FC<PolicyTimelineProps> = ({
  citations,
  supersessionStatus,
  onSelectCitation,
}) => {
  const [isExpanded, setIsExpanded] = useState<boolean>(true);

  if (!citations || citations.length === 0) return null;

  // Deduplicate by go_number and sort chronologically
  const uniqueCitationsMap = new Map<string, Citation>();
  citations.forEach((c) => {
    if (!uniqueCitationsMap.has(c.go_number)) {
      uniqueCitationsMap.set(c.go_number, c);
    }
  });

  const timelineItems = Array.from(uniqueCitationsMap.values()).sort((a, b) => {
    const dateA = a.date || "";
    const dateB = b.date || "";
    return dateA.localeCompare(dateB);
  });

  // If there's only 1 order and status is UNKNOWN / ACTIVE, we still render a clean single-order active badge or chain
  const getStatusInfo = (index: number, total: number, status?: string) => {
    const isLatest = index === total - 1;

    if (total === 1) {
      if (status === "SUPERSEDED") {
        return {
          label: "SUPERSEDED / अतिक्रमित",
          badgeClass: "bg-red-500/20 border-red-500/40 text-red-300",
          icon: AlertTriangle,
          circleClass: "bg-red-500 text-white",
        };
      }
      return {
        label: "CURRENT ACTIVE / प्रभावी",
        badgeClass: "bg-emerald-500/20 border-emerald-500/40 text-emerald-300",
        icon: CheckCircle2,
        circleClass: "bg-emerald-500 text-slate-950",
      };
    }

    if (isLatest) {
      return {
        label: "CURRENT ACTIVE / वर्तमान प्रभावी",
        badgeClass: "bg-emerald-500/20 border-emerald-500/40 text-emerald-300",
        icon: CheckCircle2,
        circleClass: "bg-emerald-500 text-slate-950 ring-4 ring-emerald-500/20",
      };
    } else {
      return {
        label: "PREVIOUS ORDER / पूर्व संशोधित आदेश",
        badgeClass: "bg-amber-500/20 border-amber-500/40 text-amber-300",
        icon: Clock,
        circleClass: "bg-amber-500 text-slate-950",
      };
    }
  };

  return (
    <div className="my-4 rounded-xl border border-slate-700/80 bg-slate-900/80 shadow-md backdrop-blur-md overflow-hidden animate-in fade-in duration-300">
      {/* Header Bar */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-4 py-2.5 flex items-center justify-between bg-slate-950/60 hover:bg-slate-950/90 text-left transition-colors border-b border-slate-800"
      >
        <div className="flex items-center gap-2">
          <GitCommit className="w-4 h-4 text-sky-400" />
          <span className="text-xs font-bold text-slate-200">
            Policy Evolution & Supersession Timeline (शासनादेश संशोधन श्रृंखला)
          </span>
          <span className="text-[10px] px-2 py-0.5 rounded-full bg-sky-500/15 text-sky-300 border border-sky-500/30">
            {timelineItems.length} {timelineItems.length === 1 ? "Order" : "Orders in Chain"}
          </span>
        </div>
        <ChevronRight
          className={`w-4 h-4 text-slate-400 transition-transform duration-200 ${
            isExpanded ? "rotate-90" : ""
          }`}
        />
      </button>

      {/* Expanded Timeline Chain */}
      {isExpanded && (
        <div className="p-4 space-y-4">
          <div className="relative pl-6 space-y-6 before:absolute before:left-2.5 before:top-2 before:bottom-2 before:w-0.5 before:bg-slate-700">
            {timelineItems.map((item, idx) => {
              const statusInfo = getStatusInfo(idx, timelineItems.length, supersessionStatus);
              const StatusIcon = statusInfo.icon;

              return (
                <div key={item.go_number} className="relative group">
                  {/* Stepper Node Icon */}
                  <div
                    className={`absolute -left-6 top-1 w-5 h-5 rounded-full flex items-center justify-center font-bold text-[10px] shadow-lg transition-transform group-hover:scale-110 ${statusInfo.circleClass}`}
                  >
                    {idx + 1}
                  </div>

                  {/* Card Content */}
                  <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800 hover:border-slate-700 transition-all space-y-2">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-xs font-bold text-sky-300">
                          {item.go_number}
                        </span>
                        <span className="text-[10px] text-slate-400 bg-slate-800/80 px-2 py-0.5 rounded">
                          {item.date || "Date Unspecified"}
                        </span>
                      </div>

                      <div className={`flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold border ${statusInfo.badgeClass}`}>
                        <StatusIcon className="w-3 h-3" />
                        <span>{statusInfo.label}</span>
                      </div>
                    </div>

                    <p className="text-[11px] text-slate-300 leading-relaxed line-clamp-2 italic">
                      "{item.exact_text_excerpt}"
                    </p>

                    <div className="flex items-center justify-between pt-1 border-t border-slate-800/60 text-[10px] text-slate-400">
                      <span>Department: {item.issuing_department} • Page {item.page_number}</span>
                      {onSelectCitation && (
                        <button
                          onClick={() => onSelectCitation(item)}
                          className="inline-flex items-center gap-1 text-sky-400 hover:text-sky-300 font-semibold transition-colors"
                          title="Open in Document Viewer"
                        >
                          <Eye className="w-3 h-3" />
                          <span>दस्तावेज़ देखें (View Document)</span>
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};
