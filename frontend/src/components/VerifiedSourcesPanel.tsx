"use client";

import React, { useState } from "react";
import { BookOpen, Flag, MapPin, Calendar, Building, Eye } from "lucide-react";
import { Citation } from "@/types";

interface VerifiedSourcesPanelProps {
  citations: Citation[];
  onSelectCitation?: (citation: Citation) => void;
  onFlagCitation?: (goNumber: string, pageNumber: number, comment?: string) => void;
}

export const VerifiedSourcesPanel: React.FC<VerifiedSourcesPanelProps> = ({
  citations,
  onSelectCitation,
  onFlagCitation,
}) => {
  const [flaggedIdx, setFlaggedIdx] = useState<number | null>(null);
  const [flagComment, setFlagComment] = useState<string>("");

  const handleFlagSubmit = (citation: Citation, idx: number, e: React.MouseEvent) => {
    e.stopPropagation();
    if (onFlagCitation) {
      onFlagCitation(citation.go_number, citation.page_number, flagComment);
    }
    setFlaggedIdx(null);
    setFlagComment("");
  };

  if (citations.length === 0) {
    return (
      <div className="glass-card rounded-2xl border border-slate-700/60 p-4 shadow-lg text-slate-400 text-center text-xs">
        <div className="flex items-center justify-center gap-2 mb-2 text-slate-400">
          <BookOpen className="w-4 h-4 text-sky-400" />
          <span className="font-bold text-slate-300 uppercase tracking-wider">Verified Sources</span>
        </div>
        <p className="text-slate-500 py-3">No citations active in current turn.</p>
      </div>
    );
  }

  return (
    <div className="glass-card rounded-2xl border border-slate-700/60 p-4 shadow-lg">
      <div className="flex items-center justify-between mb-3.5 pb-2 border-b border-slate-700/50">
        <div className="flex items-center gap-2 text-xs font-bold text-slate-200 uppercase tracking-wider">
          <BookOpen className="w-4 h-4 text-sky-400" />
          <span>Verified Sources ({citations.length})</span>
        </div>
        <span className="text-[10px] text-slate-400">Click to view document</span>
      </div>

      <div className="space-y-3 max-h-[380px] overflow-y-auto pr-1">
        {citations.map((c, idx) => (
          <div
            key={`${c.go_number}_${c.page_number}_${idx}`}
            onClick={() => onSelectCitation && onSelectCitation(c)}
            className="p-3 rounded-xl bg-slate-900/80 border border-slate-700/70 hover:border-sky-400/80 hover:bg-slate-900 hover:shadow-lg hover:shadow-sky-500/10 cursor-pointer transition-all text-xs space-y-2 group"
          >
            {/* GO Title & Page & View Badge */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-1.5">
                <span className="font-bold text-sky-300 tracking-tight group-hover:text-sky-200">
                  {c.go_number}
                </span>
                <Eye className="w-3.5 h-3.5 text-slate-500 group-hover:text-sky-400 transition-colors" />
              </div>
              <span className="px-2 py-0.5 rounded bg-slate-800 text-[10px] font-semibold text-slate-300 border border-slate-700 group-hover:border-sky-500/40">
                Page {c.page_number}
              </span>
            </div>

            {/* Department & Date Metadata */}
            <div className="flex flex-wrap items-center gap-3 text-[11px] text-slate-400">
              <span className="flex items-center gap-1">
                <Building className="w-3 h-3 text-slate-500" />
                {c.issuing_department}
              </span>
              <span className="flex items-center gap-1">
                <Calendar className="w-3 h-3 text-slate-500" />
                {c.date}
              </span>
            </div>

            {/* Verbatim Excerpt */}
            <blockquote className="p-2.5 rounded-lg bg-slate-950/70 border-l-2 border-sky-400 text-slate-300 text-[11px] leading-relaxed italic group-hover:bg-slate-950 transition-colors">
              &ldquo;{c.exact_text_excerpt}&rdquo;
            </blockquote>

            {/* Bounding Box Highlight Coordinates & Click Hint */}
            <div className="flex items-center justify-between pt-0.5">
              {c.bounding_box_coordinates && (
                <div className="flex items-center gap-1.5 text-[10px] text-emerald-400/90 font-mono">
                  <MapPin className="w-3 h-3 text-emerald-400" />
                  <span>Highlight Ready</span>
                </div>
              )}
              <span className="text-[10px] text-sky-400 font-semibold opacity-0 group-hover:opacity-100 transition-opacity">
                View in Document Viewer →
              </span>
            </div>

            {/* Flag as Incorrect Action */}
            <div className="pt-1 flex items-center justify-end border-t border-slate-800/60">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setFlaggedIdx(flaggedIdx === idx ? null : idx);
                }}
                className="flex items-center gap-1 text-[10px] text-slate-400 hover:text-red-400 transition-colors"
              >
                <Flag className="w-3 h-3" />
                <span>Flag citation</span>
              </button>
            </div>

            {/* Flag Citation Input */}
            {flaggedIdx === idx && (
              <div
                onClick={(e) => e.stopPropagation()}
                className="mt-2 pt-2 border-t border-slate-800 flex items-center gap-2"
              >
                <input
                  type="text"
                  placeholder="Reason for flag (e.g. wrong page or section)..."
                  value={flagComment}
                  onChange={(e) => setFlagComment(e.target.value)}
                  className="flex-1 bg-slate-950 border border-slate-700 rounded px-2 py-1 text-[10px] text-slate-200 focus:outline-none focus:border-red-500"
                />
                <button
                  onClick={(e) => handleFlagSubmit(c, idx, e)}
                  className="px-2 py-1 bg-red-600 hover:bg-red-500 text-white rounded text-[10px] font-bold"
                >
                  Submit
                </button>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

