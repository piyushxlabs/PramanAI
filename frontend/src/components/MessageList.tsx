"use client";

import React, { useState } from "react";
import {
  ThumbsUp,
  ThumbsDown,
  ShieldAlert,
  CheckCircle2,
  AlertCircle,
  FileText,
  Send,
  Download,
  ScrollText,
} from "lucide-react";
import { ChatMessage, Citation } from "@/types";
import { NoteSheetModal } from "@/components/NoteSheetModal";
import { PolicyTimeline } from "@/components/PolicyTimeline";

interface MessageListProps {
  messages: ChatMessage[];
  department?: string;
  activeCitations?: Citation[];
  onSelectCitation?: (citation: Citation) => void;
  onSubmitFeedback: (messageId: string, feedbackValue: boolean, comment?: string) => void;
}

export const MessageList: React.FC<MessageListProps> = ({
  messages,
  department = "Forest",
  activeCitations = [],
  onSelectCitation,
  onSubmitFeedback,
}) => {
  const [activeCommentId, setActiveCommentId] = useState<string | null>(null);
  const [feedbackComment, setFeedbackComment] = useState<string>("");
  const [currentScore, setCurrentScore] = useState<boolean>(true);
  const [selectedNoteSheetMsg, setSelectedNoteSheetMsg] = useState<ChatMessage | null>(null);

  const handleFeedbackClick = (msgId: string, isPositive: boolean) => {
    setActiveCommentId(msgId);
    setCurrentScore(isPositive);
    onSubmitFeedback(msgId, isPositive);
  };

  const handleCommentSubmit = (msgId: string) => {
    if (feedbackComment.trim()) {
      onSubmitFeedback(msgId, currentScore, feedbackComment);
    }
    setActiveCommentId(null);
    setFeedbackComment("");
  };

  const handleCitationClick = (citationNumStr: string, msgCitations?: Citation[]) => {
    if (!onSelectCitation) return;
    const num = parseInt(citationNumStr.replace(/[^\d]/g, ""), 10);
    const available = (msgCitations && msgCitations.length > 0) ? msgCitations : activeCitations;
    if (available && available.length > 0) {
      const target = available[num - 1] || available[0];
      if (target) {
        onSelectCitation(target);
      }
    }
  };

  const renderConfidenceBadge = (score?: number) => {
    if (score === undefined || score === null) return null;
    const isHigh = score >= 0.85;
    const isMed = score >= 0.6;

    const badgeClass = isHigh
      ? "bg-emerald-500/15 border-emerald-500/30 text-emerald-300"
      : isMed
      ? "bg-amber-500/15 border-amber-500/30 text-amber-300"
      : "bg-red-500/15 border-red-500/30 text-red-300";

    const label = isHigh ? "High Confidence" : isMed ? "Medium Confidence" : "Low Confidence";

    return (
      <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold border ${badgeClass}`}>
        {isHigh ? (
          <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
        ) : (
          <AlertCircle className="w-3.5 h-3.5 text-amber-400" />
        )}
        <span>
          {label} ({score.toFixed(2)})
        </span>
      </div>
    );
  };

  const renderSupersessionBadge = (status?: string) => {
    if (!status || status === "UNKNOWN") return null;
    const isCurrent = status === "CURRENT_ACTIVE" || status === "ACTIVE";

    return (
      <div
        className={`px-2.5 py-1 rounded-full text-xs font-semibold border ${
          isCurrent
            ? "bg-sky-500/15 border-sky-500/30 text-sky-300"
            : "bg-orange-500/15 border-orange-500/30 text-orange-300"
        }`}
      >
        <span>Status: {status.replace("_", " ")}</span>
      </div>
    );
  };

  const formatContentWithCitations = (text: string, msgCitations?: Citation[]) => {
    const parts = text.split(/(\[\d+\]|\(Ref:[^)]+\))/g);
    return parts.map((part, index) => {
      if (/^\[\d+\]$/.test(part) || /^\(Ref:[^)]+\)$/.test(part)) {
        return (
          <button
            key={index}
            onClick={() => handleCitationClick(part, msgCitations)}
            className="inline-flex items-center gap-1 px-1.5 py-0.5 mx-0.5 text-[11px] font-bold rounded bg-sky-500/20 text-sky-300 border border-sky-500/40 hover:bg-sky-500/30 hover:border-sky-400 cursor-pointer transition-colors"
            title={`Click to view Document Highlight for ${part}`}
          >
            <FileText className="w-3 h-3 inline text-sky-400" />
            {part}
          </button>
        );
      }
      return <span key={index}>{part}</span>;
    });
  };

  const displayMessages = messages.filter(
    (msg) => !(msg.role === "agent" && !msg.isStreaming && !msg.content.trim())
  );

  if (displayMessages.length === 0) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center p-8 text-center text-slate-400">
        <div className="w-16 h-16 rounded-2xl bg-slate-800/80 border border-slate-700/80 flex items-center justify-center mb-4 text-sky-400 shadow-xl">
          <FileText className="w-8 h-8" />
        </div>
        <h2 className="text-lg font-bold text-slate-200 mb-1">Uttarakhand Administrative Corpus Search</h2>
        <p className="text-xs text-slate-400 max-w-md">
          Ask queries in English, Hindi, or Hinglish. Every factual assertion is 100% cited verbatim from official
          indexed Government Orders with active supersession tracking.
        </p>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto px-6 py-6 space-y-6">
      {displayMessages.map((msg) => (
        <div
          key={msg.id}
          className={`flex flex-col ${msg.role === "officer" ? "items-end" : "items-start"}`}
        >
          {/* Message Bubble */}
          <div
            className={`max-w-3xl rounded-2xl p-5 shadow-lg ${
              msg.role === "officer"
                ? "bg-gradient-to-r from-sky-600 to-indigo-700 text-white rounded-br-none border border-sky-400/30"
                : "glass-card text-slate-100 rounded-bl-none border border-slate-700/60"
            }`}
          >
            {/* Header / Badges for Agent */}
            {msg.role === "agent" && (
              <div className="flex flex-wrap items-center gap-2 mb-3 pb-3 border-b border-slate-700/40">
                {renderConfidenceBadge(msg.confidence_score)}
                {renderSupersessionBadge(msg.supersession_status)}
                {msg.graceful_refusal && (
                  <div className="flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-semibold bg-red-500/20 text-red-300 border border-red-500/30">
                    <ShieldAlert className="w-3.5 h-3.5" />
                    <span>Out-of-Scope Notice</span>
                  </div>
                )}
              </div>
            )}

            {/* Message Body */}
            <div className="text-sm leading-relaxed whitespace-pre-wrap">
              {msg.role === "agent" ? (
                formatContentWithCitations(msg.content, msg.citations)
              ) : (
                msg.content
              )}
              {msg.isStreaming && (
                <span className="inline-block w-2 h-4 ml-1 bg-sky-400 animate-pulse rounded" />
              )}
            </div>

            {/* Policy Evolution & Supersession Timeline (Feature 2) */}
            {msg.role === "agent" &&
              !msg.isStreaming &&
              !msg.graceful_refusal &&
              ((msg.citations && msg.citations.length > 0) ||
                (activeCitations && activeCitations.length > 0)) && (
                <PolicyTimeline
                  citations={(msg.citations && msg.citations.length > 0) ? msg.citations : activeCitations}
                  supersessionStatus={msg.supersession_status}
                  onSelectCitation={onSelectCitation}
                />
              )}

            {/* Timestamp, Note-Sheet Export & Feedback Footer */}
            <div className="mt-3.5 pt-2 flex flex-wrap items-center justify-between gap-2 text-[11px] text-slate-400 border-t border-slate-700/30">
              <div className="flex items-center gap-3">
                <span>{msg.timestamp}</span>

                {/* Note-Sheet Export Action Button (Feature 1) */}
                {msg.role === "agent" && !msg.isStreaming && !msg.graceful_refusal && msg.content.trim().length > 0 && (
                  <button
                    onClick={() => setSelectedNoteSheetMsg(msg)}
                    className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-sky-500/15 hover:bg-sky-500/25 text-sky-300 border border-sky-500/30 transition-all font-semibold text-[11px]"
                    title="Generate Secretariat Note-Sheet PDF"
                  >
                    <ScrollText className="w-3.5 h-3.5 text-sky-400" />
                    <span>Export Note-Sheet (टिप्पणी)</span>
                  </button>
                )}
              </div>

              {msg.role === "agent" && !msg.isStreaming && (
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-slate-400">Rate answer:</span>
                  <button
                    onClick={() => handleFeedbackClick(msg.id, true)}
                    className={`p-1 rounded hover:bg-slate-700 transition-colors ${
                      msg.feedback?.score === true ? "text-emerald-400" : "text-slate-400"
                    }`}
                    title="Thumbs Up (Accurate citation)"
                  >
                    <ThumbsUp className="w-3.5 h-3.5" />
                  </button>
                  <button
                    onClick={() => handleFeedbackClick(msg.id, false)}
                    className={`p-1 rounded hover:bg-slate-700 transition-colors ${
                      msg.feedback?.score === false ? "text-red-400" : "text-slate-400"
                    }`}
                    title="Thumbs Down (Inaccurate / Incomplete)"
                  >
                    <ThumbsDown className="w-3.5 h-3.5" />
                  </button>
                </div>
              )}
            </div>

            {/* Detailed Feedback Input Form */}
            {activeCommentId === msg.id && (
              <div className="mt-3 pt-3 border-t border-slate-700/50 flex items-center gap-2">
                <input
                  type="text"
                  placeholder="Optional note for audit evaluation team..."
                  value={feedbackComment}
                  onChange={(e) => setFeedbackComment(e.target.value)}
                  className="flex-1 bg-slate-900/80 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-slate-200 placeholder:text-slate-500 focus:outline-none focus:border-sky-500"
                  onKeyDown={(e) => e.key === "Enter" && handleCommentSubmit(msg.id)}
                />
                <button
                  onClick={() => handleCommentSubmit(msg.id)}
                  className="p-1.5 bg-sky-600 hover:bg-sky-500 text-white rounded-lg text-xs"
                >
                  <Send className="w-3.5 h-3.5" />
                </button>
              </div>
            )}
          </div>
        </div>
      ))}

      {/* Secretariat Note-Sheet Modal Dialog (Feature 1) */}
      {selectedNoteSheetMsg && (
        <NoteSheetModal
          message={selectedNoteSheetMsg}
          department={department}
          onClose={() => setSelectedNoteSheetMsg(null)}
        />
      )}
    </div>
  );
};
