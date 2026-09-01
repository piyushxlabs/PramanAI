"use client";

import React, { useState } from "react";
import { Send, Square, Sparkles } from "lucide-react";
import { QueryFilters } from "@/types";
import { FacetFilters } from "@/components/FacetFilters";

interface ChatInputProps {
  onSend: (query: string, filters?: QueryFilters) => void;
  onStop: () => void;
  isStreaming: boolean;
  filters: QueryFilters;
  onFiltersChange: (filters: QueryFilters) => void;
}

const SAMPLE_PROMPTS = [
  "Forest department annual transfer policy 2018 provisions",
  "Samayojan rules for hill cadres in Uttarakhand",
  "Sanction 50 lakhs disbursement for construction project",
];

export const ChatInput: React.FC<ChatInputProps> = ({
  onSend,
  onStop,
  isStreaming,
  filters,
  onFiltersChange,
}) => {
  const [query, setQuery] = useState<string>("");
  const [isFilterOpen, setIsFilterOpen] = useState<boolean>(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim() && !isStreaming) {
      onSend(query.trim(), filters);
      setQuery("");
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="p-4 border-t border-slate-700/60 glass-panel">
      {/* Faceted Filters Section (Feature 3) */}
      <FacetFilters
        filters={filters}
        onFiltersChange={onFiltersChange}
        isOpen={isFilterOpen}
        onToggle={() => setIsFilterOpen(!isFilterOpen)}
      />

      {/* Sample Quick Prompts */}
      <div className="flex flex-wrap items-center gap-2 mb-3">
        <span className="flex items-center gap-1 text-[11px] font-semibold text-slate-400">
          <Sparkles className="w-3 h-3 text-sky-400" />
          <span>Try sample query:</span>
        </span>
        {SAMPLE_PROMPTS.map((prompt, idx) => (
          <button
            key={idx}
            type="button"
            onClick={() => setQuery(prompt)}
            disabled={isStreaming}
            className="px-2.5 py-1 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-[11px] text-slate-300 border border-slate-700 transition-colors truncate max-w-xs"
          >
            {prompt}
          </button>
        ))}
      </div>

      {/* Query Input Box */}
      <form onSubmit={handleSubmit} className="flex items-end gap-3">
        <div className="flex-1 relative">
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type administrative query in English, Hindi, or Hinglish (e.g. transfer policy rules)..."
            rows={2}
            disabled={isStreaming}
            className="w-full bg-slate-900/90 border border-slate-700/90 rounded-2xl p-3.5 text-xs text-slate-100 placeholder:text-slate-500 focus:outline-none focus:border-sky-500 transition-all resize-none shadow-inner"
          />
        </div>

        {isStreaming ? (
          <button
            type="button"
            onClick={onStop}
            className="p-3.5 rounded-2xl bg-red-600 hover:bg-red-500 text-white font-bold shadow-lg shadow-red-500/20 transition-all"
            title="Stop generation"
          >
            <Square className="w-4 h-4 fill-current" />
          </button>
        ) : (
          <button
            type="submit"
            disabled={!query.trim()}
            className="p-3.5 rounded-2xl bg-gradient-to-tr from-sky-600 to-indigo-600 hover:from-sky-500 hover:to-indigo-500 text-white font-bold shadow-lg shadow-sky-500/20 transition-all disabled:opacity-40 disabled:cursor-not-allowed"
            title="Send query"
          >
            <Send className="w-4 h-4" />
          </button>
        )}
      </form>
    </div>
  );
};
