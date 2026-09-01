"use client";

import React, { useState, useMemo } from "react";
import {
  MessageSquarePlus,
  Search,
  Trash2,
  ChevronLeft,
  Calendar,
  Clock,
  Building2,
  History,
} from "lucide-react";
import { ChatSessionItem } from "@/types";

interface HistorySidebarProps {
  isOpen: boolean;
  onClose: () => void;
  sessions: ChatSessionItem[];
  activeSessionId: string;
  onSelectSession: (sessionId: string) => void;
  onNewChat: () => void;
  onDeleteSession: (sessionId: string) => void;
  isLoading: boolean;
}

export const HistorySidebar: React.FC<HistorySidebarProps> = ({
  isOpen,
  onClose,
  sessions,
  activeSessionId,
  onSelectSession,
  onNewChat,
  onDeleteSession,
  isLoading,
}) => {
  const [searchQuery, setSearchQuery] = useState<string>("");

  // Group sessions by date
  const groupedSessions = useMemo(() => {
    const now = new Date();
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
    const yesterday = today - 86400000;
    const sevenDaysAgo = today - 7 * 86400000;

    const filtered = sessions.filter((s) =>
      s.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      s.department.toLowerCase().includes(searchQuery.toLowerCase())
    );

    const groups: { [key: string]: ChatSessionItem[] } = {
      Today: [],
      Yesterday: [],
      "Previous 7 Days": [],
      Older: [],
    };

    for (const session of filtered) {
      const sessionDate = new Date(session.updated_at || session.created_at).getTime();
      if (sessionDate >= today) {
        groups.Today.push(session);
      } else if (sessionDate >= yesterday) {
        groups.Yesterday.push(session);
      } else if (sessionDate >= sevenDaysAgo) {
        groups["Previous 7 Days"].push(session);
      } else {
        groups.Older.push(session);
      }
    }

    return groups;
  }, [sessions, searchQuery]);

  if (!isOpen) return null;

  return (
    <aside className="w-80 h-full flex flex-col bg-[#080d1a] border-r border-slate-800 shadow-2xl z-40 transition-all animate-in slide-in-from-left duration-200">
      {/* Sidebar Header */}
      <div className="p-4 border-b border-slate-800/80 flex items-center justify-between bg-slate-950/60">
        <div className="flex items-center gap-2">
          <History className="w-4 h-4 text-sky-400" />
          <h2 className="text-xs font-bold uppercase tracking-wider text-slate-200">
            Query History
          </h2>
        </div>
        <button
          onClick={onClose}
          className="p-1 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-slate-200 transition-colors"
          title="Close History Drawer"
        >
          <ChevronLeft className="w-4 h-4" />
        </button>
      </div>

      {/* Action Button: New Chat Query */}
      <div className="p-3 border-b border-slate-800/60">
        <button
          onClick={onNewChat}
          className="w-full flex items-center justify-center gap-2 py-2 px-3 rounded-xl bg-gradient-to-r from-sky-600 to-indigo-600 hover:from-sky-500 hover:to-indigo-500 text-white text-xs font-bold shadow-lg shadow-sky-600/20 border border-sky-400/30 transition-all group"
        >
          <MessageSquarePlus className="w-4 h-4 group-hover:scale-110 transition-transform" />
          <span>New Query Turn</span>
        </button>
      </div>

      {/* Search Bar */}
      <div className="p-3 border-b border-slate-800/60">
        <div className="relative">
          <Search className="w-3.5 h-3.5 text-slate-500 absolute left-3 top-2.5" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search past queries..."
            className="w-full pl-8 pr-3 py-1.5 bg-slate-900/80 border border-slate-800 rounded-lg text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-sky-500/50"
          />
        </div>
      </div>

      {/* Session History List */}
      <div className="flex-1 overflow-y-auto p-3 space-y-4">
        {isLoading && (
          <div className="space-y-2">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-14 rounded-xl bg-slate-900/40 animate-pulse" />
            ))}
          </div>
        )}

        {!isLoading && sessions.length === 0 && (
          <div className="py-12 px-4 text-center">
            <Clock className="w-8 h-8 text-slate-600 mx-auto mb-2" />
            <p className="text-xs font-semibold text-slate-300">No Past Sessions</p>
            <p className="text-[11px] text-slate-500 mt-1">
              Your verified query history and citations will appear here.
            </p>
          </div>
        )}

        {!isLoading &&
          Object.entries(groupedSessions).map(([groupTitle, groupItems]) => {
            if (groupItems.length === 0) return null;

            return (
              <div key={groupTitle} className="space-y-1.5">
                <div className="flex items-center gap-1.5 px-2 py-1 text-[10px] font-bold uppercase tracking-wider text-slate-400">
                  <Calendar className="w-3 h-3 text-slate-500" />
                  <span>{groupTitle}</span>
                </div>

                <div className="space-y-1">
                  {groupItems.map((session) => {
                    const isActive = session.session_id === activeSessionId;
                    const dateFormatted = new Date(
                      session.updated_at || session.created_at
                    ).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

                    return (
                      <div
                        key={session.session_id}
                        onClick={() => onSelectSession(session.session_id)}
                        className={`group relative flex items-center justify-between p-2.5 rounded-xl cursor-pointer transition-all border ${
                          isActive
                            ? "bg-sky-500/15 border-sky-500/50 text-slate-100 shadow-md shadow-sky-500/10"
                            : "bg-slate-900/40 border-slate-800/80 hover:bg-slate-800/60 hover:border-slate-700 text-slate-300"
                        }`}
                      >
                        <div className="flex flex-col gap-1 pr-6 flex-1 min-w-0">
                          <span className="text-xs font-medium text-slate-100 truncate leading-snug">
                            {session.title}
                          </span>
                          <div className="flex items-center gap-2 text-[10px] text-slate-400">
                            <span className="flex items-center gap-1">
                              <Building2 className="w-2.5 h-2.5 text-slate-500" />
                              <span className="truncate max-w-[80px]">
                                {session.department}
                              </span>
                            </span>
                            <span>•</span>
                            <span>{dateFormatted}</span>
                          </div>
                        </div>

                        {/* Delete Session Button on Hover */}
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onDeleteSession(session.session_id);
                          }}
                          className="opacity-0 group-hover:opacity-100 p-1.5 rounded-lg hover:bg-red-500/20 text-slate-400 hover:text-red-300 transition-all"
                          title="Delete Session Record"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    );
                  })}
                </div>
              </div>
            );
          })}
      </div>

      {/* Footer Info */}
      <div className="p-3 border-t border-slate-800/80 bg-slate-950/60 text-[10px] text-slate-400 flex items-center justify-between">
        <span>Persistent History</span>
        <span className="font-mono text-sky-400">{sessions.length} Sessions</span>
      </div>
    </aside>
  );
};
