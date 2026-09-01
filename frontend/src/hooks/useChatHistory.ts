"use client";

import { useCallback, useEffect, useState } from "react";
import { ChatMessage, ChatSessionDetail, ChatSessionItem, Citation } from "@/types";

export function useChatHistory(token: string | null) {
  const [sessions, setSessions] = useState<ChatSessionItem[]>([]);
  const [isLoadingHistory, setIsLoadingHistory] = useState<boolean>(false);
  const [activeSessionId, setActiveSessionId] = useState<string>(() =>
    `sess_${Date.now().toString(36)}_${Math.random().toString(36).substring(2, 6)}`
  );

  const fetchHistory = useCallback(async () => {
    if (!token) return;
    setIsLoadingHistory(true);
    try {
      const res = await fetch("/api/chat/history", {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      if (res.ok) {
        const data = await res.json();
        setSessions(data.sessions || []);
      }
    } catch (err) {
      console.error("Failed to fetch chat history:", err);
    } finally {
      setIsLoadingHistory(false);
    }
  }, [token]);

  useEffect(() => {
    if (token) {
      fetchHistory();
    }
  }, [token, fetchHistory]);

  const loadSession = useCallback(
    async (
      sessionId: string
    ): Promise<{
      messages: ChatMessage[];
      citations: Citation[];
      department: string;
      title: string;
    } | null> => {
      if (!token) return null;
      try {
        const res = await fetch(`/api/chat/sessions/${sessionId}`, {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });
        if (!res.ok) {
          throw new Error(`Failed to load session (HTTP ${res.status})`);
        }
        const data: ChatSessionDetail = await res.json();
        setActiveSessionId(sessionId);
        return {
          messages: data.messages || [],
          citations: data.citations || [],
          department: data.department || "Forest",
          title: data.title || "",
        };
      } catch (err) {
        console.error(`Failed to load session ${sessionId}:`, err);
        return null;
      }
    },
    [token]
  );

  const deleteSession = useCallback(
    async (sessionId: string) => {
      if (!token) return;
      try {
        const res = await fetch(`/api/chat/sessions/${sessionId}`, {
          method: "DELETE",
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });
        if (res.ok) {
          setSessions((prev) => prev.filter((s) => s.session_id !== sessionId));
          if (activeSessionId === sessionId) {
            // Generate fresh session if active was deleted
            setActiveSessionId(`sess_${Date.now().toString(36)}_${Math.random().toString(36).substring(2, 6)}`);
          }
        }
      } catch (err) {
        console.error(`Failed to delete session ${sessionId}:`, err);
      }
    },
    [token, activeSessionId]
  );

  const createNewChat = useCallback(() => {
    const freshId = `sess_${Date.now().toString(36)}_${Math.random().toString(36).substring(2, 6)}`;
    setActiveSessionId(freshId);
    return freshId;
  }, []);

  return {
    sessions,
    isLoadingHistory,
    activeSessionId,
    setActiveSessionId,
    fetchHistory,
    loadSession,
    deleteSession,
    createNewChat,
  };
}
