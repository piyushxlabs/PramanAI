"use client";

import { useCallback, useRef, useState } from "react";
import {
  ApprovalRequiredData,
  ChatMessage,
  Citation,
  GraphStepState,
  QueryFilters,
  StateUpdateData,
  ToolExecutionLog,
} from "@/types";

interface UseSSEChatOptions {
  token?: string | null;
  initialSessionId?: string;
  onTurnCompleted?: () => void;
}

export function useSSEChat(options?: UseSSEChatOptions) {
  const token = options?.token;
  const onTurnCompleted = options?.onTurnCompleted;

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [steps, setSteps] = useState<GraphStepState[]>([]);
  const [verifiedSources, setVerifiedSources] = useState<Citation[]>([]);
  const [pendingApproval, setPendingApproval] = useState<ApprovalRequiredData | null>(null);
  const [stateUpdates, setStateUpdates] = useState<StateUpdateData[]>([]);
  const [toolLogs, setToolLogs] = useState<ToolExecutionLog[]>([]);
  const [sessionId, setSessionId] = useState<string>(
    () => options?.initialSessionId || `sess_${Math.random().toString(36).substring(2, 10)}`
  );

  const abortControllerRef = useRef<AbortController | null>(null);

  const getHeaders = useCallback(() => {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (token) {
      headers["Authorization"] = `Bearer ${token}`;
    }
    return headers;
  }, [token]);

  const processSSEStream = async (response: Response, currentAgentMsgId: string) => {
    const reader = response.body?.getReader();
    if (!reader) {
      setIsStreaming(false);
      return;
    }

    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    let currentEventName = "";

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.trim();

          if (!trimmed) {
            currentEventName = "";
            continue;
          }

          if (trimmed.startsWith(":")) continue;

          if (trimmed.startsWith("event: ")) {
            currentEventName = trimmed.substring(7).trim();
            continue;
          }

          if (trimmed.startsWith("data: ")) {
            const rawJson = trimmed.substring(6).trim();
            try {
              const event = JSON.parse(rawJson);
              if (currentEventName && !event.type) {
                event.type = currentEventName;
              }
              handleStreamEvent(event, currentAgentMsgId);
            } catch (err) {
              console.warn("Failed to parse SSE event JSON:", rawJson, err);
            }
          }
        }
      }
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        console.log("Stream reading aborted by officer.");
      } else {
        console.error("Stream reading error:", err);
      }
    } finally {
      setIsStreaming(false);
      setMessages((prev) =>
        prev
          .map((msg) => (msg.id === currentAgentMsgId ? { ...msg, isStreaming: false } : msg))
          .filter((msg) => !(msg.role === "agent" && msg.content.trim() === ""))
      );
      if (onTurnCompleted) {
        onTurnCompleted();
      }
    }
  };

  const handleStreamEvent = (
    event: Record<string, unknown>,
    agentMsgId: string
  ) => {
    const type = event.type as string;

    switch (type) {
      case "text-delta": {
        const delta = (event.delta as string) || "";
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === agentMsgId
              ? { ...msg, content: msg.content + delta, isStreaming: true }
              : msg
          )
        );
        break;
      }

      case "data-graph-step":
      case "step_update": {
        const rawData = (event.data || event) as {
          node?: string;
          step?: string;
          label?: string;
          status: "started" | "active" | "completed" | "done" | "retrying";
        };
        if (rawData) {
          const nodeKey = rawData.node || rawData.step || "";
          const stepKey = rawData.step || rawData.node || "";
          const normalizedStatus: "started" | "completed" | "retrying" =
            rawData.status === "active" || rawData.status === "started"
              ? "started"
              : rawData.status === "done" || rawData.status === "completed"
              ? "completed"
              : rawData.status === "retrying"
              ? "retrying"
              : "completed";

          const stepItem: GraphStepState = {
            node: nodeKey,
            step: stepKey,
            label: rawData.label || nodeKey,
            status: normalizedStatus,
          };

          setSteps((prev) => {
            const existingIdx = prev.findIndex(
              (s) => s.node === nodeKey || s.node === stepKey || (s as { step?: string }).step === stepKey
            );
            if (existingIdx >= 0) {
              const copy = [...prev];
              copy[existingIdx] = stepItem;
              return copy;
            }
            return [...prev, stepItem];
          });
        }
        break;
      }

      case "data-state-update": {
        const data = event.data as { field: string; reducer: string; value: unknown };
        if (data) {
          setStateUpdates((prev) => [
            ...prev,
            { ...data, timestamp: new Date().toLocaleTimeString() },
          ]);

          if (data.field === "confidence_score") {
            const score = typeof data.value === "number" ? data.value : undefined;
            setMessages((prev) =>
              prev.map((msg) => (msg.id === agentMsgId ? { ...msg, confidence_score: score } : msg))
            );
          }
          if (data.field === "supersession_status") {
            const status = typeof data.value === "string" ? data.value : undefined;
            setMessages((prev) =>
              prev.map((msg) => (msg.id === agentMsgId ? { ...msg, supersession_status: status } : msg))
            );
          }
        }
        break;
      }

      case "data-approval-required": {
        const data = event.data as ApprovalRequiredData;
        if (data) {
          setPendingApproval(data);
          setMessages((prev) =>
            prev.filter((msg) => !(msg.role === "agent" && msg.content.trim() === ""))
          );
        }
        break;
      }

      case "citations": {
        const rawCitations = (event.citations as Citation[]) || [];
        if (rawCitations.length > 0) {
          setVerifiedSources(rawCitations);
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === agentMsgId
                ? { ...msg, citations: rawCitations }
                : msg
            )
          );
        }
        break;
      }

      case "finish": {
        const finishReason = (event.finishReason as string) || "success";
        setMessages((prev) =>
          prev
            .map((msg) => {
              if (msg.id !== agentMsgId) return msg;
              const isRefused = finishReason === "refused";
              const content =
                isRefused && msg.content.trim() === ""
                  ? "**सत्यापन अस्वीकृत / Verification Denied:**\n\nअधिकारी द्वारा सत्यापन अस्वीकृत कर दिया गया है। प्रक्रिया रोक दी गई है।"
                  : msg.content;
              return {
                ...msg,
                content,
                isStreaming: false,
                graceful_refusal: isRefused,
              };
            })
            .filter((msg) => !(msg.role === "agent" && msg.content.trim() === "" && finishReason !== "interrupted"))
        );
        setIsStreaming(false);
        break;
      }

      case "error": {
        const errorText = (event.errorText as string) || "An unexpected error occurred.";
        setMessages((prev) =>
          prev
            .map((msg) =>
              msg.id === agentMsgId
                ? { ...msg, content: msg.content + `\n\n⚠️ ${errorText}`, isStreaming: false }
                : msg
            )
            .filter((msg) => !(msg.role === "agent" && msg.content.trim() === ""))
        );
        setIsStreaming(false);
        break;
      }

      default:
        if (type && type.startsWith("tool-")) {
          const toolName = type.replace("tool-", "");
          const toolCallId = (event.toolCallId as string) || "";
          const state = (event.state as string) || "";
          const input = event.input as Record<string, unknown> | undefined;
          const output = event.output as Record<string, unknown> | undefined;

          setToolLogs((prev) => [
            ...prev,
            {
              toolName,
              toolCallId,
              state,
              input,
              output,
              timestamp: new Date().toLocaleTimeString(),
            },
          ]);

          if (output && Array.isArray(output.passages)) {
            setVerifiedSources(output.passages as Citation[]);
          }
        }
        break;
    }
  };

  const sendQuery = useCallback(
    async (queryText: string, department: string = "Forest", filters?: QueryFilters) => {
      if (!queryText.trim() || isStreaming) return;

      const userMsgId = `user_${Date.now()}`;
      const agentMsgId = `agent_${Date.now()}`;

      const userMessage: ChatMessage = {
        id: userMsgId,
        role: "officer",
        content: queryText,
        timestamp: new Date().toLocaleTimeString(),
        userQuery: queryText,
        queryFilters: filters,
      };

      const agentMessage: ChatMessage = {
        id: agentMsgId,
        role: "agent",
        content: "",
        timestamp: new Date().toLocaleTimeString(),
        userQuery: queryText,
        queryFilters: filters,
        isStreaming: true,
      };

      setMessages((prev) => [
        ...prev.filter((m) => !(m.role === "agent" && m.content.trim() === "")),
        userMessage,
        agentMessage,
      ]);
      setIsStreaming(true);
      setSteps([]);
      setPendingApproval(null);
      setVerifiedSources([]);

      abortControllerRef.current = new AbortController();

      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: getHeaders(),
          body: JSON.stringify({
            session_id: sessionId,
            query_text: queryText,
            officer_context: {
              department: department,
              access_scope: [department, "General"],
            },
            query_filters: filters || null,
          }),
          signal: abortControllerRef.current.signal,
        });

        if (!response.ok) {
          throw new Error(`Server returned HTTP status ${response.status}`);
        }

        await processSSEStream(response, agentMsgId);
      } catch (err: unknown) {
        if (err instanceof Error && err.name === "AbortError") {
          console.log("Turn stopped by officer.");
        } else {
          console.error("Chat streaming failed:", err);
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === agentMsgId
                ? {
                    ...msg,
                    content: "⚠️ The retrieval service encountered an error. Please try again.",
                    isStreaming: false,
                  }
                : msg
            )
          );
        }
      } finally {
        setIsStreaming(false);
        setMessages((prev) =>
          prev
            .map((msg) => (msg.id === agentMsgId ? { ...msg, isStreaming: false } : msg))
            .filter((msg) => !(msg.role === "agent" && msg.content.trim() === ""))
        );
      }
    },
    [isStreaming, sessionId, getHeaders, onTurnCompleted]
  );

  const resumeApproval = useCallback(
    async (action: "approve" | "deny", resolvedGoNumber?: string, reason?: string) => {
      if (!pendingApproval) return;

      const currentApproval = pendingApproval;
      setPendingApproval(null);
      setIsStreaming(true);

      const agentMsgId = `agent_resume_${Date.now()}`;
      const resumeMessage: ChatMessage = {
        id: agentMsgId,
        role: "agent",
        content: "",
        timestamp: new Date().toLocaleTimeString(),
        isStreaming: true,
      };

      setMessages((prev) => [
        ...prev.filter((m) => !(m.role === "agent" && m.content.trim() === "")),
        resumeMessage,
      ]);

      abortControllerRef.current = new AbortController();

      try {
        const response = await fetch("/api/hitl/resume", {
          method: "POST",
          headers: getHeaders(),
          body: JSON.stringify({
            action,
            checkpoint_id: currentApproval.checkpoint_id,
            modified_inputs: resolvedGoNumber ? { resolved_go_number: resolvedGoNumber } : null,
            reason: reason || undefined,
          }),
          signal: abortControllerRef.current.signal,
        });

        if (!response.ok) {
          throw new Error(`Resume request failed with status ${response.status}`);
        }

        await processSSEStream(response, agentMsgId);
      } catch (err: unknown) {
        if (err instanceof Error && err.name === "AbortError") {
          console.log("Resumption stopped by officer.");
        } else {
          console.error("Failed to resume approval:", err);
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === agentMsgId
                ? { ...msg, content: "⚠️ Resumption encountered an error. Please try again.", isStreaming: false }
                : msg
            )
          );
        }
      } finally {
        setIsStreaming(false);
        setMessages((prev) =>
          prev
            .map((msg) => (msg.id === agentMsgId ? { ...msg, isStreaming: false } : msg))
            .filter((msg) => !(msg.role === "agent" && msg.content.trim() === ""))
        );
      }
    },
    [pendingApproval, getHeaders, onTurnCompleted]
  );

  const submitFeedback = useCallback(
    async (messageId: string, feedbackValue: boolean, comment?: string) => {
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === messageId
            ? { ...msg, feedback: { score: feedbackValue, comment } }
            : msg
        )
      );

      try {
        await fetch("/api/feedback/score", {
          method: "POST",
          headers: getHeaders(),
          body: JSON.stringify({
            session_id: sessionId,
            feedback_value: feedbackValue,
            comment,
          }),
        });
      } catch (err) {
        console.error("Failed to submit feedback:", err);
      }
    },
    [sessionId, getHeaders]
  );

  const submitCitationFlag = useCallback(
    async (goNumber: string, pageNumber: number, comment?: string, isAccurate: boolean = false) => {
      try {
        await fetch("/api/feedback/citation", {
          method: "POST",
          headers: getHeaders(),
          body: JSON.stringify({
            session_id: sessionId,
            go_number: goNumber,
            page_number: pageNumber,
            is_accurate: isAccurate,
            comment,
          }),
        });
      } catch (err) {
        console.error("Failed to flag citation:", err);
      }
    },
    [sessionId, getHeaders]
  );

  const stopTurn = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsStreaming(false);
  }, []);

  const restoreSession = useCallback(
    (newSessionId: string, historicalMessages: ChatMessage[], historicalCitations: Citation[]) => {
      stopTurn();
      setSessionId(newSessionId);
      setMessages(historicalMessages);
      setVerifiedSources(historicalCitations);
      setSteps([]);
      setPendingApproval(null);
      setStateUpdates([]);
      setToolLogs([]);
    },
    [stopTurn]
  );

  const resetChat = useCallback(
    (newSessionId?: string) => {
      stopTurn();
      setSessionId(newSessionId || `sess_${Math.random().toString(36).substring(2, 10)}`);
      setMessages([]);
      setVerifiedSources([]);
      setSteps([]);
      setPendingApproval(null);
      setStateUpdates([]);
      setToolLogs([]);
    },
    [stopTurn]
  );

  return {
    sessionId,
    setSessionId,
    messages,
    isStreaming,
    steps,
    verifiedSources,
    pendingApproval,
    stateUpdates,
    toolLogs,
    sendQuery,
    resumeApproval,
    submitFeedback,
    submitCitationFlag,
    stopTurn,
    restoreSession,
    resetChat,
  };
}
