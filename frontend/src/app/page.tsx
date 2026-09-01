"use client";

import React, { useState, useEffect } from "react";
import { Header } from "@/components/Header";
import { HistorySidebar } from "@/components/HistorySidebar";
import { MessageList } from "@/components/MessageList";
import { SessionProgress } from "@/components/SessionProgress";
import { VerifiedSourcesPanel } from "@/components/VerifiedSourcesPanel";
import { HumanVerificationCard } from "@/components/HumanVerificationCard";
import { DebugTraceInspector } from "@/components/DebugTraceInspector";
import { DocumentViewer } from "@/components/DocumentViewer";
import { ChatInput } from "@/components/ChatInput";
import { useAuth } from "@/hooks/useAuth";
import { useChatHistory } from "@/hooks/useChatHistory";
import { useSSEChat } from "@/hooks/useSSEChat";
import { Citation, OfficerPersona, QueryFilters } from "@/types";

export default function HomePage() {
  const [department, setDepartment] = useState<string>("Forest");
  const [debugMode, setDebugMode] = useState<boolean>(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState<boolean>(true);
  const [selectedCitation, setSelectedCitation] = useState<Citation | null>(null);
  const [filters, setFilters] = useState<QueryFilters>({
    department: "Forest",
    year_range: null,
    policy_category: null,
    go_number: null,
  });

  // 1. Sovereign Authentication & Persona State
  const {
    user,
    token,
    currentPersona,
    isLoading: isAuthLoading,
    switchPersona,
  } = useAuth();

  // 2. Persistent Chat History State
  const {
    sessions,
    isLoadingHistory,
    activeSessionId,
    fetchHistory,
    loadSession,
    deleteSession,
    createNewChat,
  } = useChatHistory(token);

  // 3. SSE Chat Streaming & State Management
  const {
    sessionId,
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
  } = useSSEChat({
    token,
    initialSessionId: activeSessionId,
    onTurnCompleted: fetchHistory,
  });

  // Sync department when user / persona changes
  useEffect(() => {
    if (user?.department) {
      setDepartment(user.department);
      setFilters((prev) => ({ ...prev, department: user.department }));
    }
  }, [user]);

  // Handle switching persona from Header
  const handlePersonaSelect = async (persona: OfficerPersona) => {
    await switchPersona(persona);
    const newSessionId = createNewChat();
    resetChat(newSessionId);
    setDepartment(persona.department);
    setFilters({
      department: persona.department,
      year_range: null,
      policy_category: null,
      go_number: null,
    });
  };

  // Handle selecting a past session from HistorySidebar
  const handleSelectHistorySession = async (targetSessionId: string) => {
    const sessionData = await loadSession(targetSessionId);
    if (sessionData) {
      restoreSession(targetSessionId, sessionData.messages, sessionData.citations);
      if (sessionData.department) {
        setDepartment(sessionData.department);
        setFilters((prev) => ({ ...prev, department: sessionData.department }));
      }
    }
  };

  // Handle New Query
  const handleNewChat = () => {
    const newId = createNewChat();
    resetChat(newId);
  };

  const handleDepartmentChange = (newDept: string) => {
    setDepartment(newDept);
    setFilters((prev) => ({
      ...prev,
      department: newDept,
    }));
  };

  const handleFiltersChange = (newFilters: QueryFilters) => {
    setFilters(newFilters);
    if (newFilters.department && newFilters.department !== department) {
      setDepartment(newFilters.department);
    }
  };

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-[#0B1120] text-slate-100">
      {/* Top Sovereign Header with Persona Switcher */}
      <Header
        department={department}
        onDepartmentChange={handleDepartmentChange}
        debugMode={debugMode}
        onToggleDebug={setDebugMode}
        user={user}
        currentPersona={currentPersona}
        onSelectPersona={handlePersonaSelect}
        isAuthLoading={isAuthLoading}
        isSidebarOpen={isSidebarOpen}
        onToggleSidebar={() => setIsSidebarOpen(!isSidebarOpen)}
      />

      {/* Main Dashboard Layout */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Drawer: Persistent Chat History */}
        <HistorySidebar
          isOpen={isSidebarOpen}
          onClose={() => setIsSidebarOpen(false)}
          sessions={sessions}
          activeSessionId={sessionId}
          onSelectSession={handleSelectHistorySession}
          onNewChat={handleNewChat}
          onDeleteSession={deleteSession}
          isLoading={isLoadingHistory}
        />

        {/* Center Column: Conversation & Query Area */}
        <div className="flex-1 flex flex-col h-full overflow-hidden border-r border-slate-800/80">
          <MessageList
            messages={messages}
            department={department}
            activeCitations={verifiedSources}
            onSelectCitation={setSelectedCitation}
            onSubmitFeedback={submitFeedback}
          />

          {/* Interactive Human Verification Interrupt Card */}
          {pendingApproval && (
            <HumanVerificationCard
              approval={pendingApproval}
              onResume={resumeApproval}
            />
          )}

          {/* Bottom Chat Input with Faceted Search Filters */}
          <ChatInput
            onSend={(query, activeFilters) => sendQuery(query, department, activeFilters)}
            onStop={stopTurn}
            isStreaming={isStreaming}
            filters={filters}
            onFiltersChange={handleFiltersChange}
          />
        </div>

        {/* Right Column: Progress Stepper, Verified Sources & State Inspector */}
        <div className="w-80 lg:w-96 flex flex-col h-full overflow-y-auto p-4 space-y-4 bg-slate-950/40">
          {/* Step Stepper */}
          <SessionProgress steps={steps} isStreaming={isStreaming} />

          {/* Verified Sources Panel */}
          <VerifiedSourcesPanel
            citations={verifiedSources}
            onSelectCitation={setSelectedCitation}
            onFlagCitation={submitCitationFlag}
          />

          {/* Debug Mode State Inspector */}
          {debugMode && (
            <DebugTraceInspector
              toolLogs={toolLogs}
              stateUpdates={stateUpdates}
            />
          )}
        </div>
      </div>

      {/* Document & Bounding Box Modal Overlay */}
      {selectedCitation && (
        <DocumentViewer
          citation={selectedCitation}
          onClose={() => setSelectedCitation(null)}
        />
      )}
    </div>
  );
}
