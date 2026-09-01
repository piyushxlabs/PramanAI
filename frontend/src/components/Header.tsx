"use client";

import React from "react";
import { ShieldCheck, Bug, Building2, PanelLeft, History } from "lucide-react";

import { MASTER_DEPARTMENTS } from "@/constants/departments";
import { PersonaSwitcher } from "@/components/PersonaSwitcher";
import { OfficerPersona, UserProfile } from "@/types";

interface HeaderProps {
  department: string;
  onDepartmentChange: (dept: string) => void;
  debugMode: boolean;
  onToggleDebug: (enabled: boolean) => void;
  user: UserProfile | null;
  currentPersona: OfficerPersona;
  onSelectPersona: (persona: OfficerPersona) => void;
  isAuthLoading: boolean;
  isSidebarOpen: boolean;
  onToggleSidebar: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  department,
  onDepartmentChange,
  debugMode,
  onToggleDebug,
  user,
  currentPersona,
  onSelectPersona,
  isAuthLoading,
  isSidebarOpen,
  onToggleSidebar,
}) => {
  const isAdmin = user?.role === "ADMIN" || currentPersona.role === "ADMIN";

  return (
    <header className="glass-panel sticky top-0 z-50 border-b border-slate-700/60 px-4 sm:px-6 py-3 flex items-center justify-between shadow-xl">
      {/* Left: Sidebar Toggle & Sovereign Brand */}
      <div className="flex items-center gap-3">
        {/* Toggle Chat History Sidebar Button */}
        <button
          onClick={onToggleSidebar}
          className={`p-2 rounded-xl border transition-all ${
            isSidebarOpen
              ? "bg-sky-500/20 border-sky-500/50 text-sky-300 shadow-sm shadow-sky-500/20"
              : "bg-slate-800/80 border-slate-700/80 text-slate-300 hover:bg-slate-700/80 hover:text-white"
          }`}
          title="Toggle Chat History Drawer"
        >
          <History className="w-4 h-4" />
        </button>

        {/* Brand & Enterprise Title */}
        <div className="flex items-center gap-2.5">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-sky-600 to-indigo-600 flex items-center justify-center shadow-lg shadow-sky-500/20 border border-sky-400/30">
            <span className="text-lg font-black text-white tracking-tighter">प्र</span>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base sm:text-lg font-bold text-slate-100 tracking-tight">
                PramanAI
              </h1>
              <span className="text-[9px] sm:text-[10px] uppercase font-bold tracking-widest px-2 py-0.5 rounded-full bg-sky-500/20 text-sky-300 border border-sky-500/30">
                GEMINI • GCP
              </span>
            </div>
            <p className="hidden md:block text-[11px] text-slate-400">
              Autonomous Evidentiary GovTech Agent Fleet
            </p>
          </div>
        </div>
      </div>

      {/* Center: Persistent Advisory Mode Badge */}
      <div className="hidden lg:flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-amber-500/10 border border-amber-500/30 text-amber-300 text-xs font-semibold shadow-sm">
        <ShieldCheck className="w-4 h-4 text-amber-400" />
        <span>Advisory Mode — Human Verification Guarded</span>
      </div>

      {/* Right: Department Selector, Persona Switcher & Audit Toggle */}
      <div className="flex items-center gap-2.5 sm:gap-3.5">
        {/* Department Selector */}
        <div
          className="flex items-center gap-1.5 sm:gap-2 bg-slate-800/80 border border-slate-700/80 rounded-xl px-2.5 sm:px-3 py-1.5 text-xs text-slate-200"
          title={isAdmin ? "ITDA Admin Scope Override" : "Officer Assigned Department"}
        >
          <Building2 className="w-3.5 h-3.5 text-sky-400" />
          <select
            value={department}
            onChange={(e) => onDepartmentChange(e.target.value)}
            disabled={!isAdmin && user !== null}
            className={`bg-transparent border-none text-slate-200 text-xs font-medium focus:outline-none ${
              isAdmin ? "cursor-pointer" : "cursor-default opacity-90"
            }`}
          >
            {MASTER_DEPARTMENTS.map((dept) => (
              <option key={dept.code} value={dept.code} className="bg-slate-900 text-slate-100">
                {dept.name_en} {dept.code === currentPersona.department ? "✓" : ""}
              </option>
            ))}
          </select>
        </div>

        {/* 1-Click Officer Persona Switcher */}
        <PersonaSwitcher
          user={user}
          currentPersona={currentPersona}
          onSelectPersona={onSelectPersona}
          isLoading={isAuthLoading}
        />

        {/* Technical Audit Mode Toggle */}
        <button
          onClick={() => onToggleDebug(!debugMode)}
          className={`flex items-center gap-1.5 px-2.5 sm:px-3 py-1.5 rounded-xl text-xs font-semibold transition-all border ${
            debugMode
              ? "bg-purple-500/20 border-purple-500/50 text-purple-300 shadow-sm shadow-purple-500/20"
              : "bg-slate-800/60 border-slate-700 text-slate-400 hover:text-slate-200"
          }`}
          title="Toggle Technical Audit / State Trace Mode"
        >
          <Bug className="w-3.5 h-3.5" />
          <span className="hidden sm:inline">Audit</span>
        </button>
      </div>
    </header>
  );
};
