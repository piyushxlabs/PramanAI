"use client";

import React, { useState, useRef, useEffect } from "react";
import { ChevronDown, UserCheck, Shield, Sparkles, Building2 } from "lucide-react";
import { PRECONFIGURED_PERSONAS } from "@/constants/personas";
import { OfficerPersona, UserProfile } from "@/types";

interface PersonaSwitcherProps {
  user: UserProfile | null;
  currentPersona: OfficerPersona;
  onSelectPersona: (persona: OfficerPersona) => void;
  isLoading: boolean;
}

export const PersonaSwitcher: React.FC<PersonaSwitcherProps> = ({
  user,
  currentPersona,
  onSelectPersona,
  isLoading,
}) => {
  const [isOpen, setIsOpen] = useState<boolean>(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown on outside click
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  return (
    <div className="relative" ref={dropdownRef}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        disabled={isLoading}
        className="flex items-center gap-2.5 px-3 py-1.5 rounded-xl bg-slate-900/90 hover:bg-slate-800/90 border border-slate-700/80 hover:border-slate-600 transition-all text-left shadow-md group focus:outline-none focus:ring-1 focus:ring-sky-500/50"
        title="Switch Officer Persona for Evaluation & Testing"
      >
        {/* Officer Avatar / Role Icon */}
        <div className={`w-7 h-7 rounded-lg flex items-center justify-center border font-bold text-xs ${currentPersona.badgeBg} ${currentPersona.badgeColor}`}>
          {currentPersona.role === "ADMIN" ? (
            <Shield className="w-3.5 h-3.5 text-sky-400" />
          ) : (
            <UserCheck className="w-3.5 h-3.5 text-emerald-400" />
          )}
        </div>

        {/* Officer Info Summary */}
        <div className="hidden sm:flex flex-col">
          <div className="flex items-center gap-1.5">
            <span className="text-xs font-semibold text-slate-100 tracking-tight leading-none">
              {user?.full_name || currentPersona.full_name}
            </span>
            <span className={`text-[9px] uppercase font-bold tracking-wider px-1.5 py-0.2 rounded border ${currentPersona.badgeBg} ${currentPersona.badgeColor}`}>
              {user?.role || currentPersona.role}
            </span>
          </div>
          <span className="text-[10px] text-slate-400 leading-tight truncate max-w-[140px]">
            {user?.designation || currentPersona.designation}
          </span>
        </div>

        <ChevronDown
          className={`w-3.5 h-3.5 text-slate-400 transition-transform duration-200 ${
            isOpen ? "rotate-180 text-sky-400" : "group-hover:text-slate-200"
          }`}
        />
      </button>

      {/* Persona Selection Dropdown Menu */}
      {isOpen && (
        <div className="absolute right-0 mt-2 w-80 rounded-2xl bg-slate-900/95 border border-slate-700/90 shadow-2xl backdrop-blur-xl z-50 overflow-hidden animate-in fade-in zoom-in-95 duration-150">
          <div className="px-4 py-3 border-b border-slate-800 bg-slate-950/50 flex items-center justify-between">
            <div>
              <span className="text-xs font-bold uppercase tracking-wider text-slate-300">
                Uttarakhand Secretariat
              </span>
              <p className="text-[11px] text-slate-400">1-Click Officer Persona Switcher</p>
            </div>
            <Sparkles className="w-4 h-4 text-sky-400 animate-pulse" />
          </div>

          <div className="p-2 space-y-1 max-h-[380px] overflow-y-auto">
            {PRECONFIGURED_PERSONAS.map((persona) => {
              const isSelected =
                user?.email.toLowerCase() === persona.email.toLowerCase() ||
                currentPersona.id === persona.id;

              return (
                <button
                  key={persona.id}
                  onClick={() => {
                    onSelectPersona(persona);
                    setIsOpen(false);
                  }}
                  className={`w-full p-2.5 rounded-xl text-left transition-all flex flex-col gap-1 border ${
                    isSelected
                      ? "bg-sky-500/10 border-sky-500/40 text-slate-100 shadow-sm"
                      : "bg-transparent border-transparent hover:bg-slate-800/60 hover:border-slate-700/60 text-slate-300"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div
                        className={`w-6 h-6 rounded-md flex items-center justify-center border text-[11px] font-bold ${persona.badgeBg} ${persona.badgeColor}`}
                      >
                        {persona.full_name.charAt(0)}
                      </div>
                      <span className="text-xs font-bold text-slate-100">
                        {persona.full_name}
                      </span>
                    </div>

                    <span
                      className={`text-[9px] uppercase font-bold tracking-wider px-1.5 py-0.5 rounded border ${persona.badgeBg} ${persona.badgeColor}`}
                    >
                      {persona.role}
                    </span>
                  </div>

                  <div className="flex items-center gap-1.5 text-[10px] text-slate-400 pl-8">
                    <Building2 className="w-3 h-3 text-slate-500" />
                    <span>{persona.department} Department</span>
                    <span>•</span>
                    <span className="truncate">{persona.designation}</span>
                  </div>

                  <p className="text-[10px] text-slate-400 pl-8 line-clamp-1 italic">
                    {persona.description}
                  </p>
                </button>
              );
            })}
          </div>

          <div className="px-3.5 py-2.5 border-t border-slate-800 bg-slate-950/40 flex items-center justify-between text-[10px] text-slate-400">
            <span>Air-Gapped Sovereign JWT</span>
            <span className="font-mono text-emerald-400">HS256 Verified</span>
          </div>
        </div>
      )}
    </div>
  );
};
