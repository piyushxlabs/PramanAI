"use client";

import React, { useState } from "react";
import { AlertTriangle, Check, X, ShieldAlert } from "lucide-react";
import { ApprovalRequiredData } from "@/types";

interface HumanVerificationCardProps {
  approval: ApprovalRequiredData;
  onResume: (action: "approve" | "deny", resolvedGoNumber?: string, reason?: string) => void;
}

export const HumanVerificationCard: React.FC<HumanVerificationCardProps> = ({
  approval,
  onResume,
}) => {
  const [selectedGo, setSelectedGo] = useState<string>("");
  const [reason, setReason] = useState<string>("");
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);

  const trigger = approval.trigger;
  const candidateGos = approval.action_preview?.candidate_gos || [];

  const handleApprove = () => {
    if (isSubmitting) return;
    setIsSubmitting(true);
    onResume("approve", selectedGo || undefined, reason || undefined);
  };

  const handleDeny = () => {
    if (isSubmitting) return;
    setIsSubmitting(true);
    onResume("deny", undefined, reason || "Officer declined verification.");
  };

  return (
    <div className="mx-6 mb-4 p-5 rounded-2xl bg-amber-950/40 border-2 border-amber-500/60 shadow-2xl backdrop-blur-xl animate-in fade-in slide-in-from-bottom-3 duration-300">
      {/* Header */}
      <div className="flex items-center gap-3 mb-3">
        <div className="w-9 h-9 rounded-xl bg-amber-500/20 border border-amber-500/50 flex items-center justify-center text-amber-400">
          <AlertTriangle className="w-5 h-5" />
        </div>
        <div>
          <h3 className="text-sm font-bold text-amber-200">Human Verification Required (Node 5 Pause)</h3>
          <p className="text-xs text-amber-300/80">
            {trigger === "conflict"
              ? "Potential conflict or contradictory provisions detected between retrieved Government Orders."
              : trigger === "personal_data"
              ? "Citizen data privacy protection triggered under DPDP Act 2023."
              : "Retrieval confidence falls below sovereign threshold (0.60). Human verification required."}
          </p>
        </div>
      </div>

      {/* Trigger Details */}
      {trigger === "personal_data" && (
        <div className="p-3 mb-4 rounded-xl bg-slate-900/80 border border-amber-500/30 text-xs text-slate-300 flex items-center gap-2">
          <ShieldAlert className="w-4 h-4 text-amber-400 shrink-0" />
          <span>Sensitive citizen identifiers detected. Passages have been redacted for compliance.</span>
        </div>
      )}

      {/* Candidate GO Selector for Conflict Resolution */}
      {trigger === "conflict" && candidateGos.length > 0 && (
        <div className="mb-4 space-y-2">
          <label className="text-xs font-semibold text-amber-200 block">
            Select the Authoritative Governing GO Number:
          </label>
          <div className="space-y-2">
            {candidateGos.map((go) => (
              <label
                key={go}
                className={`flex items-center gap-2.5 p-2.5 rounded-xl border text-xs cursor-pointer transition-all ${
                  selectedGo === go
                    ? "bg-amber-500/20 border-amber-400 text-amber-100 font-bold"
                    : "bg-slate-900/60 border-slate-700 text-slate-300 hover:border-slate-500"
                } ${isSubmitting ? "opacity-60 cursor-not-allowed" : ""}`}
              >
                <input
                  type="radio"
                  name="candidate_go"
                  value={go}
                  disabled={isSubmitting}
                  checked={selectedGo === go}
                  onChange={(e) => setSelectedGo(e.target.value)}
                  className="accent-amber-500 cursor-pointer disabled:cursor-not-allowed"
                />
                <span>{go}</span>
              </label>
            ))}
          </div>
        </div>
      )}

      {/* Optional Reason / Notes */}
      <div className="mb-4">
        <label className="text-xs font-semibold text-amber-200/90 block mb-1">
          Officer Instructions / Audit Note (Optional):
        </label>
        <input
          type="text"
          placeholder="e.g. Verified with 2018 gazette notification..."
          value={reason}
          disabled={isSubmitting}
          onChange={(e) => setReason(e.target.value)}
          className="w-full bg-slate-950/80 border border-slate-700/80 rounded-xl px-3 py-2 text-xs text-slate-100 placeholder:text-slate-500 focus:outline-none focus:border-amber-500 disabled:opacity-60 disabled:cursor-not-allowed"
        />
      </div>

      {/* Actions */}
      <div className="flex items-center justify-end gap-3 pt-2 border-t border-amber-500/30">
        <button
          onClick={handleDeny}
          disabled={isSubmitting}
          className="flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-bold bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-600 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <X className="w-4 h-4 text-red-400" />
          <span>{isSubmitting ? "Stopping..." : "Deny & Stop"}</span>
        </button>

        <button
          onClick={handleApprove}
          disabled={isSubmitting}
          className="flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-bold bg-gradient-to-r from-amber-600 to-amber-500 hover:from-amber-500 hover:to-amber-400 text-slate-950 shadow-lg shadow-amber-500/20 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <Check className="w-4 h-4 stroke-[3]" />
          <span>{isSubmitting ? "Processing..." : "Approve & Synthesize"}</span>
        </button>
      </div>
    </div>
  );
};
