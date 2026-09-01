"use client";

import React from "react";
import { Filter, RotateCcw, Calendar, Building, Tag, X, ChevronDown, Check } from "lucide-react";
import { QueryFilters } from "@/types";
import { MASTER_DEPARTMENTS } from "@/constants/departments";

interface FacetFiltersProps {
  filters: QueryFilters;
  onFiltersChange: (filters: QueryFilters) => void;
  isOpen: boolean;
  onToggle: () => void;
}

const YEAR_PRESETS: { label: string; range: [number, number] | null }[] = [
  { label: "All Years (सभी वर्ष)", range: null },
  { label: "2021 – 2026 (नवीनतम)", range: [2021, 2026] },
  { label: "2015 – 2020", range: [2015, 2020] },
  { label: "2010 – 2014", range: [2010, 2014] },
  { label: "Pre-2010 (2000 – 2009)", range: [2000, 2009] },
];

const POLICY_CATEGORIES = [
  "Transfer Policy",
  "Regularization",
  "Budget Allocation",
  "Recruitment",
  "Pension",
  "Leave Rules",
  "Procurement",
];

export const FacetFilters: React.FC<FacetFiltersProps> = ({
  filters,
  onFiltersChange,
  isOpen,
  onToggle,
}) => {
  const activeFilterCount =
    (filters.department ? 1 : 0) +
    (filters.year_range ? 1 : 0) +
    (filters.policy_category ? 1 : 0) +
    (filters.go_number ? 1 : 0);

  const handleReset = () => {
    onFiltersChange({
      department: null,
      year_range: null,
      policy_category: null,
      go_number: null,
    });
  };

  const handleDeptSelect = (deptCode: string) => {
    onFiltersChange({
      ...filters,
      department: filters.department === deptCode ? null : deptCode,
    });
  };

  const handlePresetSelect = (range: [number, number] | null) => {
    onFiltersChange({
      ...filters,
      year_range: range,
    });
  };

  const handleCategorySelect = (category: string) => {
    onFiltersChange({
      ...filters,
      policy_category: filters.policy_category === category ? null : category,
    });
  };

  const currentStartYear = filters.year_range ? filters.year_range[0] : "";
  const currentEndYear = filters.year_range ? filters.year_range[1] : "";

  const handleCustomYearChange = (start: string, end: string) => {
    const s = parseInt(start, 10);
    const e = parseInt(end, 10);
    if (!isNaN(s) && !isNaN(e)) {
      onFiltersChange({ ...filters, year_range: [s, e] });
    } else if (!start && !end) {
      onFiltersChange({ ...filters, year_range: null });
    }
  };

  return (
    <div className="w-full">
      {/* Toggle Bar / Active Filters Badges */}
      <div className="flex flex-wrap items-center justify-between gap-2 px-1 mb-2">
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={onToggle}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold border transition-all ${
              isOpen || activeFilterCount > 0
                ? "bg-sky-500/20 border-sky-500/50 text-sky-200"
                : "bg-slate-800/80 border-slate-700 text-slate-300 hover:bg-slate-700"
            }`}
          >
            <Filter className="w-3.5 h-3.5 text-sky-400" />
            <span>Faceted Search Filters (फ़िल्टर)</span>
            {activeFilterCount > 0 && (
              <span className="w-4 h-4 rounded-full bg-sky-500 text-slate-950 text-[10px] font-black flex items-center justify-center">
                {activeFilterCount}
              </span>
            )}
            <ChevronDown
              className={`w-3.5 h-3.5 text-slate-400 transition-transform duration-200 ${
                isOpen ? "rotate-180" : ""
              }`}
            />
          </button>

          {/* Active filter pills */}
          {filters.department && (
            <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-indigo-500/20 border border-indigo-500/40 text-indigo-300 text-[11px] font-medium">
              <Building className="w-3 h-3 text-indigo-400" />
              <span>{filters.department}</span>
              <button
                type="button"
                onClick={() => onFiltersChange({ ...filters, department: null })}
                className="hover:text-white"
              >
                <X className="w-3 h-3" />
              </button>
            </span>
          )}

          {filters.year_range && (
            <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-emerald-500/20 border border-emerald-500/40 text-emerald-300 text-[11px] font-medium">
              <Calendar className="w-3 h-3 text-emerald-400" />
              <span>
                {filters.year_range[0]} – {filters.year_range[1]}
              </span>
              <button
                type="button"
                onClick={() => onFiltersChange({ ...filters, year_range: null })}
                className="hover:text-white"
              >
                <X className="w-3 h-3" />
              </button>
            </span>
          )}

          {filters.policy_category && (
            <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-amber-500/20 border border-amber-500/40 text-amber-300 text-[11px] font-medium">
              <Tag className="w-3 h-3 text-amber-400" />
              <span>{filters.policy_category}</span>
              <button
                type="button"
                onClick={() => onFiltersChange({ ...filters, policy_category: null })}
                className="hover:text-white"
              >
                <X className="w-3 h-3" />
              </button>
            </span>
          )}
        </div>

        {activeFilterCount > 0 && (
          <button
            type="button"
            onClick={handleReset}
            className="flex items-center gap-1 text-[11px] font-medium text-slate-400 hover:text-red-300 transition-colors"
          >
            <RotateCcw className="w-3 h-3" />
            <span>Reset Filters (फ़िल्टर हटाएं)</span>
          </button>
        )}
      </div>

      {/* Expanded Filter Drawer */}
      {isOpen && (
        <div className="p-4 mb-3 rounded-2xl bg-slate-900/90 border border-slate-700/80 shadow-2xl backdrop-blur-xl animate-in fade-in slide-in-from-top-2 duration-200 space-y-4">
          {/* Department Selection */}
          <div>
            <label className="text-[11px] font-bold text-slate-300 uppercase tracking-wider mb-2 flex items-center gap-1.5">
              <Building className="w-3.5 h-3.5 text-sky-400" />
              <span>Department Filter (विभाग चुनें):</span>
            </label>
            <div className="flex flex-wrap gap-1.5">
              {MASTER_DEPARTMENTS.map((dept) => {
                const isSelected =
                  filters.department === dept.code ||
                  filters.department === dept.name_en ||
                  filters.department === dept.short_label;
                return (
                  <button
                    key={dept.code}
                    type="button"
                    onClick={() => handleDeptSelect(dept.code)}
                    className={`px-2.5 py-1 rounded-lg text-xs font-medium border transition-all ${
                      isSelected
                        ? "bg-sky-500/20 border-sky-400 text-sky-200 font-bold"
                        : "bg-slate-800/60 border-slate-700/80 text-slate-300 hover:border-slate-500"
                    }`}
                  >
                    {dept.short_label}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Temporal / Year Range Filter */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-3 border-t border-slate-800">
            <div>
              <label className="text-[11px] font-bold text-slate-300 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                <Calendar className="w-3.5 h-3.5 text-emerald-400" />
                <span>Year Range Presets (वर्ष सीमा):</span>
              </label>
              <div className="flex flex-wrap gap-1.5">
                {YEAR_PRESETS.map((preset, idx) => {
                  const isSelected =
                    (!preset.range && !filters.year_range) ||
                    (preset.range &&
                      filters.year_range &&
                      preset.range[0] === filters.year_range[0] &&
                      preset.range[1] === filters.year_range[1]);

                  return (
                    <button
                      key={idx}
                      type="button"
                      onClick={() => handlePresetSelect(preset.range)}
                      className={`px-2.5 py-1 rounded-lg text-xs font-medium border transition-all ${
                        isSelected
                          ? "bg-emerald-500/20 border-emerald-400 text-emerald-200 font-bold"
                          : "bg-slate-800/60 border-slate-700/80 text-slate-300 hover:border-slate-500"
                      }`}
                    >
                      {preset.label}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Custom Year Inputs */}
            <div>
              <label className="text-[11px] font-bold text-slate-300 uppercase tracking-wider mb-2 block">
                Custom Year Range (कस्टम वर्ष सीमा):
              </label>
              <div className="flex items-center gap-2">
                <input
                  type="number"
                  placeholder="From (e.g. 2005)"
                  min={1950}
                  max={2026}
                  value={currentStartYear}
                  onChange={(e) => handleCustomYearChange(e.target.value, String(currentEndYear))}
                  className="w-32 bg-slate-950/80 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 placeholder:text-slate-500 focus:outline-none focus:border-sky-500"
                />
                <span className="text-slate-400 text-xs font-bold">to</span>
                <input
                  type="number"
                  placeholder="To (e.g. 2026)"
                  min={1950}
                  max={2026}
                  value={currentEndYear}
                  onChange={(e) => handleCustomYearChange(String(currentStartYear), e.target.value)}
                  className="w-32 bg-slate-950/80 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 placeholder:text-slate-500 focus:outline-none focus:border-sky-500"
                />
              </div>
            </div>
          </div>

          {/* Policy Category Chips */}
          <div className="pt-3 border-t border-slate-800">
            <label className="text-[11px] font-bold text-slate-300 uppercase tracking-wider mb-2 flex items-center gap-1.5">
              <Tag className="w-3.5 h-3.5 text-amber-400" />
              <span>Policy Category (नीति श्रेणी):</span>
            </label>
            <div className="flex flex-wrap gap-1.5">
              {POLICY_CATEGORIES.map((category) => (
                <button
                  key={category}
                  type="button"
                  onClick={() => handleCategorySelect(category)}
                  className={`px-2.5 py-1 rounded-lg text-xs font-medium border transition-all ${
                    filters.policy_category === category
                      ? "bg-amber-500/20 border-amber-400 text-amber-200 font-bold"
                      : "bg-slate-800/60 border-slate-700/80 text-slate-300 hover:border-slate-500"
                  }`}
                >
                  {category}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
