"use client";

import React, { useState } from "react";
import { Printer, Copy, Check, X, FileText, Building2, Calendar, ShieldCheck, Download } from "lucide-react";
import { ChatMessage, Citation } from "@/types";

interface NoteSheetModalProps {
  message: ChatMessage;
  department: string;
  onClose: () => void;
}

export const NoteSheetModal: React.FC<NoteSheetModalProps> = ({
  message,
  department,
  onClose,
}) => {
  const [copied, setCopied] = useState<boolean>(false);

  const citations: Citation[] = message.citations || [];

  // Derive department name dynamically from first citation if available and not 'General'
  const displayDepartment = (citations.length > 0 && citations[0].issuing_department && citations[0].issuing_department !== "General")
    ? citations[0].issuing_department
    : (department && department !== "All" && department !== "General" ? department : "उत्तराखण्ड शासन");

  const deptCode = displayDepartment.toUpperCase().replace(/[^A-Z0-9]/g, "").slice(0, 8) || "SEC";
  const fileNumber = `UK-SEC/${deptCode}/2026/NS-${message.id.slice(-6)}`;

  const currentDate = new Date().toLocaleDateString("hi-IN", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });
  const englishDate = new Date().toLocaleDateString("en-IN", {
    year: "numeric",
    month: "short",
    day: "numeric",
  });

  const subject = message.userQuery
    ? `विषय: ${message.userQuery}`
    : "विषय: शासनादेशों एवं प्रशासनिक नियमों के संबंध में विधिक परीक्षण एवं संदर्भ टिप्पणी";

  // Break message content into numbered formal secretariat paragraphs, stripping double bullet markdown
  const rawParagraphs = message.content
    .split(/\n\n+/)
    .map((p) => p.trim())
    .filter((p) => p.length > 0)
    .map((p) => p.replace(/^[-*•]\s+/, ""));

  const handlePrint = () => {
    window.print();
  };

  const handleCopyMarkdown = () => {
    let md = `# उत्तराखण्ड शासन / Government of Uttarakhand\n`;
    md += `## सचिवालय प्रशासन / Secretariat Administration\n`;
    md += `### शासकीय टिप्पणी (OFFICIAL NOTE-SHEET)\n\n`;
    md += `**पत्रांक:** ${fileNumber}\n`;
    md += `**विभाग:** ${displayDepartment}\n`;
    md += `**दिनांक:** ${currentDate} (${englishDate})\n`;
    md += `**${subject}**\n\n`;
    md += `---\n\n`;
    md += `### टिप्पणी विवरण (Note Content):\n\n`;

    rawParagraphs.forEach((para, idx) => {
      const cleanPara = para.replace(/^[-*•]\s+/, "");
      md += `${idx + 1}. ${cleanPara}\n\n`;
    });

    if (citations.length > 0) {
      md += `### संलग्न शासनादेश एवं साक्ष्य तालिका (Table of Authorities):\n\n`;
      md += `| क्र० | शासनादेश संख्या | विभाग | दिनांक | पृष्ठ | उद्धृत मूल साक्ष्य |\n`;
      md += `|---|---|---|---|---|---|\n`;
      citations.forEach((c, idx) => {
        md += `| ${idx + 1} | ${c.go_number} | ${c.issuing_department} | ${c.date} | पृ० ${c.page_number} | ${c.exact_text_excerpt.replace(/\n/g, " ")} |\n`;
      });
      md += `\n`;
    }

    md += `---\n\n`;
    md += `**हस्ताक्षर / अनुमोदक अधिकारी (Approving Officer):** ___________________\n`;
    md += `**पदनाम (Designation):** ___________________\n`;

    navigator.clipboard.writeText(md);
    setCopied(true);
    setTimeout(() => setCopied(false), 2500);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md animate-in fade-in duration-200">
      {/* Modal Container */}
      <div className="relative w-full max-w-4xl max-h-[92vh] flex flex-col rounded-2xl bg-slate-900 border border-slate-700 shadow-2xl overflow-hidden print:p-0 print:border-none print:shadow-none print:bg-white print:text-black">
        {/* Top Modal Action Bar (Hidden in Print) */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-950/80 print:hidden">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-sky-500/20 border border-sky-500/40 flex items-center justify-center text-sky-400">
              <FileText className="w-4 h-4" />
            </div>
            <div>
              <h2 className="text-sm font-bold text-slate-100">सचिवालय टिप्पणी पूर्वावलोकन (Note-Sheet Preview)</h2>
              <p className="text-xs text-slate-400">Uttarakhand Secretariat Format • Legal Print-to-PDF Ready</p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleCopyMarkdown}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition-colors"
              title="Copy as Markdown text"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
              <span>{copied ? "Copied!" : "Copy Markdown"}</span>
            </button>

            <button
              onClick={handlePrint}
              className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-xs font-semibold bg-gradient-to-r from-sky-600 to-indigo-600 hover:from-sky-500 hover:to-indigo-500 text-white shadow-md shadow-sky-500/20 transition-all"
              title="Print or save as official PDF"
            >
              <Printer className="w-3.5 h-3.5" />
              <span>Print / Save as PDF</span>
            </button>

            <button
              onClick={onClose}
              className="p-1.5 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors ml-2"
              title="Close modal"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Scrollable Printable Note-Sheet Body */}
        <div className="flex-1 overflow-y-auto p-8 text-slate-100 bg-[#0F172A] print:p-0 print:bg-white print:text-black print:overflow-visible">
          <div className="max-w-3xl mx-auto bg-slate-900/90 border border-slate-700/80 rounded-xl p-8 shadow-inner print:p-8 print:border-none print:shadow-none print:bg-white notesheet-document">
            {/* Secretariat Header */}
            <div className="text-center pb-6 border-b-2 border-slate-700 print:border-black">
              <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-slate-800 border border-slate-600 mb-2 print:bg-gray-100 print:border-gray-400 text-sky-400 print:text-gray-800 font-serif font-black text-xl">
                उत्तरा
              </div>
              <h1 className="text-xl font-bold tracking-tight text-slate-100 print:text-black font-serif">
                उत्तराखण्ड शासन / Government of Uttarakhand
              </h1>
              <h2 className="text-sm font-semibold text-slate-300 print:text-gray-800 tracking-wide mt-0.5 font-serif">
                सचिवालय प्रशासन • {displayDepartment}
              </h2>
              <div className="inline-block mt-3 px-4 py-1 rounded bg-sky-950/80 border border-sky-500/40 text-sky-300 print:bg-gray-100 print:text-black print:border-black text-xs font-bold uppercase tracking-widest">
                शासकीय टिप्पणी (OFFICIAL NOTE-SHEET)
              </div>
            </div>

            {/* Reference & Subject Box */}
            <div className="my-6 p-4 rounded-lg bg-slate-950/60 border border-slate-800 print:bg-gray-50 print:border-gray-300 text-xs space-y-2">
              <div className="flex flex-wrap items-center justify-between gap-2 text-slate-400 print:text-gray-600 font-mono">
                <span><strong>पत्रांक:</strong> {fileNumber}</span>
                <span><strong>दिनांक:</strong> {currentDate} ({englishDate})</span>
              </div>
              <div className="pt-2 border-t border-slate-800/80 print:border-gray-200">
                <p className="text-xs font-bold text-slate-200 print:text-black leading-relaxed">
                  {subject}
                </p>
              </div>
            </div>

            {/* Note Content Paragraphs */}
            <div className="space-y-4 text-xs leading-relaxed text-slate-200 print:text-black font-serif">
              {rawParagraphs.map((para, idx) => (
                <div key={idx} className="flex gap-3 items-start">
                  <span className="font-bold text-sky-400 print:text-black min-w-[20px] select-none">
                    {idx + 1}.
                  </span>
                  <p className="flex-1 whitespace-pre-wrap">{para}</p>
                </div>
              ))}
            </div>

            {/* Table of Authorities / संलग्न साक्ष्य */}
            {citations.length > 0 && (
              <div className="mt-8 pt-6 border-t border-slate-700/80 print:border-gray-400">
                <h3 className="text-xs font-bold text-slate-200 print:text-black mb-3 flex items-center gap-1.5 font-serif">
                  <ShieldCheck className="w-3.5 h-3.5 text-sky-400 print:text-black" />
                  <span>संलग्न शासनादेश एवं साक्ष्य तालिका (Table of Authorities):</span>
                </h3>
                <div className="overflow-x-auto rounded-lg border border-slate-800 print:border-gray-400">
                  <table className="w-full text-left text-[11px] border-collapse">
                    <thead>
                      <tr className="bg-slate-950/80 print:bg-gray-100 text-slate-400 print:text-gray-800 border-b border-slate-800 print:border-gray-400">
                        <th className="p-2 border-r border-slate-800 print:border-gray-400 w-8 text-center">क्र०</th>
                        <th className="p-2 border-r border-slate-800 print:border-gray-400">शासनादेश संख्या (GO No.)</th>
                        <th className="p-2 border-r border-slate-800 print:border-gray-400">विभाग</th>
                        <th className="p-2 border-r border-slate-800 print:border-gray-400">दिनांक</th>
                        <th className="p-2 border-r border-slate-800 print:border-gray-400 w-12 text-center">पृष्ठ</th>
                        <th className="p-2">उद्धृत मूल अंश (Excerpt)</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800 print:divide-gray-300">
                      {citations.map((c, idx) => (
                        <tr key={idx} className="hover:bg-slate-900/40 print:hover:bg-transparent">
                          <td className="p-2 border-r border-slate-800 print:border-gray-400 text-center font-bold text-slate-400 print:text-gray-700">
                            {idx + 1}
                          </td>
                          <td className="p-2 border-r border-slate-800 print:border-gray-400 font-semibold text-sky-300 print:text-black font-mono">
                            {c.go_number}
                          </td>
                          <td className="p-2 border-r border-slate-800 print:border-gray-400 text-slate-300 print:text-gray-800">
                            {c.issuing_department}
                          </td>
                          <td className="p-2 border-r border-slate-800 print:border-gray-400 text-slate-400 print:text-gray-700">
                            {c.date}
                          </td>
                          <td className="p-2 border-r border-slate-800 print:border-gray-400 text-center font-mono text-slate-300 print:text-gray-800">
                            {c.page_number}
                          </td>
                          <td className="p-2 text-slate-300 print:text-gray-900 leading-relaxed italic">
                            "{c.exact_text_excerpt.slice(0, 160)}..."
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Official Sign-off Block */}
            <div className="mt-12 pt-8 border-t-2 border-slate-700 print:border-black grid grid-cols-2 gap-8 text-xs">
              <div className="text-left space-y-1 text-slate-400 print:text-gray-600">
                <p className="font-semibold text-slate-300 print:text-gray-800">प्रस्तुतकर्ता (Prepared by):</p>
                <p>PramanAI Evidentiary Agent Fleet</p>
                <p className="text-[10px] text-slate-500 print:text-gray-500">Autonomous Regulatory Intelligence • Gemini & GCP</p>
                <p className="text-[10px] text-slate-500 print:text-gray-500">Timestamp: {new Date().toISOString()}</p>
              </div>

              <div className="text-right space-y-8 text-slate-300 print:text-gray-800">
                <div>
                  <p className="font-bold">हस्ताक्षर / अनुमोदक अधिकारी</p>
                  <p className="text-[11px] text-slate-400 print:text-gray-600">(Reviewing / Competent Authority)</p>
                </div>
                <div className="inline-block w-48 border-b border-slate-500 print:border-black pt-6" />
                <div className="text-[10px] text-slate-400 print:text-gray-600">
                  <p>नाम / पदनाम (Name & Designation)</p>
                  <p>मुहर / Office Seal: [ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ]</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
