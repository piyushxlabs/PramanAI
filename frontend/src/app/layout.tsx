import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PramanAI — Autonomous Evidentiary GovTech Agent Fleet",
  description:
    "Autonomous Evidentiary GovTech Agent Fleet with Multi-Modal Grounding on Gemini & Google Cloud.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased bg-[#0B1120] text-slate-100 min-h-screen">
        {children}
      </body>
    </html>
  );
}
