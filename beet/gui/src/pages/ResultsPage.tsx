import { useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import type { BeetReport } from "../types";
import {
  DeterminationBadge,
  ConfidenceMeter,
  ActionGuidance,
} from "../components/DeterminationBadge";
import {
  detectorIdToFriendlyName,
  detectorIdToIcon,
  SIGNAL_PLAIN_EXPLANATION,
} from "../api/client";

function SignalCard({ signalId }: { signalId: string }) {
  const [open, setOpen] = useState(false);
  const icon = detectorIdToIcon(signalId);
  const name = detectorIdToFriendlyName(signalId);
  const explanation =
    SIGNAL_PLAIN_EXPLANATION[signalId] ??
    "This signal contributed to the overall determination.";

  return (
    <div className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
      <div className="flex items-start gap-3">
        <span className="text-2xl" aria-hidden="true">
          {icon}
        </span>
        <div className="flex-1">
          <h4 className="font-semibold text-gray-900">{name}</h4>
          <p className="mt-1 text-sm text-gray-600">{explanation}</p>
          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            className="mt-2 text-xs text-gray-400 underline hover:text-gray-600 focus:outline-none"
            aria-expanded={open}
          >
            Technical details {open ? "▴" : "▾"}
          </button>
          {open && (
            <pre className="mt-2 rounded bg-gray-100 p-2 text-xs font-mono text-gray-700 overflow-x-auto">
              {signalId}
            </pre>
          )}
        </div>
      </div>
    </div>
  );
}

function copyToClipboard(text: string) {
  navigator.clipboard.writeText(text).catch(() => {
    /* fallback ignored */
  });
}

export function ResultsPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const report = location.state?.report as BeetReport | undefined;
  const [showJson, setShowJson] = useState(false);
  const [copied, setCopied] = useState(false);

  if (!report) {
    return (
      <div className="mx-auto max-w-2xl px-4 py-20 text-center">
        <p className="text-gray-500">No result to display.</p>
        <button
          onClick={() => navigate("/")}
          className="mt-4 rounded-xl bg-[#0D7377] px-6 py-3 font-semibold text-white hover:bg-[#0b5e62]"
        >
          Check a Text
        </button>
      </div>
    );
  }

  const isNegative = ["AMBER", "RED"].includes(report.determination);
  const whyHeader = isNegative
    ? "Why did we flag this?"
    : "Why does this look human?";

  const topSignals = report.top_signals?.slice(0, 3) ?? [];

  function handleCopySummary() {
    const label =
      report!.determination === "GREEN"
        ? "Likely Human-Written"
        : report!.determination === "RED"
        ? "Strong AI Indicators"
        : report!.determination;
    const summary = `BEET Analysis Result\n${label} (${Math.round(
      report!.p_llm * 100
    )}% likely AI-generated)\nTop signals: ${topSignals
      .map(detectorIdToFriendlyName)
      .join(", ")}`;
    copyToClipboard(summary);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  function handleDownload() {
    const blob = new Blob([JSON.stringify(report, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `beet-report-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="min-h-screen bg-[#FAFAF8]">
      <main className="mx-auto max-w-2xl px-4 py-10 space-y-6">
        {/* Determination badge */}
        <div className="rounded-2xl bg-white p-6 shadow-sm ring-1 ring-gray-200 space-y-5">
          <DeterminationBadge determination={report.determination} size="lg" />
          <ConfidenceMeter
            pLlm={report.p_llm}
            confidenceLow={report.confidence_low}
            confidenceHigh={report.confidence_high}
          />
        </div>

        {/* Why this result */}
        {topSignals.length > 0 && (
          <div className="rounded-2xl bg-white p-6 shadow-sm ring-1 ring-gray-200">
            <h2 className="mb-4 text-lg font-semibold text-gray-900">
              {whyHeader}
            </h2>
            <div className="space-y-3">
              {topSignals.map((sig) => (
                <SignalCard key={sig} signalId={sig} />
              ))}
            </div>
          </div>
        )}

        {/* What to do next */}
        <ActionGuidance determination={report.determination} />

        {/* Action buttons */}
        <div className="flex flex-wrap gap-3">
          <button
            onClick={() => navigate("/")}
            className="flex-1 rounded-xl bg-[#0D7377] px-5 py-3 font-semibold text-white hover:bg-[#0b5e62] focus:outline-none focus:ring-2 focus:ring-[#0D7377] focus:ring-offset-2"
          >
            Check Another Text
          </button>
          <button
            onClick={handleDownload}
            className="rounded-xl border-2 border-gray-200 bg-white px-5 py-3 font-semibold text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-gray-300 focus:ring-offset-2"
          >
            Download Report
          </button>
          <button
            onClick={handleCopySummary}
            className="rounded-xl border-2 border-gray-200 bg-white px-5 py-3 font-semibold text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-gray-300 focus:ring-offset-2"
          >
            {copied ? "Copied!" : "Copy Summary"}
          </button>
        </div>

        {/* Full technical report */}
        <div>
          <button
            type="button"
            onClick={() => setShowJson((v) => !v)}
            className="text-sm text-gray-400 underline hover:text-gray-600 focus:outline-none"
            aria-expanded={showJson}
          >
            View Full Technical Report {showJson ? "▴" : "▾"}
          </button>
          {showJson && (
            <pre className="mt-2 overflow-x-auto rounded-xl bg-gray-900 p-4 text-xs text-green-300 font-mono">
              {JSON.stringify(report, null, 2)}
            </pre>
          )}
        </div>
      </main>
    </div>
  );
}
