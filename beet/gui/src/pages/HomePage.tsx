import { useState, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import clsx from "clsx";
import { apiClient } from "../api/client";
import type { BeetReport, AnalyzeOptions } from "../types";
import { Tooltip } from "../components/Tooltip";

const LOADING_PHASES = [
  "Scanning for obvious patterns...",
  "Running deeper analysis...",
  "Performing thorough examination...",
];

const PROFILES = [
  {
    id: "screening" as const,
    label: "Quick Scan",
    description: "Fast, basic check",
  },
  {
    id: "default" as const,
    label: "Standard",
    description: "Recommended for most text",
  },
  {
    id: "full" as const,
    label: "Deep Analysis",
    description: "Thorough, takes longer",
  },
];

const COMMON_OCCUPATIONS = [
  "Healthcare",
  "Education",
  "Engineering",
  "Legal",
  "Marketing",
  "Finance",
  "Research",
  "Other",
];

export function HomePage() {
  const navigate = useNavigate();
  const [text, setText] = useState("");
  const [profile, setProfile] = useState<"screening" | "default" | "full">(
    "default"
  );
  const [occupation, setOccupation] = useState("");
  const [showOptions, setShowOptions] = useState(false);
  const [loading, setLoading] = useState(false);
  const [phaseIndex, setPhaseIndex] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const phaseTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const wordCount = text.trim().split(/\s+/).filter(Boolean).length;

  // Clean up timer on unmount
  useEffect(() => {
    return () => {
      if (phaseTimer.current) clearInterval(phaseTimer.current);
    };
  }, []);

  // Check onboarding (first-time user)
  const [showOnboarding, setShowOnboarding] = useState(
    () => !localStorage.getItem("beet_onboarded")
  );

  function dismissOnboarding() {
    localStorage.setItem("beet_onboarded", "1");
    setShowOnboarding(false);
  }

  async function handleAnalyze() {
    if (!text.trim()) return;
    setError(null);
    setLoading(true);
    setPhaseIndex(0);

    phaseTimer.current = setInterval(() => {
      setPhaseIndex((prev) =>
        prev < LOADING_PHASES.length - 1 ? prev + 1 : prev
      );
    }, 3000);

    try {
      const opts: AnalyzeOptions = { profile, occupation: occupation || undefined };
      const report: BeetReport = await apiClient.analyze(text, opts);

      // Save to history
      const history = JSON.parse(
        localStorage.getItem("beet_history") ?? "[]"
      ) as object[];
      history.unshift({
        id: Date.now(),
        createdAt: new Date().toISOString(),
        excerpt: text.slice(0, 80),
        determination: report.determination,
        pLlm: report.p_llm,
        report,
      });
      localStorage.setItem(
        "beet_history",
        JSON.stringify(history.slice(0, 50))
      );

      navigate("/results", { state: { report } });
    } catch (err: unknown) {
      const msg =
        err instanceof Error && "userMessage" in err
          ? (err as { userMessage: string }).userMessage
          : "Something went wrong. Please try again.";
      setError(msg);
    } finally {
      setLoading(false);
      if (phaseTimer.current) clearInterval(phaseTimer.current);
    }
  }

  async function handleFileUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;

    const ext = file.name.split(".").pop()?.toLowerCase();
    if (!["txt", "md", "docx"].includes(ext ?? "")) {
      setError(
        "We can check .txt, .md, and .docx files. This file type isn't supported yet."
      );
      return;
    }
    if (file.size > 5 * 1024 * 1024) {
      setError(
        "This file is very large. Try splitting it into smaller pieces."
      );
      return;
    }

    try {
      const content = await file.text();
      setText(content);
      setError(null);
    } catch {
      setError("We couldn't read this file. It may be corrupted.");
    }
  }

  return (
    <div className="relative min-h-screen bg-[#FAFAF8]">
      {/* Onboarding overlay */}
      {showOnboarding && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
          role="dialog"
          aria-modal="true"
          aria-label="Welcome to BEET"
        >
          <div className="w-full max-w-sm rounded-2xl bg-white p-8 shadow-2xl">
            <h2 className="mb-2 text-2xl font-bold text-gray-900">
              Welcome to BEET
            </h2>
            <p className="mb-6 text-gray-600">
              Paste any text to check if it was written by a human or AI.
            </p>
            <ol className="mb-6 space-y-3 text-gray-700">
              {[
                ["📋", "Paste text or upload a file"],
                ["🔍", "BEET analyzes the writing patterns"],
                ["✅", "Get a clear result with explanation"],
              ].map(([icon, label]) => (
                <li key={label} className="flex items-start gap-3">
                  <span className="text-xl">{icon}</span>
                  <span>{label}</span>
                </li>
              ))}
            </ol>
            <button
              onClick={dismissOnboarding}
              className="w-full rounded-xl bg-[#0D7377] py-3 font-semibold text-white hover:bg-[#0b5e62] focus:outline-none focus:ring-2 focus:ring-[#0D7377] focus:ring-offset-2"
            >
              Got it
            </button>
          </div>
        </div>
      )}

      <main className="mx-auto max-w-2xl px-4 py-16">
        {/* Header */}
        <div className="mb-10 text-center">
          <div className="mb-4 flex justify-center">
            <span
              className="flex h-14 w-14 items-center justify-center rounded-full bg-[#0D7377] text-white text-2xl font-black shadow-md"
              aria-hidden="true"
            >
              B
            </span>
          </div>
          <h1 className="text-4xl font-bold text-gray-900">BEET</h1>
          <p className="mt-2 text-gray-500">
            Check if text was written by a human or AI
          </p>
        </div>

        {/* API error banner */}
        {error && (
          <div
            role="alert"
            className="mb-4 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800"
          >
            ⚠️ {error}
            {error.includes("connect") && (
              <details className="mt-2">
                <summary className="cursor-pointer font-medium">
                  How to start the server
                </summary>
                <pre className="mt-2 rounded bg-amber-100 p-2 text-xs">
                  beet serve
                </pre>
              </details>
            )}
            <button
              className="ml-2 underline"
              onClick={() => setError(null)}
              aria-label="Dismiss error"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* Textarea */}
        <div className="rounded-2xl bg-white p-6 shadow-sm ring-1 ring-gray-200">
          <label htmlFor="text-input" className="sr-only">
            Text to analyze
          </label>
          <textarea
            id="text-input"
            rows={8}
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Paste the text you want to check here..."
            className="w-full resize-none text-gray-900 placeholder-gray-400 focus:outline-none text-base leading-relaxed"
            disabled={loading}
            aria-label="Text to analyze"
          />

          {/* Word count hint */}
          {text && wordCount < 150 && (
            <p className="mt-2 text-sm text-amber-600" role="status">
              This text is quite short ({wordCount} words). Results may be less
              reliable. For best results, paste at least 150 words.
            </p>
          )}
          {text && (
            <p className="mt-1 text-right text-xs text-gray-400">
              {wordCount} words
            </p>
          )}
        </div>

        {/* Options */}
        <div className="mt-4">
          <button
            type="button"
            onClick={() => setShowOptions((v) => !v)}
            className="text-sm text-gray-500 hover:text-gray-700 focus:outline-none"
            aria-expanded={showOptions}
          >
            Options {showOptions ? "▴" : "▾"}
          </button>

          {showOptions && (
            <div className="mt-3 rounded-xl bg-white p-5 shadow-sm ring-1 ring-gray-200 space-y-5">
              {/* Occupation */}
              <div>
                <label
                  htmlFor="occupation-select"
                  className="block text-sm font-medium text-gray-700"
                >
                  What field is this from?
                </label>
                <select
                  id="occupation-select"
                  value={occupation}
                  onChange={(e) => setOccupation(e.target.value)}
                  className="mt-1 w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm focus:border-[#0D7377] focus:outline-none focus:ring-1 focus:ring-[#0D7377]"
                >
                  <option value="">Select occupation…</option>
                  {COMMON_OCCUPATIONS.map((o) => (
                    <option key={o} value={o}>
                      {o}
                    </option>
                  ))}
                </select>
              </div>

              {/* Profile */}
              <div>
                <span className="block text-sm font-medium text-gray-700 mb-2">
                  Analysis depth{" "}
                  <Tooltip content="Quick Scan is fast. Standard is the default recommendation. Deep Analysis is thorough but takes longer.">
                    <button type="button" className="ml-1 cursor-help text-gray-400 text-xs border border-gray-300 rounded-full px-1.5 py-0.5 hover:border-gray-400 focus:outline-none focus:ring-1 focus:ring-[#0D7377]" aria-label="Help for analysis depth">?</button>
                  </Tooltip>
                </span>
                <div className="flex gap-2" role="radiogroup" aria-label="Analysis depth">
                  {PROFILES.map((p) => (
                    <button
                      key={p.id}
                      type="button"
                      role="radio"
                      aria-checked={profile === p.id}
                      onClick={() => setProfile(p.id)}
                      className={clsx(
                        "flex-1 rounded-xl border-2 py-2 px-3 text-left text-sm transition-colors focus:outline-none focus:ring-2 focus:ring-[#0D7377] focus:ring-offset-1",
                        profile === p.id
                          ? "border-[#0D7377] bg-[#0D7377]/5 text-[#0D7377]"
                          : "border-gray-200 text-gray-700 hover:border-gray-300"
                      )}
                    >
                      <div className="font-semibold">{p.label}</div>
                      <div className="text-xs opacity-70">{p.description}</div>
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Action area */}
        <div className="mt-6 text-center">
          {loading ? (
            <div className="space-y-3">
              <div
                className="inline-flex items-center gap-3 rounded-xl bg-[#0D7377]/10 px-6 py-4 text-[#0D7377]"
                role="status"
                aria-live="polite"
              >
                <svg
                  className="h-5 w-5 animate-spin"
                  viewBox="0 0 24 24"
                  fill="none"
                  aria-hidden="true"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  />
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8v8z"
                  />
                </svg>
                <span>{LOADING_PHASES[phaseIndex]}</span>
              </div>
              {phaseIndex === 2 && (
                <p className="text-sm text-gray-500">
                  Deep analysis takes a bit longer — we're examining the
                  statistical patterns in the text.
                </p>
              )}
            </div>
          ) : (
            <>
              <button
                type="button"
                onClick={handleAnalyze}
                disabled={!text.trim()}
                className="rounded-xl bg-[#0D7377] px-10 py-4 text-lg font-semibold text-white shadow-md transition-all hover:bg-[#0b5e62] focus:outline-none focus:ring-2 focus:ring-[#0D7377] focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-40"
                aria-label="Analyze the pasted text"
              >
                Check This Text
              </button>
              <div className="mt-3">
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="text-sm text-gray-500 underline hover:text-[#0D7377] focus:outline-none focus:text-[#0D7377]"
                >
                  or upload a file
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".txt,.md,.docx"
                  className="hidden"
                  onChange={handleFileUpload}
                  aria-label="Upload a text file"
                />
              </div>
            </>
          )}
        </div>
      </main>
    </div>
  );
}
