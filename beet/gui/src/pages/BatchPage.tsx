import { useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import clsx from "clsx";
import { apiClient } from "../api/client";
import type { BeetReport, Submission } from "../types";
import { DeterminationBadge } from "../components/DeterminationBadge";
import { detectorIdToFriendlyName } from "../api/client";

type DeterminationLabel = BeetReport["determination"];

const PROFILES = [
  { id: "screening" as const, label: "Quick Scan", description: "Fast, basic check" },
  { id: "default" as const, label: "Standard", description: "Recommended" },
  { id: "full" as const, label: "Deep Analysis", description: "Thorough" },
];

const DETERMINATION_ORDER: DeterminationLabel[] = [
  "GREEN",
  "YELLOW",
  "AMBER",
  "RED",
  "UNCERTAIN",
  "MIXED",
];

const DETERMINATION_COLOR_DOT: Record<DeterminationLabel, string> = {
  GREEN: "🟢",
  YELLOW: "🟡",
  AMBER: "🟠",
  RED: "🔴",
  UNCERTAIN: "⚪",
  MIXED: "🟣",
};

interface ResultRow {
  fileName: string;
  report: BeetReport;
}

export function BatchPage() {
  const navigate = useNavigate();
  const [files, setFiles] = useState<File[]>([]);
  const [profile, setProfile] = useState<"screening" | "default" | "full">("default");
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState({ completed: 0, total: 0 });
  const [results, setResults] = useState<ResultRow[]>([]);
  const [filter, setFilter] = useState<DeterminationLabel | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragOver(false);
    const dropped = Array.from(e.dataTransfer.files).filter((f) => {
      const ext = f.name.split(".").pop()?.toLowerCase();
      return ["txt", "md", "csv"].includes(ext ?? "");
    });
    setFiles((prev) => [...prev, ...dropped]);
  }

  function handleFileSelect(e: React.ChangeEvent<HTMLInputElement>) {
    const selected = Array.from(e.target.files ?? []);
    setFiles((prev) => [...prev, ...selected]);
  }

  async function handleAnalyzeAll() {
    if (!files.length) return;
    setError(null);
    setLoading(true);
    setResults([]);
    setProgress({ completed: 0, total: files.length });

    try {
      const submissions: { file: File; sub: Submission }[] = await Promise.all(
        files.map(async (f) => ({
          file: f,
          sub: { text: await f.text(), id: f.name },
        }))
      );

      const newResults: ResultRow[] = [];
      for (let i = 0; i < submissions.length; i++) {
        const { file, sub } = submissions[i];
        try {
          const report = await apiClient.analyze(sub.text, { profile });
          newResults.push({ fileName: file.name, report });
        } catch {
          // Skip failed items
        }
        setProgress({ completed: i + 1, total: files.length });
        setResults([...newResults]);
      }
    } catch (err: unknown) {
      const msg =
        err instanceof Error && "userMessage" in err
          ? (err as { userMessage: string }).userMessage
          : "Something went wrong.";
      setError(msg);
    } finally {
      setLoading(false);
    }
  }

  function exportCSV() {
    const rows = [
      ["File Name", "Result", "Confidence", "Top Signal"],
      ...results.map((r) => [
        r.fileName,
        r.report.determination,
        `${Math.round(r.report.p_llm * 100)}%`,
        r.report.top_signals?.[0]
          ? detectorIdToFriendlyName(r.report.top_signals[0])
          : "",
      ]),
    ];
    const csv = rows.map((r) => r.map((c) => `"${c}"`).join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `beet-batch-${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  const filteredResults = filter
    ? results.filter((r) => r.report.determination === filter)
    : results;

  const counts = DETERMINATION_ORDER.reduce(
    (acc, d) => {
      acc[d] = results.filter((r) => r.report.determination === d).length;
      return acc;
    },
    {} as Record<DeterminationLabel, number>
  );

  const flaggedItems = results.filter((r) =>
    ["AMBER", "RED"].includes(r.report.determination)
  );

  return (
    <div className="min-h-screen bg-[#FAFAF8]">
      <main className="mx-auto max-w-4xl px-4 py-10 space-y-8">
        <h1 className="text-2xl font-bold text-gray-900">Batch Analysis</h1>

        {error && (
          <div
            role="alert"
            className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800"
          >
            ⚠️ {error}
          </div>
        )}

        {/* Upload section */}
        <div className="rounded-2xl bg-white p-6 shadow-sm ring-1 ring-gray-200 space-y-5">
          {/* Drop zone */}
          <div
            className={clsx(
              "flex flex-col items-center justify-center rounded-xl border-2 border-dashed p-10 text-center cursor-pointer transition-colors",
              dragOver
                ? "border-[#0D7377] bg-[#0D7377]/5"
                : "border-gray-300 hover:border-[#0D7377] hover:bg-gray-50"
            )}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            role="button"
            tabIndex={0}
            aria-label="Drop files here or click to browse"
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") fileInputRef.current?.click();
            }}
          >
            <span className="text-4xl mb-3">📂</span>
            <p className="font-medium text-gray-700">
              Drop files here or click to browse
            </p>
            <p className="mt-1 text-sm text-gray-500">
              Accepts .txt, .md, .csv files
            </p>
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".txt,.md,.csv"
              className="hidden"
              onChange={handleFileSelect}
              aria-label="Select files to analyze"
            />
          </div>

          {/* File list */}
          {files.length > 0 && (
            <ul className="space-y-1">
              {files.map((f, i) => (
                <li
                  key={i}
                  className="flex items-center justify-between text-sm text-gray-600"
                >
                  <span>📄 {f.name}</span>
                  <button
                    onClick={() =>
                      setFiles((prev) => prev.filter((_, j) => j !== i))
                    }
                    className="text-red-400 hover:text-red-600 focus:outline-none"
                    aria-label={`Remove ${f.name}`}
                  >
                    ✕
                  </button>
                </li>
              ))}
            </ul>
          )}

          {/* Profile */}
          <div>
            <span className="block text-sm font-medium text-gray-700 mb-2">
              Analysis depth
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

          <button
            onClick={handleAnalyzeAll}
            disabled={!files.length || loading}
            className="w-full rounded-xl bg-[#0D7377] py-3 font-semibold text-white hover:bg-[#0b5e62] focus:outline-none focus:ring-2 focus:ring-[#0D7377] focus:ring-offset-2 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {loading
              ? `Analyzing ${progress.completed} of ${progress.total} files...`
              : `Analyze ${files.length || 0} File${files.length !== 1 ? "s" : ""}`}
          </button>

          {/* Progress bar */}
          {loading && (
            <div
              className="h-2 rounded-full bg-gray-200 overflow-hidden"
              role="progressbar"
              aria-valuenow={progress.completed}
              aria-valuemax={progress.total}
              aria-label="Analysis progress"
            >
              <div
                className="h-full rounded-full bg-[#0D7377] transition-all"
                style={{
                  width: `${
                    progress.total
                      ? (progress.completed / progress.total) * 100
                      : 0
                  }%`,
                }}
              />
            </div>
          )}
        </div>

        {/* Results dashboard */}
        {results.length > 0 && (
          <div className="space-y-6">
            {/* Summary cards */}
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
              <div className="rounded-xl bg-white p-4 shadow-sm ring-1 ring-gray-200 text-center">
                <div className="text-3xl font-bold text-gray-900">
                  {results.length}
                </div>
                <div className="text-sm text-gray-500">Total analyzed</div>
              </div>
              {DETERMINATION_ORDER.filter((d) => counts[d] > 0).map((d) => (
                <button
                  key={d}
                  onClick={() => setFilter(filter === d ? null : d)}
                  className={clsx(
                    "rounded-xl bg-white p-4 shadow-sm ring-1 ring-gray-200 text-center hover:ring-[#0D7377] transition-all focus:outline-none focus:ring-2 focus:ring-[#0D7377]",
                    filter === d && "ring-[#0D7377] ring-2"
                  )}
                >
                  <div className="text-2xl font-bold text-gray-900">
                    {counts[d]}
                  </div>
                  <div className="text-xs text-gray-500">
                    {DETERMINATION_COLOR_DOT[d]}{" "}
                    {Math.round((counts[d] / results.length) * 100)}%
                  </div>
                </button>
              ))}
            </div>

            {/* Stacked bar */}
            <div
              className="flex h-6 overflow-hidden rounded-full"
              role="img"
              aria-label="Distribution of results"
            >
              {DETERMINATION_ORDER.filter((d) => counts[d] > 0).map((d) => {
                const colorMap: Record<DeterminationLabel, string> = {
                  GREEN: "bg-green-500",
                  YELLOW: "bg-yellow-400",
                  AMBER: "bg-orange-500",
                  RED: "bg-red-600",
                  UNCERTAIN: "bg-gray-400",
                  MIXED: "bg-purple-500",
                };
                return (
                  <div
                    key={d}
                    className={clsx(colorMap[d], "transition-all")}
                    style={{
                      width: `${(counts[d] / results.length) * 100}%`,
                    }}
                    title={`${d}: ${counts[d]}`}
                  />
                );
              })}
            </div>

            {/* Filter toggles */}
            {filter && (
              <button
                onClick={() => setFilter(null)}
                className="text-sm text-[#0D7377] underline hover:text-[#0b5e62] focus:outline-none"
              >
                Clear filter
              </button>
            )}

            {/* Results table */}
            <div className="rounded-2xl bg-white shadow-sm ring-1 ring-gray-200 overflow-hidden">
              <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100">
                <h2 className="font-semibold text-gray-900">
                  Results ({filteredResults.length})
                </h2>
                <div className="flex gap-2">
                  <button
                    onClick={exportCSV}
                    className="rounded-lg border border-gray-200 px-3 py-1.5 text-sm text-gray-600 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-[#0D7377] focus:ring-offset-1"
                  >
                    Export CSV
                  </button>
                </div>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="bg-gray-50 text-left">
                    <tr>
                      <th className="px-4 py-3 font-medium text-gray-600">
                        File
                      </th>
                      <th className="px-4 py-3 font-medium text-gray-600">
                        Result
                      </th>
                      <th className="px-4 py-3 font-medium text-gray-600">
                        Confidence
                      </th>
                      <th className="px-4 py-3 font-medium text-gray-600">
                        Top Signal
                      </th>
                      <th className="px-4 py-3 font-medium text-gray-600">
                        Actions
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100">
                    {filteredResults.map((row, i) => (
                      <tr key={i} className="hover:bg-gray-50">
                        <td className="px-4 py-3 font-medium text-gray-900 max-w-[180px] truncate">
                          {row.fileName}
                        </td>
                        <td className="px-4 py-3">
                          <DeterminationBadge
                            determination={row.report.determination}
                            size="sm"
                          />
                        </td>
                        <td className="px-4 py-3 text-gray-600">
                          {Math.round(row.report.p_llm * 100)}% AI
                        </td>
                        <td className="px-4 py-3 text-gray-600">
                          {row.report.top_signals?.[0]
                            ? detectorIdToFriendlyName(
                                row.report.top_signals[0]
                              )
                            : "—"}
                        </td>
                        <td className="px-4 py-3">
                          <button
                            onClick={() =>
                              navigate("/results", {
                                state: { report: row.report },
                              })
                            }
                            className="text-[#0D7377] underline hover:text-[#0b5e62] focus:outline-none focus:ring-1 focus:ring-[#0D7377] rounded"
                          >
                            View
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Flagged items */}
            {flaggedItems.length > 0 && (
              <details className="rounded-2xl bg-white shadow-sm ring-1 ring-gray-200">
                <summary className="cursor-pointer px-5 py-4 font-semibold text-gray-900">
                  ⚠️ {flaggedItems.length} item
                  {flaggedItems.length !== 1 ? "s" : ""} need review
                </summary>
                <div className="px-5 pb-4 space-y-2">
                  {flaggedItems.map((row, i) => (
                    <div
                      key={i}
                      className="flex items-center justify-between rounded-lg border border-gray-200 p-3"
                    >
                      <div>
                        <span className="font-medium text-gray-800">
                          {row.fileName}
                        </span>
                        <span className="ml-2 text-sm text-gray-500">
                          {DETERMINATION_COLOR_DOT[row.report.determination]}{" "}
                          {row.report.determination}
                        </span>
                      </div>
                      <button
                        onClick={() =>
                          navigate("/results", {
                            state: { report: row.report },
                          })
                        }
                        className="text-sm text-[#0D7377] underline hover:text-[#0b5e62] focus:outline-none"
                      >
                        Review
                      </button>
                    </div>
                  ))}
                </div>
              </details>
            )}
          </div>
        )}

        {/* Empty state */}
        {!loading && results.length === 0 && files.length === 0 && (
          <div className="text-center py-16 text-gray-400">
            <div className="text-5xl mb-3">📂</div>
            <p>Drop files here to get started</p>
          </div>
        )}
      </main>
    </div>
  );
}
