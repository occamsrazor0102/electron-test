import { useState } from "react";
import { useNavigate } from "react-router-dom";
import type { BeetReport } from "../types";
import { DeterminationBadge } from "../components/DeterminationBadge";

interface HistoryEntry {
  id: number;
  createdAt: string;
  excerpt: string;
  determination: BeetReport["determination"];
  pLlm: number;
  report: BeetReport;
}

export function HistoryPage() {
  const navigate = useNavigate();
  const [history, setHistory] = useState<HistoryEntry[]>(() => {
    try {
      return JSON.parse(localStorage.getItem("beet_history") ?? "[]") as HistoryEntry[];
    } catch {
      return [];
    }
  });

  function clearHistory() {
    localStorage.removeItem("beet_history");
    setHistory([]);
  }

  if (history.length === 0) {
    return (
      <div className="min-h-screen bg-[#FAFAF8]">
        <main className="mx-auto max-w-2xl px-4 py-20 text-center">
          <div className="text-6xl mb-4">📋</div>
          <h1 className="text-xl font-semibold text-gray-700 mb-2">
            No analyses yet
          </h1>
          <p className="text-gray-500 mb-6">
            Check some text to get started!
          </p>
          <button
            onClick={() => navigate("/")}
            className="rounded-xl bg-[#0D7377] px-6 py-3 font-semibold text-white hover:bg-[#0b5e62] focus:outline-none focus:ring-2 focus:ring-[#0D7377] focus:ring-offset-2"
          >
            Check Text
          </button>
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#FAFAF8]">
      <main className="mx-auto max-w-3xl px-4 py-10 space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold text-gray-900">History</h1>
          <button
            onClick={clearHistory}
            className="rounded-lg border border-red-200 px-3 py-1.5 text-sm text-red-600 hover:bg-red-50 focus:outline-none focus:ring-2 focus:ring-red-400 focus:ring-offset-1"
          >
            Clear History
          </button>
        </div>

        <p className="text-sm text-gray-500 rounded-lg bg-gray-50 px-4 py-3 border border-gray-200">
          🔒 History is stored only in your browser. It is not sent anywhere.
        </p>

        <div className="rounded-2xl bg-white shadow-sm ring-1 ring-gray-200 overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-gray-50 text-left">
              <tr>
                <th className="px-4 py-3 font-medium text-gray-600">
                  Date/Time
                </th>
                <th className="px-4 py-3 font-medium text-gray-600">
                  Text Preview
                </th>
                <th className="px-4 py-3 font-medium text-gray-600">
                  Result
                </th>
                <th className="px-4 py-3 font-medium text-gray-600">
                  Confidence
                </th>
                <th className="px-4 py-3"></th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {history.map((entry) => (
                <tr key={entry.id} className="hover:bg-gray-50">
                  <td className="px-4 py-3 text-gray-500 whitespace-nowrap">
                    {new Date(entry.createdAt).toLocaleString()}
                  </td>
                  <td className="px-4 py-3 text-gray-700 max-w-[220px] truncate">
                    {entry.excerpt}
                  </td>
                  <td className="px-4 py-3">
                    <DeterminationBadge
                      determination={entry.determination}
                      size="sm"
                    />
                  </td>
                  <td className="px-4 py-3 text-gray-600">
                    {Math.round(entry.pLlm * 100)}% AI
                  </td>
                  <td className="px-4 py-3">
                    <button
                      onClick={() =>
                        navigate("/results", {
                          state: { report: entry.report },
                        })
                      }
                      className="text-[#0D7377] underline hover:text-[#0b5e62] focus:outline-none"
                    >
                      View
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </main>
    </div>
  );
}
