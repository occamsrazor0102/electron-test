import { useState, useEffect } from "react";
import { BeetApiClient } from "../api/client";
import type { HealthStatus } from "../types";

// Sourced from package.json to keep in sync with the published version
const APP_VERSION = "2.0.0";

const PROFILES = [
  {
    id: "screening",
    label: "Quick Scan",
    description: "A fast check that catches obvious AI patterns. Good for a first pass but may miss subtle cases.",
  },
  {
    id: "default",
    label: "Standard",
    description: "The recommended setting for most text. Balances speed and accuracy.",
  },
  {
    id: "full",
    label: "Deep Analysis",
    description: "Thorough examination using all available detectors. Takes longer.",
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

const DETECTOR_TIERS: {
  label: string;
  key: string;
  tier: string;
  available: boolean;
}[] = [
  { label: "Basic Pattern Detection", key: "preamble", tier: "Tier 1", available: true },
  { label: "Statistical Analysis", key: "nssi", tier: "Tier 2", available: true },
  { label: "Deep AI Comparison", key: "contrastive_gen", tier: "Tier 3", available: false },
];

export function SettingsPage() {
  const [defaultProfile, setDefaultProfile] = useState(
    () => localStorage.getItem("beet_default_profile") ?? "default"
  );
  const [defaultOccupation, setDefaultOccupation] = useState(
    () => localStorage.getItem("beet_default_occupation") ?? ""
  );
  const [apiUrl, setApiUrl] = useState(
    () => localStorage.getItem("beet_api_url") ?? "http://localhost:8000"
  );
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [testing, setTesting] = useState(false);

  useEffect(() => {
    // Save settings as they change
    localStorage.setItem("beet_default_profile", defaultProfile);
  }, [defaultProfile]);

  useEffect(() => {
    localStorage.setItem("beet_default_occupation", defaultOccupation);
  }, [defaultOccupation]);

  useEffect(() => {
    localStorage.setItem("beet_api_url", apiUrl);
  }, [apiUrl]);

  async function testConnection() {
    setTesting(true);
    setHealth(null);
    try {
      const client = new BeetApiClient(apiUrl);
      const h = await client.checkHealth();
      setHealth(h);
    } catch {
      setHealth({ connected: false, version: "unknown", availableDetectors: [] });
    } finally {
      setTesting(false);
    }
  }

  return (
    <div className="min-h-screen bg-[#FAFAF8]">
      <main className="mx-auto max-w-2xl px-4 py-10 space-y-8">
        <h1 className="text-2xl font-bold text-gray-900">Settings</h1>

        {/* Default options */}
        <section className="rounded-2xl bg-white p-6 shadow-sm ring-1 ring-gray-200 space-y-5">
          <h2 className="text-lg font-semibold text-gray-900">Default Options</h2>

          {/* Default profile */}
          <div>
            <span className="block text-sm font-medium text-gray-700 mb-2">
              Default analysis depth
            </span>
            <div className="space-y-2" role="radiogroup" aria-label="Default analysis depth">
              {PROFILES.map((p) => (
                <label
                  key={p.id}
                  className="flex cursor-pointer items-start gap-3 rounded-xl border-2 p-4 transition-colors hover:bg-gray-50"
                  style={{
                    borderColor: defaultProfile === p.id ? "#0D7377" : "#e5e7eb",
                  }}
                >
                  <input
                    type="radio"
                    name="profile"
                    value={p.id}
                    checked={defaultProfile === p.id}
                    onChange={() => setDefaultProfile(p.id)}
                    className="mt-0.5 accent-[#0D7377]"
                  />
                  <div>
                    <div className="font-semibold text-gray-900">{p.label}</div>
                    <div className="text-sm text-gray-500">{p.description}</div>
                  </div>
                </label>
              ))}
            </div>
          </div>

          {/* Default occupation */}
          <div>
            <label
              htmlFor="default-occupation"
              className="block text-sm font-medium text-gray-700"
            >
              Default occupation
            </label>
            <select
              id="default-occupation"
              value={defaultOccupation}
              onChange={(e) => setDefaultOccupation(e.target.value)}
              className="mt-1 w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm focus:border-[#0D7377] focus:outline-none focus:ring-1 focus:ring-[#0D7377]"
            >
              <option value="">None</option>
              {COMMON_OCCUPATIONS.map((o) => (
                <option key={o} value={o}>
                  {o}
                </option>
              ))}
            </select>
          </div>
        </section>

        {/* API Connection */}
        <details className="rounded-2xl bg-white shadow-sm ring-1 ring-gray-200">
          <summary className="cursor-pointer px-6 py-5 text-lg font-semibold text-gray-900">
            API Connection <span className="text-sm font-normal text-gray-500">(Advanced)</span>
          </summary>
          <div className="border-t border-gray-100 px-6 pb-6 pt-4 space-y-4">
            <div>
              <label
                htmlFor="api-url"
                className="block text-sm font-medium text-gray-700"
              >
                API server URL
              </label>
              <input
                id="api-url"
                type="url"
                value={apiUrl}
                onChange={(e) => setApiUrl(e.target.value)}
                className="mt-1 w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm font-mono focus:border-[#0D7377] focus:outline-none focus:ring-1 focus:ring-[#0D7377]"
                placeholder="http://localhost:8000"
              />
            </div>

            <div className="flex items-center gap-3">
              <button
                onClick={testConnection}
                disabled={testing}
                className="rounded-xl bg-[#0D7377] px-5 py-2 text-sm font-semibold text-white hover:bg-[#0b5e62] focus:outline-none focus:ring-2 focus:ring-[#0D7377] focus:ring-offset-2 disabled:opacity-50"
              >
                {testing ? "Testing..." : "Test Connection"}
              </button>

              {health !== null && (
                <span
                  className={`flex items-center gap-1.5 text-sm font-medium ${
                    health.connected ? "text-green-600" : "text-red-600"
                  }`}
                  role="status"
                >
                  <span
                    className={`h-2.5 w-2.5 rounded-full ${
                      health.connected ? "bg-green-500" : "bg-red-500"
                    }`}
                    aria-hidden="true"
                  />
                  {health.connected
                    ? `Connected — v${health.version}`
                    : "Not connected"}
                </span>
              )}
            </div>
          </div>
        </details>

        {/* About */}
        <section className="rounded-2xl bg-white p-6 shadow-sm ring-1 ring-gray-200 space-y-4">
          <h2 className="text-lg font-semibold text-gray-900">About</h2>
          <p className="text-sm text-gray-500">BEET v{APP_VERSION}</p>

          <div>
            <h3 className="text-sm font-medium text-gray-700 mb-2">
              Available detectors
            </h3>
            <ul className="space-y-1.5">
              {DETECTOR_TIERS.map((d) => (
                <li key={d.key} className="flex items-center gap-2 text-sm">
                  <span
                    className={d.available ? "text-green-600" : "text-gray-400"}
                    aria-hidden="true"
                  >
                    {d.available ? "✓" : "✗"}
                  </span>
                  <span
                    className={
                      d.available ? "text-gray-800" : "text-gray-400"
                    }
                  >
                    {d.label}
                  </span>
                  <span className="text-xs text-gray-400">({d.tier})</span>
                  {!d.available && (
                    <span className="text-xs text-gray-400">
                      — requires additional setup
                    </span>
                  )}
                </li>
              ))}
            </ul>
          </div>
        </section>
      </main>
    </div>
  );
}
