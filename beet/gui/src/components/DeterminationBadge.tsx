import type { DeterminationLabel } from "../types";
import { determinationToFriendlyLabel, DETERMINATION_ACTION } from "../api/client";
import clsx from "clsx";

// ─── Color config per determination ──────────────────────────────────────────

export const DETERMINATION_CONFIG: Record<
  DeterminationLabel,
  { bg: string; text: string; border: string; icon: string; short: string }
> = {
  GREEN: {
    bg: "bg-green-50",
    text: "text-green-700",
    border: "border-green-200",
    icon: "✓",
    short: "GREEN",
  },
  YELLOW: {
    bg: "bg-yellow-50",
    text: "text-yellow-700",
    border: "border-yellow-200",
    icon: "?",
    short: "YELLOW",
  },
  AMBER: {
    bg: "bg-orange-50",
    text: "text-orange-700",
    border: "border-orange-200",
    icon: "!",
    short: "AMBER",
  },
  RED: {
    bg: "bg-red-50",
    text: "text-red-700",
    border: "border-red-200",
    icon: "✕",
    short: "RED",
  },
  UNCERTAIN: {
    bg: "bg-gray-50",
    text: "text-gray-600",
    border: "border-gray-200",
    icon: "~",
    short: "UNCERTAIN",
  },
  MIXED: {
    bg: "bg-purple-50",
    text: "text-purple-700",
    border: "border-purple-200",
    icon: "✂",
    short: "MIXED",
  },
};

// ─── DeterminationBadge ───────────────────────────────────────────────────────

interface DeterminationBadgeProps {
  determination: DeterminationLabel;
  size?: "sm" | "lg";
}

export function DeterminationBadge({
  determination,
  size = "lg",
}: DeterminationBadgeProps) {
  const cfg = DETERMINATION_CONFIG[determination];
  const isLarge = size === "lg";

  return (
    <div
      className={clsx(
        "flex items-center gap-3 rounded-2xl border-2 font-semibold",
        cfg.bg,
        cfg.text,
        cfg.border,
        isLarge ? "px-6 py-4" : "px-3 py-2 text-sm"
      )}
      role="status"
      aria-label={`Determination: ${determinationToFriendlyLabel(determination)}`}
    >
      <span
        className={clsx(
          "flex items-center justify-center rounded-full border-2 font-bold",
          cfg.border,
          cfg.text,
          isLarge ? "h-10 w-10 text-xl" : "h-6 w-6 text-xs"
        )}
        aria-hidden="true"
      >
        {cfg.icon}
      </span>
      <div>
        <div className={isLarge ? "text-xl" : "text-sm"}>
          {determinationToFriendlyLabel(determination)}
        </div>
        <div className={clsx("opacity-60", isLarge ? "text-sm" : "text-xs")}>
          {cfg.short}
        </div>
      </div>
    </div>
  );
}

// ─── ConfidenceMeter ──────────────────────────────────────────────────────────

interface ConfidenceMeterProps {
  pLlm: number;
  confidenceLow: number;
  confidenceHigh: number;
}

export function ConfidenceMeter({
  pLlm,
  confidenceLow,
  confidenceHigh,
}: ConfidenceMeterProps) {
  const pct = Math.round(pLlm * 100);
  const lowPct = Math.round(confidenceLow * 100);
  const highPct = Math.round(confidenceHigh * 100);

  return (
    <div className="space-y-2">
      <div className="flex justify-between text-xs text-gray-500">
        <span>Definitely Human</span>
        <span>Definitely AI</span>
      </div>
      <div className="relative h-5 rounded-full bg-gradient-to-r from-green-400 via-yellow-300 to-red-500">
        {/* Confidence interval shading */}
        <div
          className="absolute top-0 h-full rounded-full bg-black/10"
          style={{
            left: `${lowPct}%`,
            width: `${highPct - lowPct}%`,
          }}
          aria-hidden="true"
        />
        {/* Marker */}
        <div
          className="absolute top-1/2 h-5 w-1.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-white shadow-md ring-2 ring-gray-400"
          style={{ left: `${pct}%` }}
          role="img"
          aria-label={`${pct}% likely AI-generated`}
        />
      </div>
      <p className="text-center text-sm text-gray-600">
        <strong>Confidence: {pct}% likely AI-generated</strong> (range:{" "}
        {lowPct}%–{highPct}%)
      </p>
    </div>
  );
}

// ─── ActionGuidance ───────────────────────────────────────────────────────────

export function ActionGuidance({
  determination,
}: {
  determination: DeterminationLabel;
}) {
  const cfg = DETERMINATION_CONFIG[determination];
  return (
    <div
      className={clsx(
        "rounded-xl border-2 p-4 text-sm",
        cfg.bg,
        cfg.border,
        cfg.text
      )}
    >
      <strong>What to do next:</strong> {DETERMINATION_ACTION[determination]}
    </div>
  );
}
