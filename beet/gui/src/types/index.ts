// TypeScript interfaces matching the Python contracts.py dataclasses

export type DeterminationLabel =
  | "GREEN"
  | "YELLOW"
  | "AMBER"
  | "RED"
  | "UNCERTAIN"
  | "MIXED";

export interface LayerResult {
  layer: string;
  score: number;
  contribution: number;
  detail?: string;
  signals?: Record<string, number>;
}

export interface FusionResult {
  determination: DeterminationLabel;
  p_llm: number;
  confidence_low: number;
  confidence_high: number;
  layer_results: LayerResult[];
  top_signals: string[];
  reasoning?: string;
}

export interface RouterDecision {
  profile: string;
  layers_used: string[];
  stop_reason: string;
}

export interface BeetReport {
  text_id?: string;
  text_excerpt?: string;
  word_count: number;
  determination: DeterminationLabel;
  p_llm: number;
  confidence_low: number;
  confidence_high: number;
  top_signals: string[];
  fusion_result: FusionResult;
  router_decision?: RouterDecision;
  created_at?: string;
  profile?: string;
  occupation?: string;
}

export interface AnalyzeOptions {
  profile?: "screening" | "default" | "full";
  occupation?: string;
  format?: string;
}

export interface Submission {
  id?: string;
  text: string;
  occupation?: string;
}

export interface HealthStatus {
  connected: boolean;
  version: string;
  availableDetectors: string[];
}

export interface BatchResult {
  submissions: BeetReport[];
  total: number;
  completed: number;
}
