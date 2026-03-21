import type {
  BeetReport,
  AnalyzeOptions,
  Submission,
  HealthStatus,
  DeterminationLabel,
} from "../types";

// ─── Plain-language mappings ──────────────────────────────────────────────────

export const SIGNAL_FRIENDLY_NAME: Record<string, string> = {
  preamble: "AI Assistant Greeting",
  fingerprint_vocab: "AI Vocabulary Patterns",
  prompt_structure: "Structured Like AI Output",
  voice_spec: "Tone Inconsistency",
  instruction_density: "Instruction-Heavy Writing",
  nssi: "Repetitive Patterns",
  surprisal_dynamics: "Statistical Writing Patterns",
  contrastive_lm: "Mathematical Text Analysis",
  perturbation: "Text Stability Analysis",
  token_cohesiveness: "Word Redundancy",
  contrastive_gen: "Comparison to AI Baseline",
  dna_gpt: "Predictability Analysis",
  mixed_boundary: "Mixed Authorship Detected",
  cross_similarity: "Similar to Other Submissions",
  contributor_graph: "Network Patterns",
};

export const SIGNAL_ICON: Record<string, string> = {
  preamble: "🤖",
  fingerprint_vocab: "📝",
  prompt_structure: "🏗️",
  voice_spec: "🎭",
  instruction_density: "📋",
  nssi: "🔄",
  surprisal_dynamics: "📊",
  contrastive_lm: "🔬",
  perturbation: "🧪",
  token_cohesiveness: "🧩",
  contrastive_gen: "⚖️",
  dna_gpt: "🔮",
  mixed_boundary: "✂️",
  cross_similarity: "👥",
  contributor_graph: "🕸️",
};

export const SIGNAL_PLAIN_EXPLANATION: Record<string, string> = {
  preamble:
    "The text starts with phrases that AI assistants typically use when responding to requests.",
  fingerprint_vocab:
    "Several words appear that AI models use much more frequently than human writers.",
  prompt_structure:
    "The text follows a structural template commonly seen in AI-generated content.",
  voice_spec:
    "The casual and formal elements of this text don't fit together naturally.",
  instruction_density:
    "This text has an unusually high density of commands and instructions.",
  nssi: "The text uses formulaic phrases and repetitive structures common in AI writing.",
  surprisal_dynamics:
    "The mathematical properties of word choices follow patterns typical of AI generation.",
  contrastive_lm:
    "Advanced statistical analysis shows this text sits in a region typical of AI-generated content.",
  perturbation:
    "When words are slightly changed, the text's statistical properties shift in a way typical of AI writing.",
  token_cohesiveness:
    "Individual words contribute less unique meaning than expected, suggesting AI-typical redundancy.",
  contrastive_gen:
    "This text is very similar to what an AI would produce for the same task.",
  dna_gpt:
    "An AI can predict the second half of this text unusually well from the first half.",
  mixed_boundary:
    "The writing style changes significantly at certain points, suggesting multiple authors.",
  cross_similarity:
    "This text is unusually similar to other submissions for the same task.",
  contributor_graph:
    "This contributor's submissions share patterns with a group of other contributors.",
};

export const DETERMINATION_LABEL: Record<DeterminationLabel, string> = {
  GREEN: "Likely Human-Written",
  YELLOW: "Some Patterns Detected — Probably Human",
  AMBER: "Notable AI Patterns — Review Recommended",
  RED: "Strong AI Indicators — Likely AI-Generated",
  UNCERTAIN: "Unclear — Human Review Needed",
  MIXED: "Mixed — Parts May Be AI-Generated",
};

export const DETERMINATION_ACTION: Record<DeterminationLabel, string> = {
  GREEN: "No action needed. This text appears to be human-written.",
  YELLOW:
    "This text shows minor patterns sometimes seen in AI writing, but these can also appear in human writing. Use your judgment.",
  AMBER:
    "We recommend having a human reviewer look at this text more closely. The patterns are notable but not conclusive.",
  RED: "This text has strong indicators of AI generation. We recommend further investigation before accepting it.",
  UNCERTAIN:
    "Our analysis couldn't reach a clear conclusion. This happens with very short text or unusual writing styles. Human review is recommended.",
  MIXED:
    "Parts of this text may be AI-generated. Review the highlighted sections for changes in writing style.",
};

export function detectorIdToFriendlyName(id: string): string {
  return SIGNAL_FRIENDLY_NAME[id] ?? id;
}

export function detectorIdToIcon(id: string): string {
  return SIGNAL_ICON[id] ?? "🔍";
}

export function determinationToFriendlyLabel(label: DeterminationLabel): string {
  return DETERMINATION_LABEL[label] ?? label;
}

// ─── Error types ──────────────────────────────────────────────────────────────

export class ApiConnectionError extends Error {
  userMessage = "Can't connect to the analysis server. Make sure BEET is running.";
  constructor(message?: string) {
    super(message ?? "Connection failed");
    this.name = "ApiConnectionError";
  }
}

export class ApiTimeoutError extends Error {
  userMessage = "The analysis took too long. Try a shorter text or use Quick Scan.";
  constructor(message?: string) {
    super(message ?? "Request timed out");
    this.name = "ApiTimeoutError";
  }
}

export class ApiValidationError extends Error {
  userMessage: string;
  constructor(message: string) {
    super(message);
    this.name = "ApiValidationError";
    this.userMessage = message;
  }
}

export class ApiServerError extends Error {
  userMessage = "Something went wrong on the server. Please try again.";
  constructor(message?: string) {
    super(message ?? "Server error");
    this.name = "ApiServerError";
  }
}

// ─── Mock data ────────────────────────────────────────────────────────────────

function makeMock(
  determination: DeterminationLabel,
  p_llm: number
): BeetReport {
  return {
    text_id: `mock-${determination}`,
    text_excerpt: "This is a sample text excerpt for mock mode...",
    word_count: 150,
    determination,
    p_llm,
    confidence_low: Math.max(0, p_llm - 0.12),
    confidence_high: Math.min(1, p_llm + 0.11),
    top_signals: ["fingerprint_vocab", "prompt_structure", "nssi"],
    fusion_result: {
      determination,
      p_llm,
      confidence_low: Math.max(0, p_llm - 0.12),
      confidence_high: Math.min(1, p_llm + 0.11),
      layer_results: [
        {
          layer: "stylometric",
          score: p_llm - 0.05,
          contribution: 0.4,
          detail: "Vocabulary and structure analysis",
          signals: { fingerprint_vocab: 0.8, prompt_structure: 0.6 },
        },
        {
          layer: "statistical",
          score: p_llm + 0.03,
          contribution: 0.35,
          detail: "Statistical pattern analysis",
          signals: { nssi: 0.7, surprisal_dynamics: 0.5 },
        },
      ],
      top_signals: ["fingerprint_vocab", "prompt_structure", "nssi"],
    },
    router_decision: {
      profile: "default",
      layers_used: ["stylometric", "statistical"],
      stop_reason: "convergence",
    },
    created_at: new Date().toISOString(),
    profile: "default",
  };
}

const MOCK_RESULTS: BeetReport[] = [
  makeMock("GREEN", 0.08),
  makeMock("YELLOW", 0.32),
  makeMock("AMBER", 0.61),
  makeMock("RED", 0.87),
  makeMock("UNCERTAIN", 0.5),
  makeMock("MIXED", 0.55),
];

let mockCycle = 0;

function getNextMock(): BeetReport {
  const result = MOCK_RESULTS[mockCycle % MOCK_RESULTS.length];
  mockCycle++;
  return result;
}

// ─── BeetApiClient ────────────────────────────────────────────────────────────

export class BeetApiClient {
  private baseUrl: string;
  private timeout: number;
  private mockMode: boolean;

  constructor(baseUrl = "http://localhost:8000", timeout = 60000) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.timeout = timeout;
    this.mockMode = import.meta.env.VITE_MOCK_API === "true";
  }

  private async fetchWithTimeout(
    input: RequestInfo,
    init?: RequestInit
  ): Promise<Response> {
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), this.timeout);
    try {
      const response = await fetch(input, {
        ...init,
        signal: controller.signal,
      });
      return response;
    } catch (err: unknown) {
      if (err instanceof Error && err.name === "AbortError") {
        throw new ApiTimeoutError();
      }
      throw new ApiConnectionError(String(err));
    } finally {
      clearTimeout(id);
    }
  }

  async analyze(
    text: string,
    options?: AnalyzeOptions
  ): Promise<BeetReport> {
    if (this.mockMode) {
      await new Promise((r) => setTimeout(r, 1200));
      return getNextMock();
    }
    const response = await this.fetchWithTimeout(`${this.baseUrl}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, ...options }),
    });
    if (!response.ok) {
      if (response.status >= 400 && response.status < 500) {
        const data = await response.json().catch(() => ({}));
        throw new ApiValidationError(
          (data as { detail?: string }).detail ??
            "Invalid request. Please check your input."
        );
      }
      throw new ApiServerError(`HTTP ${response.status}`);
    }
    return response.json() as Promise<BeetReport>;
  }

  async analyzeBatch(
    submissions: Submission[],
    onProgress?: (completed: number, total: number) => void
  ): Promise<BeetReport[]> {
    if (this.mockMode) {
      const results: BeetReport[] = [];
      for (let i = 0; i < submissions.length; i++) {
        await new Promise((r) => setTimeout(r, 400));
        results.push(getNextMock());
        onProgress?.(i + 1, submissions.length);
      }
      return results;
    }
    const response = await this.fetchWithTimeout(`${this.baseUrl}/batch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ submissions }),
    });
    if (!response.ok) {
      throw new ApiServerError(`HTTP ${response.status}`);
    }
    const results = (await response.json()) as BeetReport[];
    onProgress?.(results.length, results.length);
    return results;
  }

  async checkHealth(): Promise<HealthStatus> {
    if (this.mockMode) {
      return {
        connected: true,
        version: "2.0.0-mock",
        availableDetectors: [
          "preamble",
          "fingerprint_vocab",
          "prompt_structure",
          "nssi",
          "surprisal_dynamics",
        ],
      };
    }
    try {
      const response = await this.fetchWithTimeout(`${this.baseUrl}/health`, {
        method: "GET",
      });
      if (!response.ok) {
        return { connected: false, version: "unknown", availableDetectors: [] };
      }
      const data = (await response.json()) as Partial<HealthStatus>;
      return {
        connected: true,
        version: (data.version as string) ?? "unknown",
        availableDetectors: (data.availableDetectors as string[]) ?? [],
      };
    } catch {
      return { connected: false, version: "unknown", availableDetectors: [] };
    }
  }

  async getConfig(profile?: string): Promise<object> {
    if (this.mockMode) {
      return { profile: profile ?? "default", mock: true };
    }
    const url = profile
      ? `${this.baseUrl}/config?profile=${profile}`
      : `${this.baseUrl}/config`;
    const response = await this.fetchWithTimeout(url);
    if (!response.ok) throw new ApiServerError();
    return response.json() as Promise<object>;
  }
}

export const apiClient = new BeetApiClient();
