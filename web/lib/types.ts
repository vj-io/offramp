// Hand-written mirror of offramp/schemas.py (the web app's half of the contract).
// Keep field names and optionality identical; update both files together.

export type Tier = "lightning" | "super" | "ultra";
export const TIER_ORDER: Tier[] = ["lightning", "super", "ultra"]; // cheapest first

export type TaskType = "classification" | "json_extraction" | "tool_call" | "free_text" | "reasoning";
export type Decision = "MIGRATE" | "KEEP_CLOSED" | "MANUAL_REVIEW";
export type Split = "dev" | "holdout";

export type Stage =
  | "QUEUED"
  | "CLONED"
  | "SCANNED"
  | "BASELINED"
  | "EVALUATING"
  | "ITERATING"
  | "VERDICTS"
  | "PATCHED"
  | "PR_OPENED"
  | "DONE"
  | "FAILED";
// FAILED is terminal, reachable from any stage
export const STAGE_ORDER: Exclude<Stage, "FAILED">[] = [
  "QUEUED",
  "CLONED",
  "SCANNED",
  "BASELINED",
  "EVALUATING",
  "ITERATING",
  "VERDICTS",
  "PATCHED",
  "PR_OPENED",
  "DONE",
];

export interface Contract {
  task_type: TaskType;
  output: "text" | "label" | "json_schema" | "json" | "tool_call";
  tools: boolean;
  streaming: boolean;
  latency_budget_ms: number | null;
  reasoning_heavy: boolean;
  labels: string[];
  json_schema: Record<string, unknown> | null;
}

export interface PromptTemplate {
  system: string | null;
  user: string;
  variables: string[];
}

export interface CallSite {
  id: string;
  file: string;
  lines: number[]; // [start, end], 1-based inclusive
  kind: "chat" | "embedding";
  sdk: "openai" | "anthropic" | "litellm" | "langchain";
  call: string;
  function: string | null;
  source_model: string | null;
  prompt_source: "literal" | "fstring" | "variable" | "dynamic";
  prompt_ref: string | null;
  template: PromptTemplate | null;
  params: Record<string, unknown>;
  contract: Contract | null;
  confidence: number;
  flags: string[];
}

export interface AdapterSpec {
  system_prompt: string | null;
  user_prefix: string;
  user_suffix: string;
  few_shot: Record<string, unknown>[];
  response_format: Record<string, unknown> | null;
  json_mode_fallback: boolean;
  tools: Record<string, unknown>[] | null;
  tool_choice: unknown;
  reasoning: "off" | "low" | "medium" | "high";
  max_tokens: number | null;
  temperature: number;
  extra_body: Record<string, unknown>;
}

export interface Adapter {
  site_id: string;
  version: number; // 0 = Adapter Agent output, n = Iterator loop n
  tier: Tier | null;
  spec: AdapterSpec;
  patch: string | null;
  notes: string[];
  risk_flags: string[];
  change_log: string[];
}

export interface ToolCall {
  name: string;
  arguments: Record<string, unknown>;
}

export interface ModelOutput {
  text: string | null;
  tool_calls: ToolCall[];
}

// Expected output for one case (PLAN Amendment A). Not sent over SSE; mirrored for completeness.
export interface BaselineResult {
  case_id: string;
  source: "reference" | "closed_model";
  model: string; // "reference:seed" | "reference:oracle" | "reference:human" | a closed model
  output: ModelOutput;
  latency_ms: number | null; // null for references: nothing was measured
  input_tokens: number | null;
  output_tokens: number | null;
  cost_usd: number | null;
  verified_by: string[]; // e.g. ["blind:nemotron-ultra", "human:tendem"]
  recorded_at: string;
}

export interface CaseScore {
  case_id: string;
  passed: boolean;
  score: number;
  detail: string;
  error_category: string | null;
}

export interface TierMetrics {
  p50_ms: number;
  p95_ms: number;
  tokens_in: number;
  tokens_out: number;
  cost_per_1k_calls_usd: number;
}

export interface TierResult {
  site_id: string;
  tier: Tier;
  adapter_version: number;
  split: Split;
  passed: boolean;
  score: number; // parity fraction 0..1 (free_text: judge mean 1..5)
  metrics: TierMetrics;
  failing_case_ids: string[];
  case_scores?: CaseScore[]; // omitted in SSE payloads
}

// The four optional fields arrived with Amendment A; optional so older recordings still type.
export interface Verdict {
  site_id: string;
  decision: Decision;
  tier: Tier | null;
  model: string | null;
  parity_pct: number | null; // holdout parity 0..100
  best_tier: Tier | null;
  best_score: number | null;
  iterations: number;
  latency_delta_ms: number | null; // p50 after - before; null when "before" was never measured
  cost_before_per_1k: number | null;
  cost_before_note?: string | null; // "est.: <model> $in/$out per 1M (<source>, <date>) x tokens …"
  cost_after_per_1k: number | null;
  sunset_risk?: string | null; // deprecation notice for the source model, with URL
  closed_reference_model?: string | null; // GC27(c), e.g. "claude-sonnet-5.5 (via Claude Code; …)"
  // that model's holdout parity on the same references; null with a model set: it audited them (no comparison)
  closed_reference_parity_pct?: number | null;
  reason: string;
}

export type EventType =
  | "stage"
  | "site"
  | "case_summary"
  | "tier_result"
  | "iteration"
  | "verdict"
  | "pr"
  | "log"
  | "error"
  | "done";
export const EVENT_TYPES: EventType[] = [
  "stage",
  "site",
  "case_summary",
  "tier_result",
  "iteration",
  "verdict",
  "pr",
  "log",
  "error",
  "done",
];

export interface RunEvent {
  run_id: string;
  seq: number; // monotonically increasing per run; SSE `id:`
  ts: string; // ISO datetime
  stage: Stage;
  type: EventType;
  call_site_id: string | null;
  payload: Record<string, unknown>;
}

// GC18: blind check of seed references by an independent model; human audit when available.
export interface ReferenceAudit {
  checked: number;
  agreed: number;
  dropped: number;
  human_verified: number;
  auditor: string | null;
}

// `case_summary` event payload.
export interface CaseSummary {
  site_id: string;
  dev: number;
  holdout: number;
  references: number;
  reference_methods: Record<string, number>; // {"reference:seed": 20, ...}; closed answers under the model: {"claude-…": 24}
  audit: ReferenceAudit;
}

export interface RunTotals {
  sites: number;
  migrated: number;
  kept_closed: number;
  manual_review: number;
  cases: number;
  tokens_in: number;
  tokens_out: number;
  ultra_agent_calls: number;
  tf_spend_usd: number;
  est_cost_before_per_1k: number;
  est_cost_after_per_1k: number;
}

export interface RunConfig {
  repo_url: string;
  threshold: number;
  reasoning_threshold: number;
  tiers: Tier[];
  max_iterations: number;
  open_pr: boolean;
  live: boolean;
}

// GET /api/runs/{id} (Global Constraint 13). `config` is null when the snapshot was rebuilt
// from a recorded event log that does not carry the run config.
export interface RunSnapshot {
  run_id: string;
  config: RunConfig | null;
  stage: Stage;
  sites: CallSite[];
  tier_results: TierResult[];
  verdicts: Verdict[];
  totals: RunTotals | null;
  artifacts: { pr_url?: string; report_url?: string };
  // Not in GC13's snapshot yet (proposed): filled when the snapshot is rebuilt from the event log.
  case_summaries?: CaseSummary[];
}

// GET /api/runs/{id}/report
export interface RunReport {
  snapshot: RunSnapshot;
  pr_body_markdown: string | null;
}
