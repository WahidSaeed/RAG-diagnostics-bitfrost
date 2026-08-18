"use client";

import { useEffect, useRef, useState } from "react";
import { fetchEvalSet, streamEvaluation } from "@/lib/api";
import {
  EvalExample,
  EvaluationLogRecord,
  EvaluationPhase,
  EvaluationResult,
  EvaluationRow,
} from "@/lib/types";

const PHASE_LABELS: Record<EvaluationPhase, string> = {
  retrieving: "Retrieving context & generating answers",
  judging: "Scoring with RAGAS (faithfulness, relevancy, precision, recall)",
};

// Must match the number of metrics run per question in src/evaluate.py
// (Faithfulness, ResponseRelevancy, LLMContextPrecisionWithReference,
// LLMContextRecall) — used to know when a question is fully judged.
const METRIC_COUNT = 4;

type QuestionStage = "pending" | "generated" | "judging" | "done";

interface QuestionStatus {
  stage: QuestionStage;
  metricsDone: number;
  hasError: boolean;
}

const METRICS: { key: keyof EvaluationResult["mean_scores"]; label: string; hint: string }[] = [
  {
    key: "faithfulness",
    label: "Faithfulness",
    hint: "Is the answer actually supported by the retrieved context?",
  },
  {
    key: "answer_relevancy",
    label: "Answer relevancy",
    hint: "Does the answer actually address the question asked?",
  },
  {
    key: "llm_context_precision_with_reference",
    label: "Context precision",
    hint: "Of the retrieved chunks, how many were relevant?",
  },
  {
    key: "context_recall",
    label: "Context recall",
    hint: "Did retrieval surface everything needed for the reference answer?",
  },
];

function scoreColor(score: number | null): string {
  if (score === null) return "bg-zinc-400 dark:bg-zinc-600";
  if (score >= 0.7) return "bg-green-600";
  if (score >= 0.4) return "bg-orange-500";
  return "bg-red-600";
}

function ScoreCard({ label, hint, score }: { label: string; hint: string; score: number | null }) {
  const pct = score === null ? 0 : Math.round(score * 100);
  return (
    <div className="rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
      <div className="mb-1 flex items-baseline justify-between">
        <span className="text-sm font-medium text-zinc-900 dark:text-zinc-100">{label}</span>
        <span className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">
          {score === null ? "—" : score.toFixed(2)}
        </span>
      </div>
      <div className="mb-1 h-1.5 w-full overflow-hidden rounded-full bg-zinc-200 dark:bg-zinc-800">
        <div className={`h-full ${scoreColor(score)}`} style={{ width: `${pct}%` }} />
      </div>
      <p className="text-xs text-zinc-500 dark:text-zinc-400">{hint}</p>
    </div>
  );
}

function RowDetail({ row }: { row: EvaluationRow }) {
  const [open, setOpen] = useState(false);
  return (
    <li className="rounded-md border border-zinc-200 dark:border-zinc-800">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between gap-3 px-3 py-2 text-left text-sm"
      >
        <span className="text-zinc-800 dark:text-zinc-200">{row.question}</span>
        <span className="flex shrink-0 gap-3 font-mono text-xs text-zinc-500 dark:text-zinc-400">
          <span title="faithfulness">F {row.faithfulness?.toFixed(2) ?? "—"}</span>
          <span title="answer relevancy">R {row.answer_relevancy?.toFixed(2) ?? "—"}</span>
          <span title="context precision">P {row.context_precision?.toFixed(2) ?? "—"}</span>
          <span title="context recall">C {row.context_recall?.toFixed(2) ?? "—"}</span>
        </span>
      </button>
      {open && (
        <div className="flex flex-col gap-2 border-t border-zinc-200 px-3 py-2 text-xs dark:border-zinc-800">
          <div>
            <div className="mb-0.5 font-semibold text-zinc-500 dark:text-zinc-400">Reference answer</div>
            <p className="text-zinc-700 dark:text-zinc-300">{row.reference}</p>
          </div>
          <div>
            <div className="mb-0.5 font-semibold text-zinc-500 dark:text-zinc-400">Model answer</div>
            <p className="text-zinc-700 dark:text-zinc-300">{row.response}</p>
          </div>
        </div>
      )}
    </li>
  );
}

interface Progress {
  phase: EvaluationPhase;
  current: number;
  total: number;
}

function StatusIcon({ status }: { status?: QuestionStatus }) {
  if (!status || status.stage === "pending") {
    return <span className="mt-1 block h-3 w-3 shrink-0 rounded-full border border-zinc-300 dark:border-zinc-700" />;
  }
  if (status.stage === "done") {
    return (
      <span
        className={`mt-1 flex h-3 w-3 shrink-0 items-center justify-center rounded-full text-[9px] text-white ${
          status.hasError ? "bg-red-600" : "bg-green-600"
        }`}
      >
        {status.hasError ? "!" : "✓"}
      </span>
    );
  }
  // generated or judging: in progress
  return (
    <span className="mt-1 block h-3 w-3 shrink-0 animate-pulse rounded-full bg-orange-500" />
  );
}

function EvalSetEditor({
  examples,
  setExamples,
  status,
  disabled,
}: {
  examples: EvalExample[];
  setExamples: (updater: (prev: EvalExample[]) => EvalExample[]) => void;
  status: Map<string, QuestionStatus>;
  disabled: boolean;
}) {
  function update(i: number, field: keyof EvalExample, value: string) {
    setExamples((prev) => prev.map((e, idx) => (idx === i ? { ...e, [field]: value } : e)));
  }
  function remove(i: number) {
    setExamples((prev) => prev.filter((_, idx) => idx !== i));
  }
  function add() {
    setExamples((prev) => [...prev, { question: "", ground_truth: "" }]);
  }

  return (
    <div className="mb-4 flex flex-col gap-2">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
        Eval set
      </h3>
      <ul className="flex flex-col gap-2">
        {examples.map((ex, i) => (
          <li
            key={i}
            className="flex gap-2 rounded-md border border-zinc-200 p-2 dark:border-zinc-800"
          >
            <StatusIcon status={status.get(ex.question.trim())} />
            <div className="flex flex-1 flex-col gap-1">
              <input
                type="text"
                value={ex.question}
                onChange={(e) => update(i, "question", e.target.value)}
                placeholder="Question"
                disabled={disabled}
                className="rounded-md border border-zinc-300 bg-white px-2 py-1 text-sm text-zinc-900 outline-none focus:border-orange-500 disabled:opacity-60 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-100"
              />
              <textarea
                value={ex.ground_truth}
                onChange={(e) => update(i, "ground_truth", e.target.value)}
                placeholder="Reference (ground-truth) answer"
                disabled={disabled}
                rows={2}
                className="resize-y rounded-md border border-zinc-300 bg-white px-2 py-1 text-xs text-zinc-700 outline-none focus:border-orange-500 disabled:opacity-60 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-300"
              />
            </div>
            <button
              onClick={() => remove(i)}
              disabled={disabled}
              className="self-start rounded-md border border-zinc-300 px-2 py-1 text-xs text-zinc-500 hover:bg-zinc-100 disabled:opacity-40 dark:border-zinc-700 dark:hover:bg-zinc-800"
            >
              Remove
            </button>
          </li>
        ))}
      </ul>
      <button
        onClick={add}
        disabled={disabled}
        className="self-start text-xs text-orange-600 hover:underline disabled:opacity-40 dark:text-orange-400"
      >
        + Add question
      </button>
    </div>
  );
}

function truncate(s: unknown, n: number): string {
  const str = String(s ?? "");
  return str.length > n ? str.slice(0, n - 1) + "…" : str;
}

// Renders one telemetry record as a single console line. `progress` records
// drive the progress bar instead, so they're skipped here.
function formatLogLine(r: EvaluationLogRecord): string | null {
  const t = r.ts.toFixed(2).padStart(6, " ");
  switch (r.event) {
    case "run_start":
      return `[${t}s] run started — ${r.questions} questions, top_k=${r.top_k}`;
    case "run_done":
      return `[${t}s] run finished`;
    case "retrieval_done":
      return `[${t}s] retrieval    "${truncate(r.question, 50)}" — ${r.num_hits} hits in ${r.latency_ms}ms`;
    case "generation_done":
      return `[${t}s] generation   "${truncate(r.question, 50)}" — ${r.answer_chars} chars in ${r.latency_ms}ms`;
    case "metric_start":
      return `[${t}s] ${String(r.metric).padEnd(12, " ")} start  "${truncate(r.question, 40)}"`;
    case "metric_done":
      return `[${t}s] ${String(r.metric).padEnd(12, " ")} score=${Number(r.score).toFixed(3)}  ${r.latency_ms}ms  "${truncate(r.question, 40)}"`;
    case "metric_error":
      return `[${t}s] ${String(r.metric).padEnd(12, " ")} ERROR ${r.error}  "${truncate(r.question, 40)}"`;
    default:
      return null;
  }
}

function logColor(r: EvaluationLogRecord): string {
  if (r.level === "error") return "text-red-600 dark:text-red-400";
  if (r.event === "metric_start") return "text-zinc-400 dark:text-zinc-600";
  if (r.event === "run_start" || r.event === "run_done") return "text-orange-600 dark:text-orange-400";
  return "text-zinc-600 dark:text-zinc-400";
}

export default function EvaluationPanel() {
  const [examples, setExamples] = useState<EvalExample[]>([]);
  const [status, setStatus] = useState<Map<string, QuestionStatus>>(new Map());
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState<Progress | null>(null);
  const [logs, setLogs] = useState<EvaluationLogRecord[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<EvaluationResult | null>(null);
  const consoleRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetchEvalSet()
      .then(setExamples)
      .catch((err) => setError((err as Error).message));
  }, []);

  useEffect(() => {
    consoleRef.current?.scrollTo({ top: consoleRef.current.scrollHeight });
  }, [logs]);

  function applyStatusUpdate(record: EvaluationLogRecord) {
    const question = typeof record.question === "string" ? record.question : null;
    if (!question) return;

    setStatus((prev) => {
      const next = new Map(prev);
      const current: QuestionStatus = next.get(question) ?? {
        stage: "pending",
        metricsDone: 0,
        hasError: false,
      };

      if (record.event === "generation_done") {
        next.set(question, { ...current, stage: "generated" });
      } else if (record.event === "metric_start") {
        next.set(question, { ...current, stage: "judging" });
      } else if (record.event === "metric_done" || record.event === "metric_error") {
        const metricsDone = current.metricsDone + 1;
        next.set(question, {
          stage: metricsDone >= METRIC_COUNT ? "done" : "judging",
          metricsDone,
          hasError: current.hasError || record.event === "metric_error",
        });
      }
      return next;
    });
  }

  async function run() {
    if (loading) return;
    const activeExamples = examples
      .map((e) => ({ question: e.question.trim(), ground_truth: e.ground_truth.trim() }))
      .filter((e) => e.question && e.ground_truth);
    if (activeExamples.length === 0) {
      setError("Add at least one question with a reference answer before running.");
      return;
    }

    setLoading(true);
    setProgress(null);
    setLogs([]);
    setError(null);
    setResult(null);
    setStatus(new Map(activeExamples.map((e) => [e.question, { stage: "pending", metricsDone: 0, hasError: false }])));
    try {
      for await (const event of streamEvaluation({ examples: activeExamples })) {
        if (event.type === "log") {
          const { type, ...record } = event;
          void type;
          setLogs((prev) => [...prev, record]);
          if (record.event === "progress" && record.phase && record.current !== undefined && record.total !== undefined) {
            setProgress({ phase: record.phase, current: record.current, total: record.total });
          } else {
            applyStatusUpdate(record);
          }
        } else if (event.type === "result") {
          setResult({ rows: event.rows, mean_scores: event.mean_scores, log_file: event.log_file });
        } else if (event.type === "error") {
          setError(event.message);
        }
      }
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
      setProgress(null);
    }
  }

  return (
    <div className="flex flex-1 flex-col overflow-y-auto px-6 py-6">
      <p className="mb-4 text-xs text-zinc-500 dark:text-zinc-400">
        Scores the whole pipeline with RAGAS against a hand-labeled question set —
        unlike Diagnostics, which triages a single failing query. Each run costs several
        judge-LLM calls per question and takes a few minutes.
      </p>

      <EvalSetEditor examples={examples} setExamples={setExamples} status={status} disabled={loading} />

      <button
        onClick={run}
        disabled={loading}
        className="mb-2 self-start rounded-lg bg-orange-600 px-4 py-2 text-sm font-medium text-white hover:bg-orange-700 disabled:opacity-50"
      >
        {loading ? "Evaluating…" : "Run evaluation"}
      </button>

      {loading && (
        <div className="mb-4 w-full max-w-md">
          <div className="mb-1 flex justify-between text-xs text-zinc-500 dark:text-zinc-400">
            <span>{progress ? PHASE_LABELS[progress.phase] : "Starting…"}</span>
            <span>{progress ? `${progress.current}/${progress.total}` : ""}</span>
          </div>
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-zinc-200 dark:bg-zinc-800">
            <div
              className="h-full bg-orange-600 transition-all"
              style={{
                width: progress ? `${(progress.current / progress.total) * 100}%` : "5%",
              }}
            />
          </div>
        </div>
      )}

      {(loading || logs.length > 0) && (
        <div
          ref={consoleRef}
          className="mb-4 h-56 w-full overflow-y-auto rounded-md border border-zinc-200 bg-zinc-50 p-2 font-mono text-[11px] leading-5 dark:border-zinc-800 dark:bg-zinc-950"
        >
          {logs.map((r, i) => {
            const line = formatLogLine(r);
            if (!line) return null;
            return (
              <div key={i} className={logColor(r)}>
                {line}
              </div>
            );
          })}
        </div>
      )}

      {error && <p className="mb-4 text-sm text-red-600 dark:text-red-400">{error}</p>}

      {result && (
        <div className="flex flex-col gap-6">
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            {METRICS.map((m) => (
              <ScoreCard
                key={m.key}
                label={m.label}
                hint={m.hint}
                score={result.mean_scores[m.key]}
              />
            ))}
          </div>

          <p className="text-xs text-zinc-500 dark:text-zinc-400">
            Full trace written to <code className="rounded bg-zinc-100 px-1 py-0.5 dark:bg-zinc-900">{result.log_file}</code>
          </p>

          <div>
            <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
              Per-question scores
            </h3>
            <ul className="flex flex-col gap-2">
              {result.rows.map((row, i) => (
                <RowDetail key={i} row={row} />
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}
