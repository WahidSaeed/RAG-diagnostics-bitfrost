"use client";

import { useState } from "react";
import {
  diagnoseAnswerPosition,
  diagnoseIrrelevantDocuments,
  diagnoseLatency,
  diagnosePhrasingSensitivity,
} from "@/lib/api";
import {
  AnswerPositionResult,
  IrrelevantDocsResult,
  LatencyResult,
  PhrasingSensitivityResult,
  Source,
} from "@/lib/types";

type Tab = "irrelevant" | "position" | "phrasing" | "latency";

const TABS: { id: Tab; label: string; hint: string }[] = [
  { id: "irrelevant", label: "Irrelevant docs", hint: "Is the answer in the corpus at all?" },
  { id: "position", label: "Answer position", hint: "Right docs, wrong answer?" },
  { id: "phrasing", label: "Phrasing sensitivity", hint: "Right sometimes, wrong sometimes?" },
  { id: "latency", label: "Latency", hint: "Which stage is slow?" },
];

function DiagnosisBanner({ text }: { text: string | null }) {
  if (!text) return null;
  return (
    <p className="rounded-md border border-orange-300 bg-orange-50 px-3 py-2 text-sm text-orange-900 dark:border-orange-900 dark:bg-orange-950/40 dark:text-orange-200">
      {text}
    </p>
  );
}

function RunButton({ loading, disabled }: { loading: boolean; disabled?: boolean }) {
  return (
    <button
      type="submit"
      disabled={loading || disabled}
      className="rounded-lg bg-orange-600 px-4 py-2 text-sm font-medium text-white hover:bg-orange-700 disabled:opacity-50"
    >
      {loading ? "Running…" : "Run diagnosis"}
    </button>
  );
}

function TextField({
  label,
  value,
  onChange,
  placeholder,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
}) {
  return (
    <label className="flex flex-col gap-1 text-xs font-medium text-zinc-500 dark:text-zinc-400">
      {label}
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="rounded-md border border-zinc-300 bg-white px-2.5 py-1.5 text-sm text-zinc-900 outline-none focus:border-orange-500 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-100"
      />
    </label>
  );
}

function HitList({ title, hits }: { title: string; hits: Source[] }) {
  return (
    <div className="flex-1">
      <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
        {title}
      </h4>
      <ol className="flex flex-col gap-2">
        {hits.map((h, i) => (
          <li
            key={i}
            className="rounded-md border border-zinc-200 bg-zinc-50 p-2 text-xs dark:border-zinc-800 dark:bg-zinc-900"
          >
            <div className="mb-1 flex justify-between text-zinc-500 dark:text-zinc-400">
              <span>#{i + 1} · {h.episode_title}</span>
              <span>{h.score.toFixed(3)}</span>
            </div>
            <p className="line-clamp-3 text-zinc-700 dark:text-zinc-300">{h.chunk_text}</p>
          </li>
        ))}
        {hits.length === 0 && (
          <li className="text-xs text-zinc-400">No hits.</li>
        )}
      </ol>
    </div>
  );
}

function IrrelevantDocsTab() {
  const [query, setQuery] = useState("");
  const [answerSubstring, setAnswerSubstring] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<IrrelevantDocsResult | null>(null);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!query.trim() || loading) return;
    setLoading(true);
    setError(null);
    try {
      const res = await diagnoseIrrelevantDocuments({
        query,
        answerSubstring: answerSubstring.trim() || undefined,
      });
      setResult(res);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <form onSubmit={onSubmit} className="flex flex-col gap-3">
        <TextField label="Failing query" value={query} onChange={setQuery} placeholder="e.g. how does HNSW pick ef_construction?" />
        <TextField
          label="Expected answer snippet (optional)"
          value={answerSubstring}
          onChange={setAnswerSubstring}
          placeholder="A phrase you know is in the transcript"
        />
        <RunButton loading={loading} disabled={!query.trim()} />
      </form>

      {error && <p className="text-sm text-red-600 dark:text-red-400">{error}</p>}

      {result && (
        <div className="flex flex-col gap-3">
          <DiagnosisBanner text={result.diagnosis} />
          {result.vector_found_answer !== null && (
            <div className="flex gap-4 text-xs">
              <span>
                Vector search found it:{" "}
                <strong>{result.vector_found_answer ? "yes" : "no"}</strong>
              </span>
              <span>
                BM25 found it: <strong>{result.bm25_found_answer ? "yes" : "no"}</strong>
              </span>
            </div>
          )}
          <div className="flex gap-4">
            <HitList title="Vector (k-NN) hits" hits={result.vector_hits} />
            <HitList title="BM25 hits" hits={result.bm25_hits} />
          </div>
        </div>
      )}
    </div>
  );
}

function AnswerPositionTab() {
  const [query, setQuery] = useState("");
  const [answerSubstring, setAnswerSubstring] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnswerPositionResult | null>(null);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!query.trim() || !answerSubstring.trim() || loading) return;
    setLoading(true);
    setError(null);
    try {
      const res = await diagnoseAnswerPosition({ query, answerSubstring });
      setResult(res);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <form onSubmit={onSubmit} className="flex flex-col gap-3">
        <TextField label="Query" value={query} onChange={setQuery} />
        <TextField
          label="Expected answer snippet (required)"
          value={answerSubstring}
          onChange={setAnswerSubstring}
          placeholder="A phrase from the correct chunk"
        />
        <RunButton loading={loading} disabled={!query.trim() || !answerSubstring.trim()} />
      </form>

      {error && <p className="text-sm text-red-600 dark:text-red-400">{error}</p>}

      {result && (
        <div className="flex flex-col gap-3">
          <DiagnosisBanner text={result.diagnosis} />
          <div className="flex gap-6 text-sm">
            <div>
              <div className="text-xs text-zinc-500 dark:text-zinc-400">Rank before rerank</div>
              <div className="text-2xl font-semibold text-zinc-900 dark:text-zinc-100">
                {result.answer_rank ?? "not found"}
              </div>
            </div>
            <div>
              <div className="text-xs text-zinc-500 dark:text-zinc-400">Rank after rerank</div>
              <div className="text-2xl font-semibold text-zinc-900 dark:text-zinc-100">
                {result.reranked_answer_rank ?? "not found"}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function PhrasingSensitivityTab() {
  const [variants, setVariants] = useState(["", ""]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PhrasingSensitivityResult | null>(null);

  const filled = variants.map((v) => v.trim()).filter(Boolean);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (filled.length < 2 || loading) return;
    setLoading(true);
    setError(null);
    try {
      const res = await diagnosePhrasingSensitivity({ variants: filled });
      setResult(res);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <form onSubmit={onSubmit} className="flex flex-col gap-3">
        {variants.map((v, i) => (
          <div key={i} className="flex items-end gap-2">
            <div className="flex-1">
              <TextField
                label={`Phrasing ${i + 1}`}
                value={v}
                onChange={(val) =>
                  setVariants((prev) => prev.map((p, idx) => (idx === i ? val : p)))
                }
              />
            </div>
            {variants.length > 2 && (
              <button
                type="button"
                onClick={() => setVariants((prev) => prev.filter((_, idx) => idx !== i))}
                className="mb-0.5 rounded-md border border-zinc-300 px-2 py-1.5 text-xs text-zinc-500 hover:bg-zinc-100 dark:border-zinc-700 dark:hover:bg-zinc-800"
              >
                Remove
              </button>
            )}
          </div>
        ))}
        <button
          type="button"
          onClick={() => setVariants((prev) => [...prev, ""])}
          className="self-start text-xs text-orange-600 hover:underline dark:text-orange-400"
        >
          + Add another phrasing
        </button>
        <RunButton loading={loading} disabled={filled.length < 2} />
      </form>

      {error && <p className="text-sm text-red-600 dark:text-red-400">{error}</p>}

      {result && (
        <div className="flex flex-col gap-3">
          <DiagnosisBanner text={result.diagnosis} />
          <div>
            <div className="mb-1 flex justify-between text-xs text-zinc-500 dark:text-zinc-400">
              <span>Result-set overlap across phrasings</span>
              <span>{Math.round(result.overlap_ratio * 100)}%</span>
            </div>
            <div className="h-2 w-full overflow-hidden rounded-full bg-zinc-200 dark:bg-zinc-800">
              <div
                className="h-full bg-orange-600"
                style={{ width: `${Math.round(result.overlap_ratio * 100)}%` }}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function LatencyTab() {
  const [query, setQuery] = useState("");
  const [useReranking, setUseReranking] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<LatencyResult | null>(null);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!query.trim() || loading) return;
    setLoading(true);
    setError(null);
    try {
      const res = await diagnoseLatency({ query, useReranking });
      setResult(res);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }

  const stages = result ? Object.entries(result.stage_ms) : [];
  const maxMs = stages.length ? Math.max(...stages.map(([, ms]) => ms)) : 1;

  return (
    <div className="flex flex-col gap-4">
      <form onSubmit={onSubmit} className="flex flex-col gap-3">
        <TextField label="Query" value={query} onChange={setQuery} />
        <label className="flex items-center gap-2 text-sm text-zinc-700 dark:text-zinc-300">
          <input
            type="checkbox"
            checked={useReranking}
            onChange={(e) => setUseReranking(e.target.checked)}
            className="accent-orange-600"
          />
          Include reranking stage
        </label>
        <RunButton loading={loading} disabled={!query.trim()} />
      </form>

      {error && <p className="text-sm text-red-600 dark:text-red-400">{error}</p>}

      {result && (
        <div className="flex flex-col gap-3">
          <DiagnosisBanner text={result.diagnosis} />
          <div className="flex flex-col gap-2">
            {stages.map(([stage, ms]) => (
              <div key={stage}>
                <div className="mb-1 flex justify-between text-xs text-zinc-500 dark:text-zinc-400">
                  <span className={stage === result.bottleneck ? "font-semibold text-orange-600 dark:text-orange-400" : ""}>
                    {stage}
                  </span>
                  <span>{ms.toFixed(0)}ms</span>
                </div>
                <div className="h-2 w-full overflow-hidden rounded-full bg-zinc-200 dark:bg-zinc-800">
                  <div
                    className={`h-full ${stage === result.bottleneck ? "bg-orange-600" : "bg-zinc-400 dark:bg-zinc-600"}`}
                    style={{ width: `${(ms / maxMs) * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default function DiagnosticsPanel() {
  const [tab, setTab] = useState<Tab>("irrelevant");

  return (
    <div className="flex flex-1 flex-col overflow-y-auto px-6 py-6">
      <div className="mb-4 flex gap-1 border-b border-zinc-200 dark:border-zinc-800">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`border-b-2 px-3 py-2 text-sm font-medium transition-colors ${
              tab === t.id
                ? "border-orange-600 text-orange-600 dark:text-orange-400"
                : "border-transparent text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>
      <p className="mb-4 text-xs text-zinc-500 dark:text-zinc-400">
        {TABS.find((t) => t.id === tab)?.hint}
      </p>

      {tab === "irrelevant" && <IrrelevantDocsTab />}
      {tab === "position" && <AnswerPositionTab />}
      {tab === "phrasing" && <PhrasingSensitivityTab />}
      {tab === "latency" && <LatencyTab />}
    </div>
  );
}
