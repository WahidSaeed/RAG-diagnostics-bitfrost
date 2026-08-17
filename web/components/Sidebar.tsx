"use client";

import { HealthInfo, SearchMode } from "@/lib/types";

interface SidebarProps {
  searchMode: SearchMode;
  setSearchMode: (m: SearchMode) => void;
  topK: number;
  setTopK: (k: number) => void;
  model: string;
  setModel: (m: string) => void;
  health: HealthInfo | null;
}

export default function Sidebar({
  searchMode,
  setSearchMode,
  topK,
  setTopK,
  model,
  setModel,
  health,
}: SidebarProps) {
  return (
    <aside className="flex w-72 shrink-0 flex-col gap-6 border-r border-zinc-200 bg-zinc-50 p-5 dark:border-zinc-800 dark:bg-zinc-950">
      <div>
        <h2 className="mb-3 text-sm font-semibold text-zinc-900 dark:text-zinc-100">
          Search Settings
        </h2>

        <fieldset className="mb-4">
          <legend className="mb-2 text-xs font-medium text-zinc-500 dark:text-zinc-400">
            Search mode
          </legend>
          <div className="flex gap-4">
            {(["hybrid", "semantic"] as SearchMode[]).map((mode) => (
              <label
                key={mode}
                className="flex cursor-pointer items-center gap-1.5 text-sm text-zinc-700 dark:text-zinc-300"
              >
                <input
                  type="radio"
                  name="search-mode"
                  value={mode}
                  checked={searchMode === mode}
                  onChange={() => setSearchMode(mode)}
                  className="accent-orange-600"
                />
                {mode}
              </label>
            ))}
          </div>
        </fieldset>

        <div className="mb-4">
          <label className="mb-2 flex justify-between text-xs font-medium text-zinc-500 dark:text-zinc-400">
            <span>Retrieve top-k chunks</span>
            <span className="text-zinc-700 dark:text-zinc-300">{topK}</span>
          </label>
          <input
            type="range"
            min={1}
            max={15}
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
            className="w-full accent-orange-600"
          />
        </div>
      </div>

      <hr className="border-zinc-200 dark:border-zinc-800" />

      <div>
        <label className="mb-2 block text-xs font-medium text-zinc-500 dark:text-zinc-400">
          Groq model
        </label>
        <input
          type="text"
          value={model}
          onChange={(e) => setModel(e.target.value)}
          className="w-full rounded-md border border-zinc-300 bg-white px-2.5 py-1.5 text-sm text-zinc-900 outline-none focus:border-orange-500 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-100"
        />
      </div>

      <hr className="border-zinc-200 dark:border-zinc-800" />

      <div className="flex flex-col gap-1">
        <span className="text-xs font-medium text-zinc-500 dark:text-zinc-400">
          Indexed chunks
        </span>
        <span className="text-2xl font-semibold text-zinc-900 dark:text-zinc-100">
          {health ? health.doc_count.toLocaleString() : "…"}
        </span>
        <p className="mt-2 text-xs text-zinc-500 dark:text-zinc-400">
          Embedding:{" "}
          <code className="rounded bg-zinc-200 px-1 py-0.5 dark:bg-zinc-800">
            {health?.embedding_model ?? "all-MiniLM-L6-v2"}
          </code>
        </p>
        <p className="text-xs text-zinc-500 dark:text-zinc-400">
          LLM:{" "}
          <code className="rounded bg-zinc-200 px-1 py-0.5 dark:bg-zinc-800">
            {model}
          </code>
        </p>
        <a
          href="http://localhost:8080"
          target="_blank"
          rel="noopener noreferrer"
          className="mt-1 text-xs text-orange-600 hover:underline dark:text-orange-400"
        >
          ⚡ Powered by Bifrost — view dashboard
        </a>
      </div>
    </aside>
  );
}
