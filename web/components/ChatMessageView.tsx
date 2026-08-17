"use client";

import { useState } from "react";
import { ChatMessage } from "@/lib/types";

export default function ChatMessageView({ message }: { message: ChatMessage }) {
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const isUser = message.role === "user";

  return (
    <div className={`flex w-full ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-2xl rounded-2xl px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap ${
          isUser
            ? "bg-orange-600 text-white"
            : "bg-zinc-100 text-zinc-900 dark:bg-zinc-800 dark:text-zinc-100"
        }`}
      >
        {!isUser && message.sources && message.sources.length > 0 && (
          <div className="mb-3">
            <button
              onClick={() => setSourcesOpen((v) => !v)}
              className="text-xs font-medium text-orange-700 hover:underline dark:text-orange-400"
            >
              {sourcesOpen ? "▾" : "▸"} 📚 Retrieved {message.sources.length} sources
            </button>
            {sourcesOpen && (
              <div className="mt-2 flex flex-col gap-2 border-l-2 border-orange-300 pl-3 dark:border-orange-800">
                {message.sources.map((s, i) => (
                  <div key={i} className="text-xs text-zinc-600 dark:text-zinc-400">
                    <div className="font-semibold text-zinc-800 dark:text-zinc-200">
                      [{i + 1}] {s.episode_title}{" "}
                      <span className="font-normal text-zinc-400">
                        (score: {s.score.toFixed(4)})
                      </span>
                    </div>
                    <div className="mt-0.5">{s.chunk_text.slice(0, 300)}...</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
        {message.content || (
          <span className="inline-flex gap-1">
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-current [animation-delay:-0.3s]" />
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-current [animation-delay:-0.15s]" />
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-current" />
          </span>
        )}
      </div>
    </div>
  );
}
