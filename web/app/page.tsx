"use client";

import { useEffect, useRef, useState } from "react";
import Sidebar from "@/components/Sidebar";
import ChatMessageView from "@/components/ChatMessageView";
import DiagnosticsPanel from "@/components/DiagnosticsPanel";
import EvaluationPanel from "@/components/EvaluationPanel";
import { fetchHealth, streamChat } from "@/lib/api";
import { ChatMessage, HealthInfo, SearchMode } from "@/lib/types";

type View = "chat" | "diagnostics" | "evaluation";

export default function Home() {
  const [view, setView] = useState<View>("chat");
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [searchMode, setSearchMode] = useState<SearchMode>("hybrid");
  const [topK, setTopK] = useState(5);
  const [model, setModel] = useState("llama-3.3-70b-versatile");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetchHealth()
      .then((h) => {
        setHealth(h);
        setModel(h.default_model);
      })
      .catch(() => setHealth(null));
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const question = input.trim();
    if (!question || loading) return;

    setInput("");
    setLoading(true);
    setMessages((prev) => [
      ...prev,
      { role: "user", content: question },
      { role: "assistant", content: "" },
    ]);

    try {
      for await (const event of streamChat({ question, searchMode, topK, model })) {
        if (event.type === "sources") {
          setMessages((prev) => {
            const next = [...prev];
            next[next.length - 1] = { ...next[next.length - 1], sources: event.sources };
            return next;
          });
        } else if (event.type === "token") {
          setMessages((prev) => {
            const next = [...prev];
            const last = next[next.length - 1];
            next[next.length - 1] = { ...last, content: last.content + event.content };
            return next;
          });
        } else if (event.type === "error") {
          setMessages((prev) => {
            const next = [...prev];
            const last = next[next.length - 1];
            next[next.length - 1] = {
              ...last,
              content: last.content + `\n\n⚠️ ${event.message}`,
            };
            return next;
          });
        }
      }
    } catch (err) {
      setMessages((prev) => {
        const next = [...prev];
        const last = next[next.length - 1];
        next[next.length - 1] = {
          ...last,
          content: last.content + `\n\n⚠️ ${(err as Error).message}`,
        };
        return next;
      });
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex h-screen bg-white dark:bg-black">
      <Sidebar
        searchMode={searchMode}
        setSearchMode={setSearchMode}
        topK={topK}
        setTopK={setTopK}
        model={model}
        setModel={setModel}
        health={health}
      />

      <main className="flex flex-1 flex-col">
        <header className="flex items-start justify-between border-b border-zinc-200 px-6 py-4 dark:border-zinc-800">
          <div>
            <h1 className="text-xl font-semibold text-zinc-900 dark:text-zinc-100">
              🎙️ Podcast Transcripts — Ask Anything
            </h1>
            <p className="text-sm text-zinc-500 dark:text-zinc-400">
              RAG-powered Q&amp;A over 33 Podcast Transcripts. Backed by
              OpenSearch k-NN + Groq.
            </p>
          </div>
          <div className="flex shrink-0 gap-1 rounded-lg border border-zinc-200 p-1 dark:border-zinc-800">
            <button
              onClick={() => setView("chat")}
              className={`rounded-md px-3 py-1.5 text-sm font-medium ${
                view === "chat"
                  ? "bg-orange-600 text-white"
                  : "text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200"
              }`}
            >
              Chat
            </button>
            <button
              onClick={() => setView("diagnostics")}
              className={`rounded-md px-3 py-1.5 text-sm font-medium ${
                view === "diagnostics"
                  ? "bg-orange-600 text-white"
                  : "text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200"
              }`}
            >
              Diagnostics
            </button>
            <button
              onClick={() => setView("evaluation")}
              className={`rounded-md px-3 py-1.5 text-sm font-medium ${
                view === "evaluation"
                  ? "bg-orange-600 text-white"
                  : "text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200"
              }`}
            >
              Evaluation
            </button>
          </div>
        </header>

        {view === "diagnostics" ? (
          <DiagnosticsPanel />
        ) : view === "evaluation" ? (
          <EvaluationPanel />
        ) : (
          <>
            <div className="flex flex-1 flex-col gap-4 overflow-y-auto px-6 py-6">
              {messages.length === 0 && (
                <p className="mt-12 text-center text-sm text-zinc-400">
                  Ask a question about vector search to get started.
                </p>
              )}
              {messages.map((m, i) => (
                <ChatMessageView key={i} message={m} />
              ))}
              <div ref={bottomRef} />
            </div>

            <form
              onSubmit={handleSubmit}
              className="flex gap-2 border-t border-zinc-200 p-4 dark:border-zinc-800"
            >
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask a question about vector search..."
                disabled={loading}
                className="flex-1 rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm text-zinc-900 outline-none focus:border-orange-500 disabled:opacity-60 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-100"
              />
              <button
                type="submit"
                disabled={loading || !input.trim()}
                className="rounded-lg bg-orange-600 px-4 py-2 text-sm font-medium text-white hover:bg-orange-700 disabled:opacity-50"
              >
                Send
              </button>
            </form>
          </>
        )}
      </main>
    </div>
  );
}
