"use client";

import { type FormEvent, useState } from "react";

import AnswerView, { type Selection } from "@/components/AnswerView";
import CitationPanel from "@/components/CitationPanel";
import { type AskResponse, askQuestion } from "@/lib/api";

const EXAMPLES = [
  "What are Apple's main supply chain risks?",
  "How does Apple manage cybersecurity risk?",
  "What legal proceedings is Apple involved in?",
];

export default function ResearchPage() {
  const [question, setQuestion] = useState("");
  const [ticker, setTicker] = useState("AAPL");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AskResponse | null>(null);
  const [selected, setSelected] = useState<Selection | null>(null);

  async function submit(q: string) {
    if (q.trim().length < 3 || loading) return;
    setQuestion(q);
    setLoading(true);
    setError(null);
    setResult(null);
    setSelected(null);
    try {
      const tickers = ticker.trim() ? [ticker.trim().toUpperCase()] : [];
      setResult(await askQuestion(q.trim(), tickers));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  function onSubmit(e: FormEvent) {
    e.preventDefault(); // stop the browser from reloading the page
    void submit(question);
  }

  const citation = result && selected ? result.parts[selected.part].citations[selected.citation] : null;
  const source = result && citation ? result.sources[citation.source] : null;

  return (
    <main className="mx-auto max-w-6xl px-6 py-10">
      <header className="mb-6">
        <h1 className="text-2xl font-semibold tracking-tight">AlphaLens Research</h1>
        <p className="text-sm text-neutral-500">
          Ask about SEC filings. Every claim links to the passage it came from.
        </p>
      </header>

      <form onSubmit={onSubmit} className="flex gap-2">
        <input
          value={ticker}
          onChange={(e) => setTicker(e.target.value)}
          placeholder="Ticker"
          aria-label="Ticker"
          className="w-24 rounded-md border border-neutral-300 bg-transparent px-3 py-2 text-sm uppercase dark:border-neutral-700"
        />
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a question about the filing…"
          aria-label="Question"
          className="flex-1 rounded-md border border-neutral-300 bg-transparent px-3 py-2 text-sm dark:border-neutral-700"
        />
        <button
          type="submit"
          disabled={loading || question.trim().length < 3}
          className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
        >
          {loading ? "Thinking…" : "Ask"}
        </button>
      </form>

      {!result && !loading && !error && (
        <div className="mt-4 flex flex-wrap gap-2">
          {EXAMPLES.map((q) => (
            <button
              key={q}
              type="button"
              onClick={() => void submit(q)}
              className="rounded-full border border-neutral-300 px-3 py-1 text-xs text-neutral-600 hover:bg-neutral-100 dark:border-neutral-700 dark:text-neutral-300 dark:hover:bg-neutral-800"
            >
              {q}
            </button>
          ))}
        </div>
      )}

      {error && (
        <p className="mt-6 rounded-md border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-700 dark:bg-red-950/40 dark:text-red-300">
          {error}
        </p>
      )}

      {loading && (
        <p className="mt-6 text-sm text-neutral-500">
          Searching the filing and drafting a cited answer. This usually takes 10–20 seconds.
        </p>
      )}

      {result && (
        <div className="mt-8 grid gap-8 lg:grid-cols-[1fr_22rem]">
          <section>
            <AnswerView parts={result.parts} selected={selected} onSelect={setSelected} />
            <p className="mt-6 text-xs text-neutral-400">
              {result.model} · {result.input_tokens.toLocaleString()} in /{" "}
              {result.output_tokens.toLocaleString()} out tokens · {result.sources.length} sources
            </p>
          </section>
          <aside className="h-fit rounded-lg border border-neutral-200 p-4 lg:sticky lg:top-6 dark:border-neutral-800">
            <CitationPanel citation={citation} source={source} />
          </aside>
        </div>
      )}

      <p className="mt-12 text-xs text-neutral-400">Not investment advice.</p>
    </main>
  );
}
