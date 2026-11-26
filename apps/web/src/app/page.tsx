"use client";

import { useEffect, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Check = { name: string; path: string; state: "loading" | "ok" | "error"; detail?: string };

const INITIAL: Check[] = [
  { name: "API (liveness)", path: "/healthz", state: "loading" },
  { name: "Database + pgvector (readiness)", path: "/readyz", state: "loading" },
];

export default function Home() {
  const [checks, setChecks] = useState<Check[]>(INITIAL);

  useEffect(() => {
    INITIAL.forEach(async (check, i) => {
      let next: Check;
      try {
        const res = await fetch(`${API_URL}${check.path}`);
        const body = await res.json();
        next = res.ok
          ? { ...check, state: "ok", detail: JSON.stringify(body) }
          : { ...check, state: "error", detail: body.detail ?? res.statusText };
      } catch (err) {
        next = { ...check, state: "error", detail: String(err) };
      }
      setChecks((prev) => prev.map((c, j) => (j === i ? next : c)));
    });
  }, []);

  return (
    <main className="mx-auto max-w-2xl px-6 py-16">
      <h1 className="text-3xl font-semibold tracking-tight">AlphaLens</h1>
      <p className="mt-2 text-sm text-neutral-500">Walking skeleton — system status</p>

      <ul className="mt-8 divide-y divide-neutral-200 rounded-lg border border-neutral-200">
        {checks.map((c) => (
          <li key={c.path} className="flex items-start gap-3 px-4 py-3">
            <span
              aria-label={c.state}
              className={`mt-1.5 h-2.5 w-2.5 shrink-0 rounded-full ${
                c.state === "ok"
                  ? "bg-green-500"
                  : c.state === "error"
                    ? "bg-red-500"
                    : "bg-neutral-300"
              }`}
            />
            <div className="min-w-0">
              <p className="font-medium">{c.name}</p>
              <p className="truncate font-mono text-xs text-neutral-500">
                {c.path} · {c.detail ?? "checking…"}
              </p>
            </div>
          </li>
        ))}
      </ul>

      <p className="mt-10 text-xs text-neutral-400">Not investment advice.</p>
    </main>
  );
}
