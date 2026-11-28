// Mirrors apps/api/app/routers/ask.py (AskRequest / AskResponse).

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Citation = {
  source: number; // index into AskResponse.sources
  cited_text: string;
};

export type AnswerPart = {
  text: string;
  citations: Citation[];
};

export type Source = {
  chunk_id: number;
  ticker: string;
  form_type: string;
  fiscal_year: number | null;
  item_code: string;
  section_title: string | null;
  context_header: string;
  url: string;
};

export type AskResponse = {
  answer: string;
  parts: AnswerPart[];
  sources: Source[];
  model: string;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number | null; // list-price estimate; null for models without a known price
  refused: boolean;
  tickers: string[]; // companies searched; empty = all companies
  tickers_resolved: boolean; // true if inferred from the question
};

export class UnauthorizedError extends Error {}

export async function askQuestion(
  question: string,
  tickers: string[],
  accessKey: string,
): Promise<AskResponse> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  // The key is typed in by the user and kept in localStorage; it is never part of the
  // build (NEXT_PUBLIC_* values are embedded in the JavaScript anyone can read).
  if (accessKey) headers["X-Access-Key"] = accessKey;
  const res = await fetch(`${API_URL}/api/ask`, {
    method: "POST",
    headers,
    body: JSON.stringify({ question, tickers: tickers.length ? tickers : null }),
  });
  if (res.status === 401) {
    throw new UnauthorizedError("Access key missing or wrong. Enter the access key to ask questions.");
  }
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    // 422 from FastAPI returns a list of validation errors; others return a string.
    const detail = Array.isArray(body?.detail)
      ? body.detail.map((d: { msg: string }) => d.msg).join("; ")
      : (body?.detail ?? res.statusText);
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json();
}
