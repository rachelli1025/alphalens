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
  refused: boolean;
};

export async function askQuestion(question: string, tickers: string[]): Promise<AskResponse> {
  const res = await fetch(`${API_URL}/api/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, tickers: tickers.length ? tickers : null }),
  });
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
