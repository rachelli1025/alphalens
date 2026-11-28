import type { Citation, Source } from "@/lib/api";

type Props = {
  citation: Citation | null;
  source: Source | null;
};

export default function CitationPanel({ citation, source }: Props) {
  if (!citation || !source) {
    return (
      <p className="text-sm text-neutral-500">
        Click a numbered marker in the answer to see the exact passage it comes from.
      </p>
    );
  }

  return (
    <div className="space-y-3">
      <div>
        <p className="text-xs font-semibold uppercase tracking-wide text-blue-600">
          Source {citation.source + 1}
        </p>
        <p className="mt-1 text-sm font-medium">
          {source.ticker} · {source.form_type}
          {source.fiscal_year ? ` FY${source.fiscal_year}` : ""}
        </p>
        <p className="text-sm text-neutral-500">
          Item {source.item_code}
          {source.section_title ? ` · ${source.section_title}` : ""}
        </p>
      </div>

      <blockquote className="whitespace-pre-wrap border-l-4 border-blue-300 bg-blue-50 px-3 py-2 text-sm leading-relaxed dark:bg-blue-950/40">
        {citation.cited_text.trim()}
      </blockquote>

      <a
        href={source.url}
        target="_blank"
        rel="noopener noreferrer"
        className="inline-block text-sm font-medium text-blue-600 hover:underline"
      >
        Open the filing on SEC.gov →
      </a>
    </div>
  );
}
