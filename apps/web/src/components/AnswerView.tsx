"use client";

import type { ReactNode } from "react";

import type { AnswerPart } from "@/lib/api";

export type Selection = { part: number; citation: number };

// Private-use characters never appear in filing text, so they are safe delimiters
// for citation markers embedded in the joined answer string.
const OPEN = "";
const CLOSE = "";
const TOKEN = new RegExp(`(\\*\\*|${OPEN}\\d+:\\d+${CLOSE})`);

/**
 * Join all parts into one string and put a marker after each cited part.
 * Rendering the joined string (not part by part) keeps Markdown that spans
 * several parts - a bullet line or **bold** run - intact.
 */
function withMarkers(parts: AnswerPart[]): string {
  return parts
    .map((part, p) => {
      const seen = new Set<number>();
      const markers = part.citations
        .map((c, i) => {
          if (seen.has(c.source)) return ""; // same source twice in one part: one marker
          seen.add(c.source);
          return `${OPEN}${p}:${i}${CLOSE}`;
        })
        .join("");
      return part.text + markers;
    })
    .join("");
}

type Props = {
  parts: AnswerPart[];
  selected: Selection | null;
  onSelect: (sel: Selection) => void;
};

export default function AnswerView({ parts, selected, onSelect }: Props) {
  // Inline content of one line: **bold** toggles and [n] citation buttons.
  function renderInline(line: string, key: string): ReactNode[] {
    const out: ReactNode[] = [];
    let bold = false;
    line.split(TOKEN).forEach((token, i) => {
      if (token === "**") {
        bold = !bold;
        return;
      }
      const m = token.match(new RegExp(`^${OPEN}(\\d+):(\\d+)${CLOSE}$`));
      if (m) {
        const sel = { part: Number(m[1]), citation: Number(m[2]) };
        const n = parts[sel.part].citations[sel.citation].source + 1;
        const active = selected?.part === sel.part && selected?.citation === sel.citation;
        out.push(
          <button
            key={`${key}-${i}`}
            type="button"
            onClick={() => onSelect(sel)}
            className={`mx-0.5 inline-flex h-4 min-w-4 items-center justify-center rounded px-1 align-super text-[10px] font-semibold ${
              active
                ? "bg-blue-600 text-white"
                : "bg-blue-100 text-blue-700 hover:bg-blue-200 dark:bg-blue-900 dark:text-blue-200"
            }`}
            aria-label={`Show source ${n}`}
          >
            {n}
          </button>,
        );
        return;
      }
      if (token) out.push(bold ? <strong key={`${key}-${i}`}>{token}</strong> : token);
    });
    return out;
  }

  // Block structure: "- " lines become list items, blank lines separate paragraphs.
  const blocks: ReactNode[] = [];
  let listItems: ReactNode[] = [];
  const flushList = () => {
    if (listItems.length) {
      blocks.push(
        <ul key={`ul-${blocks.length}`} className="my-2 list-disc space-y-1 pl-6">
          {listItems}
        </ul>,
      );
      listItems = [];
    }
  };

  withMarkers(parts)
    .split("\n")
    .forEach((raw, i) => {
      const line = raw.trimEnd();
      if (line.startsWith("- ")) {
        listItems.push(<li key={i}>{renderInline(line.slice(2), `li-${i}`)}</li>);
        return;
      }
      flushList();
      if (line.trim()) {
        blocks.push(
          <p key={i} className="my-2">
            {renderInline(line, `p-${i}`)}
          </p>,
        );
      }
    });
  flushList();

  return <div className="leading-relaxed">{blocks}</div>;
}
