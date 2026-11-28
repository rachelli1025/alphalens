"""Self-hosted research agent loop on the Messages API (DESIGN.md §9.3, ADR 0003).

    while the model asks for tools (stop_reason == "tool_use"):
        run every requested tool, send all results back in one user message
    the last response is the answer

Rules the API depends on:
- The assistant turn is appended exactly as returned (including thinking blocks).
- All tool results of one turn go back together, in one user message.
- After MAX_TOOL_CALLS, remaining requests get an error result and tool use is turned off
  (tool_choice "none") so the model must answer with what it has.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import anthropic

from alphalens_core.agent.tools import (
    TOOL_DEFINITIONS,
    ToolContext,
    chunk_id_from_source,
    run_tool,
)
from alphalens_core.answer import AnswerPart, Citation
from alphalens_core.pricing import cost_usd

log = logging.getLogger(__name__)

MAX_TOOL_CALLS = 8

SYSTEM_PROMPT = """\
You are AlphaLens, a research assistant for US equities. Answer using the tools; do not \
rely on outside knowledge for facts about companies.

Companies with data (ticker: name):
{companies}

How to work:
- Numbers (revenue, margins, growth, rankings, comparisons): use query_financials. \
Compute ratios in SQL. Say which fiscal year and period each number is for.
- Explanations, risks, strategy, what management says: use search_filings and cite the \
passages you rely on.
- Many questions need both: get the numbers, then search for the reasons.
- If a query is rejected or fails, read the error and fix the SQL.
- If the data does not answer the question, say so plainly instead of guessing.
- Be concise: a direct answer first, then the supporting points.
- Do not give investment advice or recommendations to buy or sell."""


@dataclass
class ToolCall:
    name: str
    input: dict[str, Any]
    summary: str
    is_error: bool


@dataclass
class AgentResult:
    parts: list[AnswerPart]
    tool_calls: list[ToolCall]
    sql_queries: list[str]
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    stop_reason: str = ""
    hit_tool_limit: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "".join(p.text for p in self.parts)

    @property
    def cost_usd(self) -> float | None:
        return cost_usd(
            self.model,
            self.input_tokens,
            self.output_tokens,
            cache_write_tokens=self.cache_write_tokens,
            cache_read_tokens=self.cache_read_tokens,
        )


def _parse_answer(content: list[Any], ctx: ToolContext) -> list[AnswerPart]:
    parts = []
    for block in content:
        if getattr(block, "type", None) != "text":
            continue
        citations = []
        for c in block.citations or []:
            if getattr(c, "type", None) != "search_result_location":
                continue
            chunk_id = chunk_id_from_source(c.source)
            if chunk_id is not None and chunk_id in ctx.chunks:
                citations.append(Citation(chunk=ctx.chunks[chunk_id], cited_text=c.cited_text))
        parts.append(AnswerPart(text=block.text, citations=citations))
    return parts


def run_agent(
    client: anthropic.Anthropic,
    model: str,
    question: str,
    ctx: ToolContext,
    companies: dict[str, str],
    *,
    effort: str = "medium",
    max_tool_calls: int = MAX_TOOL_CALLS,
) -> AgentResult:
    system = SYSTEM_PROMPT.format(companies="\n".join(f"{t}: {n}" for t, n in companies.items()))
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    result = AgentResult(parts=[], tool_calls=[], sql_queries=ctx.sql_queries, model=model)
    tools_allowed = True

    while True:
        response = client.messages.create(
            model=model,
            max_tokens=16000,
            system=system,
            tools=TOOL_DEFINITIONS,
            tool_choice={"type": "auto" if tools_allowed else "none"},
            output_config={"effort": effort},
            # Caches the growing prefix (system, tools, earlier turns) between iterations.
            cache_control={"type": "ephemeral"},
            messages=messages,
        )
        usage = response.usage
        result.input_tokens += usage.input_tokens
        result.output_tokens += usage.output_tokens
        result.cache_write_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0
        result.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
        result.model = response.model
        result.stop_reason = response.stop_reason or ""

        if response.stop_reason != "tool_use":
            result.parts = _parse_answer(response.content, ctx)
            if not result.parts and response.stop_reason == "refusal":
                result.parts = [AnswerPart("The model declined to answer this question.")]
            return result

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            if len(result.tool_calls) >= max_tool_calls:
                output_content: Any = "Tool call limit reached. Answer with what you have."
                is_error, summary = True, "skipped: tool call limit"
                result.hit_tool_limit = True
            else:
                out = run_tool(ctx, block.name, block.input)
                output_content, is_error, summary = out.content, out.is_error, out.summary
            log.info("tool %s %s -> %s", block.name, block.input, summary)
            result.tool_calls.append(ToolCall(block.name, dict(block.input), summary, is_error))
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output_content,
                    "is_error": is_error,
                }
            )
        messages.append({"role": "user", "content": tool_results})
        if result.hit_tool_limit:
            tools_allowed = False
