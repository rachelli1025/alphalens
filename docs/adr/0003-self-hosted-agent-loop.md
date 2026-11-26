# ADR-0003: Run the research agent's tool-use loop inside FastAPI

- **Status:** Proposed (see DESIGN.md D2)
- **Date:** 2026-11-25

## Context
Every agent tool queries the application's own Postgres (retrieval, guarded SQL, events, prices). The loop needs per-user quotas, tracing that links to API requests, and SSE streaming of tool status to the browser.

## Decision
Implement the loop on the Anthropic Messages API inside FastAPI: model `claude-opus-5-5`, adaptive thinking, streaming, prompt caching on the system prompt and tools, a cap of 8 tool calls per turn, and refusal handling with server-side fallback.

## Consequences
- Tools run in-process next to the database, with low latency and no network hop for tool results.
- Full control over tracing, quotas, and cost accounting.
- The team owns loop correctness: tool-result ordering, error results, and stop-reason handling.

## Alternatives considered
- **Claude Managed Agents:** Anthropic hosts the loop and a sandbox, with less code to own. It is in beta and billed per session runtime on top of tokens, and every tool would still call back into the application's database.
