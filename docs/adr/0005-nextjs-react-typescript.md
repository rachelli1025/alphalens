# ADR-0005: Next.js with React and TypeScript for the frontend

- **Status:** Accepted
- **Date:** 2026-11-25

## Context
The UI has several routed pages (research, company, feed, signals, admin), streaming chat, and complex API payloads (citations, events, financial series).

## Decision
Use Next.js 15 (App Router), React 19, and TypeScript. Generate API types from the FastAPI OpenAPI schema with `openapi-typescript`.

## Consequences
- Type errors surface at compile time when backend contracts change.
- Deploys simply to Vercel.
- If SSR proves unnecessary, Vite + React + TypeScript is a drop-in alternative for the same components.
