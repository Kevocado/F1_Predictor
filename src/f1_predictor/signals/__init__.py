"""signals — one adapter per fixture signal (spec 2026-10-01-fixture-signals-design §4).

Phase 1 ships `trust` only. The contract every adapter here emits is §3's
`Signal`, and it is the shape the shared `SignalRows` component in
`frontend/src/predictor-ui/components/SignalRows.tsx` draws — read that file
before adding a field; it refuses several things a payload could plausibly
carry.
"""