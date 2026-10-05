# Trust badge — production captures

The `trust` row on the F1 session panel, in production, at both widths the
standing rule asks for. Captured against `https://f1.40-160-91-131.sslip.io/`
with Playwright (the method `predictor-ui/harness/README.md` names: "A Playwright
run that sets `viewport: { width: 390 }` measures the real thing").

| file | what |
|---|---|
| `desktop-session-panel.png` | 1440px, Australian GP selected |
| `mobile-390-session-panel.png` | 390px, same fixture |

Both show, under **CONTEXT** and above the "made after the session" badge and the
pick:

> When the model says ~10%, its picks landed 12% of the time
> `SOURCE  n=1235 · Sat 3 Oct · 11:44 AM CDT`

with the reliability bar drawn beneath the sentence.

## What the captures are evidence OF

A screenshot proves a row was *drawn*; only the text says whether it was the
*right* row. So each capture was made with the rendered `textContent` read back
out of the page and printed, and both report:

```
trust row present: true
badge line: When the model says ~10%, its picks landed 12% of the time
```

Every request outside the page's own origin is blocked, so a capture is a
property of the page rather than of whatever the network was doing.

## What to look at

- **The row is above the explainer, not below the facts block.** Spec §2 makes
  signals instant — computed from stored data, no model call — so gating them
  behind the "Get the AI summary" button would have made a free computed row a
  paid one. It is on screen with the instant block.
- **No button was pressed** to produce either capture.
- **`n=1235` is on screen next to the rate**, which is the sample the 12% is
  drawn from. This is the one market measured good enough to ship: 1,452 gradable
  pairs across 7 of 10 probability bands clear of the `n >= 30` floor.
- **The wording is past tense** — "its picks landed" — because this is a record,
  not a forecast, and the tilde on `~10%` keeps the sentence true of the whole
  0.0–0.3 band rather than of picks of exactly 10%.
- It sits directly above "MADE AFTER THE SESSION — This pick was made after the
  session started", which is the honesty badge doing its own job one line below
  the new one. Neither contradicts the other: one is a record of every resolved
  pick in the band, the other is a disclosure about *this* fixture's pick.

## Reaching this

Two defects had to be fixed first, and neither was visible from the code alone:

1. **The endpoint was unreachable from the page.** `signals_router` was mounted
   at the root while the frontend addresses the backend through `/api` in both
   production and development. PR #36.
2. **No page rendered the component.** It was vendored, exported, and drawn by
   exactly one test. PR #36.