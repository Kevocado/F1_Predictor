# Phase 1 live follow-ups — F1

Real Chromium (`chromium-headless-shell` via `playwright-core`) against this
branch's Vite dev server. F1's session panel is a panel on the races page rather
than a modal behind a click, and `RacesPage` opens on the next unplayed round by
itself — so the default view *is* the pre-game state and no test-only hook was
needed to reach it. The completed round is selected with the site's own control
(the race `<select>` on wide viewports, the race list on narrow ones).

The API is a stand-in harness process serving the repo's own
`data/public_snapshot.json` over the paths the Vite `/api` proxy forwards to.
Nothing in the page is mocked; the harness is not site code and is not in the
diff. Desktop is 1280×1000, mobile is 390×844.

| file | what it shows |
|---|---|
| `desktop-races.png`, `mobile-390-races.png` | the races page on the next unplayed round (R16, Bahrain GP): block, then the AI button, then the timing tower |
| `desktop-above-ai-button.png`, `mobile-390-above-ai-button.png` | the region above the AI button — nothing between the record strip and the button |
| `desktop-completed-round.png`, `mobile-390-completed-round.png` | a completed round (R1, Australian GP): the two real finished sentences, with the "Rebuilt after the session" badge |
| `measurement.json` | the raw DOM measurements, both viewports, both states |

## What the live DOM said

Measured inside the panel, at **both** viewports:

```
pre-game (R16)   flowRows: []      flowHeadings: []   flowText: ""
                 percentages in the block: []
                 figures in both the block and the site: []

completed (R1)   flowRows: ["P: The result is a win for Russell.",
                            "P: The pick rightness: the model's pick was right."]
                 flowHeadings: []
                 percentages in the block: []
                 figures in both the block and the site: []
```

No orphan heading in either state, and the finished state's two sentences are
present — which is the point of asserting the rows by tag and text rather than
just checking that no heading exists.

## The block carries no percentage

F1 hands the instant block no tiles and no bar (spec §A, decision 8), so the
block contributes a timing badge, a verdict sentence and the record strip, and
**no percentage at all**. Every probability on the panel belongs to the timing
tower. That is why the figure audit is clean, and it is measured
(`percentagesInBlock: []`) rather than assumed.

Stated precisely, because the first draft of the test's comment was wrong: the
block is not figure-*free*. Its record strip carries two numbers, the hits and
the settled count (`24/32` in the shot). What it carries is no percentage and
no decimal — so there is no probability on the panel that the block could be
duplicating.

## Figures compared as whole tokens

Figures are compared as whole tokens, never as substrings: `"7%"` occurs inside
`"17%"` and `"4%"` inside `"54%"`, and a substring sweep on this very panel
reports dozens of phantom duplications. F1's tower prints 50+ percentages, so
the substring form is badly misleading here.

## Regenerating

The harness scripts are not committed (they live in `/tmp`). With the Vite dev
server up and the stand-in API on `127.0.0.1:8000`:

```
node /tmp/f1-shoot.mjs http://127.0.0.1:<vite-port> <out-dir>
```
