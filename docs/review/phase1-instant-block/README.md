# Phase 1 site task — verification screenshots

Review artifacts for PR "feat: instant session record for F1", captured in real
Chromium (Playwright 1.63) against the real dev server, with a fixture API
supplying the session responses. Nothing here is loaded by the app; the folder
exists so the screenshots are reviewable, and it can be deleted in one commit
without touching the feature.

Every shot was taken while asserting its own claim in the same pass — the DOM
checks and the image come from one run, so no screenshot here is the only
evidence for a figure it shows.

| file | viewport | what it shows |
| --- | --- | --- |
| `01-desktop-block-record-before-press.png` | 1440px | The block on screen BEFORE the button, with no request: timing chip, verdict, record `4/7`. The button is still there, unpressed. |
| `02-desktop-after-press-each-figure-once.png` | 1440px | After pressing: the block is still on top and the AI summary lands BELOW it. Verdict, record label, `4/7`, `record-fill` and timing chip each appear exactly once in the panel. |
| `03-390-block-record-before-press.png` | 390px | The same pre-press state at mobile width. |
| `04-390-after-press-each-figure-once.png` | 390px | After pressing at 390px: each figure still exactly once. |
| `05-desktop-rebuilt-excluded.png` | 1440px | A rebuilt session. The header carries only the tier badge; the block's `Rebuilt after the session` disclosure is the only one; the record is a dash, never `0/0`, because every row was rebuilt. |
| `06-390-rebuilt-excluded.png` | 390px | The same rebuilt case at mobile width. |

The bars on the driver rows below the panel are the prediction table's own, and
are left alone deliberately — decision 8 keeps F1's insight per-race. The claim
under test is that the BLOCK draws no bar, which is why the assertion is scoped
to the block rather than the document.
