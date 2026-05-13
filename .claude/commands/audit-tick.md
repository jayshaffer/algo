---
description: Execute one full audit tick (Phase A discovery + Phase B execution)
---

Run one audit tick by chaining the two phase commands:

1. **Phase A (Discovery)** — read `.claude/commands/audit-discover.md` and execute everything it specifies. Honor the 5-create cap.
2. **Phase B (Execution)** — read `.claude/commands/audit-execute.md` and execute everything it specifies. Honor the 2-per-invocation cap.

Then stop and wait for the next interval.
