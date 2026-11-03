# Pilot: automatic completion results in the originating chat

State: deployed; live completion callback automatically started a follow-up in
the originating chat, which verified the evidence and acknowledged receipt.
Authorization: on 2026-10-14 the user requested that simulation completion wake
the agent and post results in the chat. This is a local controller change and CPU
notification canary, with no new scientific computation or experiment budget.

## Question and decision

Can the existing persistent controller detect terminal case coverage, retain
verifiable evidence, and deliver a results-review request to the originating
Codex thread? Accept transport separately from the receiving agent's receipt;
do not claim an end-to-end wake until the chat actually handles the event.

## Scope and handoff

Use the existing `run-watch-queue.service` and ledger. Store completion watches,
delivery state and hashed evidence in that ledger/state directory. New CLI
campaign registrations capture `CODEX_THREAD_ID`; direct API registrations and
existing work use `notify-on-completion` with an explicit originating thread.
Already finished work requires deliberate backfill. Preserve all scientific
settings, budget accounting, active workers and existing repair enrollment.

No polling model or second monitoring owner is needed. Terminal coverage triggers
one bounded `codex queue` send per controller cycle. Missing receipt evidence is
reported for verification; partial coverage stays partial. Delivery retries are
bounded and ambiguous interrupted sends are retained without automatic replay.
The receiving chat reads actual artifacts, acknowledges receipt and reports results.

Local CPU only; one synthetic completion callback to this chat, at most three
known-failure transport attempts. No new agent session. Retain code, rollback
inputs, test output and synthetic evidence under
`results/maintenance/completion-chat-20261014/`, capped at 1 GB.

## Evidence

Current-worktree targeted checks: 217 passed, covering receipt validation,
partial terminal coverage, collection waiting, restart deduplication, bounded
retries, timeout ambiguity and acknowledgement races alongside controller and
recovery regressions. Live delivery and deployment receipts will be recorded
below. The host, watcher and Codex app-server must remain reachable.

The isolated controller passed 210 tests on its retained older base; newer unrelated
resource-preemption code from the working tree was excluded. Deployed revision:
`dec696046e3bb383ad686ff5abdc6d977876ae1c`, with PID 2529587 at activation.
`zzz-completion-chat.conf` selects the new controller; `deployment.json` records
activation, configuration hash and rollback inputs. The existing repair permission
configuration is preserved. Ledger comparison preserved jobs, attempts, collection,
contracts, cases, retries, amendments, extensions, investigations and escalations.
The first complete monitoring cycle renewed the systemd watchdog, with the new
PID and no host errors; `deployment-verification.json` records that check.

The live canary used a private queue and two actual CPU subprocesses: sum of
squares for 0..999 produced 332833500, and independent validation exited zero.
No callback was emitted while it was running. Once complete, the real installed
`codex queue` accepted event `14c6ca4742129b7b7d6c0e7f` for this originating chat
on its first attempt. Its hashed evidence and delivery record are under
`canary/state/campaign-completions/` and `canary/delivery.json` in the evidence root.
The original turn was still active at acceptance. After it ended, the queued
completion automatically started a follow-up in this same chat. That follow-up
verified the bundle SHA-256 and report SHA-256, read the handoff and terminal
receipt, confirmed coverage 1/1 with no missing evidence, and ran
`ack-completion`. It returned `already_acknowledged: false`; the durable delivery
state is now `acknowledged`. This verifies automatic follow-up after an active
turn ends. Delivery while the Codex server is stopped was not tested.

Retained artifacts total approximately 235 MB, within the 1 GB cap. No scientific
jobs were launched or restarted. Interactive model token usage is unavailable.
