# Pilot: operational recovery worker permissions

State: complete; deployed and verified. Authorization: the user requested extending the recovery
worker's permissions on 2026-10-14, following the diagnosis of exp-010's
read-only investigation. This is a controller change and CPU canary, not a new
scientific experiment or authorization to reopen expired work.

## Question and decision

Can an enrolled recovery worker reproduce an operational code defect, repair it
in isolation, test and deploy the fix, and retain verifiable evidence without a
new user message? Enable the extended permission only after focused regression
tests and an actual bounded worker canary succeed.

## Frozen comparison

Default: existing read-only investigation followed by narrowly classified
recovery. Intervention: `campaign_escalation.allow_code_repairs: true` allows
operational code/environment repairs by the existing repair owner. The
investigator, campaign enrollment, attempt limits, budgets, science and deadline
checks remain unchanged. A new `repair_applied` report requires pinned patch,
old/new source identities, passing tests and a post-deployment verification log.
It does not authorize a retry or imply scientific completion.

## Budget and execution

Local CPU on nom-cool-2; no GPU experiments. Focused tests plus at most two
300-second actual Codex canary invocations, using the configured recovery model
and effort. Evidence and a new service checkout live under
`results/maintenance/operational-repair-20261014/`; cap 1 GB new artifacts.
Preserve the original service checkout and configuration for rollback. Deploy
only after readiness passes, retaining the existing shared service and ledger.
Keep existing campaign IDs enrolled and use a fresh activation timestamp so old
incidents and expired experiments are not reopened.

## Current execution and monitoring handoff

Deployment verification is complete. The existing
`run-watch-queue.service` owns production scheduling. Before deployment its PID
was 1384148, running the exp-013 controller at source
`7bf27fd2579e019930425c23644f2e5c12b6cbfc`. Read-only ledger inspection found no
active attempts. No second production watcher or scheduler is created.

The service now uses the separate maintenance controller at revision
`e15f9bd61c0ffd749a599bce0cd62e2aa73e3021`, PID 2509365. The two existing exp-013
campaign IDs remain enrolled. `allow_code_repairs` is true and `not_before` was
advanced to activation time, so historical incidents are not reissued. The
300-second worker timeout, two calls per incident and six per day are unchanged.
The effective systemd override is `zz-operational-repairs.conf`; config, prior
ledger snapshot, activation receipt and rollback inputs are in the evidence root.

## Evidence and coverage

Focused execution-service and escalation tests passed: 199 tests in the current
worktree; 192 tests on the actual deployed controller's older base. The deployed
controller preserves that base's execution service rather than bundling unrelated
newer resource-preemption changes. System/default conda Python lack pytest; checks use
`/home/filip/miniconda3/envs/py312/bin/python`.

The actual Codex worker ran twice on an isolated CPU storage-lookup fixture,
using a private database, the configured model/effort, and the real escalation
supervisor. `canary/summary.json` preserves the first failed acceptance: the
worker repaired/deployed/probed successfully, but emitted structured source
identities that the initial receipt parser rejected. The corrected parser accepts
either revision strings or absolute-path/revision objects and rejects unchanged
revisions even if their paths differ. Regression coverage includes that case.

`canary-2/summary.json` passed in 67.17 seconds without intervention: initial
probe failed, worker cloned the frozen fixture, patched it, passed four unittest
cases plus diff checking, committed and activated the repair, passed a fresh
probe, wrote hashed evidence, and was accepted as `repair_applied`. The original
tracked source files remained byte-identical. This was CPU-only; no production
queue or GPU experiment was used by either canary.

The first service activation failed because `hwa-campaign.conf` sorted after the
new numeric override and kept the old checkout. Automatic rollback restored the
original service/config; evidence is in `deployment-attempt-1/`. The corrected
override sorts last and deployment now asserts the effective working directory
before restart. `deployment.json` records the successful activation. The first
fresh heartbeat matched the new PID without host errors; ledger comparison
preserved jobs/specs/states/budgets, attempts, retries, contracts, amendments,
investigations and escalation history. The complete campaign monitoring cycle
then finished with no host errors and all those invariants still preserved;
`deployment-verification.json` records the final verification. The production
service remains the monitoring owner.

## Interpretation and next decision

The extended worker repaired the injected operational defect autonomously.
Production activation preserves existing enrollment and limits; it does not
enroll every experiment or reopen expired work. Evidence receipt validation verifies retained bytes and successful
reported checks, not arbitrary patch semantics. Unsupported source transitions,
scientific choices, expired allowance or denied access remain blockers.

## Workflow audit

Canary execution times: 65.21 and 67.17 seconds. Model usage reported by their
event logs: first 189,730 input tokens (156,544 cached) and 2,488 output; second
193,367 input (158,976 cached) and 2,427 output. Interactive token usage is
unavailable. Artifacts plus the isolated controller occupy about 243 MB, within
the 1 GB cap. Preparation, implementation and deployment were not separately
timed. No scientific computation was performed.
