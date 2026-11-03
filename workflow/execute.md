# Execute and observe

Assigned execution follows the [policy](policy.md). Launch from the pinned source
checkout and retain failed and partial attempts.

```bash
python -m ebl campaign run --manifest <plan-root>/campaign.json --output-dir <runs-root>
python -m ebl campaign run --manifest <plan-root>/campaign.json --output-dir <runs-root> --resume
```

For a slice, repeat `--stage NAME`. Dependencies must already have matching
completed records, be selected explicitly, or be supplied through declared
`--input STAGE:ROLE=PATH` bindings. Source, configuration and input fingerprints
plus artifact integrity control reuse. Completion alone never authorizes reuse.
`ebl campaign artifact --manifest ... --output-dir ... --stage NAME --kind KIND`
resolves an unchanged completed artifact without launching or writing records.

Keep one monitoring owner and inspect real artifact progress alongside process
state. Use the existing `ebl.execution.service` and worker interfaces when the
assignment needs them; do not make another campaign-specific stage runner.
Record observed failures and bounded retries; escalate scientific or budget changes.

Lifecycle stages do not support arbitrary mid-stage training resume. Completed
stage reuse and declared drift-shard continuation have separate contracts.
