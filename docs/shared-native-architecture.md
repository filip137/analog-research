# Shared native drift code

The coordinated HWA architecture migration moves the shared numerical kernels
into `hardware/drift/`, calibration and recovery algorithms into `methods/`,
artifact/cache utilities into `core/`, and shared execution/sharding into `ebl/`.
The DRN network and solver architecture is unchanged. Existing DRN recovery
callers use these owners directly.

[shared_native_modules.json](shared_native_modules.json) names the 15 files
which must remain byte-identical with HWA. From the HWA checkout, run
`python -m benchmarks.verify_shared <hwa-checkout> <drn-checkout>`. Each backend
remains independently importable; never import another worktree's code.

The migration preserves existing physical equations, state keys, seeds, counters
and fixed-episode replay. It does not adopt experiment-branch drift refinements
or change this tree's scientific contracts. Reproduce old runs using their
pinned source revisions; no blanket `training.*` compatibility aliases remain
for the migrated modules. Other DRN training modules retain their current owners.

Validation: 24 focused power-law, writable-drift and OPT recovery tests passed
on CPU; all shared module bytes match. The HWA migration pilot owns the wider
architecture review and paired CPU benchmark.
