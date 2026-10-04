"""Interactive harness used by ``labs/``, built on ``training.core``.

``epoch`` binds loaders and statistics to the core loops (``Trainer`` and
``Evaluator``), ``statistics`` holds the per-batch metrics, ``monitor`` runs
epochs and records those metrics, and ``diagnostics`` holds optional engine
observers and the BetaSize pass. Workflow-managed runs do not use this package.
"""
