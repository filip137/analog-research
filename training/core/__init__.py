"""The minimal training step that ``python -m ebl`` runs for every minibatch.

``engine`` owns the step order (``train_epoch``, ``evaluate`` and their
events), ``sgd`` computes gradients (EP, backprop, direct readout),
``optimizers`` builds the digital update backend, ``modifier`` is the seam for
temporary parameter realizations, ``batch`` normalizes loader output, ``probes``
measures evaluation passes, and ``guards`` rejects non-finite gradients.

Nothing here imports ``training.lab`` or a device model; ``build_optimizer``
imports the Tiki-Taka backend lazily, only when an update pipeline asks for it.
"""
