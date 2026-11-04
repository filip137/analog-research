"""Lazy public routing for lifecycle planning, collection and scheduling."""
from contextlib import redirect_stderr, redirect_stdout
from importlib import import_module

COMMANDS = {
    'plan': ('plan', ('plan',)),
    'collect': ('plan', ('collect',)),
    'recovery': ('recovery', ()),
    'tuning': ('tuning', ()),
    'queue': ('queue_execution', ()),
    'bounded-recovery': ('bounded_recovery', ()),
    'slurm-recovery': ('slurm_recovery', ()),
    'slurm-recovery-watch': ('slurm_recovery_watch', ()),
    'staging': ('staging', ()),
}

def run(action, argv, *, stdout, stderr):
    module, prefix = COMMANDS[action]
    entry = import_module('ebl.lifecycle.' + module)
    with redirect_stdout(stdout), redirect_stderr(stderr):
        if action == 'slurm-recovery-watch':
            if argv:
                raise ValueError('The Slurm watch consumes its request from stdin.')
            result = entry.main()
        else:
            result = entry.main([*prefix, *argv])
    return 0 if result is None else result
