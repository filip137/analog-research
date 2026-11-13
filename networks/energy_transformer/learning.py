"""Frozen small attention-learning task; inputs vary independently by sample."""
import numpy as np
from networks.energy_transformer.convergent import Parameters, Boundary


def initial_parameters():
    rng=np.random.default_rng(20261113)
    shapes=((1,1,3),(1,1,3),(2,3),(3,1),(1,3),(3,3))
    return Parameters(*[rng.normal(0.,.2 if i==5 else .08,shape)
                        for i,shape in enumerate(shapes)],kappa=1.)


def boundary(pair):
    a,b=np.asarray(pair,dtype=float)
    return Boundary(np.array([[a],[b],[0.]]),
                    np.array([[a],[b],[(a+b)/2]]),np.array([[0.],[0.],[1.]]))


def dataset(seed,count):
    return np.random.default_rng(seed).uniform(-.4,.4,(count,2))


def measured_residual(ports,boundary,nudge):
    return float(max(np.max(np.abs(ports['token_drive']-ports['z'])),
                     np.max(np.abs(ports['output_drive']-ports['y']-
                                   nudge*boundary.mask*(ports['y']-boundary.target)))))
