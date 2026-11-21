"""Strict first-study configuration; input/output paths belong to CLI requests."""

from dataclasses import asdict, dataclass, fields
import math
from typing import Mapping

from ebl.schema.base import ConfigError, RunMode

EXPERIMENT_ID='opt_cmo_drn_hwa.v1'
TEACHER_SHA256='2d74da6615135c58cf3cf9ad4cb11e7c613ff9e55fe658a47ab83b6c8d1174a9'
SOURCE_SHA256='86522e230dfac9646beffbd84d244fa7cdf7ecdbac4b9f3db2de29dcbab2b028'
# 12-MLP crossbar std3-recipe source (hwa exp/drift-all-mlps, selected at epoch 98).
SOURCE_L12_SHA256='ed53b6cbe2a7315f364291c7142ee2826c6ca4a44c21a2d5d25190b8c90569b0'
# The last 6 or all 12 decoder MLPs are DRNs. reconstruct_source rejects a
# source whose analog weights do not match the declared depth.
ANALOG_DEPTHS=(6,12)


from networks.config.opt import Solver


from networks.config.opt import Data


from methods.config.opt import Training


from methods.config.opt import Selection


from hardware.deployment_config import Deployment, MappedDeployment


from methods.config.opt import Recovery


@dataclass(frozen=True)
class Config:
    experiment_id: str=EXPERIMENT_ID
    schema_version: int=1
    stage: str='hwa'
    model: str='facebook/opt-125m'
    analog_layers: int=6
    initialization: str='full_span_quad.v1'
    device: str='cpu'
    cpu_threads: int=2
    teacher_sha256: str=TEACHER_SHA256
    source_sha256: str=SOURCE_SHA256
    solver: Solver=Solver()
    data: Data=Data()
    training: Training=Training()
    selection: Selection=Selection()
    deployment: Deployment=Deployment()
    recovery: Recovery=Recovery()
    retention: object=None
    drift_recovery: object=None


@dataclass(frozen=True)
class UpdateExecution:
    tf32: bool = True
    checkpoint_gradient_only: bool = False
    skip_zero_feedback: bool = False
    microbatch: int = 1
    token_chunk_size: int = 256


@dataclass(frozen=True)
class ExecutionConfig(Config):
    hwa_execution: UpdateExecution = UpdateExecution()


@dataclass(frozen=True)
class FaultTuningConfig(Config):
    """The fault-paper screen's own stage; a subclass keeps every older config digest unchanged."""
    fault_tuning: object=None


@dataclass(frozen=True)
class DriftTuningConfig(FaultTuningConfig):
    """Fault-tuning learner under AIHWKit CMO log-time drift; a subclass keeps older digests unchanged."""
    drift_tuning: object=None


def _parse(cls,raw,path):
    if not isinstance(raw,Mapping):
        raise ConfigError(f'Expected {path} object. Provided value: {raw!r}.')
    known={f.name for f in fields(cls)}
    if set(raw)-known:
        raise ConfigError(f'Expected known {path} fields {sorted(known)}. Provided value: {sorted(set(raw)-known)}.')
    values={}
    defaults=cls()
    for key,value in raw.items():
        default=getattr(defaults,key)
        if issubclass(cls,DriftTuningConfig) and key=='drift_tuning':
            if value is not None:
                from methods.drift_tuning import DriftTuning
                try: value=DriftTuning.parse(value)
                except ValueError as error: raise ConfigError(str(error)) from error
        elif issubclass(cls,FaultTuningConfig) and key=='fault_tuning':
            if value is not None:
                from methods.config.fault_tuning import FaultTuning
                try: value=FaultTuning.parse(value)
                except ValueError as error: raise ConfigError(str(error)) from error
        elif issubclass(cls,Config) and key=='drift_recovery':
            if value is not None:
                from methods.maintenance.config import DriftRecovery
                try: value=DriftRecovery.parse(value)
                except ValueError as error: raise ConfigError(str(error)) from error
        elif issubclass(cls,Config) and key=='deployment' and isinstance(value,Mapping) and 'source_evaluation' in value:
            from hardware.deployment_config import SourceEvaluationDeployment
            value = _parse(SourceEvaluationDeployment, value, f'{path}.{key}')
        elif issubclass(cls,Config) and key=='deployment' and isinstance(value,Mapping) and 'mapping_scale' in value:
            # Explicit deployment remap; legacy deployments keep their resolved identity.
            value=_parse(MappedDeployment,value,f'{path}.{key}')
            scale=float(value.mapping_scale)
            value=MappedDeployment(**{**asdict(value),'mapping_scale':scale})
            if not 1<=scale<=88.2/9.:
                raise ConfigError(f'Expected deployment.mapping_scale in [1, 9.8]. Provided value: {scale!r}.')
        elif issubclass(cls,Config) and key=='retention':
            if value is not None:
                from ebl.schema.retention import Retention
                try: value=Retention.parse(value)
                except ValueError as error: raise ConfigError(str(error)) from error
        elif hasattr(default,'__dataclass_fields__'):
            value=_parse(type(default),value,f'{path}.{key}')
        elif isinstance(default,tuple):
            if not isinstance(value,(tuple,list)) or not value:
                raise ConfigError(f'Expected nonempty {path}.{key} array.')
            value=tuple(value)
        elif isinstance(default,bool):
            if type(value) is not bool: raise ConfigError(f'Expected boolean {path}.{key}.')
        elif cls is Recovery and key=='max_write_attempts' and value is None:
            pass
        elif isinstance(default,int):
            if type(value) is not int: raise ConfigError(f'Expected integer {path}.{key}.')
        elif isinstance(default,float):
            if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value):
                raise ConfigError(f'Expected finite numeric {path}.{key}.')
        elif not isinstance(value,str):
            raise ConfigError(f'Expected string {path}.{key}.')
        values[key]=value
    return cls(**values)


def parse_config(raw):
    mapping=isinstance(raw,Mapping)
    c=_parse(DriftTuningConfig if mapping and 'drift_tuning' in raw else
             FaultTuningConfig if mapping and 'fault_tuning' in raw else
             ExecutionConfig if mapping and 'hwa_execution' in raw else Config,raw,'config')
    if isinstance(c, ExecutionConfig):
        execution = c.hwa_execution
        if (execution.microbatch < 1 or execution.microbatch > c.training.effective_batch
                or execution.token_chunk_size < 1):
            raise ConfigError('Invalid HWA update microbatch or token chunk size.')
    if c.experiment_id!=EXPERIMENT_ID or c.schema_version!=1 or c.stage not in ('prepare','hwa','deploy','recover','recovery_benchmark','hwa_precision_benchmark','hwa_tf32_benchmark','hwa_speed_benchmark','retention','drift_recovery','fault_tuning'):
        raise ConfigError('Expected opt_cmo_drn_hwa.v1 schema 1 and a supported stage.')
    if (c.stage=='retention') != (c.retention is not None):
        raise ConfigError('The retention stage requires its explicit settings; other stages omit them.')
    if (c.stage=='drift_recovery') != (c.drift_recovery is not None):
        raise ConfigError('Custom drift recovery requires its explicit settings only on its native stage.')
    if (c.stage=='fault_tuning') != (getattr(c,'fault_tuning',None) is not None):
        raise ConfigError('Fault tuning requires its explicit settings only on its native stage.')
    if getattr(c,'fault_tuning',None) is not None:
        f=c.fault_tuning
        nominal=c.deployment.fault_kind=='none' and c.deployment.fault_rate==0
        if not nominal and (c.deployment.fault_kind!='random' or not 0<c.deployment.fault_rate<1):
            raise ConfigError('Fault tuning reads a random-fault or a fault-free (0 %) deployment.')
        if (c.recovery.threshold_sigma!=2 or c.recovery.target_bounds!='gmin_gmax' or c.recovery.conductance_floor!='none'
                or c.recovery.max_write_attempts is not None):
            raise ConfigError('Fault tuning keeps the DRN 2-rho gate, bounded targets, no floor and no write cap.')
        if f.recovery_sequences%f.fisher_batch:
            raise ConfigError('Fisher batches must tile the recovery set.')
        from methods.config.fault_tuning import DRIFT_ONLY
        if getattr(c,'drift_tuning',None) is None and any(a.method in DRIFT_ONLY for a in f.arms):
            raise ConfigError(f'Arm methods {DRIFT_ONLY} run only in the drift-tuning stage.')
    if c.deployment.mapped and (c.stage not in ('deploy','fault_tuning') or getattr(c,'drift_tuning',None) is not None):
        raise ConfigError('An explicit deployment mapping_scale is applied only by the deploy and fault_tuning '
                          'stages; drift tuning declares its own drift_tuning.mapping_scale.')
    if getattr(c,'drift_tuning',None) is not None:
        d,f=c.drift_tuning,c.fault_tuning
        if f is None or c.deployment.fault_kind!='none' or c.deployment.fault_rate!=0:
            raise ConfigError('Drift tuning runs the fault-tuning stage on a fault-free deployment.')
        methods=[a.method for a in f.arms]
        off=set(d.off_labels)
        if not off<={a.label for a in f.arms}:
            raise ConfigError('Every drift_off_arms label must name a declared arm.')
        if d.write_clock=='continues' and d.gate=='target_rho' and not off:
            if methods.count('hold')!=1 or not set(methods)<={'hold','noisy'} or 'noisy' not in methods:
                raise ConfigError('Drift tuning takes one hold (untreated) arm and noisy learner arms.')
        else:
            drifting=[a.method for a in f.arms if a.label not in off]
            if drifting.count('hold')!=1 or len(drifting)<2:
                raise ConfigError('Drift tuning takes one drifting hold arm (A0) and at least one drifting learner.')
            if not set(methods)<={'hold','noisy','child_d','calibration','layernorm'}:
                raise ConfigError('Drift tuning arms are hold, noisy, child_d, calibration and layernorm.')
            if any(a.method not in ('hold','noisy','child_d') for a in f.arms if a.label in off):
                raise ConfigError('Drift-off arms are hold (A0-twin) or physical learners.')
        horizon=d.age(f.steps)
        if any(t>0 and t<=horizon and d.step_at(t) is None for t in d.evaluation_seconds):
            raise ConfigError('Every evaluation age inside the learning window must be a learner update age.')
        if d.projection_seconds not in d.evaluation_seconds or d.projection_seconds<=horizon:
            raise ConfigError('The projection age must be a post-window evaluation age, paired with the hold arm.')
        if d.projection_steps and d.projection_steps[-1]>f.steps:
            raise ConfigError('Projection steps must lie within the learner updates.')
    if c.drift_recovery is not None:
        if c.deployment.fault_kind!='none' or c.deployment.fault_rate!=0:
            raise ConfigError('Custom recovery requires a fault-free deployment.')
        if (c.drift_recovery.assignment_seeds != (c.deployment.assignment_seed,)
                or (c.drift_recovery.law.g_min,c.drift_recovery.law.g_max)!=(9.,88.2)):
            raise ConfigError('Custom recovery must preserve the declared DRN identity and mapping window.')
    if c.retention is not None:
        if c.deployment.fault_kind!='none' or c.deployment.fault_rate!=0:
            raise ConfigError('Scalar drift currently requires a nominal fault-free deployment.')
        if c.retention.power_law.development_sequences>c.data.development_sequences:
            raise ConfigError('Drift cohort exceeds the declared development cohort.')
    if hasattr(c.deployment,'source_evaluation') and (c.model!='facebook/opt-1.3b' or c.stage!='deploy'
            or c.deployment.source_evaluation!='cmo_24h_x1_x4' or c.deployment.assignment_seed in c.selection.assignment_seeds):
        raise ConfigError('Source quality requires a held-out 1.3B deployment and the declared CMO protocol.')
    if c.model == 'facebook/opt-1.3b':
        if (c.analog_layers not in (12, 24) or c.initialization != 'pretrained_ratio3_quad.v1'
                or c.source_sha256 != c.teacher_sha256
                or c.teacher_sha256 != 'cf7d5c970d6ddbd3b03009b397c0422e147edd5c8020d47a8d2fac0b11a3b08d'
                or c.deployment.mapped or c.stage not in ('prepare', 'hwa', 'deploy', 'hwa_precision_benchmark', 'hwa_tf32_benchmark','hwa_speed_benchmark')):
            raise ConfigError('OPT-1.3B requires its pinned pretrained teacher, native ratio3 initialization and L12/L24.')
    elif c.model!='facebook/opt-125m' or c.analog_layers not in ANALOG_DEPTHS:
        raise ConfigError('Expected OPT-125M with six or twelve DRN MLPs.')
    elif c.initialization!='full_span_quad.v1':
        raise ConfigError('Expected the assigned full-span four-device initialization.')
    for digest in (c.teacher_sha256,c.source_sha256,c.data.sha256):
        if len(digest)!=64 or any(x not in '0123456789abcdef' for x in digest):
            raise ConfigError('Expected lowercase SHA-256 identities.')
    if c.solver.settling_iterations!=4 or c.solver.gradient_iterations!=4 or c.solver.voltage_amp!=5 or c.solver.current_amp!=0.1:
        raise ConfigError('Expected the assigned T=4, K=4, A=5, B=0.1 contract.')
    if c.training.arm not in ('clean','std3') or c.training.recipe not in ('A','B'):
        raise ConfigError('Expected clean/std3 and optimizer recipe A/B.')
    if c.training.schedule not in ('constant','cosine'):
        raise ConfigError('Expected constant or cosine schedule.')
    if c.training.retention not in ('final','best_latest_final'):
        raise ConfigError('Expected final or best_latest_final checkpoint retention.')
    for value in (c.cpu_threads,c.solver.token_chunk_size,c.solver.input_gain,c.data.sequences,c.data.development_sequences,
                  c.data.mapping_sequences,c.training.epochs,c.training.effective_batch,c.training.microbatch,
                  c.training.conductance_lr,c.training.calibration_lr,c.training.digital_lr,c.training.max_wall_seconds,
                  c.training.checkpoint_every,c.selection.every_updates):
        if value<=0: raise ConfigError('Expected positive sizes, learning rates and intervals.')
    if c.data.sequence_length<2 or c.training.max_updates<0 or c.training.warmup_updates<0:
        raise ConfigError('Expected sequence length >=2 and nonnegative update counts.')
    if c.training.microbatch>c.training.effective_batch or c.data.mapping_sequences>c.data.sequences:
        raise ConfigError('Expected microbatch/mapping cohorts within the declared training budget.')
    if not 0<c.training.minimum_lr_factor<=1 or not 0<=c.selection.random_rate<=1:
        raise ConfigError('Expected LR factor in (0,1] and fault rate in [0,1].')
    if any(type(s) is not int or s<0 for s in c.selection.assignment_seeds) or len(set(c.selection.assignment_seeds))!=len(c.selection.assignment_seeds):
        raise ConfigError('Expected distinct nonnegative assignment seeds.')
    if not set(c.selection.fault_kinds)<= {'none','random'} or len(set(c.selection.fault_kinds))!=len(c.selection.fault_kinds):
        raise ConfigError('Expected distinct nominal/random selection cases.')
    if c.deployment.assignment_seed<0 or c.deployment.fault_kind not in ('none','random') or not 0<=c.deployment.fault_rate<=1:
        raise ConfigError('Expected a nonnegative deployment seed and nominal/random fault case.')
    if c.deployment.fault_kind=='none' and c.deployment.fault_rate!=0:
        raise ConfigError('Nominal deployment must have zero fault rate.')
    if c.recovery.arm not in ('hold','naive_hold','calibration','learning','learning_calibration','rewrite'):
        raise ConfigError('Unknown recovery arm.')
    if min(c.recovery.updates,c.recovery.learning_rate,c.recovery.calibration_updates,
           c.recovery.calibration_lr,c.recovery.checkpoint_every)<=0:
        raise ConfigError('Expected positive recovery budgets and learning rates.')
    if c.recovery.threshold_sigma!=2 or c.recovery.max_write_attempts not in (1,4,None):
        raise ConfigError('Expected strict 2-sigma learning with one or four attempts or null for uncapped writes.')
    capped=c.recovery.update_rule=='bounded_capped_nominal_adam'
    if (capped and c.recovery.max_write_attempts not in (1,4)) or (c.recovery.max_write_attempts==1 and not capped):
        raise ConfigError('The capped nominal-gate writer takes one or four attempts per device; '
                          'one attempt belongs only to it.')
    if c.recovery.write_noise_factor not in (0.,1.) or c.recovery.development_every<0:
        raise ConfigError('Expected recovery write noise factor 0/1 and a nonnegative development interval.')
    if c.recovery.conductance_floor not in ('none','gmin'):
        raise ConfigError('Expected recovery conductance floor none/gmin.')
    if c.recovery.update_rule not in ('thresholded_adam','direct_adam','bounded_thresholded_adam',
                                      'bounded_capped_nominal_adam'):
        raise ConfigError('Expected thresholded_adam/direct_adam/bounded_thresholded_adam/'
                          'bounded_capped_nominal_adam recovery update rule.')
    if c.recovery.target_bounds not in ('none','gmin_gmax'):
        raise ConfigError('Expected recovery target bounds none/gmin_gmax.')
    if c.recovery.update_rule in ('direct_adam','bounded_thresholded_adam'):
        if (c.recovery.target_bounds!='gmin_gmax' or c.recovery.max_write_attempts is not None
                or c.recovery.conductance_floor!='none'):
            raise ConfigError('Bounded Adam requires bounded targets, uncapped writes and no realized-state floor.')
        if c.recovery.update_rule=='bounded_thresholded_adam' and c.recovery.write_noise_factor!=1.:
            raise ConfigError('The 2-rho bounded writer requires noisy recovery writes.')
    elif c.recovery.update_rule=='bounded_capped_nominal_adam':
        if c.recovery.target_bounds!='gmin_gmax' or c.recovery.conductance_floor!='none':
            raise ConfigError('The capped nominal-gate writer requires bounded targets and no realized-state floor.')
    elif c.recovery.target_bounds!='none':
        raise ConfigError('Target clipping requires the explicit direct_adam update rule.')
    if c.recovery.lr_schedule=='constant':
        if c.recovery.warmup_updates!=0:
            raise ConfigError('A constant recovery learning rate has no warm-up.')
    elif c.recovery.lr_schedule=='warmup_cosine':
        if c.recovery.update_rule!='direct_adam' or not 0<=c.recovery.warmup_updates<c.recovery.updates:
            raise ConfigError('Warm-up cosine recovery requires direct_adam and 0 <= warmup < updates.')
    else:
        raise ConfigError('Expected recovery lr_schedule constant/warmup_cosine.')
    if c.recovery.extends_updates and (c.recovery.lr_schedule!='constant' or c.recovery.arm!='learning'
                                       or not 0<c.recovery.extends_updates<c.recovery.updates):
        raise ConfigError('A horizon extension needs constant-LR learning and 0 < extends_updates < updates.')
    if c.stage in ('hwa_precision_benchmark','hwa_tf32_benchmark','hwa_speed_benchmark') and ((c.analog_layers!=24 and c.stage!='hwa_speed_benchmark') or c.model!='facebook/opt-1.3b' or c.training.max_updates!=4 or c.training.effective_batch!=32 or c.training.microbatch!=1 or c.training.arm!='std3' or not c.device.startswith('cuda')):
        raise ConfigError('Precision benchmark requires the four-update A100 L24 std3 canary recipe.')
    if c.stage=='recovery_benchmark' and (c.training.effective_batch!=32 or c.training.microbatch!=1
            or c.recovery.arm!='learning' or c.recovery.update_rule!='direct_adam'
            or c.recovery.updates!=8 or c.recovery.learning_rate!=.003 or c.recovery.write_noise_factor!=0.):
        raise ConfigError('Recovery benchmark requires batch 32, reference microbatch 1, and eight bounded direct Adam updates at LR .003.')
    return c


def resolve_config(config,mode):
    if mode == RunMode.VALIDATE and config.model == 'facebook/opt-1.3b':
        raise ConfigError('Evaluate OPT-1.3B sources with the held-out deploy stage.')
    if mode==RunMode.VALIDATE and config.stage not in ('hwa','recover'):
        raise ConfigError('Expected hwa/recover configuration for validation.')
    return config


def compatible_config_hashes(config):
    """Accept old default-policy checkpoints without weakening new contracts."""
    from copy import deepcopy
    from ebl.runtime.store import content_hash
    values=[asdict(config)]
    def optional(keys, section=None):
        for value in list(values):
            older=deepcopy(value)
            target=older if section is None else older[section]
            for key in keys: target.pop(key, None)
            values.append(older)
    if config.drift_recovery is None:
        optional(('drift_recovery',))
    if config.retention is None:
        optional(('retention',))
    if config.recovery.extends_updates==0:
        optional(('extends_updates',), 'recovery')
    if config.recovery.lr_schedule=='constant' and config.recovery.warmup_updates==0 and config.recovery.extends_updates==0:
        optional(('lr_schedule', 'warmup_updates'), 'recovery')
    if config.recovery.update_rule=='thresholded_adam' and config.recovery.target_bounds=='none':
        optional(('update_rule', 'target_bounds'), 'recovery')
    return {content_hash(value) for value in values}


def resume_config_hashes(config):
    """Configurations whose unfinished checkpoints this run may continue.

    A declared constant-LR horizon extension also accepts states saved under the
    original horizon; only ``recovery.updates`` differs, so continuing equals one
    longer run.
    """
    hashes=compatible_config_hashes(config)
    if config.recovery.extends_updates:
        from dataclasses import replace
        original=replace(config,recovery=replace(config.recovery,updates=config.recovery.extends_updates,extends_updates=0))
        hashes|=compatible_config_hashes(original)
    return hashes
