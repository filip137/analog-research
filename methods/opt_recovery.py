"""Physical DRN gradient and update coordination."""
from types import SimpleNamespace
import math
import torch
from torch.nn import functional as F
from hardware.devices.cmo import G_MIN_US, G_MAX_US, HALF_RANGE_US, MID_US, programming_sigma
from methods.cmo_thresholded_adam import ThresholdedWriteAdam
from methods.cmo_direct_adam import CappedNominalWriteAdam, DirectWriteAdam, ThresholdedBoundedWriteAdam
from methods.core.engine import ExperimentComponents, GradientsReadyEvent, train_epoch

class RecoveryWritePort:
    """Separate the recovery noise intervention from threshold calibration."""
    def __init__(self,array,noise_factor,*,capture=False,conductance_floor='none',target_bounds='none'):
        self.array,self.noise_factor,self.capture=array,noise_factor,capture
        self.conductance_floor=conductance_floor
        self.target_bounds=target_bounds
        self.record=None

    def read(self):return self.array.read()

    def write_noise_std_q(self,command):
        target=(self.array.held+HALF_RANGE_US*command).clamp(G_MIN_US,G_MAX_US)
        return self.array.noise_scale*self.noise_factor*programming_sigma(target,2.)/HALF_RANGE_US

    def nominal_write_std_q(self):
        # Device noise scale only: the noise-free intervention keeps the same gate.
        return self.array.noise_scale*programming_sigma(MID_US,2.)/HALF_RANGE_US

    def write(self,command):
        if self.capture and bool((command!=0).any()):
            if self.target_bounds=='gmin_gmax':
                # Fixed spatial sample, independent of device identities and RNG.
                indices=torch.linspace(0,command.numel()-1,min(64,command.numel()),device=command.device).long()
                indices=indices[command.flatten()[indices]!=0]
            else:indices=(command!=0).flatten().nonzero().flatten()
            self.record={'name':self.array.name,'shape':list(command.shape),
                         'indices':indices.cpu(),'commands_q':command.flatten()[indices].detach().cpu(),
                         'pre_write_rng':self.array.generator.get_state().cpu()}
            if self.target_bounds=='gmin_gmax':
                self.record.update(sample_policy='fixed_spatial_at_most_64',
                    proposed_commands=int((command!=0).sum()),
                    pre_write_held_us=self.array.held.flatten()[indices].detach().cpu())
        return self.array.write(command,noise_factor=self.noise_factor,conductance_floor=self.conductance_floor,
                                target_bounds=self.target_bounds)

class RecoveryStep:
    """Gradients use held physical cells; only the plant changes conductances."""
    def __init__(self,model,teacher,bank,config):
        self.model,self.teacher,self.bank,self.config=model,teacher,bank,config
        self.calibration=config.recovery.arm in ('calibration','learning_calibration')
        model.requires_grad_(False)
        selected=model.parameter_groups()[1 if self.calibration else 0]
        for p in selected:p.requires_grad_(True)
        self.parameters=tuple(SimpleNamespace(state=p) for p in selected)
        names={id(p):name for name,p in model.named_parameters()}
        self.names=[names[id(p)] for p in selected]
        self.optimizer=(torch.optim.Adam(selected,lr=config.recovery.calibration_lr,foreach=False)
                        if self.calibration else None)
        self.ports=[] if self.calibration else [RecoveryWritePort(
            a,config.recovery.write_noise_factor,capture=config.recovery.development_every>0,
            conductance_floor=config.recovery.conductance_floor,target_bounds=config.recovery.target_bounds)
            for a in bank.arrays]
        self.first_write_captured=False
        self.first_write_record=None
        self.writers=[] if self.calibration else [
            DirectWriteAdam(port,learning_rate=config.recovery.learning_rate,
                            schedule=None if config.recovery.lr_schedule=='constant' else
                            ('warmup_cosine',config.recovery.warmup_updates,config.recovery.updates))
            if config.recovery.update_rule=='direct_adam' else
            ThresholdedBoundedWriteAdam(port,learning_rate=config.recovery.learning_rate,
                                       threshold_sigma=config.recovery.threshold_sigma)
            if config.recovery.update_rule=='bounded_thresholded_adam' else
            CappedNominalWriteAdam(port,learning_rate=config.recovery.learning_rate,
                                   threshold_sigma=config.recovery.threshold_sigma,
                                   cap=config.recovery.max_write_attempts)
            if config.recovery.update_rule=='bounded_capped_nominal_adam' else ThresholdedWriteAdam(
                port,learning_rate=config.recovery.learning_rate,
                threshold_sigma=config.recovery.threshold_sigma,cap=config.recovery.max_write_attempts,
                noise_scale=array.noise_scale) for array,port in zip(bank.arrays,self.ports)]

    def set_input(self,inputs,reset=False):self.inputs=inputs
    def set_target(self,targets):pass
    def compute_equilibrium(self):pass

    def compute_gradient(self):
        self.model.train()
        self.model.zero_grad(set_to_none=True)
        view=self.bank.view(self.model)
        loss_sum=0.0
        for batch in self.inputs.split(self.config.training.microbatch):
            batch=batch.to(self.config.device)
            with torch.no_grad():reference=self.teacher(batch)[:,:-1]
            prediction=self.model(batch,view)
            loss=F.kl_div(prediction.flatten(0,1).log_softmax(-1),reference.flatten(0,1).softmax(-1),reduction='batchmean')
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite physical-training KL.')
            fraction=len(batch)/len(self.inputs)
            (loss*fraction).backward()
            loss_sum+=float(loss.detach())*fraction
        self.last={'teacher_kl':loss_sum,'sequences':len(self.inputs),'primary_state':'held_apparent'}
        return tuple(p.state.grad for p in self.parameters)

    def step(self):
        if self.optimizer is not None:
            self.optimizer.step()
        else:
            for writer,parameter in zip(self.writers,self.parameters):writer.step(parameter.state.grad)
            self.bank.view(self.model)  # Reject invalid coupled denominators immediately.
            if not self.first_write_captured and any(p.record is not None for p in self.ports):
                self.first_write_record=[p.record for p in self.ports if p.record is not None]
                self.first_write_captured=True
                for port in self.ports:port.capture=False;port.record=None

    def components(self):
        return ExperimentComponents(self,self,self,self.parameters,self,self,clamp_after_update=False)

    def observe(self,event):
        if isinstance(event,GradientsReadyEvent):
            norms=torch.stack([p.state.grad.detach().norm() for p in self.parameters])
            self.last['gradient_norm']=float(norms.norm())

    def state_dict(self):
        return {'writers':[w.state_dict() for w in self.writers],
                'write_noise_factor':self.config.recovery.write_noise_factor,
                'conductance_floor':self.config.recovery.conductance_floor,
                'update_rule':self.config.recovery.update_rule,'target_bounds':self.config.recovery.target_bounds,
                'first_write_captured':self.first_write_captured,
                'optimizer':None if self.optimizer is None else self.optimizer.state_dict()}

    def load_state_dict(self,state):
        for key,default in (('update_rule','thresholded_adam'),('target_bounds','none')):
            if state.get(key,default)!=getattr(self.config.recovery,key):
                raise ValueError(f'Recovery {key} continuation contract differs.')
        if state.get('write_noise_factor',1.)!=self.config.recovery.write_noise_factor:
            raise ValueError('Recovery write-noise continuation contract differs.')
        if state.get('conductance_floor','none')!=self.config.recovery.conductance_floor:
            raise ValueError('Recovery conductance-floor continuation contract differs.')
        if len(state['writers'])!=len(self.writers) or (state['optimizer'] is None)!=(self.optimizer is None):
            raise ValueError('Recovery optimizer mode differs.')
        for writer,saved in zip(self.writers,state['writers']):writer.load_state_dict(saved)
        if self.optimizer is not None:self.optimizer.load_state_dict(state['optimizer'])
        self.first_write_captured=state.get('first_write_captured',False)
        if self.first_write_captured:
            for port in self.ports:port.capture=False
