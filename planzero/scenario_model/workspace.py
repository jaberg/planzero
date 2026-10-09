from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro
from jax.typing import ArrayLike
from numpyro.distributions.distribution import Distribution

from .base import (
    ElementAnnualKey,
    ElementCarryKey,
    ElementGeneralKey,
    InitialCarry,
    NdarrayDim,
    NextCarry,
    Observation,
    ObservationValid,
    ObservationWeight,
    Phase,
    Subphase,
    VarKey,
    scan_step_ii_key,
    years_dim,
)
from .computation import Computation, Model, ScanStorage


def dist_scan_set_next(comp:Computation, scan_storage:ScanStorage, var_key:ElementCarryKey, dist:Distribution):

    nc_key = NextCarry(carry_key=var_key)
    obs_nc_key = Observation(prior_var_key=nc_key)
    obsvalid_nc_key = ObservationValid(obs_var_key=obs_nc_key)

    obs = comp._ndarray_d.get(obs_nc_key)
    obs_valid = comp._ndarray_d.get(obsvalid_nc_key)

    if comp.phase == Phase.Prior:
        if obs or obs_valid:
            raise NotImplementedError()
        else:
            # TODO: check that the shape is correct
            scan_storage.next_carry_dist_d[var_key] = dist
            sample = numpyro.sample(
                    comp.sample_site(nc_key),
                    fn=dist,
                    rng_key=scan_storage._split_rng_key())
            if isinstance(sample, jnp.ndarray):
                scan_storage.next_carry_d[var_key] = sample
            else:
                raise NotImplementedError(sample)

    elif comp.phase == Phase.Posterior:
        if obs or obs_valid:
            raise NotImplementedError()
        else:
            # We assume here that the posterior mcmc samples
            # only cover n_prior_years initial years.
            # When asked for posterior samples over more years
            # we revert to drawing from the prior distribution (
            # which is typically informed by time-invariant
            # posterior samples.

            this_idx_ = scan_storage.this_X_d[scan_step_ii_key]
            assert this_idx_.shape == (1,)
            this_idx = this_idx_[0]
            n_prior_years = comp.model.mv.n_prior_years

            # the mask is a scalar bool, all or nothing
            switching_scalar = this_idx >= n_prior_years
            scan_storage.next_carry_dist_d[var_key] = dist.mask(switching_scalar)

            # draw a sample regardless of whether we need it or not
            # on this scan step
            if dist.shape()[-1] != comp.n_mcmc:
                expanded_dist = dist.expand_by((comp.n_mcmc,))
                moveaxis = True
            else:
                expanded_dist = dist
                moveaxis = False

            sample_mcmc_left = numpyro.sample(
                    comp.sample_site(nc_key),
                    fn=expanded_dist,
                    rng_key=scan_storage._split_rng_key())
            if moveaxis:
                sample_mcmc_right = jnp.moveaxis(sample_mcmc_left, 0, -1)
            else:
                sample_mcmc_right = sample_mcmc_left

            # bring in the next carry values from central storage
            # for the n_prior_years steps that they cover
            nc_data = comp._ndarray_d[nc_key][
                        jnp.minimum(this_idx, n_prior_years - 1)]
            this_sample = jnp.where(switching_scalar, sample_mcmc_right, nc_data)

            scan_storage.next_carry_d[var_key] = this_sample
    else:
        assert 0


class ValGeneral:

    ws: Workspace
    attr: str

    @property
    def comp(self) -> Computation:
        return self.ws.comp

    def __init__(self, ws:Workspace, attr:str):
        self.ws = ws
        self.attr = attr

    def __getitem__(self, item:str|ElementGeneralKey|ElementAnnualKey) -> jnp.ndarray:
        if isinstance(item, str):
            var_key = self.ws._general_key(item)
        else:
            var_key = item
        del item

        if self.attr == 'val':
            if self.ws.subphase in (Subphase.Prep, Subphase.Proc):
                sample_if_necessary = True
            elif self.ws.subphase == Subphase.Step:
                sample_if_necessary = False
            else:
                raise NotImplementedError(self.ws.subphase)
            return self.ws.comp.get_ndarray(var_key, sample_if_necessary)
        elif self.attr == 'obs_val':
            obs_key = Observation(prior_var_key=var_key)
            if self.ws.subphase in (Subphase.Prep, Subphase.Proc):
                sample_if_necessary = True
            elif self.ws.subphase == Subphase.Step:
                sample_if_necessary = False
            else:
                raise NotImplementedError(self.ws.subphase)
            return self.ws.comp.get_ndarray(obs_key, sample_if_necessary)
        else:
            raise NotImplementedError(self.attr)


    def __setitem__(self, item:str|ElementGeneralKey, value:jnp.ndarray):
        if isinstance(item, str):
            var_key = self.ws._general_key(item)
        else:
            var_key = item
        del item

        if self.attr == 'val':
            self.ws.comp.ndarray_d[var_key] = value

        elif self.attr == 'obs_val':
            obs_var_key = Observation(prior_var_key=var_key)
            self.ws.comp.ndarray_d[obs_var_key] = value

        elif self.attr == 'obs_valid':
            obsvalid_var_key = ObservationValid(
                   obs_var_key=Observation(prior_var_key=var_key))
            self.ws.comp.ndarray_d[obsvalid_var_key] = value

        elif self.attr == 'obs_weight':
            obs_var_key = ObservationWeight(
                   obs_var_key=Observation(
                       prior_var_key=var_key))
            self.ws.comp.ndarray_d[obs_var_key] = value

        else:
            raise NotImplementedError(self.attr)


class ValAnnual:

    ws: Workspace
    attr: str

    @property
    def comp(self) -> Computation:
        return self.ws.comp

    def __init__(self, ws:Workspace, attr:str):
        self.ws = ws
        self.attr = attr

    def __getitem__(self, item:str|ElementAnnualKey) -> jnp.ndarray:
        if isinstance(item, str):
            var_key = self.ws._annual_key(item)
        else:
            var_key = item
        del item

        if self.attr == 'val':
            if self.ws.subphase in (Subphase.Prep, Subphase.Proc):
                sample_if_necessary = True
            elif self.ws.subphase == Subphase.Step:
                sample_if_necessary = False
            else:
                raise NotImplementedError(self.ws.subphase)
            return self.ws.comp.get_ndarray(var_key, sample_if_necessary)
        else:
            raise NotImplementedError(self.attr)

    def __setitem__(self, item:str|ElementAnnualKey, value:jnp.ndarray):
        if isinstance(item, str):
            var_key = self.ws._annual_key(item)
        else:
            var_key = item
        del item

        assert value.shape[0] == self.ws.n_years

        if self.attr == 'val':
            self.ws.comp.ndarray_d[var_key] = value
        elif self.attr == 'obs_val':
            obs_key = Observation(prior_var_key=var_key)
            self.ws.comp.ndarray_d[obs_key] = value
        else:
            raise NotImplementedError(self.attr)


class ValInitialCarry:

    ws: Workspace
    attr: str

    def __init__(self, ws:Workspace, attr:str):
        self.ws = ws
        self.attr = attr

    def __getitem__(self, item:str|ElementCarryKey) -> jnp.ndarray:
        if isinstance(item, str):
            var_key = self.ws._carry_key(item)
        else:
            var_key = item
        del item

        if self.attr == 'val':
            return self.ws.comp.ndarray_d[InitialCarry(carry_key=var_key)]
        else:
            raise NotImplementedError(self.attr)

    def __setitem__(self, item:str|ElementCarryKey, value:jnp.ndarray):
        if isinstance(item, str):
            var_key = self.ws._carry_key(item)
        else:
            var_key = item
        del item

        if self.attr == 'val':
            self.ws.comp.ndarray_d[InitialCarry(carry_key=var_key)] = value
        else:
            raise NotImplementedError(self.attr)


class ValThisCarry:

    ws: Workspace
    attr: str

    def __init__(self, ws:Workspace, attr:str):
        self.ws = ws
        self.attr = attr

    def __getitem__(self, item:str|ElementCarryKey) -> jnp.ndarray:
        if isinstance(item, str):
            var_key = self.ws._carry_key(item)
        else:
            var_key = item
        del item

        assert self.ws.scan_storage is not None

        if self.attr == 'val':
            rval = self.ws.scan_storage.this_carry_d[var_key]
            return rval
        else:
            raise NotImplementedError(f'{self.attr}_scan_get_this')

    def __setitem__(self, item:str|ElementCarryKey, value:jnp.ndarray):
        if isinstance(item, str):
            var_key = self.ws._carry_key(item)
        else:
            var_key = item
        del item
        assert self.ws.scan_storage is not None

        if self.attr == 'val':
            self.ws.scan_storage.this_carry_d[var_key] = value
        else:
            raise NotImplementedError(self.attr)


class DistGeneral:

    ws: Workspace
    attr: str

    def __init__(self, ws:Workspace, attr:str):
        self.ws = ws
        self.attr = attr

    def __getitem__(self, item:str|ElementGeneralKey) -> Distribution:
        if isinstance(item, str):
            var_key = self.ws._general_key(item)
        else:
            var_key = item
        del item

        if self.attr == 'dist':
            return self.ws.comp.dist_d[var_key]
        elif self.attr == 'obs_dist':
            return self.ws.comp.dist_d[Observation(prior_var_key=var_key)]
        else:
            raise NotImplementedError(self.attr)

    def __setitem__(self, item:str|ElementGeneralKey, dist:Distribution) -> None:
        if isinstance(item, str):
            var_key = self.ws._general_key(item)
        else:
            var_key = item
        del item

        if self.attr == 'dist':
            # ws.obs.general[item] = dist
            self.ws.comp.dist_d[var_key] = dist
        elif self.attr == 'obs_dist':
            # ws.obs_dist.general[item] = dist
            self.ws.comp.dist_d[Observation(prior_var_key=var_key)] = dist
        else:
            raise NotImplementedError(self.attr)


class DistInitialCarry:

    ws: Workspace
    attr: str

    def __init__(self, ws:Workspace, attr:str):
        self.ws = ws
        self.attr = attr

    def __getitem__(self, item:str|ElementCarryKey) -> Distribution:
        if isinstance(item, str):
            var_key = self.ws._carry_key(item)
        else:
            var_key = item
        del item

        if self.attr == 'dist':
            return self.ws.comp.dist_d[InitialCarry(carry_key=var_key)]
        else:
            raise NotImplementedError(self.attr)

    def __setitem__(self, item:str|ElementCarryKey, dist:Distribution):
        if isinstance(item, str):
            var_key = self.ws._carry_key(item)
        else:
            var_key = item
        del item

        if self.attr == 'dist':
            self.ws.comp.dist_d[InitialCarry(carry_key=var_key)] = dist
        else:
            raise NotImplementedError(self.attr)


class DistNextCarry:

    ws: Workspace
    attr: str

    def __init__(self, ws:Workspace, attr:str):
        self.ws = ws
        self.attr = attr

    def __getitem__(self, item:str|ElementCarryKey) -> Distribution:
        if isinstance(item, str):
            var_key = self.ws._carry_key(item)
        else:
            var_key = item
        del item

        if self.attr == 'dist':
            if self.ws.scan_storage is None:
                raise NotImplementedError()
            else:
                return self.ws.scan_storage.next_carry_dist_d[var_key]
        else:
            raise NotImplementedError(self.attr)

    def __setitem__(self, item:str|ElementCarryKey, dist:Distribution):
        if isinstance(item, str):
            var_key = self.ws._carry_key(item)
        else:
            var_key = item
        del item

        if self.attr == 'dist':
            if self.ws.scan_storage is None:
                raise NotImplementedError()
            else:
                dist_scan_set_next(self.ws.comp, self.ws.scan_storage, var_key, dist)
        else:
            raise NotImplementedError(f'{self.attr}_scan_set_this')


class ValNextCarry:

    ws: Workspace
    attr: str

    def __init__(self, ws:Workspace, attr:str):
        self.ws = ws
        self.attr = attr

    def __getitem__(self, item:str|ElementCarryKey) -> jnp.ndarray:
        if isinstance(item, str):
            var_key = self.ws._carry_key(item)
        else:
            var_key = item
        del item

        assert self.ws.scan_storage is not None

        if self.attr == 'val':
            return self.ws.scan_storage.next_carry_d[var_key]
            raise NotImplementedError(var_key)
        else:
            raise NotImplementedError(self.attr)

    def __setitem__(self, item:str|ElementCarryKey, value:jnp.ndarray):
        if isinstance(item, str):
            var_key = self.ws._carry_key(item)
        else:
            var_key = item
        del item
        assert self.ws.scan_storage is not None

        if self.attr == 'val':
            self.ws.scan_storage.next_carry_d[var_key] = value
        else:
            raise NotImplementedError(f'{self.attr}_scan_set_this')


class ValThisY:

    ws: Workspace
    attr: str

    def __init__(self, ws:Workspace, attr:str):
        self.ws = ws
        self.attr = attr

    def __getitem__(self, item:str|ElementAnnualKey) -> jnp.ndarray:
        if isinstance(item, str):
            var_key = self.ws._annual_key(item)
        else:
            var_key = item
        del item
        assert self.ws.scan_storage is not None

        if self.attr == 'val':
            return self.ws.scan_storage.this_Y_d[var_key]
        else:
            raise NotImplementedError(self.attr)

    def __setitem__(self, item:str|ElementAnnualKey, value:jnp.ndarray):
        if isinstance(item, str):
            var_key = self.ws._annual_key(item)
        else:
            var_key = item
        del item
        assert self.ws.scan_storage is not None

        if self.attr == 'val':
            self.ws.scan_storage.this_Y_d[var_key] = value
        else:
            raise NotImplementedError(self.attr)


class WorkspaceAttrVal:

    ws: Workspace
    attr: str

    def __init__(self, ws:Workspace, attr:str):
        self.ws = ws
        self.attr = attr

    @property
    def general(self) -> ValGeneral:
        return ValGeneral(self.ws, self.attr)

    @property
    def annual(self) -> ValAnnual:
        return ValAnnual(self.ws, self.attr)

    @property
    def initial_carry(self) -> ValInitialCarry:
        return ValInitialCarry(self.ws, self.attr)

    @property
    def this_carry(self) -> ValThisCarry:
        if self.ws.subphase == Subphase.Step:
            return ValThisCarry(self.ws, self.attr)
        else:
            raise AttributeError(
                    f"this_carry not available in subphase {self.ws.subphase}")

    @property
    def next_carry(self) -> ValNextCarry:
        if self.ws.subphase == Subphase.Step:
            return ValNextCarry(self.ws, self.attr)
        else:
            raise AttributeError(
                    f"next_carry not available in subphase {self.ws.subphase}")

    @property
    def this_Y(self) -> ValThisY:
        if self.ws.subphase == Subphase.Step:
            return ValThisY(self.ws, self.attr)
        else:
            raise AttributeError(
                    f"this_Y not available in subphase {self.ws.subphase}")


class WorkspaceAttrDist:

    ws: Workspace
    attr: str

    def __init__(self, ws:Workspace, attr:str):
        self.ws = ws
        self.attr = attr

    @property
    def general(self) -> DistGeneral:
        return DistGeneral(self.ws, self.attr)

    @property
    def initial_carry(self) -> DistInitialCarry:
        return DistInitialCarry(self.ws, self.attr)

    @property
    def next_carry(self) -> DistNextCarry:
        if self.ws.subphase == Subphase.Step:
            return DistNextCarry(self.ws, self.attr)
        else:
            raise AttributeError(
                    f"next_carry not available in subphase {self.ws.subphase}")


class Workspace:

    year_0: int
    n_years: int
    phase: Phase
    comp: Computation
    model: Model
    elem_id: str
    subphase: Subphase
    scan_storage: ScanStorage|None

    def __init__(
            self,
            year_0:int,
            n_years:int,
            phase:Phase,
            comp:Computation,
            elem_id:str,
            subphase:Subphase,
            scan_storage:ScanStorage|None,
            ):
        self.year_0 = year_0
        self.n_years = n_years
        assert n_years >= 0
        self.phase = phase
        self.comp = comp
        self.model = self.comp.model
        self.elem_id = elem_id
        self.subphase = subphase
        self.scan_storage = scan_storage

    def _general_key(self, name:str) -> ElementGeneralKey:
        return ElementGeneralKey(elem_id=self.elem_id, name=name)

    def _carry_key(self, name:str) -> ElementCarryKey:
        return ElementCarryKey(elem_id=self.elem_id, name=name)

    def _annual_key(self, name:str) -> ElementAnnualKey:
        return ElementAnnualKey(elem_id=self.elem_id, name=name)

    @property
    def dist(self) -> WorkspaceAttrDist:
        return WorkspaceAttrDist(self, 'dist')

    @property
    def val(self) -> WorkspaceAttrVal:
        return WorkspaceAttrVal(self, 'val')

    @property
    def obs_dist(self) -> WorkspaceAttrDist:
        return WorkspaceAttrDist(self, 'obs_dist')

    @property
    def obs_val(self) -> WorkspaceAttrVal:
        return WorkspaceAttrVal(self, 'obs_val')

    @property
    def obs_valid(self) -> WorkspaceAttrVal:
        return WorkspaceAttrVal(self, 'obs_valid')

    @property
    def obs_weight(self) -> WorkspaceAttrVal:
        return WorkspaceAttrVal(self, 'obs_weight')

    @property
    def shape(self) -> WorkspaceAttrVal:
        return WorkspaceAttrVal(self, 'shape')

    @property
    def n_mcmc(self) -> int:
        if self.phase == Phase.Prior:
            raise NotImplementedError()
        else:
            return self.comp.n_mcmc

    @property
    def mcmc_shape(self) -> tuple:
        if self.phase == Phase.Prior:
            return ()
        else:
            return (self.comp.n_mcmc,)

    @property
    def mcmc_broadcast_shape(self) -> tuple:
        if self.phase == Phase.Prior:
            return ()
        else:
            return (1,)

    def mcmc_broadcast_dist(self, dist:Distribution) -> Distribution:
        if self.phase == Phase.Prior:
            return dist
        else:
            if isinstance(dist, Distribution):
                return right_pad_universal(dist)

    def mcmc_broadcast_ndarray(self, nd:jnp.ndarray) -> jnp.ndarray:
        if self.phase == Phase.Prior:
            return nd
        else:
            return jnp.expand_dims(nd, axis=-1)

    def mcmc_expand(self, dist):
        if self.phase == Phase.Prior:
            return dist
        else:
            padded = right_pad_universal(dist)
            return padded.expand(dist.shape() + (self.n_mcmc,))



def right_pad_universal(dist):
    """
    Right-pads ANY NumPyro distribution's batch_shape with (1,).
    Safely handles both scalar parameters and matrix parameters.
    """
    new_params = {}

    # Look up the exact event dimensions required for each parameter
    for param_name, constraint in dist.arg_constraints.items():
        val = getattr(dist, param_name)
        val = jnp.asarray(val)  # Ensure it is a JAX array

        # constraint.event_dim tells us how many dimensions are at the end
        # of the array that belong to the "event".
        # We calculate the insertion axis to be right BEFORE the event dimensions.
        event_dim = constraint.event_dim
        insert_axis = val.ndim - event_dim

        new_params[param_name] = jnp.expand_dims(val, axis=insert_axis)

    # Re-instantiate the distribution using the reshaped parameters.
    # Because we are calling the constructor anew, NumPyro will automatically
    # calculate and set the correct _batch_shape for us!
    return type(dist)(**new_params)

class WorkSpacePrep_Dist_InitialCarry:
    """object to represent `ws.dist.initial_carry`"""

    def __setitem__(self, item:ElementCarryKey|str, dist:Distribution) -> None:
        if isinstance(item, str):
            var_key = self.wsd.ws._carry_key(item)
        else:
            var_key = item
        ic_key = InitialCarry(carry_key=var_key)
        obs = self.comp.storage_nd.get(observation(ic_key))
        obs_valid = self.comp.storage_nd.get(
                observation_valid(observation(ic_key)))

        if self.comp.phase == Phase.Prior:
            if obs is None and obs_valid is None:
                # TODO: check that the shape is correct
                self.comp.storage_dist[ic_key] = dist

                # Assume, for now, that we need the sampled value
                # because otherwise we won't know until inside the scan logic
                # at which time it is too late.
                # If this assumption is violated, add an argument to
                # @define_carry e.g. sample=False
                self.comp.storage_nd[ic_key] = numpyro.sample(
                        self.comp.sample_site(ic_key),
                        fn=dist,
                        rng_key=self.comp._split_rng_key(),
                        )
            else:
                raise NotImplementedError()
        else:
            assert self.comp.phase == Phase.Posterior
            if obs is None and obs_valid is None:
                # don't define the distribution this time
                # because the provided distribution was the prior.

                assert ic_key in self.comp.storage_nd
                # TODO: check that the shape is correct
                #       the shape should be [n_mcmc] + [var shape]
            else:
                raise NotImplementedError()



class WorkSpacePrep_Val_InitialCarry:
    """object to represent `ws.val.initial_carry`"""

    def __setitem__(self, item:ElementCarryKey|str, value:jnp.ndarray) -> None:
        # TODO: check if the element, subphase defines item
        # TODO: check that the shape is correct
        if isinstance(item, str):
            var_key = self.wsv.ws._carry_key(item)
        else:
            var_key = item
        ic_key = InitialCarry(carry_key=var_key)
        ndm = self.comp.model.mv.carry_nd[ic_key].definition_metadata
        assert value.shape == tuple(ndm.value_type.shape)
        self.comp.storage_nd[ic_key] = value


class WorkSpacePrep_ObsDist_General:

    def __setitem__(self, item:VarKey, obs_dist:Distribution) -> None:

        if self.comp.phase == Phase.Prior:
            assert item not in self.comp.storage_nd
            assert observation(item) not in self.comp.storage_nd
            assert observation_valid(observation(item)) not in self.comp.storage_nd
            # TODO: assignments of ^^ should also
            # raise errors if storage_dist[observation(item)] is set,
            # as I believe they are mathematically mutually exclusive

            self.comp.storage_dist[observation(item)] = obs_dist

            # must be defined already
            prior_dist = self.comp.storage_dist[item]
            obs_weight = self.comp.storage_nd[observation_weight(observation(item))]

            expected_log_prob_plus_const = -kl_divergence(obs_dist, prior_dist)
            numpyro.factor(self.comp.sample_site(item),
                           obs_weight * expected_log_prob_plus_const)
        else:
            pass


class WorkSpacePrep_ObsWeight_Attr:

    ws_od: WorkSpaceProc_ObsWeight
    comp: Computation

    def __init__(self, ws_od:WorkSpaceProc_ObsWeight):
        self.ws_od = ws_od
        self.comp = self.ws_od.ws.comp


class WorkSpacePrep_ObsWeight_General:

    def __setitem__(self, item:VarKey, obs_weight:ArrayLike) -> None:
        self.comp.storage_nd[observation_weight(observation(item))] = obs_weight


class WorkSpacePrep_Shape_General:

    def __getitem__(self, item:str|ElementGeneralKey) -> list[int|NdarrayDim]:
        if isinstance(item, str):
            var_key = self.ws_shp.ws._general_key(item)
        else:
            var_key = item
        del item
        vm = self.comp.model.mv.general_nd[var_key]
        prior_shape = vm.definition_metadata.value_type.shape
        if self.comp.phase == Phase.Prior:
            rval = []
            for shape_ii in prior_shape:
                if isinstance(shape_ii, int):
                    rval.append(shape_ii)
                elif shape_ii == years_dim:
                    rval.append(self.comp.n_years)
                else:
                    raise NotImplementedError(shape_ii)
            return rval
        else:
            assert self.comp.phase == Phase.Posterior
            # shape could be same as prior shape
            # or left-extended with 1 for broadcasting over mcmc dim
            # or left-extended with mcmc_dim
            # ... it might be worth offering a way to provide a hint
            # in the @define because otherwise I don't think it's knowable
            # prior to having the actual value to check.
            if var_key in self.comp.storage_nd:
                return list(self.comp.storage_nd[var_key].shape)
            else:
                raise NotImplementedError(var_key)


