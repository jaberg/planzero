# this resolves circular type references
from __future__ import annotations

import jax.numpy as jnp
import jax.random as jrandom
import numpyro
from jax.lax import scan as jax_scan
from jax.typing import ArrayLike
from numpyro.contrib.control_flow import scan as numpyro_scan
from numpyro.distributions.distribution import Distribution
from numpyro.infer import MCMC, NUTS

from .base import (
    GroupedPosterior,
    InitialCarry,
    ModelVariables,
    NdarrayDefinitionMetadata,
    NdarrayDefinitionMetadata_w_Properties,
    NdarrayDim,
    NdarrayType,
    NdarrayType_w_Properties,
    NdarrayVariableMetadata,
    NextCarry,
    Phase,
    Subphase,
    VarKey,
    final_carry,
    initial_carry,
    new_named_key,
    next_carry,
    observation,
    observation_valid,
    years_dim,
)


class WorkSpacePrep_Dist_Attr:

    wsd: WorkSpacePrep_Dist | WorkSpaceProc_Dist
    comp: Computation

    def __init__(self, wsd:WorkSpacePrep_Dist|WorkSpaceProc_Dist):
        self.wsd = wsd
        self.comp = self.wsd.ws.comp


class WorkSpacePrep_Dist_General(WorkSpacePrep_Dist_Attr):
    """object to represent `ws.dist.general`"""

    def __setitem__(self, item:VarKey, dist:Distribution) -> None:
        # TODO: check that the shape is correct
        # TODO: check that the item is supposed to be sampled

        obs = self.comp.storage_nd.get(observation(item))
        obs_valid = self.comp.storage_nd.get(observation_valid(item))

        if self.comp.phase == Phase.Prior:

            if obs is None and obs_valid is None:
                # TODO: check that the shape is correct
                self.comp.storage_dist[item] = dist
                self.comp.storage_nd[item] = numpyro.sample(
                        self.comp.model.mv.sample_sites[item],
                        fn=dist,
                        rng_key=self.comp._split_rng_key(),
                        )
            elif obs_valid is None:
                # don't define the distribution if the observation
                # is provided, because it no longer makes sense
                #
                # but do store the observation as the sample
                #
                # TODO: check that obs' shape is correct
                assert obs is not None
                self.comp.storage_nd[item] = obs
                numpyro.sample(
                        self.comp.model.mv.sample_sites[item],
                        fn=dist,
                        rng_key=self.comp._split_rng_key(),
                        obs=obs,
                        )
            else:
                # don't define the distribution if the observation
                # is provided, because it no longer makes sense
                assert obs_valid is not None
                assert obs is not None
                # mask the distribution
                # mask the observation in the sample call?
                # store dist jnp.where(obs_valid, obs, an_actual_sample)
                #
                # Note that according to ...
                # https://num.pyro.ai/en/stable/primitives.html#sample
                # ... obs_mask parameter should *not* be used with MCMC
                # ... so think about what this model needs in terms of
                # .... symantics.
                raise NotImplementedError()
        else:
            assert self.comp.phase == Phase.Posterior
            if obs is None and obs_valid is None:
                # don't define the distribution this time
                # because the provided distribution was the prior.

                assert item in self.comp.storage_nd
                # TODO: check that the shape is correct
                #       the shape should be [n_mcmc] + [var shape]

            elif obs_valid is None:
                assert obs is not None
                # TODO: check that the shape is correct
                #       obs' shape should be [var shape]
                #       with no mcmc samples.

                # technically this might be okay in some scenarios
                # but if it happens for now, it's an error.
                assert item not in self.comp.storage_nd

                self.comp.storage_nd[item] = obs
            else:
                assert obs_valid is not None
                assert obs is not None
                # mask the distribution
                # mask the observation in the sample call?
                # store dist jnp.where(obs_valid, obs, an_actual_sample)
                self.comp.storage_nd[item] = numpyro.sample(
                        self.comp.model.mv.sample_sites[item],
                        fn=dist,
                        rng_key=self.comp._split_rng_key(),
                        obs=obs,
                        obs_mask=obs_valid
                        )


class WorkSpacePrep_Dist_InitialCarry(WorkSpacePrep_Dist_Attr):
    """object to represent `ws.dist.initial_carry`"""

    def __setitem__(self, item:VarKey, dist:Distribution) -> None:
        obs = self.comp.storage_nd.get(observation(initial_carry(item)))
        obs_valid = self.comp.storage_nd.get(observation_valid(initial_carry(item)))

        if self.comp.phase == Phase.Prior:
            if obs is None and obs_valid is None:
                # TODO: check that the shape is correct
                # TODO: check that the item is supposed to be sampled
                self.comp.storage_dist[initial_carry(item)] = dist

                self.comp.storage_nd[initial_carry(item)] = numpyro.sample(
                        self.comp.model.mv.sample_sites[initial_carry(item)],
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

                assert initial_carry(item) in self.comp.storage_nd
                # TODO: check that the shape is correct
                #       the shape should be [n_mcmc] + [var shape]
            else:
                raise NotImplementedError()



class WorkSpacePrep_Val_Attr:

    wsv: WorkSpacePrep_Val | WorkSpaceProc_Val
    comp: Computation

    def __init__(self, wsv:WorkSpacePrep_Val | WorkSpaceProc_Val):
        self.wsv = wsv
        self.comp = self.wsv.ws.comp


class WorkSpacePrep_Val_General(WorkSpacePrep_Val_Attr):
    """object to represent `ws.val.general`"""

    def __getitem__(self, item:VarKey) -> ArrayLike:
        return self.comp.storage_nd[item]

    def __setitem__(self, item:VarKey, value:ArrayLike) -> None:
        # TODO: check if the element, subphase defines item
        # TODO: check that the shape is correct
        # TODO: check that the item is not supposed to be sampled
        #
        # TODO: if item is an `observation(obs_var_key)`
        #    check that obs_var_key has nothing in storage_nd or storage_dist
        #    because observations have to be defined before sampling.
        #    ditto if item is an `observation_valid(obs_var_key)`
        self.comp.storage_nd[item] = value


class WorkSpacePrep_Val_InitialCarry(WorkSpacePrep_Val_Attr):
    """object to represent `ws.val.initial_carry`"""

    def __setitem__(self, item:VarKey, value:ArrayLike) -> None:
        # TODO: check if the element, subphase defines item
        # TODO: check that the shape is correct
        ndm = self.comp.model.mv.this_carry_nd[item].definition_metadata
        assert not ndm.sampled
        assert value.shape == tuple(ndm.value_type.shape)
        self.comp.storage_nd[initial_carry(item)] = value


class WorkSpaceStep_Dist_Attr:

    wsd: WorkSpaceStep_Dist
    comp: Computation
    scan_storage: ScanStorage

    def __init__(self, wsd:WorkSpaceStep_Dist):
        self.wsd = wsd
        self.comp = self.wsd.ws.comp
        self.scan_storage = wsd.ws.scan_storage


class WorkSpaceStep_Dist_ThisY(WorkSpaceStep_Dist_Attr):
    """object to represent `ws.dist.this_Y`"""

    def __getitem__(self, item:VarKey) -> Distribution:
        # check if the element, subphase defines item
        # check that the shape is correct
        #
        raise NotImplementedError()

    def __setitem__(self, item:VarKey, value:Distribution) -> None:
        # check if the element, subphase defines item
        # check that the shape is correct
        #
        raise NotImplementedError()


class WorkSpaceStep_Dist_NextCarry(WorkSpaceStep_Dist_Attr):
    """object to represent `ws.dist.next_carry`"""

    def __getitem__(self, item:VarKey) -> Distribution:
        return self.scan_storage.next_carry_dist_d[item]

    def __setitem__(self, item:VarKey, dist:Distribution) -> None:
        # TODO: check if the element, subphase defines item
        # TODO: check that the shape is correct
        # TODO: check that the item is supposed to be sampled

        obs = self.comp.storage_nd.get(observation(item))
        obs_valid = self.comp.storage_nd.get(observation_valid(item))

        if self.comp.phase == Phase.Prior:
            if obs or obs_valid:
                raise NotImplementedError()
            else:
                # TODO: check that the shape is correct
                self.scan_storage.next_carry_dist_d[item] = dist
                self.scan_storage.next_carry_d[item] = numpyro.sample(
                        self.comp.model.mv.sample_sites[next_carry(item)],
                        fn=dist,
                        rng_key=self.scan_storage._split_rng_key())
        else:
            if obs or obs_valid:
                raise NotImplementedError()
            else:
                this_idx = self.scan_storage.this_X_d[str(scan_step_ii_key)]
                n_prior_years = self.comp.model.mv.n_prior_years

                self.scan_storage.next_carry_dist_d[item] = dist.mask(
                        this_idx >= n_prior_years)

                # TODO: require `dist` have a leading mcmc dim?
                sample = numpyro.sample(
                        self.comp.model.mv.sample_sites[next_carry(item)],
                        fn=dist,
                        rng_key=self.scan_storage._split_rng_key())

                self.scan_storage.next_carry_d[item] = jnp.where(
                        this_idx >= n_prior_years,
                        sample,
                        self.comp.storage_nd[NextCarry(carry_key=item)][
                            jnp.minimum(this_idx, n_prior_years - 1)])


class WorkSpaceStep_Val_Attr:

    wsv: WorkSpaceStep_Val
    comp: Computation
    scan_storage: ScanStorage

    def __init__(self, wsv:WorkSpaceStep_Val):
        self.wsv = wsv
        self.comp = self.wsv.ws.comp
        self.scan_storage = wsv.ws.scan_storage


class WorkSpaceStep_Val_General(WorkSpaceStep_Val_Attr):
    """object to represent `ws.val.general`"""

    def __getitem__(self, item:VarKey) -> ArrayLike:
        return self.comp.storage_nd[item]


class WorkSpaceStep_Val_ThisCarry(WorkSpaceStep_Val_Attr):
    """object to represent `ws.val.this_carry`"""

    def __getitem__(self, item:VarKey) -> ArrayLike:
        return self.scan_storage.this_carry_d[item]


class WorkSpaceStep_Val_NextCarry(WorkSpaceStep_Val_Attr):
    """object to represent `ws.val.next_carry`"""

    def __getitem__(self, item:VarKey) -> ArrayLike:
        # TODO: check if the element, subphase defines item
        # TODO: check that the shape is correct
        raise NotImplementedError()

    def __setitem__(self, item:VarKey, value:ArrayLike) -> None:
        # TODO: check if the element, subphase defines item
        # TODO: check that the shape is correct
        # TODO: check that the item is not supposed to be sampled
        self.scan_storage.next_carry_d[item] = value


class WorkSpaceStep_Val_ThisX(WorkSpaceStep_Val_Attr):
    """object to represent `ws.val.this_X`"""

    def __getitem__(self, item:VarKey) -> ArrayLike:
        raise NotImplementedError()


class WorkSpaceStep_Val_ThisY(WorkSpaceStep_Val_Attr):
    """object to represent `ws.val.this_Y`"""

    def __getitem__(self, item:VarKey) -> ArrayLike:
        raise NotImplementedError()

    def __setitem__(self, item:VarKey, value:ArrayLike) -> None:
        # TODO: check if the element, subphase defines item
        # TODO: check that the shape is correct
        # TODO: check that the item is not supposed to be sampled
        #
        self.scan_storage.this_Y_d[item] = value


class WorkSpacePrep_Attr:

    ws: WorkSpace_Prep

    def __init__(self, ws:WorkSpace_Prep):
        self.ws = ws


class WorkSpacePrep_Dist(WorkSpacePrep_Attr):
    """object to represent `ws.dist`"""

    @property
    def general(self) -> WorkSpacePrep_Dist_General:
        return WorkSpacePrep_Dist_General(self)

    @property
    def initial_carry(self) -> WorkSpacePrep_Dist_InitialCarry:
        return WorkSpacePrep_Dist_InitialCarry(self)


class WorkSpacePrep_Val(WorkSpacePrep_Attr):
    """object to represent `ws.val`"""

    @property
    def general(self) -> WorkSpacePrep_Val_General:
        return WorkSpacePrep_Val_General(self)

    @property
    def initial_carry(self) -> WorkSpacePrep_Val_InitialCarry:
        return WorkSpacePrep_Val_InitialCarry(self)


class WorkSpaceStep_Attr:

    ws: WorkSpace_Step

    def __init__(self, ws:WorkSpace_Step):
        self.ws = ws


class WorkSpaceStep_Dist(WorkSpaceStep_Attr):
    """object to represent `ws.dist`"""

    @property
    def this_Y(self) -> WorkSpaceStep_Dist_ThisY:
        return WorkSpaceStep_Dist_ThisY(self)

    @property
    def next_carry(self) -> WorkSpaceStep_Dist_NextCarry:
        return WorkSpaceStep_Dist_NextCarry(self)


class WorkSpaceStep_Val(WorkSpaceStep_Attr):
    """object to represent `ws.val`"""

    @property
    def general(self) -> WorkSpaceStep_Val_General:
        return WorkSpaceStep_Val_General(self)

    @property
    def this_carry(self) -> WorkSpaceStep_Val_ThisCarry:
        return WorkSpaceStep_Val_ThisCarry(self)

    @property
    def next_carry(self) -> WorkSpaceStep_Val_NextCarry:
        return WorkSpaceStep_Val_NextCarry(self)

    @property
    def this_X(self) -> WorkSpaceStep_Val_ThisX:
        return WorkSpaceStep_Val_ThisX(self)

    @property
    def this_Y(self) -> WorkSpaceStep_Val_ThisY:
        return WorkSpaceStep_Val_ThisY(self)


class WorkSpaceProc_Attr:

    ws: WorkSpace_Proc

    def __init__(self, ws:WorkSpace_Proc):
        self.ws = ws


class WorkSpaceProc_Dist(WorkSpaceProc_Attr):
    """object to represent `ws.dist`"""

    @property
    def general(self) -> WorkSpacePrep_Dist_General:
        return WorkSpacePrep_Dist_General(self)


class WorkSpaceProc_Val(WorkSpaceProc_Attr):
    """object to represent `ws.val`"""

    @property
    def general(self) -> WorkSpacePrep_Val_General:
        return WorkSpacePrep_Val_General(self)

    # TODO: initial_carry
    # TODO: final_carry


class WorkSpace:

    year_0: int
    n_years: int
    phase: Phase
    comp: Computation
    model: Model

    def __init__(
            self,
            year_0:int,
            n_years:int,
            phase:Phase,
            comp:Computation,
            ):
        self.year_0 = year_0
        self.n_years = n_years
        assert n_years >= 0
        self.phase = phase
        self.comp = comp
        self.model = self.comp.model


class WorkSpace_Prep(WorkSpace):

    @property
    def dist(self) -> WorkSpacePrep_Dist:
        return WorkSpacePrep_Dist(self)

    @property
    def val(self) -> WorkSpacePrep_Val:
        return WorkSpacePrep_Val(self)


class WorkSpace_Step(WorkSpace):

    scan_storage: ScanStorage

    def __init__(self, scan_storage:ScanStorage, **kwargs):
        super().__init__(**kwargs)
        self.scan_storage = scan_storage

    @property
    def dist(self) -> WorkSpaceStep_Dist:
        return WorkSpaceStep_Dist(self)

    @property
    def val(self) -> WorkSpaceStep_Val:
        return WorkSpaceStep_Val(self)


class WorkSpace_Proc(WorkSpace):

    @property
    def dist(self) -> WorkSpaceProc_Dist:
        return WorkSpaceProc_Dist(self)

    @property
    def val(self) -> WorkSpaceProc_Val:
        return WorkSpaceProc_Val(self)


class ScanStorage:

    this_carry_d: dict[VarKey, ArrayLike]
    this_X_d: dict[VarKey, ArrayLike]
    next_carry_d: dict[VarKey, ArrayLike]
    this_Y_d: dict[VarKey, ArrayLike]

    next_carry_dist_d: dict[VarKey, Distribution]
    this_Y_dist_d: dict[VarKey, Distribution]

    def __init__(self, this_carry_d, this_X_d):
        self.this_carry_d = this_carry_d
        self.this_X_d = this_X_d

        self.next_carry_d = {}
        self.this_Y_d = {}

        self.next_carry_dist_d = {}
        self.this_Y_dist_d = {}

    def carry_rng(self):
        if '__rng_key' not in self.next_carry_d:
            self.next_carry_d['__rng_key'] = self.this_carry_d['__rng_key']

    def _split_rng_key(self):
        self.next_carry_d['__rng_key'], rval = jrandom.split(
                self.next_carry_d['__rng_key'])
        return rval


class Computation:

    model: Model
    storage_dist: dict[VarKey, Distribution]
    storage_nd: dict[VarKey, ArrayLike]
    rng_key: ArrayLike
    phase: Phase
    n_mcmc: int|None
    has_run: bool

    def __init__(
            self,
            model:Model,
            grouped_samples:dict[str, jnp.ndarray]|None,
            rng_key:ArrayLike):

        self.model = model
        self.storage_dist = {}
        self.storage_nd = {}
        self.rng_key = rng_key
        self.phase = (Phase.Prior
                      if grouped_samples is None
                      else Phase.Posterior)
        self.n_mcmc = None
        self.has_run = False

        if grouped_samples is not None:
            self._load_grouped_samples(grouped_samples)

    def _load_grouped_samples(self, grouped_samples:dict[str, jnp.ndarray]):
        rlookup = {
                val: key
                for key, val in self.model.mv.sample_sites.items()}
        assert len(rlookup) == len(self.model.mv.sample_sites)

        for key_str, grouped_sample in grouped_samples.items():
            var_key = rlookup[key_str]
            assert len(grouped_sample.shape) >= 2
            (n_groups, n_saved_samples, *shape) = grouped_sample.shape
            if self.n_mcmc is None:
                self.n_mcmc = n_saved_samples * n_groups
            else:
                assert self.n_mcmc == n_saved_samples * n_groups
            if isinstance(var_key, NextCarry):
                self.storage_nd[GroupedPosterior(prior_var_key=var_key)] = grouped_sample
                (n_groups, n_saved_samples, n_prior_scan_steps, *annual_shape) \
                        = grouped_sample.shape
                assert n_prior_scan_steps == self.model.mv.n_prior_years
                reshaped = grouped_sample.reshape(
                        [n_groups * n_saved_samples, n_prior_scan_steps]
                        + annual_shape)
                transposed = reshaped.transpose(
                        [1, 0] + list(range(2, len(annual_shape))))
                self.storage_nd[var_key] = transposed

                # post-conditions:
                assert transposed.shape[0] == n_prior_scan_steps
                assert transposed.shape[1] == self.n_mcmc
                assert transposed.shape[2:] == tuple(annual_shape), (
                        transposed.shape[2:], annual_shape)

                # and a separate pull_from_sample_mask array of shape [n_years]
                # and these both need to be set up as Xs for the scan
                # and then the ws.dist.next_carry __setitem__ needs to
                # put either the this_X from the sample or the random draw
                # into the scan_storage.next_carry_d, depending on the
                # pull_from_sample_mask.
                # ... and that's all assuming that these dists are not observed,
                # .... which I'm sure they will be sometimes! I think that
                # complicates the logic, but doesn't break the approach.
            else:
                self.storage_nd[GroupedPosterior(prior_var_key=var_key)] = grouped_sample
                self.storage_nd[var_key] \
                        = grouped_sample.reshape([n_groups * n_saved_samples] + shape)
        if self.n_mcmc is None:
            raise NotImplementedError('grouped_samples was empty dict')

    def _split_rng_key(self) -> ArrayLike:
        self.rng_key, key = jrandom.split(self.rng_key)
        return key

    def _initial_carry_d(self):
        rval = {}
        for var_key, val in self.storage_nd.items():
            if isinstance(var_key, InitialCarry):
                if self.phase == Phase.Prior:
                    rval[var_key.carry_key] = val
                else:
                    assert self.phase == Phase.Posterior
                    # In general, the posterior scan operates on
                    # ndarrays with a leading mcmc dimension.
                    # If this is a performance problem, consider
                    # adding a hint to the @define_carry that it isn't
                    # the case for a particular variable.
                    ndm = self.model.mv.this_carry_nd[var_key.carry_key].definition_metadata
                    reqd_shape = [self.n_mcmc] + ndm.value_type.shape
                    if val.shape == tuple(ndm.value_type.shape):
                        bcast_value = jnp.zeros(reqd_shape, dtype=val.dtype)
                        rval[var_key.carry_key] = bcast_value
                    else:
                        assert val.shape == tuple(reqd_shape), (
                                var_key, val.shape, reqd_shape)
                        rval[var_key.carry_key] = val

        return rval

    def _X_d(self):
        # I'm not sure what heuristic / policy to use here.
        # First try: all var_keys in general_nd whose first shape dim
        # is the inference_years_dim.
        rval = {
                str(var_key): self.storage_nd[var_key]
                for var_key, nvm in self.model.mv.general_nd.items()
                if (
                    nvm.definition_metadata.value_type.shape
                    and (nvm.definition_metadata.value_type.shape[0] == years_dim)
                    and var_key in self.storage_nd)
                }
        return rval

    def _elem_items(self):
        yield from self.model.model_elements.items()

    def _run_prep(self):
        for elem_id, elem in self._elem_items():
            try:
                ws = WorkSpace_Prep(
                        comp=self,
                        year_0=self.model.mv.year_0,
                        n_years=self.n_years,
                        phase=self.phase,
                        )
                elem.model_element_prepare(ws)
            except Exception as err:
                err.add_note(f'element_id={elem_id}')
                raise

    def _run_step(self):

        def scan_step(this_carry_d, this_X_d):
            scan_storage = ScanStorage(this_carry_d, this_X_d)
            scan_storage.carry_rng()
            for elem_id, elem in self._elem_items():
                try:
                    ws = WorkSpace_Step(
                            comp=self,
                            year_0=self.model.mv.year_0,
                            n_years=self.n_years,
                            phase=self.phase,
                            scan_storage=scan_storage)
                    elem.model_element_annual_step(ws)
                except Exception as err:
                    err.add_note(f'element_id={elem_id}')
                    raise
            return scan_storage.next_carry_d, scan_storage.this_Y_d

        initial_carry_d = self._initial_carry_d()
        initial_carry_d['__rng_key'] = self._split_rng_key()
        X_d = self._X_d()

        if self.phase == Phase.Prior:
            final_carry_d, Y_d = numpyro_scan(scan_step, initial_carry_d, X_d)
        else:
            final_carry_d, Y_d = jax_scan(scan_step, initial_carry_d, X_d)

        # TODO: check for collisions
        self.storage_nd.update(Y_d)
        self.rng_key = final_carry_d['__rng_key']

    def _run_proc(self):
        for elem_id, elem in self._elem_items():
            try:
                ws = WorkSpace_Proc(
                        comp=self,
                        year_0=self.model.mv.year_0,
                        n_years=self.n_years,
                        phase=self.phase,
                        )
                elem.model_element_postprocess(ws)
            except Exception as err:
                err.add_note(f'element_id={elem_id}')
                raise
    @property
    def n_years(self):
        return (self.model.mv.n_prior_years
                if self.phase == Phase.Prior
                else self.model.mv.n_posterior_years)

    def run(self):
        assert not self.has_run
        self._run_prep()
        self._run_step()
        self._run_proc()
        self.has_run = True

def run_mcmc(
        model:Model,
        seed:int,
        num_warmup:int,
        thinning:int,
        num_samples:int,
        ):
    def trace_fn():
        # TODO verify
        # the seed value is ignored
        # when running via MCMC
        obj = Computation(model=model, rng_key=jrandom.key(1), grouped_samples=None)
        obj.run()
    rng_key = jrandom.key(seed=seed)
    mcmc = MCMC(
            NUTS(trace_fn),
            num_warmup=num_warmup,
            thinning=thinning,
            num_samples=num_samples)
    mcmc.run(rng_key=rng_key)
    return mcmc


def _subphase_from_f(f):
    if f.__name__ == 'model_element_prepare':
        return Subphase.Prep
    elif f.__name__ == 'model_element_annual_step':
        return Subphase.Step
    elif f.__name__ == 'model_element_postprocess':
        return Subphase.Proc
    else:
        raise NotImplementedError(f)

_deco_attr_define = 'model_element_define'

def define(
        var_key:VarKey|property, *,
        sampled:bool,
        shape:list[int|NdarrayDim|property],
        dtype:str='float64',
        ):
    def deco(f):
        if not hasattr(f, _deco_attr_define):
            setattr(f, _deco_attr_define, {})
        assert var_key not in f.model_element_define
        ndm = NdarrayDefinitionMetadata_w_Properties(
                value_type=NdarrayType_w_Properties(
                    shape=shape,
                    dtype=dtype),
                sampled=sampled,
                subphase=_subphase_from_f(f))
        f.model_element_define[var_key] = ndm
        return f
    return deco


_deco_attr_annual = 'model_element_annual_step'

def define_annual(
        var_key:VarKey, *,
        sampled:bool,
        annual_shape:list[int|NdarrayDim],
        dtype:str='float64',
        ):
    def deco(f):
        if not hasattr(f, _deco_attr_annual):
            setattr(f, _deco_attr_annual, {})
        assert var_key not in f.model_element_annual_step
        ndm = NdarrayDefinitionMetadata(
                value_type=NdarrayType(
                    shape=[years_dim] + list(annual_shape),
                    dtype=dtype),
                sampled=sampled,
                subphase=_subphase_from_f(f))
        f.model_element_annual_step[var_key] = ndm
        return f
    return deco

_deco_attr_carry = 'model_element_define_carry'

def define_carry(
        var_key:VarKey, *,
        initial_sampled:bool,
        next_sampled:bool,
        shape:list[int|NdarrayDim],
        dtype:str='float64',
        ):
    def deco(f):
        if not hasattr(f, _deco_attr_carry):
            setattr(f, _deco_attr_carry, {})
        assert var_key not in f.model_element_define_carry
        initial_ndm = NdarrayDefinitionMetadata(
                value_type=NdarrayType(
                    shape=shape,
                    dtype=dtype),
                sampled=initial_sampled,
                subphase=_subphase_from_f(f))
        next_ndm = NdarrayDefinitionMetadata(
                value_type=NdarrayType(
                    shape=shape,
                    dtype=dtype),
                sampled=next_sampled,
                subphase=_subphase_from_f(f))
        f.model_element_define_carry[var_key] = (
                initial_ndm, next_ndm)
        return f
    return deco



class ModelElement:
    """
    Inherit from this to define a model
    """

    _identifier: str|None = None

    @property
    def identifier(self) -> str:
        # must be unique within a model
        return self._identifier or self.__class__.__name__

    def model_element_prepare(self, ws:WorkSpace_Prep):
        pass

    def model_element_annual_step(self, ws:WorkSpace_Step):
        pass

    def model_element_postprocess(self, ws:WorkSpace_Proc):
        pass


years_key = new_named_key('years')
scan_step_ii_key = new_named_key('scan_step_ii')


class ModelBuiltIns(ModelElement):
    @define(years_key, sampled=False, shape=[years_dim])
    @define(scan_step_ii_key, sampled=False, shape=[years_dim])
    def model_element_prepare(self, ws:WorkSpace_Prep):
        ws.val.general[years_key] = jnp.arange(
                ws.model.mv.year_0,
                ws.model.mv.year_0 + ws.n_years)
        ws.val.general[scan_step_ii_key] = jnp.arange(0, ws.n_years)


def _resolve_properties_ndarray_definition(
        element,
        ndmp:NdarrayDefinitionMetadata_w_Properties,
        ) -> NdarrayDefinitionMetadata:

    def shape_elem_fn(shape_elem:int|NdarrayDim|property) -> int|NdarrayDim:
        if isinstance(shape_elem, (int, NdarrayDim)):
            return shape_elem
        else:
            return shape_elem.__get__(element, type(element))

    def shape_fn(shape) -> list[int|NdarrayDim]:
        return [shape_elem_fn(shape_elem) for shape_elem in shape]

    def value_type_fn(value_type) -> NdarrayType:
        return NdarrayType(
                shape=shape_fn(value_type.shape),
                dtype=value_type.dtype)
    return NdarrayDefinitionMetadata(
            value_type=value_type_fn(ndmp.value_type),
            sampled=ndmp.sampled,
            subphase=ndmp.subphase)


class Model:
    """
    Try to make the elements work mostly as a set.
    Try to make this work like a bag of elements that self-assemble.
    That way, documentation of the model can be created by documenting
    the elements.
    I'm afraid that if there is central structuring of the model, it will
    just be impossible to understand.
    Discourage thinking of elements as attributes of the model that can
    be accessed individually.
    """

    model_elements: dict[str, ModelElement]
    mv: ModelVariables

    def __init__(self, first_year:int, n_prior_years:int, n_posterior_years:int):
        self.model_elements = {}
        self.mv = ModelVariables(
                year_0=first_year,
                n_prior_years=n_prior_years,
                n_posterior_years=n_posterior_years,
                )
        self.add_element(ModelBuiltIns())

    def _add_define_d(
            self,
            element:ModelElement,
            defining_subphase:Subphase,
            define_d):

        def add_var_key_ndm(var_key:VarKey, ndm):
            self.mv.general_nd[var_key] = NdarrayVariableMetadata(
                    definition_metadata=_resolve_properties_ndarray_definition(
                        element, ndm),
                    defining_element_id=element.identifier,
                    defining_subphase=defining_subphase)
            if ndm.sampled:
                assert var_key not in self.mv.sample_sites, var_key
                self.mv.sample_sites[var_key] = str(var_key)

        for var_key, ndm in define_d.items():
            if isinstance(var_key, VarKey):
                add_var_key_ndm(var_key, ndm)
            elif isinstance(var_key, property):
                var_key_prop = var_key.__get__(element, type(element))
                if isinstance(var_key_prop, VarKey):
                    add_var_key_ndm(var_key_prop, ndm)
                elif isinstance(var_key_prop, dict):
                    for int_key in var_key_prop.values():
                        add_var_key_ndm(int_key, ndm)
                else:
                    raise NotImplementedError(var_key_prop)
            else:
                raise NotImplementedError(var_key)


    def _add_define_annual_d(self, element_id, defining_subphase, annual_d):
        for var_key, ndm in annual_d.items():
            self.mv.general_nd[var_key] = NdarrayVariableMetadata(
                    definition_metadata=ndm,
                    defining_element_id=element_id,
                    defining_subphase=defining_subphase)
            if ndm.sampled:
                assert var_key not in self.mv.sample_sites, var_key
                self.mv.sample_sites[var_key] = str(var_key)

            print(element_id, defining_subphase, var_key, ndm)

            if defining_subphase in (Subphase.Prep, Subphase.Step):
                this_ndm = NdarrayVariableMetadata(
                        definition_metadata=NdarrayDefinitionMetadata(
                            value_type=NdarrayType(
                                shape=ndm.value_type.shape[1:],
                                dtype=ndm.value_type.dtype),
                            sampled=ndm.sampled,
                            subphase=defining_subphase,
                            ),
                        defining_element_id=element_id,
                        defining_subphase=defining_subphase)
                if defining_subphase == Subphase.Prep:
                    self.mv.this_X_nd[var_key] = this_ndm
                else:
                    self.mv.this_Y_nd[var_key] = this_ndm

    def _add_define_carry_d(self, element_id, carry_d):
        for var_key, (initial_ndm, next_ndm) in carry_d.items():

            initial_var_key = initial_carry(var_key)

            self.mv.general_nd[initial_var_key] = NdarrayVariableMetadata(
                    definition_metadata=initial_ndm,
                    defining_element_id=element_id,
                    defining_subphase=Subphase.Prep)

            if initial_ndm.sampled:
                assert initial_var_key not in self.mv.sample_sites, initial_var_key
                self.mv.sample_sites[initial_var_key] = str(initial_var_key)

            self.mv.this_carry_nd[var_key] = NdarrayVariableMetadata(
                    definition_metadata=NdarrayDefinitionMetadata(
                        value_type=initial_ndm.value_type,
                        sampled=False,
                        subphase=Subphase.Step,
                        ),
                    defining_element_id=element_id,
                    defining_subphase=Subphase.Step)

            self.mv.next_carry_nd[var_key] = NdarrayVariableMetadata(
                    definition_metadata=next_ndm,
                    defining_element_id=element_id,
                    defining_subphase=Subphase.Step)

            if next_ndm.sampled:
                next_var_key = next_carry(var_key)
                assert next_var_key not in self.mv.sample_sites, next_var_key
                self.mv.sample_sites[next_var_key] = str(next_var_key)

            final_var_key = final_carry(var_key)
            self.mv.general_nd[final_var_key] = NdarrayVariableMetadata(
                    definition_metadata=next_ndm,
                    defining_element_id=element_id,
                    defining_subphase=Subphase.Proc)

    def add_element(self, element):
        assert element.identifier
        assert element.identifier not in self.model_elements
        self.model_elements[element.identifier] = element
        self._add_define_d(
                element,
                Subphase.Prep,
                getattr(element.model_element_prepare, _deco_attr_define, {}))
        assert not hasattr(element.model_element_annual_step, _deco_attr_define)
        self._add_define_d(
                element,
                Subphase.Proc,
                getattr(element.model_element_postprocess, _deco_attr_define, {}))

        self._add_define_carry_d(
                element.identifier,
                getattr(element.model_element_prepare, _deco_attr_carry, {}))
        assert not hasattr(element.model_element_annual_step, _deco_attr_carry)
        assert not hasattr(element.model_element_postprocess, _deco_attr_carry)

        self._add_define_annual_d(
                element.identifier,
                Subphase.Prep,
                getattr(element.model_element_prepare, _deco_attr_annual, {}))
        self._add_define_annual_d(
                element.identifier,
                Subphase.Step,
                getattr(element.model_element_annual_step, _deco_attr_annual, {}))
        self._add_define_annual_d(
                element.identifier,
                Subphase.Proc,
                getattr(element.model_element_postprocess, _deco_attr_annual, {}))
