# this resolves circular type references
from __future__ import annotations

import jax.random as jrandom
import numpyro
from jax.lax import scan as jax_scan
from jax.typing import ArrayLike
from numpyro.distributions.distribution import Distribution

from .base import (
    GroupedPosterior,
    InitialCarry,
    Model,
    Posterior,
    VarKey,
    g_inference_years_dim,
    g_posterior_years_dim,
    initial_carry,
    posterior,
)
from .prior import (
    InferenceWorkSpace_Post,
    InferenceWorkSpace_Prep,
    InferenceWorkSpace_Step,
    ScanStorage,
    WorkSpace,
)


class WorkSpacePrepPosterior_Dist_Attr:

    wsd: WorkSpacePrepPosterior_Dist

    def __init__(self, wsd:WorkSpacePrepPosterior_Dist):
        self.wsd = wsd

    @property
    def pc(self) -> PosteriorComputation:
        return self.wsd.ws.pc


class WorkSpacePrepPosterior_Dist_General(WorkSpacePrepPosterior_Dist_Attr):
    """object to represent `ws.dist.general`"""

    def __setitem__(self, item:VarKey, value:Distribution) -> None:
        # A general distribution assignment variable
        # is either observed or unobserved.
        #
        # If it was observed, then it was only observed for the conditioning years.
        #
        # If it was not observed, then values should already be present from mcmc.
        #
        # check if the element, subphase defines item
        # check that the shape is correct
        
        dm = self.pc.model.mv.general_nd[item].definition_metadata
        if dm.value_type.shape and (dm.value_type.shape[0] == g_inference_years_dim):
            raise NotImplementedError()
        else:
            obs_var_key = dm.observation
            if obs_var_key:
                # this variable was observed, and is not a timeseries
                # we leave the distribution undefined
                # and we should set the value to the observed value
                obs_value = self.pc.storage_nd[obs_var_key]
                # we should check that it has the right shape,
                # and broadcast it over the mcmc dimension
                self.pc.storage_nd[item] = obs_value[None, ...]
            else:
                # this variable was not observed, and is not a timeseries
                #
                # we leave the distribution un-defined because all we have
                # is samples from the posterior,
                # which are already supposed to be in storage_nd
                assert item in self.pc.storage_nd, item


    def __getitem__(self, item:VarKey) -> Distribution:
        raise NotImplementedError()


class WorkSpacePrepPosterior_Dist_InitialCarry(WorkSpacePrepPosterior_Dist_Attr):
    """object to represent `ws.dist.initial_carry`"""

    def __setitem__(self, item:VarKey, value:Distribution) -> None:
        # check if the element, subphase defines item
        # check that the shape is correct
        # check that we should be sampling this variable
        vm = self.pc.model.mv.initial_carry_nd[item]
        dm = vm.definition_metadata
        assert g_inference_years_dim not in dm.value_type.shape, item

        obs_var_key = dm.observation
        if obs_var_key:
            # this variable was observed
            # we leave the distribution undefined
            # and we should set the value to the observed value
            obs_value = self.pc.storage_nd[obs_var_key]
            # we should check that it has the right shape,
            # and broadcast it over the mcmc dimension
            self.pc.storage_nd[initial_carry(item)] = obs_value[None, ...]
        else:
            # this variable was not observed, and is not a timeseries
            #
            # we leave the distribution un-defined because all we have
            # is samples from the posterior,
            # which are already supposed to be in storage_nd
            assert initial_carry(item) in self.pc.storage_nd

    def __getitem__(self, item:VarKey) -> Distribution:
        raise NotImplementedError()

class WorkSpacePrepPosterior_Val_Attr:

    wsv: WorkSpacePrepPosterior_Val

    def __init__(self, wsv:WorkSpacePrepPosterior_Val):
        self.wsv = wsv

    @property
    def pc(self) -> PosteriorComputation:
        return self.wsv.ws.pc


class WorkSpacePrepPosterior_Val_General(WorkSpacePrepPosterior_Val_Attr):
    """object to represent `ws.val.general`"""

    def __setitem__(self, item:VarKey, value:ArrayLike) -> None:
        # check if the element, subphase defines item
        # check that the shape is correct
        # check that we should NOT be sampling this variable
        #
        self.pc.storage_nd[item] = value


class WorkSpacePrepPosterior_Val_InitialCarry(WorkSpacePrepPosterior_Val_Attr):
    """object to represent `ws.val.initial_carry`"""

    def __setitem__(self, item:VarKey, value:ArrayLike) -> None:
        # check if the element, subphase defines item
        # check that the shape is correct
        # it may include a leading mcmc dimension
        # check that we should NOT be sampling this variable
        self.pc.storage_nd[initial_carry(item)] = value


class WorkSpacePrepPosterior_Val_AnnualX(WorkSpacePrepPosterior_Val_Attr):
    """object to represent `ws.val.annual_X`"""

    def __setitem__(self, item:VarKey, value:ArrayLike) -> None:
        # check if the element, subphase defines item
        # check that the shape is correct
        # check that we should NOT be sampling this variable
        #
        self.pc.storage_nd[item] = value


class WorkSpaceStepPosterior_Dist_Attr:

    wsd: WorkSpaceStepPosterior_Dist

    def __init__(self, wsd:WorkSpaceStepPosterior_Dist):
        self.wsd = wsd

    @property
    def pc(self) -> PosteriorComputation:
        return self.wsd.ws.pc

    @property
    def scan_storage(self) -> ScanStorage:
        return self.wsd.ws.scan_storage


class WorkSpaceStepPosterior_Dist_ThisY(WorkSpaceStepPosterior_Dist_Attr):
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


class WorkSpaceStepPosterior_Dist_NextCarry(WorkSpaceStepPosterior_Dist_Attr):
    """object to represent `ws.dist.next_carry`"""

    def __getitem__(self, item:VarKey) -> Distribution:
        return self.scan_storage.next_carry_dist_d[item]

    def __setitem__(self, item:VarKey, value:Distribution) -> None:
        # check if the element, subphase defines item
        # check that the shape is correct
        #
        self.scan_storage.next_carry_dist_d[item] = value

        self.scan_storage.next_carry_d[item] = numpyro.sample(
                self.pc.model.mv.sample_sites[next_carry(item)],
                fn=value,
                rng_key=self.scan_storage._split_rng_key(),
                )


class WorkSpaceStepPosterior_Val_Attr:

    wsv: WorkSpaceStepPosterior_Val

    def __init__(self, wsv:WorkSpaceStepPosterior_Val):
        self.wsv = wsv

    @property
    def pc(self) -> PosteriorComputation:
        return self.wsv.ws.pc

    @property
    def scan_storage(self) -> ScanStorage:
        return self.wsv.ws.scan_storage


class WorkSpaceStepPosterior_Val_General(WorkSpaceStepPosterior_Val_Attr):
    """object to represent `ws.val.general`"""

    def __getitem__(self, item:VarKey) -> ArrayLike:
        return self.pc.storage_nd[item]


class WorkSpaceStepPosterior_Val_ThisCarry(WorkSpaceStepPosterior_Val_Attr):
    """object to represent `ws.val.this_carry`"""

    def __getitem__(self, item:VarKey) -> ArrayLike:
        return self.scan_storage.this_carry_d[item]


class WorkSpaceStepPosterior_Val_NextCarry(WorkSpaceStepPosterior_Val_Attr):
    """object to represent `ws.val.next_carry`"""

    def __getitem__(self, item:VarKey) -> ArrayLike:
        # check if the element, subphase defines item
        # check that the shape is correct
        #
        raise NotImplementedError()


    def __setitem__(self, item:VarKey, value:ArrayLike) -> None:
        # check if the element, subphase defines item
        # check that the shape is correct
        #
        self.scan_storage.next_carry_d[item] = value


class WorkSpaceStepPosterior_Val_ThisX(WorkSpaceStepPosterior_Val_Attr):
    """object to represent `ws.val.this_X`"""

    def __getitem__(self, item:VarKey) -> ArrayLike:
        # check if the element, subphase defines item
        # check that the shape is correct
        #
        raise NotImplementedError()


class WorkSpaceStepPosterior_Val_ThisY(WorkSpaceStepPosterior_Val_Attr):
    """object to represent `ws.val.this_Y`"""

    def __getitem__(self, item:VarKey) -> ArrayLike:
        # check if the element, subphase defines item
        # check that the shape is correct
        #
        raise NotImplementedError()

    def __setitem__(self, item:VarKey, value:ArrayLike) -> None:
        # check if the element, subphase defines item
        # check that the shape is correct
        #
        self.scan_storage.this_Y_d[item] = value


class WorkSpacePrepPosterior_Attr:

    ws: PosteriorWorkSpace_Prep

    def __init__(self, ws:PosteriorWorkSpace_Prep):
        self.ws = ws


class WorkSpacePrepPosterior_Dist(WorkSpacePrepPosterior_Attr):
    """object to represent `ws.dist`"""

    @property
    def general(self) -> WorkSpacePrepPosterior_Dist_General:
        return WorkSpacePrepPosterior_Dist_General(self)

    @property
    def initial_carry(self) -> WorkSpacePrepPosterior_Dist_InitialCarry:
        return WorkSpacePrepPosterior_Dist_InitialCarry(self)


class WorkSpacePrepPosterior_Val(WorkSpacePrepPosterior_Attr):
    """object to represent `ws.val`"""

    @property
    def general(self) -> WorkSpacePrepPosterior_Val_General:
        return WorkSpacePrepPosterior_Val_General(self)

    @property
    def initial_carry(self) -> WorkSpacePrepPosterior_Val_InitialCarry:
        return WorkSpacePrepPosterior_Val_InitialCarry(self)

    @property
    def annual_X(self) -> WorkSpacePrepPosterior_Val_AnnualX:
        return WorkSpacePrepPosterior_Val_AnnualX(self)


class WorkSpaceStepPosterior_Attr:

    ws: PosteriorWorkSpace_Step

    def __init__(self, ws:PosteriorWorkSpace_Step):
        self.ws = ws


class WorkSpaceStepPosterior_Dist(WorkSpaceStepPosterior_Attr):
    """object to represent `ws.dist`"""

    @property
    def this_Y(self) -> WorkSpaceStepPosterior_Dist_ThisY:
        return WorkSpaceStepPosterior_Dist_ThisY(self)

    @property
    def next_carry(self) -> WorkSpaceStepPosterior_Dist_NextCarry:
        return WorkSpaceStepPosterior_Dist_NextCarry(self)


class WorkSpaceStepPosterior_Val(WorkSpaceStepPosterior_Attr):
    """object to represent `ws.val`"""

    @property
    def general(self) -> WorkSpaceStepPosterior_Val_General:
        return WorkSpaceStepPosterior_Val_General(self)

    @property
    def this_carry(self) -> WorkSpaceStepPosterior_Val_ThisCarry:
        return WorkSpaceStepPosterior_Val_ThisCarry(self)

    @property
    def next_carry(self) -> WorkSpaceStepPosterior_Val_NextCarry:
        return WorkSpaceStepPosterior_Val_NextCarry(self)

    @property
    def this_X(self) -> WorkSpaceStepPosterior_Val_ThisX:
        return WorkSpaceStepPosterior_Val_ThisX(self)

    @property
    def this_Y(self) -> WorkSpaceStepPosterior_Val_ThisY:
        return WorkSpaceStepPosterior_Val_ThisY(self)


class WorkSpacePostPosterior_Attr:

    ws: PosteriorWorkSpace_Post

    def __init__(self, ws:PosteriorWorkSpace_Post):
        self.ws = ws


class WorkSpacePostPosterior_Dist_Attr:

    wsd: WorkSpacePostPosterior_Dist

    def __init__(self, wsd:WorkSpacePostPosterior_Dist):
        self.wsd = wsd

    @property
    def pc(self) -> PosteriorComputation:
        return self.wsd.ws.pc


class WorkSpacePostPosterior_Dist_General(WorkSpacePostPosterior_Dist_Attr):
    """object to represent `ws.dist.general`"""

    def __setitem__(self, item:VarKey, value:Distribution) -> None:
        # check if the element, subphase defines item
        # check that the shape is correct
        #
        self.pc.storage_dist[item] = value

        # define the numpyro sample as well
        obs_var_key = self.pc.model.mv.general_nd[item].definition_metadata.observation
        if obs_var_key:
            obs = self.pc.storage_nd[obs_var_key]
        else:
            obs = None

        self.pc.storage_nd[item] = numpyro.sample(
                self.pc.model.mv.sample_sites[item],
                fn=value,
                rng_key=self.pc._split_rng_key(),
                obs=obs)


class WorkSpacePostPosterior_Dist(WorkSpacePostPosterior_Attr):
    """object to represent `ws.dist`"""

    @property
    def general(self) -> WorkSpacePostPosterior_Dist_General:
        return WorkSpacePostPosterior_Dist_General(self)


class WorkSpacePostPosterior_Val_Attr:

    wsv: WorkSpacePostPosterior_Val

    def __init__(self, wsv:WorkSpacePostPosterior_Val):
        self.wsv = wsv

    @property
    def pc(self) -> PosteriorComputation:
        return self.wsv.ws.pc


class WorkSpacePostPosterior_Val_General(WorkSpacePostPosterior_Val_Attr):
    """object to represent `ws.val.general`"""

    def __getitem__(self, item:VarKey) -> ArrayLike:
        return self.pc.storage_nd[item]


class WorkSpacePostPosterior_Val_AnnualY(WorkSpacePostPosterior_Val_Attr):
    """object to represent `ws.val.annual_Y`"""

    def __getitem__(self, item:VarKey) -> ArrayLike:
        return self.pc.storage_nd[item]


class WorkSpacePostPosterior_Val(WorkSpacePostPosterior_Attr):
    """object to represent `ws.val`"""

    @property
    def general(self) -> WorkSpacePostPosterior_Val_General:
        return WorkSpacePostPosterior_Val_General(self)

    @property
    def annual_Y(self) -> WorkSpacePostPosterior_Val_AnnualY:
        return WorkSpacePostPosterior_Val_AnnualY(self)



class PosteriorWorkSpace(WorkSpace):
    pc: PosteriorComputation

    def __init__(self, pc:PosteriorComputation):
        self.pc = pc


class PosteriorWorkSpace_Prep(PosteriorWorkSpace):

    @property
    def dist(self) -> WorkSpacePrepPosterior_Dist:
        return WorkSpacePrepPosterior_Dist(self)

    @property
    def val(self) -> WorkSpacePrepPosterior_Val:
        return WorkSpacePrepPosterior_Val(self)


class PosteriorWorkSpace_Step(PosteriorWorkSpace):

    scan_storage: ScanStorage

    def __init__(self, pc:PosteriorComputation, scan_storage:ScanStorage):
        super().__init__(pc=pc)
        self.scan_storage = scan_storage

    @property
    def dist(self) -> WorkSpaceStepPosterior_Dist:
        return WorkSpaceStepPosterior_Dist(self)

    @property
    def val(self) -> WorkSpaceStepPosterior_Val:
        return WorkSpaceStepPosterior_Val(self)


class PosteriorWorkSpace_Post(PosteriorWorkSpace):

    @property
    def dist(self) -> WorkSpacePostPosterior_Dist:
        return WorkSpacePostPosterior_Dist(self)

    @property
    def val(self) -> WorkSpacePostPosterior_Val:
        return WorkSpacePostPosterior_Val(self)

    @property
    def obs(self) -> WorkSpacePostPosterior_Obs:
        return WorkSpacePostPosterior_Obs(self)

    @property
    def obs_valid(self) -> WorkSpacePostPosterior_ObsValid:
        return WorkSpacePostPosterior_ObsValid(self)


WorkSpace_Prep = InferenceWorkSpace_Prep | PosteriorWorkSpace_Prep
WorkSpace_Step = InferenceWorkSpace_Step | PosteriorWorkSpace_Step
WorkSpace_Proc = InferenceWorkSpace_Post | PosteriorWorkSpace_Post


class PosteriorComputation:

    model: Model

    storage_dist: dict[VarKey, Distribution]
    storage_nd: dict[VarKey, ArrayLike]
    storage_obj: dict[VarKey, object]
    rng_key: ArrayLike
    n_saved_samples: int

    def __init__(
            self,
            model:Model,
            grouped_samples:dict[str, ArrayLike],
            rng_key:ArrayLike,
            ):
        self.model = model
        self.storage_dist = {}
        self.storage_nd = {}
        self.storage_obj = {}
        self.rng_key = rng_key
        self.n_saved_samples = 0
        
        rlookup = {
                val: key
                for key, val in model.mv.sample_sites.items()}
        for key_str, grouped_sample in grouped_samples.items():
            var_key = rlookup[key_str]
            self.storage_nd[GroupedPosterior(prior_var_key=var_key)] = grouped_sample
            assert len(grouped_sample.shape) >= 2
            (n_groups, n_saved_samples, *shape) = grouped_sample.shape
            self.storage_nd[var_key] \
                    = grouped_sample.reshape([n_groups * n_saved_samples] + shape)
            if self.n_saved_samples:
                assert self.n_saved_samples == n_saved_samples
            else:
                self.n_saved_samples = n_saved_samples

    def _split_rng_key(self) -> ArrayLike:
        self.rng_key, key = jrandom.split(self.rng_key)
        return key

    def _initial_carry_d(self):
        initial_carry_d = {
                var_key.carry_key: val
                for var_key, val in self.storage_nd.items()
                if isinstance(var_key, InitialCarry)
                }
        return initial_carry_d

    def _len_scan1(self):
        return min(
                self.model.mv.n_conditioning_years,
                self.model.mv.n_posterior_years)

    def _X_d(self):
        # I'm not sure what heuristic / policy to use here.
        # First try: all var_keys in general_nd whose first shape dim
        # is the inference_years_dim.
        #
        # This inclusion logic should exactly match
        # InferenceComputation._X_d, consider using the same function
        rval = {}
        len_scan = self._len_scan1()
        for var_key, nvm in self.model.mv.general_nd.items():
            def_shape = nvm.definition_metadata.value_type.shape
            if (def_shape
                and (def_shape[0] in {g_posterior_years_dim, g_inference_years_dim})
                and var_key in self.storage_nd):
                # the following val is what went into the dict
                # for prior inference
                # We should include the same
                val = self.storage_nd[var_key]
                if len(val.shape) == len(def_shape):
                    rval[var_key] = val[:len_scan]
                elif len(val.shape) == len(def_shape) + 1:
                    assert val.shape[0] == self.n_saved_samples
                    idxs = list(range(len(val.shape)))
                    idxs[:2] = [1, 0]
                    rval[var_key] = val.transpose(tuple(idxs))[:len_scan]
                else:
                    raise NotImplementedError()
                assert rval[var_key].shape[0] == len_scan
            else:
                pass
            
        return rval

    def _len_scan2(self):
        return max(self.model.mv.n_posterior_years,
                   - self.model.mv.n_conditioning_years,
                   0)

    def _X2_d(self):
        rval = {}
        len_scan1 = self._len_scan1()
        len_scan2 = self._len_scan2()
        assert len_scan2, 'do not call if scan2 len is 0'

        for var_key, nvm in self.model.mv.general_nd.items():
            def_shape = nvm.definition_metadata.value_type.shape
            if (def_shape
                and (def_shape[0] == g_posterior_years_dim)
                and var_key in self.storage_nd):

                # the goal here is to pass the remaining time-slices
                # of variables the longer length
                val = self.storage_nd[var_key]
                if len(val.shape) == len(def_shape):
                    rval[var_key] = val[len_scan1:len_scan1 + len_scan2]
                elif len(val.shape) == len(def_shape) + 1:
                    assert val.shape[0] == self.n_saved_samples
                    idxs = list(range(len(val.shape)))
                    idxs[:2] = [1, 0]
                    all_tranposed = val.transpose(tuple(idxs))
                    rval[var_key] = all_transposed[len_scan1:len_scan1 + len_scan2]
                else:
                    raise NotImplementedError()

                assert rval[var_key].shape[0] == len_scan2
            else:
                pass
            
        return rval


    def run_once(self):
        inference_elements = self.model._inference_elements

        for element_id, inference_element in inference_elements.items():
            try:
                ws = PosteriorWorkSpace_Prep(pc=self)
                inference_element.annual_scan_prep_fn(ws)
            except Exception as err:
                err.add_note(f'element_id={element_id}')
                raise

        def scan_step(this_carry_d, this_X_d):
            scan_storage = ScanStorage(this_carry_d, this_X_d)
            scan_storage.carry_rng()
            for element_id, inference_element in inference_elements.items():
                try:
                    ws = InferenceWorkSpace_Step(
                            ic=self,
                            scan_storage=scan_storage)
                    inference_element.annual_scan_step_fn(ws)
                except Exception as err:
                    err.add_note(f'element_id={element_id}')
                    raise
            return scan_storage.next_carry_d, scan_storage.this_Y_d

        initial_carry_d = self._initial_carry_d()
        initial_carry_d['__rng_key'] = self._split_rng_key()
        X_d = self._X_d()

        final_carry_d, Y_d = jax_scan(scan_step, initial_carry_d, X_d)
        print('final carry')
        print(final_carry_d)
        print('Y_d')
        print(Y_d)
