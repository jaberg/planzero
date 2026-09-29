# this resolves circular type references
from __future__ import annotations

import jax.random as jrandom
import numpyro
from jax.typing import ArrayLike
from numpyro.contrib.control_flow import scan as numpyro_scan
from numpyro.distributions.distribution import Distribution

from .base import InitialCarry, Model, VarKey, initial_carry, next_carry


class WorkSpacePrepInference_Dist_Attr:

    wsd: WorkSpacePrepInference_Dist

    def __init__(self, wsd:WorkSpacePrepInference_Dist):
        self.wsd = wsd

    @property
    def ic(self) -> InferenceComputation:
        return self.wsd.ws.ic


class WorkSpacePrepInference_Dist_General(WorkSpacePrepInference_Dist_Attr):
    """object to represent `ws.dist.general`"""

    def __setitem__(self, item:VarKey, value:Distribution) -> None:
        # check if the element, subphase defines item
        # check that the shape is correct
        #
        self.ic.storage_dist[item] = value

        obs_var_key = self.ic.model.mv.general_nd[item].definition_metadata.observation
        if obs_var_key:
            obs = self.ic.storage_nd[obs_var_key]
        else:
            obs = None

        self.wsd.ws.ic.storage_nd[item] = numpyro.sample(
                self.ic.model.mv.sample_sites[item],
                fn=value,
                rng_key=self.ic._split_rng_key(),
                obs=obs,
                )


class WorkSpacePrepInference_Dist_InitialCarry(WorkSpacePrepInference_Dist_Attr):
    """object to represent `ws.dist.initial_carry`"""

    def __setitem__(self, item:VarKey, value:Distribution) -> None:
        # check if the element, subphase defines item
        # check that the shape is correct

        self.ic.storage_dist[initial_carry(item)] = value

        self.wsd.ws.ic.storage_nd[initial_carry(item)] = numpyro.sample(
                self.ic.model.mv.sample_sites[initial_carry(item)],
                fn=value,
                rng_key=self.ic._split_rng_key(),
                )

class WorkSpacePrepInference_Val_Attr:

    wsv: WorkSpacePrepInference_Val

    def __init__(self, wsv:WorkSpacePrepInference_Val):
        self.wsv = wsv

    @property
    def ic(self) -> InferenceComputation:
        return self.wsv.ws.ic


class WorkSpacePrepInference_Val_General(WorkSpacePrepInference_Val_Attr):
    """object to represent `ws.val.general`"""

    def __setitem__(self, item:VarKey, value:ArrayLike) -> None:
        # check if the element, subphase defines item
        # check that the shape is correct
        #
        self.ic.storage_nd[item] = value


class WorkSpacePrepInference_Val_InitialCarry(WorkSpacePrepInference_Val_Attr):
    """object to represent `ws.val.initial_carry`"""

    def __setitem__(self, item:VarKey, value:ArrayLike) -> None:
        # check if the element, subphase defines item
        # check that the shape is correct
        #
        self.ic.storage_nd[initial_carry(item)] = value


class WorkSpacePrepInference_Val_AnnualX(WorkSpacePrepInference_Val_Attr):
    """object to represent `ws.val.annual_X`"""

    def __setitem__(self, item:VarKey, value:ArrayLike) -> None:
        # check if the element, subphase defines item
        # check that the shape is correct
        #
        self.ic.storage_nd[item] = value


class WorkSpaceStepInference_Dist_Attr:

    wsd: WorkSpaceStepInference_Dist

    def __init__(self, wsd:WorkSpaceStepInference_Dist):
        self.wsd = wsd

    @property
    def ic(self) -> InferenceComputation:
        return self.wsd.ws.ic

    @property
    def scan_storage(self) -> ScanStorage:
        return self.wsd.ws.scan_storage


class WorkSpaceStepInference_Dist_ThisY(WorkSpaceStepInference_Dist_Attr):
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


class WorkSpaceStepInference_Dist_NextCarry(WorkSpaceStepInference_Dist_Attr):
    """object to represent `ws.dist.next_carry`"""

    def __getitem__(self, item:VarKey) -> Distribution:
        return self.scan_storage.next_carry_dist_d[item]

    def __setitem__(self, item:VarKey, value:Distribution) -> None:
        # check if the element, subphase defines item
        # check that the shape is correct
        #
        self.scan_storage.next_carry_dist_d[item] = value

        self.scan_storage.next_carry_d[item] = numpyro.sample(
                self.ic.model.mv.sample_sites[next_carry(item)],
                fn=value,
                rng_key=self.scan_storage._split_rng_key(),
                )


class WorkSpaceStepInference_Val_Attr:

    wsv: WorkSpaceStepInference_Val

    def __init__(self, wsv:WorkSpaceStepInference_Val):
        self.wsv = wsv

    @property
    def ic(self) -> InferenceComputation:
        return self.wsv.ws.ic

    @property
    def scan_storage(self) -> ScanStorage:
        return self.wsv.ws.scan_storage


class WorkSpaceStepInference_Val_General(WorkSpaceStepInference_Val_Attr):
    """object to represent `ws.val.general`"""

    def __getitem__(self, item:VarKey) -> ArrayLike:
        return self.ic.storage_nd[item]


class WorkSpaceStepInference_Val_ThisCarry(WorkSpaceStepInference_Val_Attr):
    """object to represent `ws.val.this_carry`"""

    def __getitem__(self, item:VarKey) -> ArrayLike:
        return self.scan_storage.this_carry_d[item]


class WorkSpaceStepInference_Val_NextCarry(WorkSpaceStepInference_Val_Attr):
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


class WorkSpaceStepInference_Val_ThisX(WorkSpaceStepInference_Val_Attr):
    """object to represent `ws.val.this_X`"""

    def __getitem__(self, item:VarKey) -> ArrayLike:
        # check if the element, subphase defines item
        # check that the shape is correct
        #
        raise NotImplementedError()


class WorkSpaceStepInference_Val_ThisY(WorkSpaceStepInference_Val_Attr):
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


class WorkSpacePrepInference_Attr:

    ws: InferenceWorkSpace_Prep

    def __init__(self, ws:InferenceWorkSpace_Prep):
        self.ws = ws


class WorkSpacePrepInference_Dist(WorkSpacePrepInference_Attr):
    """object to represent `ws.dist`"""

    @property
    def general(self) -> WorkSpacePrepInference_Dist_General:
        return WorkSpacePrepInference_Dist_General(self)

    @property
    def initial_carry(self) -> WorkSpacePrepInference_Dist_InitialCarry:
        return WorkSpacePrepInference_Dist_InitialCarry(self)


class WorkSpacePrepInference_Val(WorkSpacePrepInference_Attr):
    """object to represent `ws.val`"""

    @property
    def general(self) -> WorkSpacePrepInference_Val_General:
        return WorkSpacePrepInference_Val_General(self)

    @property
    def initial_carry(self) -> WorkSpacePrepInference_Val_InitialCarry:
        return WorkSpacePrepInference_Val_InitialCarry(self)

    @property
    def annual_X(self) -> WorkSpacePrepInference_Val_AnnualX:
        return WorkSpacePrepInference_Val_AnnualX(self)


class WorkSpaceStepInference_Attr:

    ws: InferenceWorkSpace_Step

    def __init__(self, ws:InferenceWorkSpace_Step):
        self.ws = ws


class WorkSpaceStepInference_Dist(WorkSpaceStepInference_Attr):
    """object to represent `ws.dist`"""

    @property
    def this_Y(self) -> WorkSpaceStepInference_Dist_ThisY:
        return WorkSpaceStepInference_Dist_ThisY(self)

    @property
    def next_carry(self) -> WorkSpaceStepInference_Dist_NextCarry:
        return WorkSpaceStepInference_Dist_NextCarry(self)


class WorkSpaceStepInference_Val(WorkSpaceStepInference_Attr):
    """object to represent `ws.val`"""

    @property
    def general(self) -> WorkSpaceStepInference_Val_General:
        return WorkSpaceStepInference_Val_General(self)

    @property
    def this_carry(self) -> WorkSpaceStepInference_Val_ThisCarry:
        return WorkSpaceStepInference_Val_ThisCarry(self)

    @property
    def next_carry(self) -> WorkSpaceStepInference_Val_NextCarry:
        return WorkSpaceStepInference_Val_NextCarry(self)

    @property
    def this_X(self) -> WorkSpaceStepInference_Val_ThisX:
        return WorkSpaceStepInference_Val_ThisX(self)

    @property
    def this_Y(self) -> WorkSpaceStepInference_Val_ThisY:
        return WorkSpaceStepInference_Val_ThisY(self)


class WorkSpacePostInference_Attr:

    ws: InferenceWorkSpace_Post

    def __init__(self, ws:InferenceWorkSpace_Post):
        self.ws = ws


class WorkSpacePostInference_Dist_Attr:

    wsd: WorkSpacePostInference_Dist

    def __init__(self, wsd:WorkSpacePostInference_Dist):
        self.wsd = wsd

    @property
    def ic(self) -> InferenceComputation:
        return self.wsd.ws.ic


class WorkSpacePostInference_Dist_General(WorkSpacePostInference_Dist_Attr):
    """object to represent `ws.dist.general`"""

    def __setitem__(self, item:VarKey, value:Distribution) -> None:
        # check if the element, subphase defines item
        # check that the shape is correct
        #
        self.ic.storage_dist[item] = value

        # define the numpyro sample as well
        obs_var_key = self.ic.model.mv.general_nd[item].definition_metadata.observation
        if obs_var_key:
            obs = self.ic.storage_nd[obs_var_key]
        else:
            obs = None

        self.ic.storage_nd[item] = numpyro.sample(
                self.ic.model.mv.sample_sites[item],
                fn=value,
                rng_key=self.ic._split_rng_key(),
                obs=obs)


class WorkSpacePostInference_Dist(WorkSpacePostInference_Attr):
    """object to represent `ws.dist`"""

    @property
    def general(self) -> WorkSpacePostInference_Dist_General:
        return WorkSpacePostInference_Dist_General(self)


class WorkSpacePostInference_Val_Attr:

    wsv: WorkSpacePostInference_Val

    def __init__(self, wsv:WorkSpacePostInference_Val):
        self.wsv = wsv

    @property
    def ic(self) -> InferenceComputation:
        return self.wsv.ws.ic


class WorkSpacePostInference_Val_General(WorkSpacePostInference_Val_Attr):
    """object to represent `ws.val.general`"""

    def __getitem__(self, item:VarKey) -> ArrayLike:
        return self.ic.storage_nd[item]


class WorkSpacePostInference_Val_AnnualY(WorkSpacePostInference_Val_Attr):
    """object to represent `ws.val.annual_Y`"""

    def __getitem__(self, item:VarKey) -> ArrayLike:
        return self.ic.storage_nd[item]


class WorkSpacePostInference_Val(WorkSpacePostInference_Attr):
    """object to represent `ws.val`"""

    @property
    def general(self) -> WorkSpacePostInference_Val_General:
        return WorkSpacePostInference_Val_General(self)

    @property
    def annual_Y(self) -> WorkSpacePostInference_Val_AnnualY:
        return WorkSpacePostInference_Val_AnnualY(self)


class WorkSpace:
    pass


class InferenceWorkSpace(WorkSpace):
    ic: InferenceComputation

    def __init__(self, ic:InferenceComputation):
        self.ic = ic


class InferenceWorkSpace_Prep(InferenceWorkSpace):

    @property
    def dist(self) -> WorkSpacePrepInference_Dist:
        return WorkSpacePrepInference_Dist(self)

    @property
    def val(self) -> WorkSpacePrepInference_Val:
        return WorkSpacePrepInference_Val(self)


class InferenceWorkSpace_Step(InferenceWorkSpace):

    scan_storage: ScanStorage

    def __init__(self, ic:InferenceComputation, scan_storage:ScanStorage):
        super().__init__(ic=ic)
        self.scan_storage = scan_storage

    @property
    def dist(self) -> WorkSpaceStepInference_Dist:
        return WorkSpaceStepInference_Dist(self)

    @property
    def val(self) -> WorkSpaceStepInference_Val:
        return WorkSpaceStepInference_Val(self)


class InferenceWorkSpace_Post(InferenceWorkSpace):

    @property
    def dist(self) -> WorkSpacePostInference_Dist:
        return WorkSpacePostInference_Dist(self)

    @property
    def val(self) -> WorkSpacePostInference_Val:
        return WorkSpacePostInference_Val(self)


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


class InferenceComputation:

    model: Model

    storage_dist: dict[VarKey, Distribution]
    storage_nd: dict[VarKey, ArrayLike]
    storage_obj: dict[VarKey, object]
    rng_key: ArrayLike

    def __init__(self, model:Model, seed:int):
        self.model = model

        self.storage_dist = {}
        self.storage_nd = {}
        self.storage_obj = {}
        self.rng_key = jrandom.key(seed=seed)

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

    def _X_d(self):
        # I'm not sure what heuristic / policy to use here.
        # First try: all var_keys in general_nd whose first shape dim
        # is the inference_years_dim.
        rval = {
                var_key: self.storage_nd[var_key]
                for var_key, nvm in self.model.mv.general_nd.items()
                if (
                    nvm.definition_metadata.value_type.shape
                    and (
                        nvm.definition_metadata.value_type.shape[0]
                        == self.model.inference_years_dim)
                    and var_key in self.storage_nd)
                }
        return rval

    def run_once(self):
        inference_elements = self.model._inference_elements
        for element_id, inference_element in inference_elements.items():
            try:
                ws = InferenceWorkSpace_Prep(ic=self)
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

        final_carry_d, Y_d = numpyro_scan(scan_step, initial_carry_d, X_d)
        print('final carry')
        print(final_carry_d)
        print('Y_d')
        print(Y_d)

        # TODO: check for collisions
        self.storage_nd.update(Y_d)
        self.rng_key = final_carry_d['__rng_key']

        for element_id, inference_element in inference_elements.items():
            try:
                ws = InferenceWorkSpace_Post(ic=self)
                inference_element.annual_scan_post_fn(ws)
            except Exception as err:
                err.add_note(f'element_id={element_id}')
                raise

def run_mcmc(
        model:Model,
        seed:int,
        num_warmup:int,
        thinning:int,
        num_samples:int,
        ):
    from numpyro.infer import MCMC, NUTS
    def trace_fn():
        # the seed value is ignored
        # when running via MCMC
        obj = InferenceComputation(model=model, seed=1)
        obj.run_once()

    rng_key = jrandom.key(seed=seed)
    mcmc = MCMC(NUTS(trace_fn),
                num_warmup=num_warmup,
                thinning=thinning,
                num_samples=num_samples)
    mcmc.run(rng_key=rng_key)
    mcmc.print_summary()
    grouped_samples = mcmc.get_samples(group_by_chain=True)
    for key, val in grouped_samples.items():
        print(key)
        print(val.shape)
        print()
    return mcmc
