# this resolves circular type references
from __future__ import annotations

from jax.lax import scan as jax_scan

from .scenario_model import *


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
        # check if the element, subphase defines item
        # check that the shape is correct
        #
        self.pc.storage_dist[item] = value

        obs_var_key = self.pc.model.mv.general_nd[item].definition_metadata.observation
        if obs_var_key:
            obs = self.pc.storage_nd[obs_var_key]
        else:
            obs = None

        self.wsd.ws.pc.storage_nd[item] = numpyro.sample(
                self.pc.model.mv.sample_sites[item],
                fn=value,
                rng_key=self.pc._split_rng_key(),
                obs=obs,
                )


class WorkSpacePrepPosterior_Dist_InitialCarry(WorkSpacePrepPosterior_Dist_Attr):
    """object to represent `ws.dist.initial_carry`"""

    def __setitem__(self, item:VarKey, value:Distribution) -> None:
        # check if the element, subphase defines item
        # check that the shape is correct

        self.pc.storage_dist[initial_carry(item)] = value

        self.wsd.ws.pc.storage_nd[initial_carry(item)] = numpyro.sample(
                self.pc.model.mv.sample_sites[initial_carry(item)],
                fn=value,
                rng_key=self.pc._split_rng_key(),
                )

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
        #
        self.pc.storage_nd[item] = value


class WorkSpacePrepPosterior_Val_InitialCarry(WorkSpacePrepPosterior_Val_Attr):
    """object to represent `ws.val.initial_carry`"""

    def __setitem__(self, item:VarKey, value:ArrayLike) -> None:
        # check if the element, subphase defines item
        # check that the shape is correct
        #
        self.pc.storage_nd[initial_carry(item)] = value


class WorkSpacePrepPosterior_Val_AnnualX(WorkSpacePrepPosterior_Val_Attr):
    """object to represent `ws.val.annual_X`"""

    def __setitem__(self, item:VarKey, value:ArrayLike) -> None:
        # check if the element, subphase defines item
        # check that the shape is correct
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


WorkSpace_Prep = InferenceWorkSpace_Prep | PosteriorWorkSpace_Prep
WorkSpace_Step = InferenceWorkSpace_Step | PosteriorWorkSpace_Step
WorkSpace_Proc = InferenceWorkSpace_Post | PosteriorWorkSpace_Post


class PosteriorComputation:

    model: Model

    storage_dist: dict[VarKey, Distribution]
    storage_nd: dict[VarKey, ArrayLike]
    storage_obj: dict[VarKey, object]
    rng_key: ArrayLike

    def __init__(
            self,
            model:Model,
            grouped_samples:dict[str, ArrayLike],
            ):
        self.model = model
        self.storage_dist = {}
        self.storage_nd = {}
        self.storage_obj = {}
        
        rlookup = {
                val: key
                for key, val in model.mv.sample_sites.items()}
        for key_str, grouped_sample in grouped_samples.items():
            var_key = rlookup[key_str]
            self.storage_nd[GroupedPosterior(prior_var_key=var_key)] = grouped_sample
            assert len(grouped_sample.shape) >= 2
            (n_groups, n_saved_scenarios, *shape) = grouped_sample.shape
            self.storage_nd[Posterior(prior_var_key=var_key)] \
                    = grouped_sample.reshape([n_groups * n_saved_scenarios] + shape)

    def _split_rng_key(self) -> ArrayLike:
        self.rng_key, key = jrandom.split(self.rng_key)
        return key

    def run_once(self):
        jax_scan
