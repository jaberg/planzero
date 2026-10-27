# this resolves circular type references
from __future__ import annotations

import jax.numpy as jnp
import jax.random as jrandom
import numpyro
from jax.lax import scan as jax_scan
from jax.typing import ArrayLike
from numpyro.contrib.control_flow import scan as numpyro_scan
from numpyro.distributions import Distribution, kl
from numpyro.infer import MCMC, NUTS

from .base import (
    ElementAnnualKey,
    ElementCarryKey,
    ElementGeneralKey,
    FinalCarry,
    GroupedPosterior,
    InitialCarry,
    NdarrayDim,
    NdarrayVariableMetadata,
    NextCarry,
    Observation,
    ObservationValid,
    ObservationWeight,
    Phase,
    Subphase,
    UnassignedDefinitionError,
    VarKey,
    mcmc_dim,
    rng_var_key,
    years_dim,
)
from .scenario_model import (
    ModelElement,
    ResolvedAccessMetadata,
    ScenarioModel,
)


class ScanStorage_ThisAnnual_NdarrayD:
    """Dict-like access by key for annual ndarrays in a scan.

    Covers both kinds of annual variable declared with define_annual:
    inputs defined prior to the scan (on model_element_prepare) and
    outputs defined by the scan step itself (on model_element_annual_step).
    """
    scanstor: ScanStorage

    def __init__(self, scanstor:ScanStorage):
        self.scanstor = scanstor

    def __getitem__(self, item:VarKey) -> jnp.ndarray:
        return self.scanstor.this_annual_get_ndarray(item, sample_if_necessary=True)

    def __setitem__(self, item:VarKey, value:jnp.ndarray):
        self.scanstor.this_annual_set_ndarray(item, value)



class ScanStorage:

    this_carry_d: dict[VarKey, jnp.ndarray]
    next_carry_d: dict[VarKey, jnp.ndarray]
    next_carry_dist_d: dict[VarKey, Distribution]

    # Annual variables at this scan step, i.e. this_annual. Both dicts hold
    # one value per variable for the current step only.
    #
    # - this_annual_input_d: annual variables defined prior to the scan with
    #   define_annual on model_element_prepare. The scan slices these and
    #   feeds them in one step at a time.
    # - this_annual_output_d: annual variables defined by the scan step
    #   itself with define_annual on model_element_annual_step. The scan
    #   stacks these into full timeseries.
    this_annual_input_d: dict[VarKey, jnp.ndarray]
    this_annual_output_d: dict[VarKey, jnp.ndarray]

    this_annual_d: ScanStorage_ThisAnnual_NdarrayD
    this_annual_dist_d: dict[VarKey, Distribution]

    def __init__(self, this_carry_d, this_annual_input_d, sample_site_fn, phase):
        self.this_carry_d = this_carry_d
        self.this_annual_input_d = this_annual_input_d

        self.next_carry_d = {}
        self.next_carry_dist_d = {}

        self.this_annual_output_d = {}
        self.this_annual_d = ScanStorage_ThisAnnual_NdarrayD(scanstor=self)
        self.this_annual_dist_d = {}

        self.sample_site_fn = sample_site_fn
        self.phase = phase

    def carry_rng(self):
        if rng_var_key not in self.next_carry_d:
            self.next_carry_d[rng_var_key] = self.this_carry_d[rng_var_key]

    def _split_rng_key(self):
        self.next_carry_d[rng_var_key], rval = jrandom.split(
                self.next_carry_d[rng_var_key])
        return rval

    def this_annual_get_ndarray(
            self,
            var_key:VarKey,
            sample_if_necessary:bool,
            default:jnp.ndarray|None=None,
            ) -> jnp.ndarray:
        if var_key in self.this_annual_input_d:
            # annual variable defined prior to the scan: its value at this
            # step is the input slice.
            return self.this_annual_input_d[var_key]

        # annual variable defined by the scan step itself.
        try:
            return self.this_annual_output_d[var_key]
        except KeyError as err:
            if (var_key not in self.this_annual_dist_d or not sample_if_necessary):
                if default is not None:
                    if self.phase == Phase.Prior:
                        return default
                    elif self.phase == Phase.Posterior:
                        return jnp.expand_dims(default, -1)
                    else:
                        raise NotImplementedError(self.phase)
                err.add_note(f'var_key in dist_d = {var_key in self.this_annual_dist_d}')
                err.add_note(f'sample_if_necessary = {sample_if_necessary}')
                raise
        self.this_annual_set_ndarray_from_dist(var_key)
        return self.this_annual_output_d[var_key]

    def this_annual_set_ndarray(self, var_key:VarKey, value:jnp.ndarray) -> None:
        if var_key in self.this_annual_input_d:
            raise NotImplementedError(
                    f'{var_key} is an annual variable defined prior to the scan'
                    ' (define_annual on model_element_prepare); it cannot be'
                    ' assigned during a scan step.'
                    ' Assign it during prepare via ws.val.annual.')
        self.this_annual_output_d[var_key] = value

    def this_annual_set_ndarray_from_dist(self, var_key:VarKey) -> None:
        assert var_key in self.this_annual_dist_d
        if var_key in self.this_annual_output_d:
            # if it's already set, this method can't promise it's
            # set to the right distribution
            raise NotImplementedError()

        # accessing an undefined value of a general key after the dist
        # has been set for that key, triggers sampling.

        obs_var_key = Observation(prior_var_key=var_key)
        obs_val = self.this_annual_output_d.get(obs_var_key)
        obs_dist = self.this_annual_dist_d.get(obs_var_key)

        if obs_val or obs_dist:
            raise NotImplementedError()

        dist = self.this_annual_dist_d[var_key]

        dist_sample = numpyro.sample(
                self.sample_site_fn(var_key),
                fn=dist,
                rng_key=self._split_rng_key(),
                )
        if isinstance(dist_sample, jnp.ndarray):
            self.this_annual_set_ndarray(var_key, dist_sample)
        else:
            raise NotImplementedError(dist_sample)


def build_general_prior_shape(comp:Computation, var_key:VarKey, val_shape:tuple) -> list[int]:
    metadata_shape = comp.model.mv.general_nd[var_key].definition_metadata.value_type.shape
    prior_shape_list:list[int] = []
    for ii, dim_ii in enumerate(metadata_shape):
        if isinstance(dim_ii, NdarrayDim):
            comp.ndarray_dim_d.setdefault(dim_ii, val_shape[ii])
            prior_shape_list.append(comp.ndarray_dim_d[dim_ii])
        elif isinstance(dim_ii, int):
            prior_shape_list.append(dim_ii)
        else:
            raise NotImplementedError(dim_ii)
    return prior_shape_list


def build_annual_prior_shape(comp:Computation, var_key:ElementAnnualKey, val_shape:tuple) -> list[int]:
    """Return the shape of value assigned to var_key after calling `comp.set_ndarray`
    with an argument (distribution or ndarray) of shape `val_shape`
    in phase `Phase.Prior`
    """
    metadata_annual_shape = comp.model.mv.annual_nd[var_key].definition_metadata.value_type.shape
    metadata_shape = [comp.model.n_prior_years] + metadata_annual_shape
    if len(val_shape) < len(metadata_shape):
        broadcast_val_shape = (1,) * (len(metadata_shape) - len(val_shape)) + val_shape
    else:
        broadcast_val_shape = val_shape
    del val_shape

    prior_shape_list:list[int] = []
    if len(broadcast_val_shape) == len(metadata_shape):
        for ii, dim_ii in enumerate(metadata_shape):
            if isinstance(dim_ii, NdarrayDim):
                comp.ndarray_dim_d.setdefault(dim_ii, broadcast_val_shape[ii])
                prior_shape_list.append(comp.ndarray_dim_d[dim_ii])
            elif isinstance(dim_ii, int):
                prior_shape_list.append(dim_ii)
            else:
                raise NotImplementedError(dim_ii)
    else:
        raise ValueError(f"val_shape {broadcast_val_shape} has too many dimensions")
    return prior_shape_list


def build_annual_posterior_shape(comp:Computation, var_key:ElementAnnualKey, val_shape:tuple) -> list[int]:
    """Return the shape of value assigned to var_key after calling `comp.set_ndarray`
    with an argument (distribution or ndarray) of shape `val_shape`
    in phase `Phase.Posterior`
    """
    metadata_annual_shape = comp.model.mv.annual_nd[var_key].definition_metadata.value_type.shape
    metadata_shape = (
            [comp.model.n_posterior_years]
            + metadata_annual_shape
            + [comp.n_mcmc])
    if len(val_shape) < len(metadata_shape):
        # broadcast val_shape by left-padding shape with (1,)
        broadcast_val_shape = (1,) * (len(metadata_shape) - len(val_shape)) + val_shape
    else:
        broadcast_val_shape = val_shape
    del val_shape

    posterior_shape_list:list[int] = []
    if len(broadcast_val_shape) == len(metadata_shape):
        for ii, dim_ii in enumerate(metadata_shape):
            if isinstance(dim_ii, NdarrayDim):
                comp.ndarray_dim_d.setdefault(dim_ii, broadcast_val_shape[ii])
                posterior_shape_list.append(comp.ndarray_dim_d[dim_ii])
            elif isinstance(dim_ii, int):
                posterior_shape_list.append(dim_ii)
            else:
                raise NotImplementedError(dim_ii)
    else:
        raise ValueError(f"val_shape {broadcast_val_shape} has too many dimensions")
    return posterior_shape_list


def storage_nd_check_ElementGeneralKey(comp:Computation, var_key:ElementGeneralKey, val:jnp.ndarray) -> bool:
    prior_shape = tuple(build_general_prior_shape(comp, var_key, val.shape))

    if comp.phase == Phase.Prior:
        assert val.shape == prior_shape, (var_key, val.shape, prior_shape)
        return False
    elif comp.phase == Phase.Posterior:
        if val.shape == prior_shape:
            return True
        elif (val.shape == prior_shape + (1,)
              or val.shape == prior_shape + (comp.n_mcmc,)):
            return False
        else:
            raise NotImplementedError(val.shape, prior_shape, comp.n_mcmc)
    else:
        raise NotImplementedError(comp.phase)


def storage_nd_check_ElementAnnualKey(comp:Computation, var_key:VarKey, val:jnp.ndarray) -> bool:
    metadata_shape = comp.model.mv.general_nd[var_key].definition_metadata.value_type.shape
    general_shape = tuple(
            comp.n_years if dim_ii == years_dim else dim_ii
            for dim_ii in metadata_shape)
    if comp.phase == Phase.Prior:
        assert val.shape == general_shape, (
                val.shape, comp.n_years, general_shape)
        return False
    elif comp.phase == Phase.Posterior:
        if val.shape == general_shape:
            return True
        elif (val.shape == general_shape + (1,)
              or val.shape == general_shape + (comp.n_mcmc,)):
            return False
        else:
            raise NotImplementedError(val.shape)
    else:
        raise NotImplementedError(comp.phase)


def storage_nd_check_InitialCarry(comp:Computation, var_key:InitialCarry, val:jnp.ndarray) -> bool:
    annual_shape = tuple(
            comp.model.mv.carry_nd[var_key.carry_key].definition_metadata.value_type.shape)
    if comp.phase == Phase.Prior:
        assert val.shape == annual_shape
        return False
    elif comp.phase == Phase.Posterior:
        if val.shape == annual_shape:
            return True
        elif (val.shape == annual_shape + (1,)
              or val.shape == annual_shape + (comp.n_mcmc,)):
            return False
        else:
            # it may be that the shape should be right-padded for broadcasting
            # over mcmc dimension
            # or it may be that the shape needs to be explicitly unrolled over the
            # the mcmc dimension because of the recursive carry definition.
            raise NotImplementedError(var_key, val.shape)
    else:
        raise NotImplementedError(comp.phase)


def storage_nd_check_NextCarry(comp:Computation, var_key:NextCarry, val:jnp.ndarray) -> bool:
    annual_shape = tuple(
            comp.model.mv.carry_nd[var_key.carry_key].definition_metadata.value_type.shape)
    if comp.phase == Phase.Prior:
        raise NotImplementedError()
    elif comp.phase == Phase.Posterior:
        # a next_carry prior distribution will lead to this assignment
        # when loading saved MCMC samples
        if val.shape == (comp.model.n_prior_years,) + annual_shape + (comp.n_mcmc,):
            return False
        else:
            raise NotImplementedError(val.shape)
    else:
        raise NotImplementedError(comp.phase)


def storage_nd_check_observation_shape(comp, prior_var_key, obs_shape) -> bool:
    """Is it okay for an obs, obs_valid, or obs_weight, obs_dist to have obs_shape?
    as it relates to prior_var_key?
    """
    if isinstance(prior_var_key, ElementGeneralKey):
        # the observation can have fewer, equal, or more dimensions than the
        # observed variable. If there are more, they represent multiple observations
        # If there are fewer, they represent broadcast observations.
        metadata_prior_shape = comp.model.mv.general_nd[prior_var_key].definition_metadata.value_type.shape

        if comp.phase == Phase.Prior:
            n_obs_dim = len(obs_shape)
            # make a version of the obs_shape that matches the length of prior_shape
            if n_obs_dim < len(metadata_prior_shape):
                padded_obs_shape = (1,) * (len(metadata_prior_shape) - n_obs_dim) + obs_shape
            elif n_obs_dim == len(metadata_prior_shape):
                padded_obs_shape = obs_shape
            else:
                if len(metadata_prior_shape) == 0:
                    padded_obs_shape = ()
                else:
                    padded_obs_shape = obs_shape[-len(metadata_prior_shape):]

            assert len(padded_obs_shape) == len(metadata_prior_shape), (
                    padded_obs_shape, metadata_prior_shape)

            # ensure that it is broadcastable
            for obs_shape_ii, prior_shape_ii in zip(padded_obs_shape, metadata_prior_shape):
                if isinstance(prior_shape_ii, int):
                    assert obs_shape_ii in (1, prior_shape_ii), (obs_shape_ii, prior_shape_ii)
            return False

        elif comp.phase == Phase.Posterior:
            # it is only okay to pass the same shape as in the prior
            # or the same shape with a (1,) tagged on the end of the shape
            return False
        else:
            raise NotImplementedError(comp.phase)
        # The observation should not have mcmc dimension in Posterior phase
        # it should be automatically right-padded by the storage_nd_set

    elif isinstance(prior_var_key, ElementAnnualKey):
        # the observation can have fewer, equal, or more dimensions than the
        # observed variable. If there are more, they represent multiple observations
        # If there are fewer, they represent broadcast observations.
        metadata_annual_shape = comp.model.mv.annual_nd[prior_var_key].definition_metadata.value_type.shape
        if comp.phase == Phase.Prior:
            # why isn't this followed yet?
            # it is because an annual variable had no data observations
            # but MCMC samples are installed as observations by set_phase_posterior
            raise NotImplementedError()

        elif comp.phase == Phase.Posterior:
            n_obs_dim = len(obs_shape)
            if n_obs_dim == 1 + len(metadata_annual_shape) + 1:
                assert obs_shape[0] == comp.model.n_posterior_years, (
                        obs_shape, comp.model.n_posterior_years)

                obs_annual_shape = obs_shape[1:-1]
                # too strict, but works for now
                # TODO: accept broadcasting
                assert obs_annual_shape == tuple(metadata_annual_shape), (
                        obs_annual_shape, metadata_annual_shape)
            else:
                raise NotImplementedError(prior_var_key, metadata_annual_shape, obs_shape)
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError(prior_var_key)


def storage_nd_check_Observation(comp:Computation, obs_key:Observation, val:jnp.ndarray) -> bool:
    prior_var_key = obs_key.prior_var_key
    # the order of things is for the model element define the distribution lastly
    # and for the computation system to then set the sampled value
    assert prior_var_key not in comp._ndarray_d, prior_var_key
    assert prior_var_key not in comp._dist_d, prior_var_key

    pad_right = storage_nd_check_observation_shape(comp, prior_var_key, val.shape)
    return pad_right


def storage_nd_check_ObservationValid(comp:Computation, var_key:ObservationValid, val:jnp.ndarray) -> bool:
    assert val.dtype == jnp.bool
    # the order of things is for the model element to define the
    # valid mask first, before setting obs_val or obs_dist,
    # in order, at least possibly, to validate the observed data
    # at the time of setting the obs_val or obs_dist
    obs_var_key = var_key.obs_var_key
    assert obs_var_key not in comp._ndarray_d, var_key
    assert obs_var_key not in comp._dist_d, var_key
    pad_right = storage_nd_check_observation_shape(comp, obs_var_key.prior_var_key, val.shape)
    return pad_right


def storage_nd_check_ObservationWeight(comp:Computation, var_key:ObservationValid, val:jnp.ndarray) -> bool:
    # the order of things is for the model element to define the
    # valid mask first, before setting obs_val or obs_dist,
    # in order, at least possibly, to validate the observed data
    # at the time of setting the obs_val or obs_dist
    obs_var_key = var_key.obs_var_key
    assert obs_var_key not in comp._ndarray_d, var_key
    assert obs_var_key not in comp._dist_d, var_key
    if isinstance(obs_var_key.prior_var_key, (ElementGeneralKey, ElementCarryKey)):
        assert val.shape == () or val.shape == (1,)
    elif isinstance(obs_var_key.prior_var_key, ElementAnnualKey):
        assert val.shape == () or val.ndim == 1 or (val.ndim == 2 and val.shape[1] == 1), val.shape
    else:
        raise NotImplementedError()
    return False


def set_dist_ElementGeneralKey_posterior_obs_valid(
        comp:Computation,
        var_key:ElementGeneralKey,
        dist:Distribution,
        obs_val:jnp.ndarray,
        obs_valid:jnp.ndarray,
        ):
    dist_shape = dist.shape()
    prior_shape = tuple(build_general_prior_shape(comp, var_key, dist_shape))
    if dist_shape == prior_shape:
        mcmc_left = numpyro.sample(
                comp.sample_site(var_key),
                fn=dist.expand_by((comp.n_mcmc,)),
                rng_key=comp._split_rng_key())
        assert isinstance(mcmc_left, jnp.ndarray)
        mcmc_right = jnp.moveaxis(mcmc_left, 0, -1)
    elif dist_shape == prior_shape + (comp.n_mcmc,):
        mcmc_right = numpyro.sample(
                comp.sample_site(var_key),
                fn=dist,
                rng_key=comp._split_rng_key())
        assert isinstance(mcmc_right, jnp.ndarray)
    elif dist_shape == prior_shape + (1,):
        # I'm not sure, but I *think* that because of the semantics
        # of the mcmc dim, it does not make sense to broadcast
        # a sample over that axis, even if all of the parameters
        # are set to broadcast in the mcmc dimension.
        left_extradim = numpyro.sample(
                comp.sample_site(var_key),
                fn=dist.expand_by((comp.n_mcmc,)),
                rng_key=comp._split_rng_key())
        assert isinstance(left_extradim, jnp.ndarray)
        right_extradim = jnp.moveaxis(left_extradim, 0, -1)
        mcmc_right = right_extradim[..., 0, :]
    else:
        raise NotImplementedError(dist_shape, prior_shape, var_key)

    assert obs_valid.ndim == obs_val.ndim, (
            obs_valid.shape, obs_val.shape)
    assert obs_val.ndim >= mcmc_right.ndim
    updated_sample = jnp.where(obs_valid, obs_val, mcmc_right)
    comp._ndarray_d[var_key] = updated_sample


def set_dist_ElementAnnualKey_posterior_obs_valid(
        comp:Computation,
        var_key:ElementAnnualKey,
        dist:Distribution,
        obs_val:jnp.ndarray,
        obs_valid:jnp.ndarray,
        ):
    dist_shape = dist.shape()
    posterior_shape = tuple(build_annual_posterior_shape(comp, var_key, dist_shape))
    if dist_shape == posterior_shape[:-1]:
        # this is convenience to the user - they ignore the MCMC
        # dimension so they can use the same form of parameterization
        # in both Prior and Posterior phases
        mcmc_left = numpyro.sample(
                comp.sample_site(var_key),
                fn=dist.expand_by((comp.n_mcmc,)),
                rng_key=comp._split_rng_key())
        assert isinstance(mcmc_left, jnp.ndarray)
        mcmc_right = jnp.moveaxis(mcmc_left, 0, -1)
        assert obs_valid.ndim == obs_val.ndim, (
                obs_valid.shape, obs_val.shape)
        assert obs_val.ndim >= mcmc_right.ndim
        if obs_val.ndim > mcmc_right.ndim:
            # see long comment below
            raise NotImplementedError(obs_val.shape, mcmc_right.shape)
        updated_sample = jnp.where(obs_valid, obs_val, mcmc_right)
        comp._ndarray_d[var_key] = updated_sample

    elif dist_shape == posterior_shape:
        mcmc_right = numpyro.sample(
                comp.sample_site(var_key),
                fn=dist,
                rng_key=comp._split_rng_key())
        assert isinstance(mcmc_right, jnp.ndarray)
        assert obs_valid.ndim == obs_val.ndim, (
                obs_valid.shape, obs_val.shape)
        assert obs_val.ndim >= mcmc_right.ndim
        if obs_val.ndim > mcmc_right.ndim:
            # This needs thought.
            # Is it okay for a random variable element of a dist
            # to broadcast over observations?
            # If yes, then what is the value in the posterior of the observed variable?
            # Maybe a "posterior_valid" mask is required, in order to indicate
            # the elements of a sample that are not observed. It might be
            # easy to forget to use the mask unless the posterior were only
            # accessible as e.g. ws.masked_dist.annual[foo]
            # and ws.dist_valid.annual[foo]
            #
            # If no, then does the number of observations of each variable
            # need to be entered into the shape metadata?
            # It doesn't seem practical or necessary.
            #
            # Another possibility is to populate the MCMC dimension of observed
            # values by uniformly sampling from among
            # observed historical trajectories. This approach is practical
            # and defensible (at least generalizes the one-observation case),
            # if not unassailably obviously uniquely correct.
            #
            # Whatever the solution, it should work coherently for both numerical
            # observations, as well as distributional observations (`obs_dist`).
            raise NotImplementedError(obs_val.shape, mcmc_right.shape)
        updated_sample = jnp.where(obs_valid, obs_val, mcmc_right)
        comp._ndarray_d[var_key] = updated_sample
    else:
        raise NotImplementedError(dist_shape, posterior_shape, var_key)


def set_dist_ElementGeneralKey(
        comp:Computation,
        var_key:ElementGeneralKey|ElementAnnualKey,
        dist:Distribution):
    obs_var_key = Observation(prior_var_key=var_key)
    obsvalid_var_key = ObservationValid(obs_var_key=obs_var_key)
    obsweight_var_key = ObservationWeight(obs_var_key=obs_var_key)

    obs = comp._ndarray_d.get(obs_var_key)
    obs_valid = comp._ndarray_d.get(obsvalid_var_key)
    obs_dist = comp._dist_d.get(obs_var_key)
    obs_weight = comp._ndarray_d.get(obsweight_var_key)

    have_obs_val = obs is not None
    have_obs_valid = obs_valid is not None
    have_obs_dist = obs_dist is not None
    if have_obs_dist:
        assert obs_weight is not None
    if have_obs_val and obs_weight is not None:
        raise NotImplementedError()

    # TODO: check that the shape is correct
    # shape correctness does not depend on obs
    if comp.phase == Phase.Prior:

        if not (have_obs_val or have_obs_dist):

            assert not have_obs_valid
            comp._dist_d[var_key] = dist

        elif have_obs_val and not have_obs_valid:
            comp._ndarray_d[var_key] = obs
            numpyro.sample(
                    comp.sample_site(var_key),
                    fn=dist,
                    rng_key=comp._split_rng_key(),
                    obs=obs,
                    )

        elif have_obs_val and have_obs_valid:
            # think carefully before storing
            # either the distribution in _dist_d
            # *or* drawn / observed values in _ndarray_d
            #
            # don't define the distribution if the observation
            # is provided, because it no longer makes sense
            # mask the distribution
            # mask the observation in the sample call?
            # store dist jnp.where(obs_valid, obs, an_actual_sample)
            #
            # Note that according to ...
            # https://num.pyro.ai/en/stable/primitives.html#sample
            # ... obs_mask parameter should *not* be used with MCMC
            # ... so think about what this model needs in terms of
            # .... symantics.
            numpyro.sample(
                    comp.sample_site(var_key),
                    fn=dist.mask(obs_valid),
                    rng_key=comp._split_rng_key(),
                    obs=obs,
                    )

        elif have_obs_dist and have_obs_valid:
            # ... some type of masking should be done?
            raise NotImplementedError()

        elif have_obs_dist and not have_obs_valid:
            neg_expected_log_prob_plus_const = (
                   kl.kl_divergence(obs_dist, dist) * obs_weight)
            numpyro.factor(
                    comp.sample_site(var_key),
                    -neg_expected_log_prob_plus_const.sum())
            comp._dist_d[var_key] = dist  # dist is prior

        else:
            raise NotImplementedError(
                    'observation of both values and distribution')

    elif comp.phase == Phase.Posterior:
        if not (have_obs_val or have_obs_dist):
            # don't define the distribution this time
            # because the provided distribution was the prior.

            # TODO: verify that what's in _ndarray_d was actually
            #       put there by the loading of posterior samples.
            #
            #       If instead it is e.g. from a previous assignment to this key
            #       then that previous assignment should be replaced!
            if var_key not in comp._ndarray_d:
                # this can happen for items that are not sampled
                # in the prior computation such as the NIR2025
                # reference distributions
                # TODO: check that the shape is correct
                comp._dist_d[var_key] = dist
            else:
                assert var_key in comp._ndarray_d
            # TODO: check that the shape is correct
            #       the shape should be [n_mcmc] + [var shape]

        elif have_obs_val and not have_obs_valid:
            # TODO: check that the shape is correct
            #       obs' shape should be [var shape]
            #       with no mcmc samples.

            # technically this might be okay in some scenarios
            # but if it happens for now, it's an error.
            assert var_key not in comp._ndarray_d
            comp._ndarray_d[var_key] = obs

        elif have_obs_val and have_obs_valid:
            # draw a sample, but then mix it with the observations
            # according to obs_valid. (Subroutine because there is
            # some special-casing based on shapes.)
            if isinstance(var_key, ElementGeneralKey):
                set_dist_ElementGeneralKey_posterior_obs_valid(
                        comp, var_key, dist,
                        obs_val=obs,
                        obs_valid=obs_valid)
            elif isinstance(var_key, ElementAnnualKey):
                set_dist_ElementAnnualKey_posterior_obs_valid(
                        comp, var_key, dist,
                        obs_val=obs,
                        obs_valid=obs_valid)
            else:
                # could be fine, haven't checked
                raise NotImplementedError()

        elif have_obs_dist and have_obs_valid:
            raise NotImplementedError()

        elif have_obs_dist and not have_obs_valid:
            # simply use the obs_dist as the posterior
            comp._dist_d[var_key] = obs_dist

        else:
            raise NotImplementedError(
                    'observation of both values and distribution')
    else:
        raise NotImplementedError(comp.phase)


def set_dist_ElementAnnualKey(comp:Computation, var_key:ElementAnnualKey, dist:Distribution):
    return set_dist_ElementGeneralKey(comp, var_key, dist)


def set_dist_Observation(
        comp:Computation,
        obs_key:Observation,
        dist:Distribution,
        ) ->None:
    prior_var_key = obs_key.prior_var_key
    # the order of things is
    # 1. set observation values / distribution
    # 2. set the prior on the variable
    #
    # In step 2, this runtime will ensure samples from prior or posterior
    #    are assigned to comp._ndarray_d[prior_var_key]
    assert prior_var_key not in comp._ndarray_d, prior_var_key
    assert prior_var_key not in comp._dist_d, prior_var_key

    if obs_key in comp._ndarray_d:
        raise NotImplementedError(
                'setting both distribution and values for observation',
                obs_key)

    pad_right = storage_nd_check_observation_shape(comp, prior_var_key, dist.shape())
    if pad_right:
        # I suspect there is some pytree magic to do this for any distribution
        # but I don't know what it is. I think all of the parameters of the dist
        # must be right-padded.
        raise NotImplementedError()
    comp._dist_d[obs_key] = dist


def set_dist_InitialCarry(comp:Computation, ic_key:InitialCarry, dist:Distribution):
    obs_var_key = Observation(prior_var_key=ic_key)
    obsvalid_var_key = ObservationValid(obs_var_key=obs_var_key)

    obs = comp._ndarray_d.get(obs_var_key)
    obs_valid = comp._ndarray_d.get(obsvalid_var_key)
    obs_dist = comp._dist_d.get(obs_var_key)
    if obs_dist:
        # set up a call to numpyro.factor
        raise NotImplementedError()

    if comp.phase == Phase.Prior:
        # TODO: check that the shape is correct
        # shape correctness does not depend on obs

        if obs is None and obs_valid is None:
            comp._dist_d[ic_key] = dist

        elif obs_valid is None:
            # don't define the distribution if the observation
            # is provided, because it no longer makes sense
            #
            # but do store the observation as the sample
            #
            # TODO: check that obs' shape is correct
            assert obs is not None
            comp._ndarray_d[ic_key] = obs
            numpyro.sample(
                    comp.sample_site(ic_key),
                    fn=dist,
                    rng_key=comp._split_rng_key(),
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
        assert comp.phase == Phase.Posterior
        if obs is None and obs_valid is None:
            # don't define the distribution this time
            # because the provided distribution was the prior.

            # TODO: verify that what's in _ndarray_d was actually
            #       put there by the loading of posterior samples.
            #
            #       If instead it is e.g. from a previous assignment to this key
            #       then that previous assignment should be replaced!
            if ic_key not in comp._ndarray_d:
                # this can happen for items that are not sampled
                # in the prior computation such as the NIR2025
                # reference distributions
                # TODO: check that the shape is correct
                comp._dist_d[ic_key] = dist
            else:
                # There are samples loaded from mcmc in _ndarray_d
                pass
                # TODO: check that the shape is correct

        elif obs_valid is None:
            assert obs is not None
            # TODO: check that the shape is correct
            #       obs' shape should be [var shape]
            #       with no mcmc samples.

            # technically this might be okay in some scenarios
            # but if it happens for now, it's an error.
            assert ic_key not in comp._ndarray_d

            comp._ndarray_d[ic_key] = obs
        else:
            assert obs is not None
            raise NotImplementedError()


class ComputationDistD:
    """Dict-like access by key for all ndarrays in a Computation
    """
    comp: Computation

    def __init__(self, comp:Computation):
        self.comp = comp

    def __contains__(self, item:VarKey):
        try:
            self.comp.get_dist(item)
            return True
        except KeyError:
            return False

    def __getitem__(self, item:VarKey) -> Distribution:
        return self.comp.get_dist(item)

    def __setitem__(self, item:VarKey, value:Distribution):
        self.comp.set_dist(item, value)

    def items(self):
        return self.comp._dist_d.items()


class ComputationNdarrayD:
    """Dict-like access by key for all ndarrays in a Computation
    """
    comp: Computation

    def __init__(self, comp:Computation):
        self.comp = comp

    def __contains__(self, item:VarKey):
        try:
            self.comp.get_ndarray(item, sample_if_necessary=True)
            return True
        except KeyError:
            return False

    def __getitem__(self, item:VarKey) -> jnp.ndarray:
        return self.comp.get_ndarray(item, sample_if_necessary=True)

    def __setitem__(self, item:VarKey, value:jnp.ndarray):
        self.comp.set_ndarray(item, value)

    def items_no_sample_if_necessary(self):
        return self.comp._ndarray_d.items()


class Computation:
    """
    Things that are discovered by running a model, either as Prior or Posterior.

    All elements in storage_nd have a trailing n_mcmc dimension during Posterior
    computations.
    """

    model: ScenarioModel
    _dist_d: dict[VarKey, Distribution]
    _ndarray_d: dict[VarKey, jnp.ndarray]
    rng_key: ArrayLike
    phase: Phase
    has_run: bool
    sample_sites: dict[VarKey, str]
    ndarray_dim_d: dict[NdarrayDim, int]
    _resolved_accesses: list[ResolvedAccessMetadata]

    n_mcmc: int

    def __init__(
            self,
            model:ScenarioModel,
            rng_key:ArrayLike|int):

        self.model = model
        self._dist_d = {}
        self._ndarray_d = {}
        self._assigned_keys:set[VarKey] = set()
        if isinstance(rng_key, int):
            self.rng_key = jrandom.key(rng_key)
        else:
            self.rng_key = rng_key
        self.phase = Phase.Prior
        self.has_run = False
        self.sample_sites = {}
        self.ndarray_dim_d = {
                years_dim: model.n_prior_years,
                mcmc_dim: model.num_samples,
                }
        self._defined_var_keys = (
                set(model.mv.general_nd)
                | set(model.mv.carry_nd)
                | set(model.mv.annual_nd))
        self._resolved_accesses = model.resolve_accesses()
        self._access_default_d = {
                (access.elem_id, access.subphase, access.var_key): access.default
                for access in self._resolved_accesses
                if access.default is not None
                }

    def is_var_defined(self, var_key:VarKey) -> bool:
        return var_key in self._defined_var_keys

    def access_default(
            self,
            elem_id:str,
            subphase:Subphase,
            var_key:VarKey,
            ) -> ArrayLike|None:
        """The default declared for this read, if no element defines var_key.

        Defaults only apply when the key is undefined for the whole model.
        If the key has a definer, a missing value is a bug in the model or in
        the dependency ordering and must still raise.
        """
        if self.is_var_defined(var_key):
            return None
        return self._access_default_d.get((elem_id, subphase, var_key))

    @property
    def dist_d(self) -> ComputationDistD:
        return ComputationDistD(self)

    @property
    def ndarray_d(self) -> ComputationNdarrayD:
        return ComputationNdarrayD(self)

    def get_dist(self, item:VarKey) -> Distribution:
        return self._dist_d[item]

    def set_dist(self, var_key:VarKey, dist:Distribution):
        if isinstance(var_key, ElementGeneralKey):
            set_dist_ElementGeneralKey(self, var_key, dist)
        elif isinstance(var_key, ElementAnnualKey):
            set_dist_ElementAnnualKey(self, var_key, dist)
        elif isinstance(var_key, InitialCarry):
            set_dist_InitialCarry(self, var_key, dist)
        elif isinstance(var_key, Observation):
            set_dist_Observation(self, var_key, dist)
        else:
            raise NotImplementedError(var_key)

    def get_ndarray(
            self,
            var_key:VarKey,
            sample_if_necessary,
            default:jnp.ndarray|None=None,
            ) -> jnp.ndarray:
        try:
            return self._ndarray_d[var_key]
        except KeyError as err:
            if (var_key not in self._dist_d or not sample_if_necessary):
                if default is not None:
                    return default
                err.add_note(f'var_key in dist_d = {var_key in self._dist_d}')
                err.add_note(f'sample_if_necessary = {sample_if_necessary}')
                err.add_note(f'phase = {self.phase}')
                raise
        self.set_ndarray_from_dist(var_key)
        return self._ndarray_d[var_key]

    def set_ndarray(self, item:VarKey, value:jnp.ndarray):
        if isinstance(item, ElementGeneralKey):
            pad_right = storage_nd_check_ElementGeneralKey(self, item, value)

        elif isinstance(item, ElementAnnualKey):
            pad_right = storage_nd_check_ElementAnnualKey(self, item, value)

        elif isinstance(item, InitialCarry):
            pad_right = storage_nd_check_InitialCarry(self, item, value)

        elif isinstance(item, NextCarry):
            pad_right = storage_nd_check_NextCarry(self, item, value)

        elif isinstance(item, Observation):
            pad_right = storage_nd_check_Observation(self, item, value)

        elif isinstance(item, ObservationValid):
            pad_right = storage_nd_check_ObservationValid(self, item, value)

        elif isinstance(item, ObservationWeight):
            pad_right = storage_nd_check_ObservationWeight(self, item, value)

        elif isinstance(item, GroupedPosterior):
            pad_right = False

        else:
            raise NotImplementedError(item)

        if pad_right:
            padded = value[..., None]
        else:
            padded = value
        self._ndarray_d[item] = padded

    def set_ndarray_from_dist(self, var_key:VarKey):
        assert var_key in self._dist_d
        if var_key in self._ndarray_d:
            # if it's already set, this method can't promise it's
            # set to the right distribution
            raise NotImplementedError()

        # accessing an undefined value of a general key after the dist
        # has been set for that key, triggers sampling.

        obs_var_key = Observation(prior_var_key=var_key)
        obs_val = self._ndarray_d.get(obs_var_key)
        obs_dist = self._dist_d.get(obs_var_key)

        if obs_val or obs_dist:
            raise NotImplementedError()

        if self.phase == Phase.Prior:
            dist = self._dist_d[var_key]
            left_mcmc = False

        elif self.phase == Phase.Posterior:
            # A sample is being accesssed in the posterior that
            # was not accessed in the prior. This happens for
            # e.g. the NIR2025 reference distribution.
            dist = self._dist_d[var_key]
            left_mcmc = False
        else:
            raise NotImplementedError(self.phase)

        dist_sample = numpyro.sample(
                self.sample_site(var_key),
                fn=dist,
                rng_key=self._split_rng_key(),
                )
        if isinstance(dist_sample, jnp.ndarray):
            if left_mcmc:
                transposed_sample = jnp.moveaxis(dist_sample, 0, -1)
            else:
                transposed_sample = dist_sample
            self.set_ndarray(var_key, transposed_sample)
        else:
            raise NotImplementedError()


    def _workspace(self,
                   elem_id:str,
                   subphase:Subphase,
                   scan_storage:ScanStorage|None=None
                   ) -> Workspace:
        assert (subphase == Subphase.Step) == (scan_storage is not None)
        return Workspace(
                comp=self,
                year_0=self.model.mv.year_0,
                n_years=self.n_years,
                phase=self.phase,
                elem_id=elem_id,
                subphase=subphase,
                scan_storage=scan_storage,
                )
    @property
    def n_years(self) -> int:
        return (self.model.mv.n_prior_years
                if self.phase == Phase.Prior
                else self.model.mv.n_posterior_years)

    def sample_site(self, item:VarKey) -> str:
        self.sample_sites[item] = str(item)
        return self.sample_sites[item]

    def _load_general_grouped_sample(self, var_key, grouped_sample):
        self.set_ndarray(GroupedPosterior(prior_var_key=var_key), grouped_sample)
        (n_groups, n_saved_samples, *prior_shape) = grouped_sample.shape

        # reshape to combine groups and saved samples as if single chain mcmc
        reshaped = grouped_sample.reshape([n_groups * n_saved_samples] + prior_shape)
        # transpose mcmc dimension to the end
        transposed = jnp.moveaxis(reshaped, 0, -1)

        self.set_ndarray(var_key, transposed)

    def _load_nextcarry_grouped_sample(self, var_key, grouped_sample):
        self.set_ndarray(GroupedPosterior(prior_var_key=var_key), grouped_sample)
        (n_groups, n_saved_samples, n_prior_scan_steps, *annual_shape) \
                = grouped_sample.shape
        assert n_prior_scan_steps == self.model.mv.n_prior_years
        # reshape to combine groups and saved samples as if single chain mcmc
        reshaped = grouped_sample.reshape(
                [n_groups * n_saved_samples, n_prior_scan_steps]
                + annual_shape)
        # transpose mcmc dimension to the end
        transposed = jnp.moveaxis(reshaped, 0, -1)

        # post-conditions:
        assert transposed.shape[0] == n_prior_scan_steps
        assert transposed.shape[-1] == self.n_mcmc
        assert transposed.shape[1:-1] == tuple(annual_shape)

        self.set_ndarray(var_key, transposed)

        # and a separate pull_from_sample_mask array of shape [n_years]
        # and these both need to be set up as annual inputs for the scan
        # and then the ws.dist.next_carry __setitem__ needs to
        # put either the this_annual input from the sample or the random draw
        # into the scan_storage.next_carry_d, depending on the
        # pull_from_sample_mask.
        # ... and that's all assuming that these dists are not observed,
        # .... which I'm sure they will be sometimes! I think that
        # complicates the logic, but doesn't break the approach.

    def _load_annual_grouped_sample(self, var_key:ElementAnnualKey, grouped_sample):
        self.set_ndarray(GroupedPosterior(prior_var_key=var_key), grouped_sample)

        (n_groups, n_saved_samples, n_prior_years, *annual_shape) \
                = grouped_sample.shape
        assert n_prior_years == self.model.mv.n_prior_years
        # reshape to combine groups and saved samples as if single chain mcmc
        reshaped = grouped_sample.reshape(
                [n_groups * n_saved_samples, n_prior_years]
                + annual_shape)
        # transpose mcmc dimension to the end
        transposed = jnp.moveaxis(reshaped, 0, -1)

        # post-conditions:
        assert transposed.shape[0] == n_prior_years
        assert transposed.shape[-1] == self.n_mcmc
        assert transposed.shape[1:-1] == tuple(annual_shape)

        nvm = self.model.mv.annual_nd[var_key]

        # an annual value can be defined in essentially two ways:
        # 1. all at once in either of model_element_prepare or model_element_postprocess
        # 2. stepwise in a model_element_annual_step
        #
        if nvm.defining_subphase in (Subphase.Prep, Subphase.Proc):

            # If this annual value has been defined all at once, then `transposed`
            # should be used as if the following appeared prior to its definition
            # ws.obs_valid.annual[var_key] = year_idxs < n_prior_years
            # ws.obs_val.annual[var_key] = jnp.where( year_idxs < n_prior_years, transposed, 0)

            # don't lookup scan_step_ii_key because it isn't defined yet
            scan_step_ii = jnp.arange(self.model.n_posterior_years)
            expanded_mask = jnp.reshape(
                    scan_step_ii < self.model.n_prior_years,
                    (self.model.n_posterior_years,) + (1,) * (transposed.ndim - 1))
            self.set_ndarray(
                    ObservationValid(
                        obs_var_key=Observation(
                            prior_var_key=var_key)),
                    expanded_mask)
            padding_years = self.model.n_posterior_years - self.model.n_prior_years
            assert padding_years >= 0
            padding = jnp.zeros((padding_years,) + transposed.shape[1:])
            self.set_ndarray(
                    Observation(prior_var_key=var_key),
                    jnp.concatenate([transposed, padding]))
        elif nvm.defining_subphase == Subphase.Step:
            # If this annual value has been defined stepwise, then the
            # technique used in _load_nextcarry_grouped_sample can be adapted.
            raise NotImplementedError()
        else:
            raise NotImplementedError(nvm.defining_subphase)

    def set_phase_posterior(
            self,
            n_mcmc,
            sample_sites:dict[VarKey, str],
            grouped_samples:dict[str, jnp.ndarray]):
        self.phase = Phase.Posterior
        self.n_mcmc = n_mcmc
        self.sample_sites = sample_sites
        rlookup = {
                val: key
                for key, val in self.sample_sites.items()}
        assert len(rlookup) == len(self.sample_sites)
        self.ndarray_dim_d[years_dim] = self.model.n_posterior_years

        for key_str, grouped_sample in grouped_samples.items():
            var_key = rlookup[key_str]
            assert len(grouped_sample.shape) >= 2
            (n_groups, n_saved_samples) = grouped_sample.shape[:2]
            assert n_groups == 1
            assert n_saved_samples >= self.n_mcmc
            if n_saved_samples >  self.n_mcmc:
                grouped_sample = grouped_sample[:, :self.n_mcmc]

            #
            # Posterior calculations are carried out with trailing mcmc dimension.
            #
            # storage_nd requires every variable to be ndarray so that every variable
            # can have a trailing mcmc dimension.
            #
            # MCMC dimension is trailing so that implicit broadcasting of leading
            # dimensions and the most idiomatic
            # indexing e.g. ws.val.general[foo][ii] still work as they should.
            #
            # This mcmc dimension is not meant to be hidden from user. They can
            # see it when they access `shape` attribute, and they can write
            # code that branches on ws.phase.
            #

            if isinstance(var_key, (ElementGeneralKey, InitialCarry)):
                self._load_general_grouped_sample(var_key, grouped_sample)
            elif isinstance(var_key, NextCarry):
                self._load_nextcarry_grouped_sample(var_key, grouped_sample)
            elif isinstance(var_key, ElementAnnualKey):
                self._load_annual_grouped_sample(var_key, grouped_sample)
            else:
                raise NotImplementedError(var_key, grouped_sample.shape)

    def _split_rng_key(self) -> ArrayLike:
        self.rng_key, key = jrandom.split(self.rng_key)
        return key

    def _initial_carry_d(self):
        rval = {}
        for var_key, val in self._ndarray_d.items():
            if isinstance(var_key, InitialCarry):
                carry_key = var_key.carry_key
                if self.phase == Phase.Prior:
                    rval[carry_key] = val
                elif self.phase == Phase.Posterior:
                    if val.shape[-1] == self.n_mcmc:
                        rval[carry_key] = val
                    elif val.shape[-1] == 1:
                        shape = val.shape[:-1] + (self.n_mcmc,)
                        rval[carry_key] = jnp.full(
                                shape, val, dtype=val.dtype)
                    else:
                        # how did this get into _ndarray_d?
                        raise NotImplementedError(val.shape)
                else:
                    raise NotImplementedError(self.phase)
        return rval

    def _sample_from_dists_that_might_be_accessed(self):
        # I'm not sure what heuristic / policy to use here.
        # First try: all var_keys in general_nd whose first shape dim
        # is the inference_years_dim.
        for key_var in self.model.mv.carry_nd:
            if isinstance(key_var, ElementCarryKey):
                ic_key = InitialCarry(carry_key=key_var)
            else:
                raise NotImplementedError()
            if ic_key in self._ndarray_d:
                pass
            elif ic_key in self._dist_d:
                self.set_ndarray_from_dist(ic_key)
            else:
                nvm = self.model.mv.general_nd[ic_key]
                if (self.phase == Phase.Prior
                        and nvm.definition_metadata.posterior_only):
                    # no initial carry is required during Phase.Prior
                    pass
                else:
                    raise NotImplementedError(key_var)

        for access in self._resolved_accesses:
            if (access.subphase == Subphase.Step
                and access.var_key in self._dist_d # it's been assigned a prior distribution
                and access.var_key not in self._ndarray_d): # hasn't beens sampled yet
                self.set_ndarray_from_dist(access.var_key)

    def _annual_input_d(self) -> dict[VarKey, jnp.ndarray]:
        """Annual variables defined prior to the scan: one value per
        variable per year, to be sliced and fed into the scan one step
        at a time.
        """
        rval = {
                var_key: self._ndarray_d[var_key]
                for var_key, nvm in self.model.mv.annual_nd.items()
                if nvm.defining_subphase == Subphase.Prep
                }
        return rval

    def _elem_items(self, subphase:Subphase):
        for elem_id in self.model.element_order(subphase):
            yield elem_id, self.model.model_elements[elem_id]

    def _reset_assignments(self) -> None:
        self._assigned_keys.clear()

    def _record_assignment(self, var_key:VarKey) -> None:
        self._assigned_keys.add(var_key)

    def _expected_defined_keys(
            self,
            element:ModelElement,
            subphase:Subphase,
            ) -> list[tuple[VarKey, NdarrayVariableMetadata]]:
        """Definitions this element is expected to write in this subphase.

        Keys are resolved once by add_element, so using the model metadata
        automatically handles string, property, and dict-property declarations.
        """
        elem_id = element.identifier
        if subphase == Subphase.Prep:
            return [
                    (var_key, nvm)
                    for var_key, nvm in self.model.mv.general_nd.items()
                    if nvm.defining_element_id == elem_id
                    and nvm.defining_subphase == Subphase.Prep]
        elif subphase == Subphase.Step:
            return (
                    [(var_key, nvm)
                     for var_key, nvm in self.model.mv.annual_nd.items()
                     if nvm.defining_element_id == elem_id
                     and nvm.defining_subphase == Subphase.Step]
                    + [(var_key, nvm)
                       for var_key, nvm in self.model.mv.carry_nd.items()
                       if nvm.defining_element_id == elem_id])
        elif subphase == Subphase.Proc:
            # FinalCarry has no writable workspace accessor yet.
            return [
                    (var_key, nvm)
                    for var_key, nvm in self.model.mv.general_nd.items()
                    if nvm.defining_element_id == elem_id
                    and nvm.defining_subphase == Subphase.Proc
                    and not isinstance(var_key, FinalCarry)]
        else:
            raise NotImplementedError(subphase)

    def _check_defined_assignments(
            self,
            element:ModelElement,
            subphase:Subphase,
            ) -> None:
        missing = []
        for var_key, nvm in self._expected_defined_keys(element, subphase):
            if (self.phase == Phase.Prior
                    and nvm.definition_metadata.posterior_only):
                continue
            if var_key not in self._assigned_keys:
                missing.append(var_key)
        if missing:
            method_name = {
                    Subphase.Prep: 'model_element_prepare',
                    Subphase.Step: 'model_element_annual_step',
                    Subphase.Proc: 'model_element_postprocess',
                    }[subphase]
            names = ', '.join(sorted(str(key) for key in missing))
            raise UnassignedDefinitionError(
                    f'{element.identifier}: {method_name} did not assign'
                    f' declared key(s): {names}')

    def _run_prep(self):
        for elem_id, elem in self._elem_items(Subphase.Prep):
            try:
                self._reset_assignments()
                elem.model_element_prepare(self._workspace(elem_id, Subphase.Prep))
                self._check_defined_assignments(elem, Subphase.Prep)
            except Exception as err:
                err.add_note(f'element_id={elem_id}')
                raise

    def _run_step(self):

        step_elem_items = list(self._elem_items(Subphase.Step))

        def scan_step(this_carry_d, this_annual_input_d):
            scan_storage = ScanStorage(this_carry_d, this_annual_input_d, self.sample_site, self.phase)
            scan_storage.carry_rng()
            for elem_id, elem in step_elem_items:
                try:
                    ws = self._workspace(elem_id, Subphase.Step, scan_storage)
                    self._reset_assignments()
                    elem.model_element_annual_step(ws)
                    self._check_defined_assignments(elem, Subphase.Step)
                except Exception as err:
                    err.add_note(f'element_id={elem_id}')
                    raise
            return scan_storage.next_carry_d, scan_storage.this_annual_output_d

        self._sample_from_dists_that_might_be_accessed()

        initial_carry_d = self._initial_carry_d()
        initial_carry_d[rng_var_key] = self._split_rng_key()
        annual_input_d = self._annual_input_d()

        if self.phase == Phase.Prior:
            final_carry_d, annual_d = numpyro_scan(
                    scan_step, initial_carry_d, annual_input_d)
        else:
            final_carry_d, annual_d = jax_scan(
                    scan_step, initial_carry_d, annual_input_d)

        # TODO: check for collisions
        self._ndarray_d.update(annual_d)
        # TODO: store all of the final_carry_d and final_carry_dist_d
        self.rng_key = final_carry_d[rng_var_key]

    def _run_proc(self):
        for elem_id, elem in self._elem_items(Subphase.Proc):
            try:
                ws = self._workspace(elem_id, Subphase.Proc)
                self._reset_assignments()
                elem.model_element_postprocess(ws)
                self._check_defined_assignments(elem, Subphase.Proc)
            except Exception as err:
                err.add_note(f'element_id={elem_id}')
                raise

    def run(self):
        assert not self.has_run
        self._run_prep()
        self._run_step()
        self._run_proc()
        self.has_run = True


def run_mcmc(
        model:ScenarioModel,
        seed:int|ArrayLike,
        progress_bar=True,
        ):
    def trace_fn():
        # TODO verify
        # the seed value is ignored
        # when running via MCMC
        obj = Computation(model=model, rng_key=jrandom.key(1))
        obj.run()
    if isinstance(seed, int):
        rng_key = jrandom.key(seed=seed)
    else:
        rng_key = seed
    mcmc = MCMC(
            NUTS(trace_fn,
                 target_accept_prob=0.995,
                 #init_strategy=init_to_sample,
                 ),
            num_warmup=model.num_warmup,
            thinning=model.thinning,
            num_samples=model.num_samples,
            progress_bar=progress_bar)
    mcmc.run(rng_key=rng_key)
    extra_fields = mcmc.get_extra_fields()
    if "diverging" in extra_fields:
        n_divergences = jnp.sum(extra_fields["diverging"])
        assert n_divergences <= model.n_divergences_acceptable, (
                n_divergences, model.n_divergences_acceptable)
    return mcmc


def compute_model(
        model:ScenarioModel,
        seed_or_key:int|ArrayLike,
        ) -> Computation:
    if isinstance(seed_or_key, int):
        seed_or_key = jrandom.key(seed_or_key)
    mcmc_submodels = model.mcmc_submodels()

    comp_prior = Computation(model=model, rng_key=jrandom.key(1))
    comp_prior.run()

    if len(mcmc_submodels) > 1:
        raise NotImplementedError()
    else:
        saved_sample_sites = {}
        all_grouped_samples = {}
        for mcmc_group, mcmc_model in mcmc_submodels.items():
            seed_or_key, key = jrandom.split(seed_or_key)
            mcmc = run_mcmc(model, seed=key)
            grouped_samples = mcmc.get_samples(group_by_chain=True)
            for var_key, sample_site in comp_prior.sample_sites.items():
                if sample_site in grouped_samples:
                    saved_sample_sites[var_key] = sample_site
            assert len(saved_sample_sites) == len(grouped_samples)
            all_grouped_samples.update(grouped_samples)
            print(grouped_samples)
    comp_posterior = Computation(
            model=model,
            rng_key=seed_or_key)
    comp_posterior.set_phase_posterior(
            n_mcmc=model.num_samples,
            sample_sites=saved_sample_sites,
            grouped_samples=all_grouped_samples)
    comp_posterior.run()
    return comp_posterior

from .workspace import Workspace
