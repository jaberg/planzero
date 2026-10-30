
import jax.numpy as jnp
import jinja2
import pint
from numpyro import distributions as dists

from .enums import GHG, PT, SPE, IPCC_Sector
from .ghgvalues import GWP_100
from .html import coderef_url
from .nir_model import NIR_ModelElement
from .sc_3210013001 import (
    Livestock,
    Livestock_nonsums,
)
from .scenario_model import (
    AnnualKey,
    CarryKey,
    GeneralKey,
    ModelElement,
    Phase,
    VersionID,
    Workspace,
    access_annual,
    access_this_carry,
    define,
    define_annual,
    define_carry,
    years_key,
)
from .ureg import u

# This could be modelled / loaded. It's in statscan
cattle_per_farm = 160 # look this up somewhere


class Bovaer_Adoption_Limit(ModelElement):

    def __init__(self):
        pass
    
    @property
    def max_increase_rate(self) -> pint.Quantity:
        return 5.0 * u.percent / u.year

    @property
    def prob_max_increase_rate(self) -> tuple[float, float]:
        return (
                self.max_increase_rate.magnitude / 2,
                self.max_increase_rate.magnitude * 2)

    @property
    def organic_fraction(self) -> float:
        # TODO: make this an STS, it can change over time
        # Approximately 1.5% of Canada's milk is organic, and about 0.7%
        # of beef is organic.
        return 0.01

    @property
    def short_description(self) -> str:
        return f"Assume Bovaer will only be adopted by, at most, {self.max_increase_rate} of cattle operations, up to a maximum of {(1 - self.organic_fraction) * 100:.1f}%"

    @property
    def description(self) -> str:
        low, high = self.prob_max_increase_rate
        return f"""Assume that no more than {self.max_increase_rate} of
        farmers will switch to administering Bovaer
        in any given year, but that adoption
        can ultimately rise to {(1 - self.organic_fraction) * 100:.1f}%,
        the remainder of whom are organic farmers who won't adopt it.
        </p>
        When used in probabilistic models, assume instead that
        {low}-{high}% of farmers will switch to Bovaer in any given year.
        <p>
        """

    @property
    def version_id(self) -> VersionID:
        return (1,)

    @property
    def bovine_fraction_on_bovaer(self) -> CarryKey:
        return self.carry_key('bovaer_fraction')

    @property
    def max_possible_fraction_on_bovaer(self) -> CarryKey:
        return self.carry_key('max_possible_fraction')

    @define_carry(max_possible_fraction_on_bovaer, shape=[])
    @define_annual('max_increase_fraction', annual_shape=[])
    def model_element_prepare(self, ws:Workspace):
        ws.val.initial_carry[self.max_possible_fraction_on_bovaer] = jnp.zeros(())
        ws.dist.annual['max_increase_fraction'] = dists.Uniform().expand_by(
                (ws.n_years,))

    @access_annual('max_increase_fraction')
    @access_this_carry(bovine_fraction_on_bovaer, shape=[], default=jnp.zeros(()))
    def model_element_annual_step(self, ws:Workspace):
        low, high = self.prob_max_increase_rate
        max_fraction_percent = (
                low + (high - low) * ws.val.this_annual['max_increase_fraction'])
        ws.val.next_carry[self.max_possible_fraction_on_bovaer] \
                = jnp.minimum(
                    (ws.val.this_carry[self.bovine_fraction_on_bovaer]
                     + max_fraction_percent / 100),
                    (1 - self.organic_fraction))

    @property
    def posts_developing_this_page(self) -> list[str]:
        return ['ProbabilisticBovaer',
                'ModellingBovaer']


class Bovaer_Production_Emissions(NIR_ModelElement):

    @property
    def short_description(self) -> str:
        return f"""Suppose that embedded/production emission of Bovaer is about {self.rate}."""

    @property
    def description(self) -> str:
        return f"""In deterministic models, this barrier estimates the
        production emissions of Bovaer at {self.rate}.
        </p>
        <p>
        In the probabilistic version of this barrier,
        the logic is based on a "20-50x" reduction in CO2e compared with
        methane emitted from cattle, as suggested by an LLM-AI chat,
        which was the original source for the {self.rate} value.
        For each sampled scenario, a reduction factor
        is sampled uniformly from 20 to 50.
        </p>
        <p>This barrier does not reflect manufacturer or 3rd party
        estimates of emission rates nor does
        it include any change in production emission efficiency over time.
        """

    def __init__(
            self,
            annual_headcounts_key,
            annual_bovaer_fraction,
            ):
        super().__init__()
        self._ktCO2e_ca_key = self.annual_key('ktCO2e_ca')
        self._ktCO2e_pt_keys = {
                pt: self.annual_key(f'ktCO2e_{pt.value}')
                for pt in PT if pt != PT.XX}
        self._annual_headcounts_key = annual_headcounts_key
        self._annual_bovaer_fraction = annual_bovaer_fraction

    @property
    def ktCO2e_ca_key(self) -> AnnualKey:
        return self._ktCO2e_ca_key

    @property
    def ktCO2e_pt_keys(self) -> dict[PT, AnnualKey]:
        return self._ktCO2e_pt_keys

    @property
    def annual_headcounts(self) -> AnnualKey:
        return self._annual_headcounts_key

    @property
    def annual_bovaer_fraction(self) -> AnnualKey:
        return self._annual_bovaer_fraction

    @property
    def version_id(self) -> VersionID:
        return (1,)
    
    @access_annual(annual_headcounts)
    @access_annual(annual_bovaer_fraction)
    @define('bovaer_production_emission_factor', prior_shape=[])
    @define_annual(ktCO2e_ca_key, annual_shape=[])
    @define_annual(ktCO2e_pt_keys, annual_shape=[])
    def model_element_postprocess(self, ws:Workspace):
        # in units of kg CH4 / head / year
        ws.dist.general['bovaer_production_emission_factor'] = dists.Uniform(20, 50)

        # sum over livestock types
        heads_pt = ws.val.annual[self.annual_headcounts].sum(axis=1)

        # sum over regions
        heads_ca = heads_pt.sum(axis=1)

        kt_per_kg = 1 / 1_000_000
        pt_scale = 10

        loc = (
                heads_ca
                   * ws.val.annual[self.annual_bovaer_fraction]
                   * (ws.val.general['bovaer_production_emission_factor']
                      * kt_per_kg
                      * float(GWP_100[GHG.CH4].magnitude)))
        ws.dist.annual[self.ktCO2e_ca_key] = dists.Normal(
                loc, scale=jnp.sqrt(13 * (pt_scale ** 2)))
        for ii, (_, pt_key) in enumerate(self.ktCO2e_pt_keys.items()):
            loc = (
                    heads_pt[:, ii]
                       * ws.val.annual[self.annual_bovaer_fraction]
                       * (ws.val.general['bovaer_production_emission_factor']
                          * kt_per_kg
                          * float(GWP_100[GHG.CH4].magnitude)))
            ws.dist.annual[pt_key] = dists.Normal(loc, scale=pt_scale)

    @property
    def ktCO2e_emissions_ca(
            self,
            ) -> dict[tuple[IPCC_Sector, GHG], AnnualKey]:
        rval = {}
        rval[IPCC_Sector.SCS__Manufacturing__Other, GHG.CO2] = self.ktCO2e_ca_key
        return rval

    @property
    def ktCO2e_emissions_pt(
            self,
            ) -> dict[tuple[IPCC_Sector, GHG], dict[PT, AnnualKey]]:
        rval = {}
        rval[IPCC_Sector.SCS__Manufacturing__Other, GHG.CO2] = self.ktCO2e_pt_keys
        return rval

# TODO: there will be a cost for monitoring
# https://www.mn.uio.no/geo/english/about/news-and-events/news/2025/combined-drone-satelite-data-and-ground-based-measurements-methane-emissions.html
# It can apparently be done pretty well with drones
# There are approximately 70_000 cattle operations in Canada
# Monitoring might cost 10-20 million / year?


class Bovaer_Monitoring(NIR_ModelElement):

    @property
    def short_description(self) -> str:
        return f"""Assume administering and monitoring costs
    {self.paperwork_monitoring} for paperwork and {self.onsite_monitoring} for
    on-site inspection"""

    @property
    def description(self) -> str:
        return f"""Assume administering and monitoring costs
    {self.paperwork_monitoring} for paperwork and {self.onsite_monitoring} for
    on-site inspection. Furthermore, estimate that these amounts are paid
    in labour costs, from which income tax is collected back at a rate of
    {self.income_tax_rate * 100}%.
    """

    def __init__(self, annual_headcounts_key, annual_bovaer_fraction):
        super().__init__()
        self._baseline_subsidy_ca_key = self.annual_key('baseline_subsidy_ca')
        self._annual_headcounts_key = annual_headcounts_key
        self._annual_bovaer_fraction = annual_bovaer_fraction

    @property
    def paperwork_monitoring(self) -> object:
        return 1000 * u.CAD / u.farm / u.year

    @property
    def onsite_monitoring(self) -> object:
        return 3000 * u.CAD / u.farm / u.year

    @property
    def income_tax_rate(self) -> float:
        return .25

    @property
    def baseline_subsidy_ca_key(self) -> AnnualKey:
        return self._baseline_subsidy_ca_key

    @property
    def subsidy_ca_key(self) -> AnnualKey:
        return self.annual_key('subsidy_ca')

    @property
    def annual_headcounts_key(self) -> AnnualKey:
        return self._annual_headcounts_key

    @property
    def annual_bovaer_fraction(self) -> AnnualKey:
        return self._annual_bovaer_fraction
    
    @property
    def version_id(self) -> VersionID:
        return (1,)

    @property
    def admin_ca(self) -> AnnualKey:
        return self.annual_key('admin_ca')

    @property
    def admin_ca_tax(self) -> AnnualKey:
        return self.annual_key('admin_ca_tax')

    @property
    def onsite_ca(self) -> AnnualKey:
        return self.annual_key('onsite_ca')

    @property
    def onsite_ca_tax(self) -> AnnualKey:
        return self.annual_key('onsite_ca_tax')

    @access_annual(baseline_subsidy_ca_key, annual_shape=[], default=0)
    @access_annual(annual_headcounts_key, annual_shape=[])
    @define_annual(admin_ca, annual_shape=[], posterior_only=True)
    @define_annual(onsite_ca, annual_shape=[], posterior_only=True)
    @define_annual(admin_ca_tax, annual_shape=[], posterior_only=True)
    @define_annual(onsite_ca_tax, annual_shape=[], posterior_only=True)
    def model_element_postprocess(self, ws:Workspace):
        if ws.phase == Phase.Prior:
            return
        heads = ws.val.annual[self.annual_headcounts_key]
        heads_ca = heads.sum(axis=(1, 2))
        # (years, maybe mcmc dim)
        farms_on_bovaer_ca = (
                heads_ca / cattle_per_farm
                * ws.val.annual[self.annual_bovaer_fraction])
        ws.val.annual[self.admin_ca] = (
                farms_on_bovaer_ca * self.paperwork_monitoring.magnitude)
        ws.val.annual[self.onsite_ca] = (
                farms_on_bovaer_ca * self.onsite_monitoring.magnitude)

        ws.val.annual[self.admin_ca_tax] = (
                ws.val.annual[self.admin_ca] * self.income_tax_rate)
        ws.val.annual[self.onsite_ca_tax] = (
                ws.val.annual[self.onsite_ca] * self.income_tax_rate)

    @property
    def CAD_cashflow(
            self,
            ) -> dict[tuple[SPE, SPE, str], AnnualKey]:
        return {
                (SPE.BovaerSubsidy,
                 SPE.BovaerMonitoringCompanies,
                 'salaries for admin staff',
                 ): self.admin_ca,
                (SPE.BovaerSubsidy,
                 SPE.BovaerMonitoringCompanies,
                 'salaries for onsite staff',
                 ): self.onsite_ca,
                (SPE.BovaerMonitoringCompanies,
                 SPE.BovaerSubsidy, # imagine income tax is applied back here
                 'tax from admin staff',
                 ): self.admin_ca_tax,
                (SPE.BovaerMonitoringCompanies,
                 SPE.BovaerSubsidy, # imagine income tax is applied back here
                 'tax from onsite staff',
                 ): self.onsite_ca_tax,
                }


class Bovaer_Farm_Subsidy(NIR_ModelElement):

    def __init__(self,
                 def_bovine_fraction_on_bovaer:CarryKey,
                 max_fraction:CarryKey,
                 annual_headcounts_key:AnnualKey,
                 def_annual_bovaer_fraction:AnnualKey):
        self._bovine_fraction_on_bovaer_key = def_bovine_fraction_on_bovaer
        self._max_fraction = max_fraction
        self._annual_headcounts_key = annual_headcounts_key
        self._annual_bovaer_fraction = def_annual_bovaer_fraction

    @property
    def short_description(self) -> str:
        return f"""Pay farmers {self.subsidy_rate} to administer Bovaer."""

    @property
    def description(self) -> str:
        return f"""Model a tax-funded government subsidy program to pay
        cattle farmers {self.subsidy_rate} to administer Bovaer to all
        cattle on their farms, and comply with monitoring protocols.
        It is assumed that this subsidy amount is deductible, not
        subject to income tax.
        """

    @property
    def subsidy_rate(self) -> object:
        return 5000 * u.CAD / u.farm / u.year

    @property
    def bovaer_budget(self) -> AnnualKey:
        return self.annual_key('bovaer_budget')

    @property
    def bovine_fraction_on_bovaer(self) -> CarryKey:
        return self._bovine_fraction_on_bovaer_key

    @property
    def annual_bovaer_fraction(self) -> AnnualKey:
        return self._annual_bovaer_fraction

    @property
    def annual_headcounts(self) -> AnnualKey:
        return self._annual_headcounts_key

    @property
    def max_fraction_on_bovaer(self) -> CarryKey:
        return self._max_fraction

    @property
    def version_id(self) -> VersionID:
        return (1,)

    @define_carry(bovine_fraction_on_bovaer, shape=[])
    def model_element_prepare(self, ws:Workspace):
        ws.val.initial_carry[self.bovine_fraction_on_bovaer] = jnp.zeros(())

    @access_annual(bovaer_budget, default=0)
    @define_annual(annual_bovaer_fraction, annual_shape=[])
    def model_element_annual_step(self, ws:Workspace):
        ws.val.next_carry[self.bovine_fraction_on_bovaer] = jnp.where(
                ws.val.this_annual[self.bovaer_budget] > 0,
                ws.val.this_carry[self.max_fraction_on_bovaer],
                ws.val.this_carry[self.bovine_fraction_on_bovaer])
        ws.val.this_annual[self.annual_bovaer_fraction] \
                = ws.val.next_carry[self.bovine_fraction_on_bovaer]

    @property
    def subsidy_pt(self) -> AnnualKey:
        return self.annual_key('subsidy_pt')

    @property
    def subsidy_ca(self) -> AnnualKey:
        return self.annual_key('subsidy_ca')

    @property
    def subsidy_pt_tax(self) -> AnnualKey:
        return self.annual_key('subsidy_pt_tax')

    @property
    def subsidy_ca_tax(self) -> AnnualKey:
        return self.annual_key('subsidy_ca_tax')

    @access_annual(annual_headcounts)
    @define_annual(subsidy_pt, annual_shape=[13])
    @define_annual(subsidy_ca, annual_shape=[])
    @define_annual(subsidy_pt_tax, annual_shape=[13])
    @define_annual(subsidy_ca_tax, annual_shape=[])
    def model_element_postprocess(self, ws:Workspace):
        # num_samples, cattle_types, PT 14
        subsidy_per_head = self.subsidy_rate.magnitude / cattle_per_farm

        # sum over livestock types
        heads_pt = ws.val.annual[self.annual_headcounts].sum(axis=1)

        bovaer_heads_pt = (
                heads_pt
                * ws.val.annual[self.annual_bovaer_fraction][:, None, ...])

        ws.val.annual[self.subsidy_pt] = subsidy_per_head * bovaer_heads_pt[:, :13]
        ws.val.annual[self.subsidy_ca] = subsidy_per_head * bovaer_heads_pt.sum(axis=1)

        ws.val.annual[self.subsidy_pt_tax] = ws.val.annual[self.subsidy_pt] * .2
        ws.val.annual[self.subsidy_ca_tax] = ws.val.annual[self.subsidy_ca] * .2


    @property
    def CAD_cashflow(
            self,
            ) -> dict[tuple[SPE, SPE, str], AnnualKey]:
        return {
                (SPE.BovaerSubsidy,
                 SPE.CattleFarmersFeedingBovaer,
                 'Bovaer farm subsidy',
                 ): self.subsidy_ca,
                (SPE.CattleFarmersFeedingBovaer,
                 SPE.BovaerSubsidy, # imagine income tax is applied back here
                 'Farm income tax',
                 ): self.subsidy_ca_tax,
                }


class Bovaer_Purchase_Cost(NIR_ModelElement):
    """
    Define subsidy rates for the purchase cost of Bovaer
    """

    def __init__(self, annual_headcounts_key, annual_bovaer_fraction):
        super().__init__()
        self._baseline_subsidy_ca_key = self.annual_key('baseline_subsidy_ca')
        self._annual_headcounts_key = annual_headcounts_key
        self._annual_bovaer_fraction = annual_bovaer_fraction

    @property
    def bovaer_cost(self) -> dict[object, object]:
        # https://www.producer.com/livestock/new-methane-feed-additive-pleases-producers
        return {
            Livestock.Bulls: .50 * u.CAD / u.day / u.cattle,
            Livestock.DairyCows: 0.50 * u.CAD / u.day / u.cattle,
            Livestock.BeefCows: 0.50 * u.CAD / u.day / u.cattle,
            Livestock.DairyHeifers: .35 * u.CAD / u.day / u.cattle,
            Livestock.BeefHeifers: .35 * u.CAD / u.day / u.cattle,
            Livestock.SlaughterHeifers: .35 * u.CAD / u.day / u.cattle,
            Livestock.Steers: .35 * u.CAD / u.day / u.cattle,
            Livestock.Calves: .20 * u.CAD / u.day / u.cattle,
        }

    @property
    def annual_bovaer_fraction(self) -> AnnualKey:
        return self._annual_bovaer_fraction

    @property
    def annual_headcounts(self) -> AnnualKey:
        return self._annual_headcounts_key

    @property
    def version_id(self) -> VersionID:
        return (1,)

    @property
    def cost_ca(self) -> AnnualKey:
        return self.annual_key('cost_ca')

    @access_annual(annual_bovaer_fraction)
    @access_annual(annual_headcounts)
    @define_annual('bovaer_cost_pt', annual_shape=[13])
    @define_annual(cost_ca, annual_shape=[])
    def model_element_postprocess(self, ws:Workspace):
        # num_samples, cattle_types, PT 14

        # shape (years, livestock_types, PTs, maybe MCMC)
        heads_pt = ws.val.annual[self.annual_headcounts]

        bovaer_cost_per_year = jnp.array(
                [self.bovaer_cost[lt].magnitude for lt in Livestock_nonsums]
                ) * 365

        # shape (years, PTs, maybe MCMC)
        bovaer_costs_pt = (
                (heads_pt
                 * bovaer_cost_per_year[:, None, ws.mcmc_broadcast_index]
                 ).sum(axis=1) # livestock types, leaving (years, PT, maybe MCMC)
                * ws.val.annual[self.annual_bovaer_fraction][:, None])

        ws.val.annual['bovaer_cost_pt'] = bovaer_costs_pt
        ws.val.annual[self.cost_ca] = bovaer_costs_pt.sum(axis=1)

    @property
    def CAD_cashflow(
            self,
            ) -> dict[tuple[SPE, SPE, str], AnnualKey]:
        return {
                (SPE.BovaerSubsidy,
                 SPE.BovaerProducers,
                 'Bovaer purchase',
                 ): self.cost_ca,
                }


class Scale_Bovaer(ModelElement):
    """<p>The "Scale Bovaer" strategy implements an assumption that
    cattle farmers who are modelled as being open to Bovaer usage
    (according to the assumptions in
     <a href="{{coderef_url(Bovaer_Adoption_Limit)}}">Bovaer Adoption Limit</a>)
    are subsized by public funds, and go for it. This adoption is modelled as a nation-wide
    rate of adoption, not province-by-province.
    It is assumed that no statistically significant population of farmers
    reverts to Bovaer-free farming.
    </p>
    """
    # TODO: add a see-also type mechanism, to look at the effects
    # on the various barriers affected by this strategy.

    # used by SiteInference constructor to identify strategies that
    # should be ablated
    is_strategy:bool = True

    def __init__(self, def_bovaer_budget:AnnualKey, fixed_year:int|None=2030):
        self._bovaer_budget = def_bovaer_budget
        self.fixed_year = 2030

    @staticmethod
    def element_identifier(fixed_year:int|None) -> str:
        if fixed_year is None:
            return 'Scale_Bovaer'
        else:
            return f'Scale_Bovaer_{fixed_year}'

    @property
    def identifier(self) -> str:
        return self.element_identifier(self.fixed_year)

    @property
    def short_description(self) -> str:
        return "Subsidize farmers to administer Bovaer."

    def see_also_html(self, context_vars) -> list[str]:
        TODO_include = "'bovaer dairy safety NIH': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC8603004'"
        sources = [
            ('<a'
             ' {% if sim_name %}'
             ' href="/models/sim/{{ sim_name }}/barriers/Bovaer_Adoption_Limit/">Bovaer'
             ' {% else %}'
             ' href="/models/prob/{{ site_inference_name }}/barriers/Bovaer_Adoption_Limit/">Bovaer'
             ' {% endif %}'
             ' Adoption Limit</a>, which is the model barrier that sets the rate'
             ' of adoption for this strategy'),
            ('<a'
            ' href="https://github.com/jaberg/planzero/blob/main/planzero/cattle.py">cattle.py</a>,'
            ' which defines several of the barriers relating to this strategy'),
            ('<a'
             ' href="{{coderef_url(myself.__class__)}}">{{coderef_filepath(myself.__class__)}}</a>,'
             ' the implementation of this strategy'),
        ]
        render_vars = dict(
            context_vars,
            Bovaer_Adoption_Limit=Bovaer_Adoption_Limit,
            myself=self,
            )
        return [jinja2.Template(source=source).render(render_vars)
                for source in sources]

    @property
    def description_html(self) -> str:
        if self.__doc__ is None:
            raise NotImplementedError()
        else:
            template = jinja2.Template(source=self.__doc__)
            rval = template.render(
                Bovaer_Adoption_Limit=Bovaer_Adoption_Limit,
                coderef_url=coderef_url,
                )
            return rval

    @property
    def bovaer_budget(self) -> AnnualKey:
        return self._bovaer_budget

    @property
    def bovaer_program_launch(self) -> GeneralKey:
        return self.general_key('bovaer_program_launch')

    @property
    def version_id(self) -> VersionID:
        return (1,)

    @define(bovaer_program_launch, prior_shape=[])
    @define_annual(bovaer_budget, annual_shape=[])
    def model_element_prepare(self, ws:Workspace):
        if self.fixed_year is not None:
            ws.val.general[self.bovaer_program_launch] = jnp.array(self.fixed_year)[ws.mcmc_broadcast_index]
        else:
            ws.dist.general[self.bovaer_program_launch] = dists.Uniform(2027, 2035)
        # (years, optionally MCMC dimension)
        ws.val.annual[self.bovaer_budget] = jnp.where(
                ws.val.annual[years_key] < ws.val.general[self.bovaer_program_launch],
                0,
                float('inf'))
