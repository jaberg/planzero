
from pydantic import Field, computed_field

from ..ureg import u
from ..enums import IPCC_Sector, PT
from ..base import DynamicElement
from .. import sts
from .. import objtensor
from ..html import coderef_url

strategies = {} # classname -> Singleton instance


class Strategy2(DynamicElement):

    @classmethod
    def __init_subclass__(cls):
        super().__init_subclass__()
        strategies[cls.__name__] = cls()

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.tags.add('strategy')

    @computed_field
    def ipcc_sector_values(self) -> list[str]:
        return [sec.value for sec in self.ipcc_sectors]

import jinja2

class Scale_Bovaer(Strategy2):
    """<p>The "Scale Bovaer" strategy implements an assumption that
    cattle farmers who are modelled as being open to Bovaer usage
    (according to the assumptions in
     <a href="{{coderef_url(Bovaer_Adoption_Limit)}}">Bovaer Adoption Limit</a>)
    actually go for it. This adoption is modelled as a nation-wide
    proportionality, not province-by-province.</p>
    """
    # TODO: add a see-also type mechanism, to look at the effects
    # on the various barriers affected by this strategy.

    @computed_field
    def short_description(self) -> str:
        return f"Model that farmers who are open to using Bovaer actually start administering it."

    def see_also_html(self, context_vars) -> list[str]:
        sources = [
            ('<a'
             'href="/simulations/{{sim_name}}/barriers/Bovaer_Adoption_Limit/">Bovaer'
             'Adoption Limit</a>, which is the model barrier that sets the rate'
             'of adoption for this strategy'),
            ('<a'
            'href="https://github.com/jaberg/planzero/blob/main/planzero/cattle.py">cattle.py</a>,'
            'which defines several of the barriers relating to this strategy'),
            ('<a'
             'href="{{coderef_url(myself.__class__)}}">{{coderef_filepath(myself.__class__)}}</a>,'
             'the implementation of this strategy'),
        ]
        from ..cattle import Bovaer_Adoption_Limit
        render_vars = dict(
            context_vars,
            Bovaer_Adoption_Limit=Bovaer_Adoption_Limit,
            myself=self,
            )
        return [jinja2.Template(source=source).render(render_vars)
                for source in sources]

    @computed_field
    def description_html(self) -> str:
        from ..cattle import Bovaer_Adoption_Limit

        template = jinja2.Template(source=self.__doc__)
        rval = template.render(
            Bovaer_Adoption_Limit=Bovaer_Adoption_Limit,
            coderef_url=coderef_url,
            )
        return rval

    @computed_field
    def extra_ipcc_sectors(self) -> list[object]:
        # TODO: https://github.com/jaberg/planzero/issues/72
        # would eliminate need for this
        return [
            IPCC_Sector.Enteric_Fermentation,
            IPCC_Sector.Other_Product_Manufacture_and_Use,
        ]


    @computed_field
    def research(self) -> dict[str, str]:
        return {
            'bovaer dairy safety NIH': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC8603004'
        }

    def on_add_project(self, state):
        with state.defining(self) as ctx:
            obj = sts.SparseTimeSeries(default_value=0 * u.dimensionless, t_unit=u.year)
            ctx.bovine_population_fraction_on_bovaer = obj
        # XXX wait until after Cattle_Population nearcasting until there's support
        # for temporally overlapping initialization
        return 2030 * u.years

    def step(self, state, current):
        # The strategy here, is to use as much Bovaer as farmers will take
        max_fraction = state.latest.max_fraction_of_cattle_on_bovaer
        current.bovine_population_fraction_on_bovaer = max_fraction
        return state.t_now + 1 * u.years
