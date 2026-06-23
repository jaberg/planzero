import datetime
import json
import os

import numpy as np

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import jinja2

app = FastAPI()

htmlroot = 'html'

app.mount("/assets", StaticFiles(directory=f"{htmlroot}/assets/"), name="assets")
app.mount("/images", StaticFiles(directory=f"{htmlroot}/images/"), name="images")

templates = Jinja2Templates(
    env=jinja2.Environment(
        undefined=jinja2.StrictUndefined,
        loader=jinja2.FileSystemLoader(htmlroot),
        ))

import planzero
import planzero.blog
import planzero.ipcc_home
import planzero.est_nir
import planzero.enums

u = planzero.ureg

HOME_SHOW_UNPUBLISHED_POSTS = (os.environ['PLANZERO_HOME_SHOW_UNPUBLISHED_POSTS'] == '1')

def app_cache(f):
    if planzero.my_functools.USE_DISK_CACHE:
        # this branch is used in deployed code
        return planzero.my_functools.cache(f)
    else:
        # this branch is used in dev mode,
        # where, coincidentally, it's preferred to
        # not cache endpoints *at all* so that I can see
        # changes in html templates rendered via jinja
        # without reloading anything
        return f


@app.get("/ipcc-sectors/", response_class=HTMLResponse)
async def get_ipcc_sectors(request: Request, error_text:str=None):
    return templates.TemplateResponse(
        request=request,
        name="ipcc-sectors.html",
        context=dict(
            default_context,
            active_tab='ipcc_sectors',
            error_text=error_text,
            npv_unit='MCAD',
            nph_unit='exajoule',
            ipcc_home=planzero.ipcc_home
            ),
    )


def url_for_catpath(catpath):
    return f'/ipcc-sectors/{catpath}/'.replace(' ', '_')


def filepath_for_catpath(catpath):
    return f'{htmlroot}/ipcc-sectors/{catpath}.html'.replace(' ', '_')


def templatepath_for_catpath(catpath):
    return f'/ipcc-sectors/{catpath}.html'.replace(' ', '_')


def have_page_for_catpath(catpath):
    try:
        filepath = filepath_for_catpath(catpath)
        open(filepath).close()
        return True
    except IOError:
        return False


@app_cache
def get_ipcc_sector_html(catpath: str):
    if not have_page_for_catpath(catpath):
        return None
    return templates.get_template(templatepath_for_catpath(catpath)).render(dict(
        default_context,
        active_tab='ipcc_sectors',
        stakeholders=planzero.strategies.stakeholders,
        catpath=catpath,
        est_nir=planzero.est_nir,
        ))


@app.get("/ipcc-sectors/{category}/", response_class=HTMLResponse)
@app.get("/ipcc-sectors/{category}/{subcategory}/", response_class=HTMLResponse)
@app.get("/ipcc-sectors/{category}/{subcategory}/{subsubcategory}/", response_class=HTMLResponse)
async def get_ipcc_sectors_category(
    request: Request,
    category,
    subcategory:str=None,
    subsubcategory:str=None):

    if subsubcategory is not None:
        catpath=f'{category}/{subcategory}/{subsubcategory}'
    elif subcategory is not None:
        catpath = f'{category}/{subcategory}'
    else:
        catpath = f'{category}'

    html = get_ipcc_sector_html(catpath)
    if html:
        return HTMLResponse(content=html)
    else:
        return await get_ipcc_sectors(
            request, 
            error_text=f"Sorry, we don't have the analysis page for {catpath} yet")


@app.get("/models/sim/{sim_name}/barriers/{barrier_name}/", response_class=HTMLResponse)
async def get_simulation_barrier_impact(request: Request, sim_name: str, barrier_name: str):
    sim = planzero.sim.simulation_result(sim_name)
    return templates.TemplateResponse(
        request=request,
        name="scenario_barrier.html",
        context=dict(
            default_context,
            sim=sim,
            active_tab='simulations',
            sim_name=sim_name,
            barrier_name=barrier_name,
            ),
    )


@app_cache
def get_simulations_page_html(ident:str):
    site_sim = planzero.sim.site_simulations[ident]
    sim_result = planzero.sim.simulation_result(ident)
    sectors_by_de = sim_result.state.ipcc_sectors_by_dynamic_element()

    def ipcc_sectors_from_dynelem(dynelem):
        sectors = sectors_by_de.get(dynelem.identifier, set())
        if len(sectors) < 5:
            return sectors
        else:
            return [] # TODO: better version of "many"

    return templates.get_template("scenario_template.html").render(
        dict(
            default_context,
            active_tab='models',
            ident=ident,
            ipcc_sectors_from_dynelem=ipcc_sectors_from_dynelem,
            site_sim=site_sim,
            sim_result=sim_result,
            ))


@app.get("/models/sim/{ident}/", response_class=HTMLResponse)
async def get_simulation_page(ident:str, request: Request):
    html = get_simulations_page_html(ident)
    return HTMLResponse(content=html)


@app_cache
def get_models_prob_page_html(ident:str):
    site_inference = planzero.prob.site_inferences[ident]
    return templates.get_template("models_prob.html").render(
        dict(
            default_context,
            active_tab='models',
            ident=ident,
            site_inference=site_inference,
            ))

@app.get("/models/prob/{ident}/", response_class=HTMLResponse)
async def get_models_prob_page(ident:str, request: Request):
    html = get_models_prob_page_html(ident)
    return HTMLResponse(content=html)


# models/prob/{ident}/sector/{sector_value}

@app_cache
def get_models_prob_sector_page_html(ident:str, sector_value:str):
    site_inference = planzero.prob.site_inferences[ident]
    return templates.get_template("models_prob_sector.html").render(
        dict(
            default_context,
            active_tab='models',
            ident=ident,
            site_inference=site_inference,
            sector=planzero.enums.IPCC_Sector(sector_value),
            ))


@app.get("/models/prob/{ident}/sectors/{sector_value}", response_class=HTMLResponse)
async def get_models_prob_page(ident:str, sector_value:str, request: Request):
    html = get_models_prob_sector_page_html(ident, sector_value=sector_value)
    return HTMLResponse(content=html)


@app.get("/models/sim/{sim_name}/ipcc-sectors/{category}/", response_class=HTMLResponse)
@app.get("/models/sim/{sim_name}/ipcc-sectors/{category}/{subcategory}/", response_class=HTMLResponse)
@app.get("/models/sim/{sim_name}/ipcc-sectors/{category}/{subcategory}/{subsubcategory}/", response_class=HTMLResponse)
async def get_simulation_ipcc_sectors_category(
    request: Request,
    sim_name: str,
    category: str,
    subcategory: str = None,
    subsubcategory: str = None):

    if subsubcategory is not None:
        catpath=f'{category}/{subcategory}/{subsubcategory}'
    elif subcategory is not None:
        catpath = f'{category}/{subcategory}'
    else:
        catpath = f'{category}'

    sim = planzero.sim.simulation_result(sim_name)
    chart = sim.echart_ipcc_sector(catpath)

    return templates.TemplateResponse(
        request=request,
        name="scenario_ipcc_sector.html",
        context=dict(
            default_context,
            active_tab='simulations',
            sim_name=sim_name,
            ipcc_sector=planzero.enums.IPCC_Sector.from_catpath(catpath),
            catpath=catpath,
            chart=chart,
            ),
    )

@app_cache
def get_simulations_strategy_impact_html(sim_name: str, strategy_name: str):
    sim = planzero.sim.simulation_result(sim_name)
    baseline_state = sim.state
    ablated_state = sim.ablations.get(strategy_name)
    if not ablated_state:
        raise HTTPException(status_code=404, detail="Strategy not found in this simulation")
    
    # Calculate impact (baseline - ablated)
    # This assumes we want to show emissions saved
    sim_years_ints = np.arange(1990, 2090)
    sim_years = [tt * u.years for tt in sim_years_ints]
    
    impact_chart = sim.strategy_impact_echart(strategy_name)
    subsidies_chart = sim.strategy_subsidies_echart(strategy_name)

    # Simple total emissions comparison for cost calculation
    baseline_total = baseline_state.compute_annual_emissions().total()
    ablated_total = ablated_state.compute_annual_emissions().total()

    # Simple total subsidy comparison
    subsidy_baseline_total = baseline_state.compute_annual_subsidies().total()
    subsidy_ablated_total = ablated_state.compute_annual_subsidies().total()

    try:
        cost_per_tCO2e = (
            (subsidy_baseline_total - subsidy_ablated_total).sum()
            / (ablated_total - baseline_total).sum()).to(u.CAD / u.tonne_CO2e)
    except AssertionError:
        # this happens in Planet_Model
        cost_per_tCO2e = float('nan') * u.CAD / u.tonne_CO2e

    assert len(list(planzero.blog.blogs_by_tag(strategy_name)))

    context = dict(
        default_context,
        active_tab='simulations',
        sim_name=sim_name,
        strategy_name=strategy_name,
        strategy_class=baseline_state.projects[strategy_name].__class__,
        description_html=baseline_state.projects[strategy_name].description_html,
        impact_chart=impact_chart,
        subsidies_chart=subsidies_chart,
        cost_per_tCO2e=cost_per_tCO2e,
        )
    strategy_obj = baseline_state.projects[strategy_name]
    context['see_also'] = strategy_obj.see_also_html(context)
    return templates.get_template('strategy_impact.html').render(context)


@app.get("/models/sim/{sim_name}/strategies/{strategy_name}/", response_class=HTMLResponse)
async def get_simulations_strategy_impact(request: Request, sim_name: str, strategy_name: str):
    html = get_simulations_strategy_impact_html(sim_name, strategy_name)
    return HTMLResponse(content=html)


@app.get("/strategies/", response_class=HTMLResponse)
async def get_strategies(request: Request):
    sims_by_dynelems = {}
    sectors_by_dynelems = {}
    for sitesim_name, sitesim in planzero.sim.site_simulations.items():
        if not sitesim.show_on_simulations_page:
            continue
        sim_result = planzero.sim.simulation_result(sitesim_name)
        sectors_by_de = sim_result.state.ipcc_sectors_by_dynamic_element()
        for dynelem in sitesim.dynamic_elements():
            sims_by_dynelems.setdefault(dynelem.__class__.__name__, set())\
                    .add(sitesim_name)
            sectors_by_dynelems.setdefault(dynelem.__class__.__name__, set())\
                    .update(sectors_by_de[dynelem.identifier])
            sectors_by_dynelems[dynelem.__class__.__name__].update(
                dynelem.extra_ipcc_sectors)
    return templates.TemplateResponse(
        request=request,
        name="strategies.html",
        context=dict(
            default_context,
            active_tab='strategies',
            npv_unit='MCAD',
            nph_unit='exajoule',
            sims_by_dynelems=sims_by_dynelems,
            sectors_by_dynelems=sectors_by_dynelems,
            ),
    )


@app_cache
def get_blog_html(post_name: str):
    blog = planzero.blog._blogs_by_url_filename.get(post_name)
    prev_url_filename = None
    next_url_filename = None
    for ii, obj in enumerate(planzero.blog._blogs_sorted_by_date):
        if obj is blog:
            if ii:
                next_url_filename = planzero.blog._blogs_sorted_by_date[ii - 1].url_filename
            if ii + 1 < len(planzero.blog._blogs_sorted_by_date):
                prev_url_filename = planzero.blog._blogs_sorted_by_date[ii + 1].url_filename
            break

    return templates.get_template(f"/blog/{post_name}.html").render(dict(
        default_context,
        active_tab='blog',
        blog=blog,
        prev_url_filename=prev_url_filename,
        next_url_filename=next_url_filename,
        ))


@app.get("/blog/{post_name}", response_class=HTMLResponse)
@app.get("/post/{post_name}", response_class=HTMLResponse)
async def get_blog(request: Request, post_name:str):
    try:
        html = get_blog_html(post_name)
        return HTMLResponse(content=html)
    except IOError:
        raise HTTPException(status_code=404, detail="url not recognized")

## MODELS

@app_cache
def get_models_html():
    return templates.get_template('models.html').render(dict(
        default_context,
        active_tab='models',
        ))

@app.get("/models/", response_class=HTMLResponse)
async def get_models(request: Request):
    html = get_models_html()
    return HTMLResponse(content=html)


## PREDICTIONS

@app_cache
def get_predictions_html():
    return templates.get_template('predictions.html').render(dict(
        default_context,
        active_tab='predictions',
        ))

@app.get("/predictions/", response_class=HTMLResponse)
async def get_predictions(request: Request):
    html = get_predictions_html()
    return HTMLResponse(content=html)


## GLOSSARY

@app_cache
def get_glossary_html():
    return templates.get_template('glossary.html').render(dict(
        default_context,
        active_tab='glossary',
        ))

@app.get("/glossary/", response_class=HTMLResponse)
async def get_glossary(request: Request):
    html = get_glossary_html()
    return HTMLResponse(content=html)


@app.get("/about/", response_class=HTMLResponse)
async def get_about(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="about.html",
        context=dict(
            default_context,
            active_tab='about',
            ),
    )


@app.get("/index.html", response_class=HTMLResponse)
@app.get("/posts/", response_class=HTMLResponse)
@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request, unpublished:bool=HOME_SHOW_UNPUBLISHED_POSTS):
    return templates.TemplateResponse(
        request=request,
        name="blog.html",
        #name="index.html",
        context=dict(
            default_context,
            fade_in_intro=True,
            blogs_sorted_by_date=planzero.blog._blogs_sorted_by_date,
            active_tab='blog',
            unpublished=unpublished,
            ),
    )

default_context = dict(
    int=int,
    float=float,
    min=min,
    max=max,
    sorted=sorted,
    enumerate=enumerate,
    isinstance=isinstance,
    u=u,
    have_page_for_catpath=have_page_for_catpath,
    url_for_catpath=url_for_catpath,
    json=json,
    datetime=datetime,
    ipcc_canada=planzero.ipcc_canada,
    stakeholders=planzero.strategies.stakeholders,
    discount_rate=.02,
    planzero=planzero,
    CO2=planzero.blog.latex(r'\mathrm{CO}_2'),
    CH4=planzero.blog.latex(r'\mathrm{CH}_4'),
    NF3=planzero.blog.latex(r'\mathrm{NF}_3'),
    N2O=planzero.blog.latex(r"\mathrm N_2 \mathrm O"),
    CO2e=planzero.blog.latex(r'\mathrm{CO}_2\mathrm e '),
    degrees=planzero.blog.latex(r'^\circ'),
    siteref=planzero.glossary.siteref,
    coderef_url=planzero.html.coderef_url,
    coderef_filepath=planzero.html.coderef_filepath,
    fade_in_intro=False,
    printcname=(lambda cname: cname.replace('_', ' ')),
    blogs_by_tag=planzero.blog.blogs_by_tag,
    latex=planzero.blog.latex,
    )

