import json
import datetime
import os

import numpy as np

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

htmlroot = 'html'

app.mount("/assets", StaticFiles(directory=f"{htmlroot}/assets/"), name="assets")
app.mount("/images", StaticFiles(directory=f"{htmlroot}/images/"), name="images")

templates = Jinja2Templates(directory=htmlroot)

import planzero
import planzero.blog
import planzero.ipcc_home
import planzero.est_nir
import planzero.enums
from planzero import get_peval

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


@app.get("/strategies/{strategy_name}/", response_class=HTMLResponse)
async def get_strategy_eval(request: Request, strategy_name:str):
    peval = get_peval()
    strategy = peval.comparisons[strategy_name].project
    comparison = peval.comparisons[strategy_name]
    strategy_page = strategy.strategy_page(comparison)
    return templates.TemplateResponse(
        request=request,
        name=f"strategy_page.html",
        context=dict(
            default_context,
            peval=peval,
            active_tab='strategies',
            strategy=strategy,
            comparison=comparison,
            strategy_page=strategy_page,
            ),
    )

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
        peval=get_peval(),
        stakeholders=planzero.strategies.stakeholders,
        catpath=catpath,
        blogs_by_tag=planzero.blog.blogs_by_tag,
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


@app.get("/barriers/", response_class=HTMLResponse)
async def get_barriers(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="barriers.html",
        context=dict(
            default_context,
            active_tab='barriers',
            ),
    )

@app.get("/scenarios/{scenario_name}/barriers/{barrier_name}/", response_class=HTMLResponse)
async def get_scenario_strategy_impact(request: Request, scenario_name: str, barrier_name: str):
    sim = planzero.sim.sim_scenario(scenario_name)
    return templates.TemplateResponse(
        request=request,
        name="scenario_barrier.html",
        context=dict(
            default_context,
            sim=sim,
            active_tab='scenarios',
            scenario_name=scenario_name,
            barrier_name=barrier_name,
            ),
    )


@app.get("/scenarios/", response_class=HTMLResponse)
async def get_scenarios(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="scenarios.html",
        context=dict(
            default_context,
            active_tab='scenarios',
            ),
    )


@app.get("/scenarios/{ident}/", response_class=HTMLResponse)
async def get_scenario_page(ident:str, request: Request):
    return templates.TemplateResponse(
        request=request,
        name="scenario_template.html",
        context=dict(
            default_context,
            active_tab='scenarios',
            ident=ident,
            ),
    )


@app.get("/scenarios/{scenario_name}/ipcc-sectors/{category}/", response_class=HTMLResponse)
@app.get("/scenarios/{scenario_name}/ipcc-sectors/{category}/{subcategory}/", response_class=HTMLResponse)
@app.get("/scenarios/{scenario_name}/ipcc-sectors/{category}/{subcategory}/{subsubcategory}/", response_class=HTMLResponse)
async def get_scenario_ipcc_sectors_category(
    request: Request,
    scenario_name: str,
    category: str,
    subcategory: str = None,
    subsubcategory: str = None):

    if subsubcategory is not None:
        catpath=f'{category}/{subcategory}/{subsubcategory}'
    elif subcategory is not None:
        catpath = f'{category}/{subcategory}'
    else:
        catpath = f'{category}'

    sim = planzero.sim.sim_scenario(scenario_name)
    chart = sim.echart_ipcc_sector(catpath)

    return templates.TemplateResponse(
        request=request,
        name="scenario_ipcc_sector.html",
        context=dict(
            default_context,
            active_tab='scenarios',
            scenario_name=scenario_name,
            ipcc_sector=planzero.enums.IPCC_Sector.from_catpath(catpath),
            catpath=catpath,
            chart=chart,
            ),
    )


@app.get("/scenarios/{scenario_name}/strategies/{strategy_name}/", response_class=HTMLResponse)
async def get_scenario_strategy_impact(request: Request, scenario_name: str, strategy_name: str):
    sim = planzero.sim.sim_scenario(scenario_name)
    baseline_state = sim.state
    ablated_state = sim.ablations.get(strategy_name)
    if not ablated_state:
        raise HTTPException(status_code=404, detail="Strategy not found in this scenario")
    
    # Calculate impact (baseline - ablated)
    # This assumes we want to show emissions saved
    sim_years_ints = np.arange(1990, 2090)
    sim_years = [tt * u.years for tt in sim_years_ints]
    
    # Simple total emissions comparison
    baseline_total = baseline_state.sts['Predicted_Annual_Emitted_CO2e_mass']
    ablated_total = ablated_state.sts['Predicted_Annual_Emitted_CO2e_mass']

    impact_data = planzero.sim.EChartSeriesData(
        ablated_total - baseline_total, # ablated - baseline = amount saved if ablated > baseline
        times=sim_years,
        v_unit=u.Mt_CO2e,
        url=None, # TODO: link to this class's code on github
        )

    impact_chart = planzero.sim.StackedAreaEChart(
        div_id='impact_chart',
        title=planzero.sim.EChartTitle(
            text=f'Emissions Impact: {strategy_name}',
            subtext=f'Annual Mt CO2e saved in {scenario_name}'),
        xAxis=planzero.sim.EChartXAxis(data=sim_years_ints.tolist()),
        yAxis=[planzero.sim.EChartYAxis(name='Emissions Saved (Mt CO2e)')],
        stacked_series=[
            planzero.sim.EChartSeriesStackElem(
                name='Emissions Avoided',
                data=impact_data,
            )
        ],
        other_series=[])

    # Simple total subsidy comparison
    subsidy_baseline_total = baseline_state.sts['AnnualSubsidyTotal']
    subsidy_ablated_total = ablated_state.sts['AnnualSubsidyTotal']

    subsidy_comparison_data = planzero.sim.EChartSeriesData(
        subsidy_baseline_total - subsidy_ablated_total,
        times=sim_years,
        v_unit=u.giga_CAD,
        url=None, # TODO: link to this class's code on github
        )

    subsidies_chart = planzero.sim.StackedAreaEChart(
        div_id='subsidies_chart',
        title=planzero.sim.EChartTitle(
            text=f'Subsidies Impact: {strategy_name}',
            subtext=f'Annual cost of subsidies in {scenario_name}'),
        xAxis=planzero.sim.EChartXAxis(data=sim_years_ints.tolist()),
        yAxis=[planzero.sim.EChartYAxis(name='Subsidies Required (CAD, Billions)')],
        stacked_series=[
            planzero.sim.EChartSeriesStackElem(
                name='Cost Incurred',
                data=subsidy_comparison_data,
            )
        ],
        other_series=[])

    cost_per_tCO2e = (
        (subsidy_baseline_total - subsidy_ablated_total).sum()
        / (ablated_total - baseline_total).sum()).to(u.CAD / u.tonne_CO2e)

    return templates.TemplateResponse(
        request=request,
        name="strategy_impact.html",
        context=dict(
            default_context,
            active_tab='scenarios',
            scenario_name=scenario_name,
            strategy_name=strategy_name,
            impact_chart=impact_chart,
            subsidies_chart=subsidies_chart,
            cost_per_tCO2e=cost_per_tCO2e,
            ),
    )


@app.get("/strategies/", response_class=HTMLResponse)
async def get_strategies(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="strategies.html",
        context=dict(
            default_context,
            peval=get_peval(),
            active_tab='strategies',
            npv_unit='MCAD',
            nph_unit='exajoule',
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
async def get_blog(request: Request, post_name:str):
    try:
        html = get_blog_html(post_name)
        return HTMLResponse(content=html)
    except IOError:
        raise HTTPException(status_code=404, detail="url not recognized")


@app.get("/about/", response_class=HTMLResponse)
async def get_about(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="about.html",
        context=dict(
            default_context,
            active_tab='about',
            blogs_by_tag=planzero.blog.blogs_by_tag,
            ),
    )

@app.get("/glossary/", response_class=HTMLResponse)
async def get_about(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="glossary.html",
        context=dict(
            default_context,
            active_tab='glossary',
            blogs_by_tag=planzero.blog.blogs_by_tag,
            ),
    )

@app.get("/index.html", response_class=HTMLResponse)
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
            peval=get_peval(),
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
    N2O=planzero.blog.latex(r"\mathrm N_2 \mathrm O"),
    CO2e=planzero.blog.latex(r'\mathrm{CO}_2\mathrm e '),
    degrees=planzero.blog.latex(r'^\circ'),
    siteref=planzero.glossary.siteref,
    )

