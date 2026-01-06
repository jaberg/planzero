import json
import datetime

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

htmlroot = 'html5up-massively'

app.mount("/assets", StaticFiles(directory=f"{htmlroot}/assets/"), name="assets")
app.mount("/images", StaticFiles(directory=f"{htmlroot}/images/"), name="images")


templates = Jinja2Templates(directory=htmlroot)

from planzero import ipcc_canada, stakeholders, base
u = base.u

peval = base.ProjectEvaluation(
    projects={prj.idea.name: prj for prj in [
        base.ComboA(idea=stakeholders.ideas.combo_a),
        base.NationalBovaerMandate(idea=stakeholders.ideas.national_bovaer_mandate),
        base.BatteryTugWithAuxSolarBarges(idea=stakeholders.ideas.battery_tugs_w_aux_solar_barges),
        base.Force_Government_ZEVs(),
    ]},
    common_projects=[
        base.GeometricBovinePopulationForecast(),
        base.GeometricHumanPopulationForecast(),
        base.IPCC_Forest_Land_Model(),
        base.IPCC_Transport_Marine_DomesticNavigation_Model(),
        base.IPCC_Transport_RoadTransportation_LightDutyGasolineTrucks(),
        base.PacificLogBargeForecast(),
        base.AtmosphericChemistry(),
    ],
)
peval.run_until(2125 * u.years)


@app.get("/strategies/{strategy_name}/", response_class=HTMLResponse)
async def get_strategy_eval(request: Request, strategy_name:str):
    return templates.TemplateResponse(
        request=request,
        name=f"strategies/{strategy_name}.html",
        context=dict(
            default_context,
            active_tab='strategies',
            strategy_name=strategy_name,
            comparison=peval.comparisons[strategy_name],
            project=peval.comparisons[strategy_name].project,
            state_A=peval.comparisons[strategy_name].state_A,
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
        open(filepath_for_catpath(catpath))
        return True
    except IOError:
        return False


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

    if have_page_for_catpath(catpath):
        return templates.TemplateResponse(
            request=request,
            name=templatepath_for_catpath(catpath),
            context=dict(
                default_context,
                active_tab='ipcc_sectors',
                stakeholders=stakeholders,
                catpath=catpath,
                ),
        )
    else:
        rval = await get_ipcc_sectors(
            request, 
            error_text=f"Sorry, we don't have the analyis page for {catpath} yet")
        return rval


@app.get("/strategies/", response_class=HTMLResponse)
async def get_strategies(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="strategies.html",
        context=dict(
            default_context,
            active_tab='strategies',
            npv_unit='MCAD',
            nph_unit='exajoule',
            ),
    )

@app.get("/blog/{post_name}", response_class=HTMLResponse)
async def get_strategies(request: Request, post_name:str):
    return templates.TemplateResponse(
        request=request,
        name=f"/blog/{post_name}.html",
        context=dict(
            default_context,
            active_tab='blog',
            ),
    )

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
@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="blog.html",
        #name="index.html",
        context=dict(
            default_context,
            fade_in_intro=True,
            active_tab='blog',
            ),
    )

default_context = dict(
    int=int,
    float=float,
    min=min,
    max=max,
    sorted=sorted,
    peval=peval,
    u=u,
    have_page_for_catpath=have_page_for_catpath,
    url_for_catpath=url_for_catpath,
    json=json,
    datetime=datetime,
    ipcc_canada=ipcc_canada,
    stakeholders=stakeholders,
    base=base,
    discount_rate=.02,
    )

