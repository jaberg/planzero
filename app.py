import json
import datetime

from fastapi import FastAPI, Request
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

planzero.blog.init_blogs_by_url_filename()
u = planzero.ureg

peval = planzero.standard_project_evaluation()
peval.run_until(2125 * u.years)


@app.get("/strategies/{strategy_name}/", response_class=HTMLResponse)
async def get_strategy_eval(request: Request, strategy_name:str):

    strategy = peval.comparisons[strategy_name].project
    comparison = peval.comparisons[strategy_name]
    strategy_page = strategy.strategy_page(comparison)
    return templates.TemplateResponse(
        request=request,
        name=f"strategy_page.html",
        context=dict(
            default_context,
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
                stakeholders=planzero.strategies.stakeholders,
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
async def get_blog(request: Request, post_name:str):
    blog = planzero.blog._blogs_by_url_filename.get(post_name)
    try:
        return templates.TemplateResponse(
            request=request,
            name=f"/blog/{post_name}.html",
            context=dict(
                default_context,
                active_tab='blog',
                blog=blog,
                ),
        )
    except IOError:
        return templates.TemplateResponse(
            request=request,
            name=f"/blog/blog_template.html",
            context=dict(
                default_context,
                active_tab='blog',
                blog=blog,
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

if 0:
    @app.get("/foo/", response_class=HTMLResponse)
    async def get_about(request: Request):
        return templates.TemplateResponse(
            request=request,
            name="foo.html",
            context=dict(
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
    ipcc_canada=planzero.ipcc_canada,
    stakeholders=planzero.strategies.stakeholders,
    discount_rate=.02,
    planzero=planzero,
    )

