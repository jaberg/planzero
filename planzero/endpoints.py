import os

HOME_SHOW_UNPUBLISHED_POSTS = (os.environ['PLANZERO_HOME_SHOW_UNPUBLISHED_POSTS'] == '1')

def completed_prob_registry():
    from . import (
            challenge,  # noqa: F401   # HACK to ensure loaded
            nir2025_site,
            #nir_ar2_site,
            nir_static_normals_site,
            scaling_site,
    )
    rval = {}

    nir = nir2025_site.NIR2025()
    rval[nir.url_name] = nir

    static_normals = nir_static_normals_site.Static_Normals_2024_12_31()
    rval[static_normals.url_name] = static_normals

    scaling = scaling_site.ScalingSiteInference()
    rval[scaling.url_name] = scaling

    return rval


def endpoints():
    from . import (
            blog,
            challenge,  # noqa: F401
            ipcc_canada,
            sim,
    )

    prob_registry = completed_prob_registry()

    rval = []

    rval.extend([
        f"/post/{url_filename}/"
        for url_filename, post in blog._blogs_by_url_filename.items()
        if post.published or HOME_SHOW_UNPUBLISHED_POSTS
    ])

    for sim_name, site_sim in sorted(sim.site_simulations.items()):
        rval.append(f"/models/sim/{sim_name}/")
        
        for dynelem in site_sim.dynamic_elements():
            if 'strategy' in dynelem.tags:
                rval.append(f"/models/sim/{sim_name}/strategies/{dynelem.identifier}/")

            if 'barrier' in dynelem.tags:
                rval.append(f"/models/sim/{sim_name}/barriers/{dynelem.identifier}/")

        for catpath in ipcc_canada.catpaths:
            rval.append(f"/models/sim/{sim_name}/ipcc-sectors/{catpath}/")

    for url_name, site_inf in sorted(prob_registry.items()):
        if not site_inf.show_on_models_page:
            continue
        rval.append(f"/models/prob/{url_name}/")

        for catpath in ipcc_canada.catpaths:
            rval.append(f"/models/prob/{url_name}/sectors/{catpath}/")

        for strategy_name in site_inf.strategies:
            rval.append(f"/models/prob/{url_name}/strategies/{strategy_name}/")

        for barrier_name in site_inf.barriers:
            rval.append(f"/models/prob/{url_name}/barriers/{barrier_name}/")

    rval.extend([
        "/",
        "/ipcc-sectors/",
        "/models",
        "/model-explorer/",
        "/metrics/",
        "/strategies/",
        "/glossary/",
        "/about/",
    ])

    assert len(ipcc_canada.catpaths) == 71
    rval.extend([
        f"/ipcc-sectors/{catpath}/"
        for catpath in ipcc_canada.catpaths])


    return rval
