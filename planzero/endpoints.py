from . import (
    ipcc_canada,
    enums,
    blog,
    strategies,
    barriers,
    sim,
    prob,
    )


def endpoints():
    rval = []

    rval.extend([
        f"/blog/{url_filename}/"
        for url_filename in blog._blogs_by_url_filename])


    for sim_name, site_sim in sorted(sim.site_simulations.items()):
        rval.append(f"/models/sim/{sim_name}/")
        
        for dynelem in site_sim.dynamic_elements():
            if 'strategy' in dynelem.tags:
                rval.append(f"/models/sim/{sim_name}/strategies/{dynelem.identifier}/")

            if 'barrier' in dynelem.tags:
                rval.append(f"/models/sim/{sim_name}/barriers/{dynelem.identifier}/")

        for catpath in ipcc_canada.catpaths:
            rval.append(f"/models/sim/{sim_name}/ipcc-sectors/{catpath}/")

    for model_name, site_inf in sorted(prob.site_inferences.items()):
        rval.append(f"/models/prob/{model_name}/")

        for catpath in ipcc_canada.catpaths:
            rval.append(f"/models/sim/{sim_name}/sectors/{catpath}/")

    rval.extend([
        "/",
        "/ipcc-sectors/",
        "/strategies/",
        "/models",
        "/glossary/",
        "/about/",
    ])

    assert len(ipcc_canada.catpaths) == 71
    rval.extend([
        f"/ipcc-sectors/{catpath}/"
        for catpath in ipcc_canada.catpaths])


    return rval
