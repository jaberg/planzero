from . import (
    ipcc_canada,
    enums,
    blog,
    strategies,
    barriers,
    sim,
    )


def endpoints():
    rval = []

    for sim_name, site_sim in sorted(sim.site_simulations.items()):
        rval.append(f"/simulations/{sim_name}/")
        
        for dynelem in site_sim.dynamic_elements():
            if 'strategy' in dynelem.tags:
                rval.append(f"/simulations/{sim_name}/strategies/{dynelem.identifier}/")

            if 'barrier' in dynelem.tags:
                rval.append(f"/simulations/{sim_name}/barriers/{dynelem.identifier}/")

        for catpath in ipcc_canada.catpaths:
            rval.append(f"/simulations/{sim_name}/ipcc-sectors/{catpath}/")

    rval.extend([
        "/",
        "/ipcc-sectors/",
        "/strategies/",
        "/simulations/",
        "/glossary/",
        "/about/",
    ])

    assert len(ipcc_canada.catpaths) == 71
    rval.extend([
        f"/ipcc-sectors/{catpath}/"
        for catpath in ipcc_canada.catpaths])


    rval.extend([
        f"/blog/{url_filename}/"
        for url_filename in blog._blogs_by_url_filename])

    return rval
