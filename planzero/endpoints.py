from . import (
    get_peval,
    ipcc_canada,
    enums,
    blog,
    strategies,
    barriers,
    sim,
    )


def endpoints():
    rval = [
        "/",
        "/ipcc-sectors/",
        "/strategies/",
        "/simulations/",
        "/glossary/",
        "/about/",
    ]

    assert len(ipcc_canada.catpaths) == 71
    rval.extend([
        f"/ipcc-sectors/{catpath}/"
        for catpath in ipcc_canada.catpaths])

    for sim_name, site_sim in sim.site_simulations.items():
        rval.append(f"/simulations/{sim_name}/")
        
        for dynelem in site_sim.dynamic_elements():
            if 'strategy' in dynelem.tags:
                rval.append(f"/simulations/{sim_name}/strategies/{dynelem.identifier}/")

            if 'barrier' in dynelem.tags:
                rval.append(f"/simulations/{sim_name}/barriers/{dynelem.identifier}/")

        for catpath in ipcc_canada.catpaths:
            rval.append(f"/simulations/{sim_name}/ipcc-sectors/{catpath}/")


    rval.extend([
        f"/blog/{url_filename}/"
        for url_filename in blog._blogs_by_url_filename])


    # TODO: deprecate this
    rval.extend([
        f"/strategies/{idea_name}/"
        for idea_name in get_peval().projects])

    return rval
