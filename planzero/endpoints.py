from . import (
    get_peval,
    ipcc_canada,
    enums,
    blog,
    strategies,
    barriers
    )


def endpoints():
    rval = [
        "/",
        "/ipcc-sectors/",
        "/strategies/",
        "/scenarios/",
        "/about/",
    ]

    assert len(ipcc_canada.catpaths) == 71
    rval.extend([
        f"/ipcc-sectors/{catpath}/"
        for catpath in ipcc_canada.catpaths])

    for scenario in enums.StandardScenarios:
        rval.append(f"/scenarios/{scenario.value}/")
        
        for strategy in strategies.strategies:
            rval.append(f"/scenarios/{scenario.value}/strategies/{strategy}/")

        for barrier in barriers.barriers:
            rval.append(f"/scenarios/{scenario.value}/barriers/{barrier}/")

        for catpath in ipcc_canada.catpaths:
            rval.append(f"/scenarios/{scenario.value}/ipcc-sectors/{catpath}/")


    rval.extend([
        f"/blog/{url_filename}/"
        for url_filename in blog._blogs_by_url_filename])


    # TODO: deprecate this
    rval.extend([
        f"/strategies/{idea_name}/"
        for idea_name in get_peval().projects])

    return rval
