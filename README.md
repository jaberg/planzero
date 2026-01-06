What might Canada's future look like in terms of technology deployment vs emissions reductions?
This repo drives [https://planzero.ca](planzero.ca) but it can also be used as
a library if you want to e.g. explore your own strategies privately.

# Contributing

## Research how the public problems of sectoral emissions relate to private problems

Pick an IPCC sector from [here](https://www.planzero.ca/ipcc-sectors/) that doesn't have analysis yet.

Click its curve on the stacked area chart, to trigger a page lookup.
I just did it and I see the url is
"https://www.planzero.ca/ipcc-sectors/Stationary_Combustion_Sources/Commercial_and_Institutional/".
If I pull out the end part without including the trailing slash, that's what
I've called a "catpath" (mnemonic: for CAT-egory PATH) in the source code, and it's
where you should put the analysis page for this IPCC sector within the source tree:
`html/ipcc-sectors/<catpath>.html` so e.g. `html/ipcc-sectors/Stationary_Combustion_Sources/Commercial_and_Institutional.html`.
Make that page (probably starting from another research page by copying it), and fill it with the results of your research.

The **definition-of-done** for such a page is that it breaks down the sources or
drivers of sectoral emissions in such a way as to turn it from a **public problem into
private problems**. The public problem is always the same: too much sectoral
emissions. The private problems are always different: the work is to identify
relevant stakeholders, the decisions they make, the interactions they have with one
another, the size of the stakeholder populations and so on, with enough granularity to do two things:

1. go and talk to these stakeholders to confirm that they do indeed make such
   decisions and have such interactions
1. Come up with ideas for emission reduction that resonate with these
   stakeholders


## Contribute ideas for emissions reduction

Have a read through any of the [IPCC-Sector pages](https://www.planzero.ca/ipcc-sectors/) that have analysis
and think about what might be done to reduce those emissions.
Lots of ideas have already been thought of, and more come up every day as
technology possibilities continue to evolve.

Submit the idea ideally in one of two ways:
1. [Github Issue](https://github.com/jaberg/planzero/issues?q=state%3Aopen%20label%3A%22Emission%20Reduction%20Idea%22) tagged as "Emissions Reduction Idea"
2. Directly as an idea in the `planzero/strategies/ideas.py` file. See that
   file for examples.

The **definition-of-done** for these ideas is that their cost or financial
feasibility and environmental impact can be modelled reasonably.
The modelling itself is the next step. This definition of done is vague, but
if no one can see how to model an idea, then it needs more work.


## Model a strategy for addressing emissions

Pick an idea for emissions reduction, and modify the Python simulation
implemented in the `planzero` directory to be able to run with and without
a `Project` that implements the strategy in terms of simulation variables.


The **definition-of-done** for a strategy model is that leverages the best
available information and serves to support
a decision about whether to implement that strategy. Does it convince
a company to start an internal project, a project developer to invest time and
effort, an entrepreneur to start a business, a customer to buy differently,
a policy-maker to draft new policy?
If not, why is it not effective and what might make the model more compelling?

## Get the Message Out, Reduce Emissions

The types of contribution above are about funneling information and feedback
*into* the model, but the purpose of the model is to focus emission reduction
efforts in the world. For yourself: start a business, change your business, buy differently,
live differently, etc. Beyond yourself: use this model in your own work, tell
others about it. And if there are changes to this project that would make it
more useful or useable, let us know! (via e.g. github issue to start)

