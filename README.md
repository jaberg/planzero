[![Tests](https://github.com/jaberg/planzero/actions/workflows/test.yaml/badge.svg)](https://github.com/jaberg/planzero/actions/workflows/test.yaml)

What might Canada's future look like in terms of technology deployment vs emissions reductions?
This repo drives [https://planzero.ca](planzero.ca).

# How to use this repo locally

1. Configure git on your computer, clone this repo.
1. Get docker or equivalent.
1. Run `make tmux` or look at the tmux target in the Makefile and run that command.
1. Optionally, configure your own development tools and environment, fork the repo, contribute pull-requests on GitHub, etc.

The Dockerfile contains a development image with the programs and configuration required for development, at least as I develop.
If you want to follow my example, configure neovim and tmux locally on your computer, and then that `make tmux` command will link
your local neovim, tmux, and git configurations into the development environment and away you go.
