# intro.jl — run this first in any new Julia session
#
#   julia> include("intro.jl")
#
# This activates the project environment and greets you with the project info.

using DrWatson
@quickactivate "PCBPaper"

# Link to local PCB.jl package (dev mode)
# First time only: ] dev ../PenalizedComponentBlends.jl
using PenalizedComponentBlends

println("""
Project : PCBPaper
Path    : $(projectdir())
Data    : $(datadir())

Key directories:
  scripts/experiments/ → run simulations
  scripts/analysis/    → process results
  scripts/figures/     → generate paper figures
  data/sims/           → auto-saved simulation results
  data/exp_pro/        → processed results
  paper/figures/       → final figures for paper
""")
