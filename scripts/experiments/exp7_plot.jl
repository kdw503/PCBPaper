# scripts/experiments/exp7_plot.jl
#
# Re-draw exp5 figures from saved JLD2 data (no recomputation needed).
# Also saves W images per method using TestData.imsave_data.
#
# Run interactively:
#   julia> include("scripts/experiments/exp5_plot.jl")

using DrWatson
@quickactivate "PCBPaper"

using CairoMakie
using JLD2
using TestData
using LCSVD

# ── Load data ────────────────────────────────────────────────────────────────
method_labels = []

fname="exp7_pareto_frontier_bpdn.jld2"
d = load(projectdir("scripts", "data", fname))
params        = d["params"]
fvs_bpdn      = d["fvs"]
sparHs_bpdn   = d["sparHs"]
push!(method_labels, d["method_name"])
@unpack dataset, imgsz, lengthT, noc, maxiter, tol, lpowrng = params
maxiter_bpdn = maxiter

fname="exp7_pareto_frontier_pcb.jld2"
d = load(projectdir("scripts", "data", fname))
params        = d["params"]
fvs_pcb       = d["fvs"]
sparHs_pcb    = d["sparHs"]
push!(method_labels, d["method_name"])
@unpack dataset, imgsz, lengthT, noc, k, αₘ, αₙ, σ₀, r, maxiter, inner_maxiter, tol, inner_tol, apowrng = params
maxiter_pcb = maxiter

# ── Convergence figure ────────────────────────────────────────────────────────
CairoMakie.activate!()
colors  = Makie.wong_colors()
markers = [:circle, :xcross]

fig = Figure(size = (900, 600));

ax1 = Axis(fig[1, 1];
    xlabel         = "L1 norm",
    ylabel         = "Fit value",
    title          = "",
    yscale         = identity,
    limits         = (nothing, nothing),
    xticklabelsize = 12,
    yticklabelsize = 12,
)

legend_entries = []
sc1 = scatter!(ax1, sparHs_bpdn, fvs_bpdn; color = colors[1], marker = markers[1])
push!(legend_entries, sc1)
text!(ax1, sparHs_bpdn, fvs_bpdn;
    text   = string.(collect(lpowrng)),
    offset = (4, 4),
    fontsize = 9,
    color  = colors[1],
)

sc2 = scatter!(ax1, sparHs_pcb, fvs_pcb; color = colors[2], marker = markers[2])
push!(legend_entries, sc2)
text!(ax1, sparHs_pcb, fvs_pcb;
    text   = string.(collect(apowrng)),
    offset = (4, 4),
    fontsize = 9,
    color  = colors[2],
)

Legend(fig[2, :], legend_entries, method_labels;
    orientation = :horizontal,
    labelsize   = 11,
    tellwidth   = false,
)

Label(fig[0, :];
    text     = "Sparse coding Pareto frontier (dataset=$dataset, inner_maxiter=$inner_maxiter, inner_tol=$inner_tol, p=$noc, k=$k, σ₀=$σ₀, r=$r)",
    fontsize = 13,
    font     = :bold,
)

# ── Save figure ────────────────────────────────────────────────────────
convfigdir = projectdir("scripts", "figures")
mkpath(convfigdir)
save(joinpath(convfigdir, "exp7_pareto_frontier.png"), fig; px_per_unit = 2)
#save(joinpath(convfigdir, "exp6_sparse_coding.pdf"), fig)
@info "Saved → scripts/figures/exp7_pareto_frontier.png / .pdf"
