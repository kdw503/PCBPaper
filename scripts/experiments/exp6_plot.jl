# scripts/experiments/exp6_plot.jl
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
dataset = :natural; αₙ=5e-1; inner_maxiter=5000
fname="exp6_sparse_coding_$(dataset)_ah$(αₙ)_imiter$(inner_maxiter)_long_nstep.jld2"
d = load(projectdir("scripts", "data", fname))

params        = d["params"]
method_labels = d["method_labels"]
times_all     = d["times"]
fvals_all     = d["fvals"]
iters_all     = d["iters"]
W_all         = d["W_all"]
H_all         = d["H_all"]
fv_all        = d["fv_all"]
m, n, p       = d["m"], d["n"], d["p"]

@unpack dataset, imgsz, lengthT, noc, k, αₘ, αₙ, σ₀, r, maxiter, inner_maxiter, tol, inner_tol = params

# ── Save W images ─────────────────────────────────────────────────────────────
figdir = projectdir("scripts", "figures", "pcb")
mkpath(figdir)

method_tags = [
    "rL1 AD-LBFGS",
#    "rL1 LBFGS",
    "AD-SGD",
    "AD-ADAM",
    "Joint-SGD",
#    "Joint-ADAM",
]
for (i, (iters,times,fv,W,H,tag)) in enumerate(zip(iters_all,times_all,fv_all,W_all,H_all,method_tags))
    rt    = last(times)
    fname = joinpath(figdir, "exp6_$(tag)_am$(αₘ)_an$(αₙ)_f$(fv)_it$(iters)_rt$(round(rt;digits=1))")
    imsave_data(dataset, fname, W, H, imgsz, 100; saveH=false, verbose=false)
end
@info "Saved W images → scripts/figures/pcb/"

# ── Convergence figure ────────────────────────────────────────────────────────
CairoMakie.activate!()
colors  = Makie.wong_colors()
markers = [:circle, :diamond, :rect, :utriangle, :xcross, :star5]
styles  = [:solid, :dash, :solid, :dash, :solid, :dash]
yminlimit = minimum(minimum(fv) for fv in fvals_all) - 5
ymaxlimit = yminlimit + (maximum(last(fv) for fv in fvals_all) - yminlimit) * 1.5
# yminlimit = 10^2.881
# ymaxlimit = 10^2.886

fig = Figure(size = (900, 480));

ax1 = Axis(fig[1, 1];
    xlabel         = "Wall-clock time (s)",
    ylabel         = "Exact L1 objective",
    title          = "Objective vs time",
    yscale         = log10,
    limits         = (nothing, nothing),
#    limits         = (nothing, (yminlimit, ymaxlimit)),
    xticklabelsize = 12,
    yticklabelsize = 12,
)

legend_entries = []
for (i, (times,fvals)) in enumerate(zip(times_all,fvals_all))
    ln = lines!(ax1, times, fvals; color = colors[i], linewidth = 2, linestyle = styles[i])
    push!(legend_entries, ln)
end

ax2 = Axis(fig[1, 2];
    xlabel         = "Outer iteration",
    ylabel         = "Objective value",
    title          = "Objective vs iteration",
    yscale         = log10,
    limits         = (nothing, nothing),
#    limits         = (nothing, (yminlimit, ymaxlimit)),
    xticklabelsize = 12,
    yticklabelsize = 12,
)

for (i, fvals) in enumerate(fvals_all)
    iters = 0:length(fvals)-1
    lines!(ax2, iters, fvals; color = colors[i], linewidth = 2, linestyle = styles[i])
end

Legend(fig[2, :], legend_entries, method_labels;
    orientation = :horizontal,
    labelsize   = 11,
    tellwidth   = false,
)

Label(fig[0, :];
    text     = "PCB vs LCSVD  (dataset=$dataset, m=$m, n=$n, p=$p, k=$k, σ₀=$σ₀, r=$r)",
    fontsize = 13,
    font     = :bold,
)

# ── Save figure ────────────────────────────────────────────────────────
convfigdir = projectdir("scripts", "figures")
mkpath(convfigdir)
save(joinpath(convfigdir, "exp6_sparse_coding.png"), fig; px_per_unit = 2)
#save(joinpath(convfigdir, "exp6_sparse_coding.pdf"), fig)
@info "Saved → scripts/figures/exp6_sparse_coding.png / .pdf"
