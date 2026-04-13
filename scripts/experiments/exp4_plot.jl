# scripts/experiments/exp4_plot.jl
#
# Re-draw exp4 figures from saved JLD2 data (no recomputation needed).
# Also saves W images per method using TestData.imsave_data.
#
# Run interactively:
#   julia> include("scripts/experiments/exp4_plot.jl")

using DrWatson
@quickactivate "PCBPaper"

using CairoMakie
using JLD2
using TestData
using LCSVD

# ── Load data ────────────────────────────────────────────────────────────────
d = load(projectdir("scripts", "data", "exp4_obj_convergence_randn.jld2"))

params        = d["params"]
method_labels = d["method_labels"]
method_names  = d["method_names"]
times_all     = d["times"]
fvals_all     = d["fvals"]
W_all         = d["W_all"]
H_all         = d["H_all"]
fv_all        = d["fv_all"]
m, n, p       = d["m"], d["n"], d["p"]

@unpack dataset, noc, k, αₘ, αₙ, σ₀, r, maxiter = params

skip_mtds = ["L1_AD_LBFGS", "L1_LBFGS", "L1_ADMM", "L1_FISTA", "rL1_SC_AD_SAGA", "rL1_SC_SAGA"]
skip_mtd_idx = Int[]
plot_mtd_lbl = String[]
for (i, (mtd,lbl)) in enumerate(zip(method_names,method_labels))
    mtd in skip_mtds && push!(skip_mtd_idx, i)
    mtd in skip_mtds || push!(plot_mtd_lbl, lbl)
end

# ── Convergence figure ────────────────────────────────────────────────────────
CairoMakie.activate!()

# 16 methods: 6 deterministic (solid) + 5 AD stochastic (dashed) + 5 Joint (dotted)
# AD and Joint pairs share the same hue (tab10[1..5]) for easy visual grouping.
_cm    = Makie.to_colormap(:tab10)
colors  = [_cm[1:6]; _cm[1:5]; _cm[1:5]]
#  "RelaxedL1_AD_LBFGS", "RelaxedL1_LBFGS", "L1_AD_LBFGS", "L1_LBFGS", "L1_ADMM", "L1_FISTA"
#  "rL1_SC_AD_SVRG", "rL1_SC_AD_SAGA", "rL1_SC_AD_LBFGS", "rL1_SC_AD_SGD", "rL1_SC_AD_ADAM"
#  "rL1_SC_SVRG", "rL1_SC_SAGA", "rL1_SC_LBFGS", "rL1_SC_SGD", "rL1_SC_ADAM"
markers = [:circle, :diamond, :rect, :hexagon, :utriangle, :dtriangle,
           :circle, :diamond, :rect, :hexagon, :utriangle,
           :circle, :diamond, :rect, :hexagon, :utriangle]
styles  = [:solid, :solid, :solid, :solid, :solid, :solid,
           :dash,  :dash,  :dash,  :dash,  :dash,
           :dot,   :dot,   :dot,   :dot,   :dot]

fig = Figure(size = (1050, 540))
yminlimit = minimum(minimum(fv) for fv in fvals_all) - 5
ymaxlimit = yminlimit + (maximum(last(fv) for fv in fvals_all) - yminlimit) * 1.5
# yminlimit = 10^2.881
# ymaxlimit = 10^2.886
markersize = 6
num_plot_methods = 16  # Only plot the first 11 methods for clarity

ax1 = Axis(fig[1, 1];
    xlabel         = "Wall-clock time (s)",
    ylabel         = "Exact L1 objective",
    title          = "Objective vs time",
    limits         = (nothing, (yminlimit, ymaxlimit)),
    yscale         = log10,
    xticklabelsize = 12,
    yticklabelsize = 12,
)

legend_entries = []
for (i, (times, fvals)) in enumerate(zip(times_all, fvals_all))
    i in skip_mtd_idx && continue  # Skip specified methods
    # ln = scatterlines!(ax1, times, fvals; color = colors[i], linewidth = 2,
    #                    linestyle = styles[i], marker = markers[i], markersize = markersize)
    ln = lines!(ax1, times, fvals; color = colors[i], linewidth = 2, linestyle = styles[i])
    push!(legend_entries, ln)
end

ax2 = Axis(fig[1, 2];
    xlabel         = "Outer iteration",
    ylabel         = "Exact L1 objective",
    title          = "Objective vs iteration",
    limits         = ((0, 300), (yminlimit, ymaxlimit)),
    yscale         = log10,
    xticklabelsize = 12,
    yticklabelsize = 12,
)

for (i, fvals) in enumerate(fvals_all)
    i in skip_mtd_idx && continue  # Skip specified methods
    iters = 0:length(fvals)-1
    # scatterlines!(ax2, iters, fvals; color = colors[i], linewidth = 2,
    #               linestyle = styles[i], marker = markers[i], markersize = markersize)
    lines!(ax2, iters, fvals; color = colors[i], linewidth = 2, linestyle = styles[i])
end

# σ threshold lines
y_top = maximum(maximum, fvals_all)
for (ε, ls) in [(1e-2, :dash), (1e-4, :dot)]
    t_thresh = ceil(Int, log(ε) / (2 * log(r)))
    lab = "σ→$(ε*100)%\n(iter $t_thresh)"
    vlines!(ax2, t_thresh; color = :gray, linestyle = ls, linewidth = 1.5)
    text!(ax2, t_thresh + 2, y_top; text = lab, fontsize = 9, color = :gray,
          align = (:left, :top))
end

# 4-row legend to fit 16 entries
Legend(fig[2, :], legend_entries, plot_mtd_lbl;
    orientation = :horizontal,
    nbanks      = 4,
    labelsize   = 11,
    tellwidth   = false,
)

Label(fig[0, :];
    text     = "PCB objective convergence  (dataset=$dataset, m=$m, n=$n, p=$p, k=$k, σ₀=$σ₀, r=$r, maxiter=$maxiter)",
    fontsize = 13,
    font     = :bold,
)

# ── Save figure ───────────────────────────────────────────────────────────────
figdir = projectdir("scripts", "figures")
mkpath(figdir)

save(joinpath(figdir, "exp4_obj_convergence_$(dataset).png"), fig; px_per_unit = 2)
#save(joinpath(figdir, "exp4_obj_convergence_$(dataset).pdf"), fig)

@info "Saved → scripts/figures/exp4_obj_convergence_$(dataset).png / .pdf"
