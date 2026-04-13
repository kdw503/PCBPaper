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
d = load(projectdir("scripts", "data", "exp4_obj_convergence_fakecells.jld2"))
# d = load(projectdir("scripts", "data", "exp4_obj_convergence_fakecells.jld2"))

params        = d["params"]
method_labels = d["method_labels"]
method_names  = d["method_names"]
times_all     = d["times"]
fvals_all     = d["fvals"]
W_all         = d["W_all"]
H_all         = d["H_all"]
fv_all        = d["fv_all"]
m, n, p       = d["m"], d["n"], d["p"]

@unpack dataset, noc, factor, SNR, imgsz0, k, αₘ, αₙ, σ₀, r, maxiter = params
sqfactor = Int(floor(sqrt(factor)))
imgsz    = (sqfactor * imgsz0[1], sqfactor * imgsz0[2])

# ── Save W images per method ──────────────────────────────────────────────────
pcbfigdir = projectdir("scripts", "figures", "pcb")
mkpath(pcbfigdir)

dataset == :fakecells && for (i, (W1, H1, fv, mname)) in enumerate(zip(W_all, H_all, fv_all, method_names))
    iters = d["iters"][i]
    rt    = last(times_all[i])
    fprex = "pcb_$(mname)_$(SNR)db$(factor)f$(noc)sisvd"
    fname = joinpath(pcbfigdir, "$(fprex)_am$(αₘ)_an$(αₙ)_f$(fv)_it$(iters)_rt$(rt)")
    imsave_data(dataset, fname, W1, H1, imgsz, 100; saveH=false, verbose=false)
end

dataset == :fakecells && @info "Saved W images → scripts/figures/pcb/"

# ── Convergence figure ────────────────────────────────────────────────────────
CairoMakie.activate!()
colors  = Makie.wong_colors()
markers = [:circle, :diamond, :rect, :hexagon, :utriangle, :star5]

fig = Figure(size = (900, 480))
ymaxlimit = maximum(map(fvs->fvs[2],fvals_all))
yminlimit = minimum(map(fvs->minimum(fvs),fvals_all))-5

ax1 = Axis(fig[1, 1];
    xlabel         = "Wall-clock time (s)",
    ylabel         = "Exact L1 objective",
    title          = "Objective vs time",
    limits         = (nothing, (yminlimit,ymaxlimit)),
    yscale         = log10,
    xticklabelsize = 12,
    yticklabelsize = 12,
)

legend_entries = []
for (i, (times, fvals)) in enumerate(zip(times_all, fvals_all))
    ln = scatterlines!(ax1, times, fvals; color = colors[i], linewidth = 2, linestyle = styles[i], marker = markers[i], markersize = 8)
    push!(legend_entries, ln)
end

ax2 = Axis(fig[1, 2];
    xlabel         = "Outer iteration",
    ylabel         = "Exact L1 objective",
    title          = "Objective vs iteration",
    limits         = (nothing, (yminlimit,ymaxlimit)),
    yscale         = log10,
    xticklabelsize = 12,
    yticklabelsize = 12,
)

for (i, fvals) in enumerate(fvals_all)
    iters = 0:length(fvals)-1
    scatterlines!(ax2, iters, fvals; color = colors[i], linewidth = 2, linestyle = styles[i], marker = markers[i], markersize = 8)
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

Legend(fig[2, :], legend_entries, method_labels;
    orientation = :horizontal,
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
