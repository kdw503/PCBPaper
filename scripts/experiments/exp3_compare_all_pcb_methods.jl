# scripts/experiments/exp3_compare_all_pcb_methods.jl
#
# Experiment 3: Runtime comparison of all five PCB methods
#
#   Methods compared
#   ─────────────────────────────────────────────────────
#   :RelaxedL1_AD_LBFGS   relaxed L1, alternating direction L-BFGS
#   :RelaxedL1_LBFGS      relaxed L1, joint L-BFGS
#   :L1_AD_LBFGS          exact L1, alternating direction L-BFGS
#   :L1_ADMM              exact L1, ADMM inner solver
#   :L1_FISTA             exact L1, FISTA inner solver
#
# Run interactively:
#   julia> include("scripts/experiments/exp3_compare_all_pcb_methods.jl")
#
# Figures saved to: scripts/figures/exp3_compare_all_pcb_methods.png / .pdf

using DrWatson
@quickactivate "PCBPaper"

using PenalizedComponentBlends
using LinearAlgebra, Statistics, Random
using CairoMakie

# ── Experiment parameters ────────────────────────────────────────────────────
const METHODS = [
    :RelaxedL1_AD_LBFGS,
    :RelaxedL1_LBFGS,
    :L1_AD_LBFGS,
    :L1_ADMM,
    :L1_FISTA,
]

const METHOD_LABELS = [
    "rL1\nAD-LBFGS",
    "rL1\nLBFGS",
    "L1\nAD-LBFGS",
    "L1\nADMM",
    "L1\nFISTA",
]

params = Dict(
    :sizes         => [50, 100, 200],
    :p             => 5,
    :αₘ            => 1e-2,
    :αₙ            => 1e-2,
    :maxiter       => 30,
    :inner_maxiter => 500,
    :n_reps        => 5,
    :seed          => 42,
)

# ── Timing helper ────────────────────────────────────────────────────────────
function time_method(method, X, p; αₘ, αₙ, maxiter, inner_maxiter, n_reps)
    times = Vector{Float64}(undef, n_reps)
    for r in 1:n_reps
        times[r] = @elapsed pcb(X, p;
            pcb_method    = method,
            αₘ            = αₘ,
            αₙ            = αₙ,
            maxiter       = maxiter,
            inner_maxiter = inner_maxiter,
        )
    end
    return median(times)
end

# ── Run experiment ───────────────────────────────────────────────────────────
function run_experiment(params)
    @unpack sizes, p, αₘ, αₙ, maxiter, inner_maxiter, n_reps, seed = params

    # times[method_idx, size_idx] = median runtime (s)
    times = zeros(length(METHODS), length(sizes))

    for (si, sz) in enumerate(sizes)
        rng  = MersenneTwister(seed)
        W_gt = randn(rng, sz, p)
        H_gt = randn(rng, p, sz)
        X    = W_gt * H_gt + 0.01 * randn(rng, sz, sz)

        @info "Matrix size $sz × $sz"
        for (mi, method) in enumerate(METHODS)
            t = time_method(method, X, p; αₘ, αₙ, maxiter, inner_maxiter, n_reps)
            times[mi, si] = t
            @info "  $(rpad(string(method), 22)) → $(round(t * 1000; digits=1)) ms"
        end
    end

    return times
end

times = run_experiment(params)
times_ms = times .* 1000   # → milliseconds

# ── Figure ───────────────────────────────────────────────────────────────────
sizes      = params[:sizes]
n_methods  = length(METHODS)
n_sizes    = length(sizes)

colors  = Makie.wong_colors()   # colour-blind friendly palette
markers = [:circle, :diamond, :rect, :utriangle, :star5]

fig = Figure(size = (900, 400))

# ── Panel 1: line plot — runtime vs matrix size ───────────────────────────
ax1 = Axis(fig[1, 1];
    xlabel         = "Matrix size (m = n)",
    ylabel         = "Median runtime (ms)",
    title          = "Runtime vs matrix size",
    xticks         = sizes,
    xticklabelsize = 12,
    yticklabelsize = 12,
)

for mi in 1:n_methods
    lines!(ax1, sizes, times_ms[mi, :];
        color     = colors[mi],
        linewidth = 2,
        label     = replace(METHOD_LABELS[mi], "\n" => " "),
    )
    scatter!(ax1, sizes, times_ms[mi, :];
        color      = colors[mi],
        marker     = markers[mi],
        markersize = 10,
    )
end

axislegend(ax1; position = :lt, labelsize = 11)

# ── Panel 2: bar chart at largest matrix size ─────────────────────────────
sz_last  = last(sizes)
si_last  = n_sizes
bar_vals = times_ms[:, si_last]

ax2 = Axis(fig[1, 2];
    xlabel         = "Method",
    ylabel         = "Median runtime (ms)",
    title          = "Runtime at m = n = $sz_last",
    xticks         = (1:n_methods, METHOD_LABELS),
    xticklabelsize = 11,
    yticklabelsize = 12,
)

barplot!(ax2, 1:n_methods, bar_vals;
    color      = colors[1:n_methods],
    strokecolor = :white,
    strokewidth = 1,
)

# ── Shared title ──────────────────────────────────────────────────────────
Label(fig[0, :];
    text      = "PCB method speed comparison  (p=$(params[:p]), maxiter=$(params[:maxiter]))",
    fontsize  = 13,
    font      = :bold,
)

# ── Save ─────────────────────────────────────────────────────────────────────
figdir = projectdir("scripts", "figures")
mkpath(figdir)

save(joinpath(figdir, "exp3_compare_all_pcb_methods.png"), fig; px_per_unit = 2)
save(joinpath(figdir, "exp3_compare_all_pcb_methods.pdf"), fig)

@info "Saved → scripts/figures/exp3_compare_all_pcb_methods.png / .pdf"
