# scripts/experiments/exp1_sparsity_vs_accuracy.jl
#
# Experiment 1: Reconstruction accuracy vs sparsity penalty
# Sweeps αₘ = αₙ over a grid and records reconstruction error + sparsity.
#
# Run interactively:
#   julia> include("scripts/experiments/exp1_sparsity_vs_accuracy.jl")
#
# Results saved to: data/sims/exp1/

using DrWatson
@quickactivate "PCBPaper"

using PCB
using LinearAlgebra, Statistics, JLD2

# ── Experiment parameters ────────────────────────────────────────────────────
params = Dict(
    :m      => 50,           # rows of X
    :n      => 40,           # cols of X
    :p_true => 5,            # true rank
    :p      => 5,            # components to fit
    :αs     => [0.0, 1e-3, 1e-2, 1e-1, 1.0],  # sparsity grid
    :n_reps => 10,           # repetitions per α (random X)
    :maxiter => 100,
    :seed   => 42,
)

# ── Helper ───────────────────────────────────────────────────────────────────
sparsity(A; thresh=1e-2) = mean(abs.(A) .< thresh)

function run_experiment(params)
    @unpack m, n, p_true, p, αs, n_reps, maxiter, seed = params

    results = []
    for α in αs
        for rep in 1:n_reps
            # Random low-rank data
            rng_seed = seed + rep
            W_gt = randn(m, p_true)
            H_gt = randn(p_true, n)
            X    = W_gt * H_gt + 0.01 * randn(m, n)

            result = pcb(X, p; αₘ=α, αₙ=α, maxiter)

            push!(results, Dict(
                :α          => α,
                :rep        => rep,
                :rel_err    => norm(result.W * result.H - X) / norm(X),
                :sparsity_W => sparsity(result.W),
                :sparsity_H => sparsity(result.H),
                :converged  => result.converged,
                :iterations => result.iterations,
            ))
        end
        @info "α=$α done ($(n_reps) reps)"
    end
    return results
end

# ── Run & save ───────────────────────────────────────────────────────────────
results = run_experiment(params)

# savename auto-generates filename from params dict
fname = datadir("sims", "exp1", savename(params, "jld2"))
mkpath(dirname(fname))
@tagsave fname @strdict results params
@info "Saved → $fname"
