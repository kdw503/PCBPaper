# scripts/experiments/exp2_scalability.jl
#
# Experiment 2: Runtime scalability vs matrix size and number of components
#
# Run interactively:
#   julia> include("scripts/experiments/exp2_scalability.jl")
#
# Results saved to: data/sims/exp2/

using DrWatson
@quickactivate "PCBPaper"

using PCB
using LinearAlgebra, BenchmarkTools, JLD2

params = Dict(
    :sizes => [50, 100, 200, 500],   # matrix dimension m=n
    :ps    => [3, 5, 10],            # number of components
    :maxiter => 20,                  # small for benchmarking
)

function run_benchmark(params)
    @unpack sizes, ps, maxiter = params
    results = []
    for sz in sizes, p in ps
        X = rand(sz, sz)
        t = @belapsed pcb($X, $p; maxiter=$maxiter) samples=3 evals=1
        push!(results, Dict(:sz => sz, :p => p, :time_s => t))
        @info "sz=$sz p=$p → $(round(t; digits=3))s"
    end
    return results
end

results = run_benchmark(params)

fname = datadir("sims", "exp2", savename(params, "jld2"))
mkpath(dirname(fname))
@tagsave fname @strdict results params
@info "Saved → $fname"
