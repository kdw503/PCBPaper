# scripts/analysis/analyze_exp1.jl
#
# Load exp1 simulation results and produce a summary DataFrame.
# Output saved to: data/exp_pro/exp1_summary.jld2
#
# Run after exp1_sparsity_vs_accuracy.jl

using DrWatson
@quickactivate "PCBPaper"

using JLD2, DataFrames, Statistics

# ── Load all exp1 results ────────────────────────────────────────────────────
files = readdir(datadir("sims", "exp1"); join=true)
all_results = []
for f in filter(endswith(".jld2"), files)
    d = load(f)
    append!(all_results, d["results"])
end

df = DataFrame(all_results)

# ── Summarise by α ───────────────────────────────────────────────────────────
summary = combine(groupby(df, :α),
    :rel_err    => mean => :mean_rel_err,
    :rel_err    => std  => :std_rel_err,
    :sparsity_W => mean => :mean_sparsity_W,
    :sparsity_H => mean => :mean_sparsity_H,
    :converged  => mean => :conv_rate,
)

println(summary)

# ── Save processed result ────────────────────────────────────────────────────
mkpath(datadir("exp_pro"))
wsave(datadir("exp_pro", "exp1_summary.jld2"), @strdict summary df)
@info "Saved summary → $(datadir("exp_pro", "exp1_summary.jld2"))"
