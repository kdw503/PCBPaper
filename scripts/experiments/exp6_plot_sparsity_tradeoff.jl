# scripts/figures/fig1_sparsity_tradeoff.jl
#
# Figure 1: Reconstruction error vs sparsity tradeoff curve
# Output: paper/figures/fig1_sparsity_tradeoff.pdf
#
# Run after analyze_exp1.jl

using DrWatson
@quickactivate "PCBPaper"

using JLD2, Plots, DataFrames

# ── Load processed data ───────────────────────────────────────────────────────
d = load(datadir("exp_pro", "exp1_summary.jld2"))
summary = d["summary"]

# ── Plot ─────────────────────────────────────────────────────────────────────
αs = summary.α

p1 = plot(αs, summary.mean_rel_err;
    ribbon     = summary.std_rel_err,
    xlabel     = "Sparsity penalty α",
    ylabel     = "Relative reconstruction error",
    xscale     = :log10,
    label      = "PCB",
    lw         = 2,
    fillalpha  = 0.2,
    legend     = :topleft,
    title      = "Accuracy vs Sparsity",
)

p2 = plot(αs, summary.mean_sparsity_W;
    xlabel  = "Sparsity penalty α",
    ylabel  = "Sparsity (fraction < 1e-2)",
    xscale  = :log10,
    label   = "W",
    lw      = 2,
)
plot!(p2, αs, summary.mean_sparsity_H; label="H", lw=2, ls=:dash)

fig = plot(p1, p2; layout=(1,2), size=(800, 350), dpi=300)

# ── Save ─────────────────────────────────────────────────────────────────────
mkpath(plotsdir())
mkpath(projectdir("paper", "figures"))

savefig(fig, plotsdir("fig1_sparsity_tradeoff.png"))          # quick preview
savefig(fig, projectdir("paper", "figures", "fig1_sparsity_tradeoff.pdf"))  # paper

@info "Saved → paper/figures/fig1_sparsity_tradeoff.pdf"
