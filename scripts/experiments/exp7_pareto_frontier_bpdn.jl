# scripts/experiments/exp4_obj_convergence.jl
#
# Experiment 4: Objective value decrease over time — all five PCB methods
#
# Run interactively:
#   julia> include("scripts/experiments/exp4_obj_convergence.jl")
#
# Figures saved to: scripts/figures/exp4_obj_convergence.png / .pdf

using DrWatson
@quickactivate "PCBPaper"

using PenalizedComponentBlends
using LinearAlgebra, Random
using CairoMakie
using JLD2
using TestData
using LCSVD

if Sys.iswindows()
    datapath="C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca\\paper\\sparse_coding"
elseif Sys.isunix()
    datapath=ENV["MYSTORAGE"]*"/work/julia/sca/paper/sparse_coding"
end
datadir_ = projectdir("scripts", "data")
mkpath(datadir_)
figdir = projectdir("scripts", "figures", "bpdn")
mkpath(figdir)

# ── Parameters ───────────────────────────────────────────────────────────────
params = Dict(
    :method        => :BPDN,
    :dataset       => :natural,
    :imgsz         => (12, 12),
    :lengthT       => 100000,
    :noc           => 72,
    :maxiter       => 100, # 100,
    :tol           => 1e-10,
    :lpowrng       => -1:0.2:1, # -1:0.5:5
)

# ── Data ─────────────────────────────────────────────────────────────────────
@unpack method, dataset, imgsz, lengthT, noc, maxiter, tol, lpowrng = params

X = load(joinpath(datapath, "X_whitened_Hspar","natural_SC_l3.0_iter50.jld2"),"X_whitened")

# ── BPDN ─────────────────────────────────────────────────────────────────
results = map(lpowrng) do lpow
    @info "BPDN with λ=$(lpow)"
    λ = 10.0^lpow
    W, H = LCSVD.sparse_coding(X, noc, λ, Dict(); initmethod=:randcolX, max_iter=maxiter, lr=1e-1)
    fv = LCSVD.fitd(X, W*H)
    fname="exp7_pareto_frontier_BPDN_lpow$(lpow)"
    LCSVD.normalizeW!(W,H)
    imsave_data(dataset, joinpath(figdir, fname*".png"), W, H, imgsz, 100; saveH=false, verbose=false)
    LCSVD.normalizeWH!(W,H); sparH = norm(H,1)
    jldsave(joinpath(datadir_, fname*".jld2"); lpow, λ, fv, sparH)
    (W=W, H=H, fv=fv, sparH=sparH)
end

# ── Save data ─────────────────────────────────────────────────────────────────

fname="exp7_pareto_frontier_bpdn.jld2"
jldsave(joinpath(datadir_, fname);
    params,
    method_name  = string(method),
    fvs    = [r.fv  for r in results],
    sparHs = [r.sparH  for r in results],
    W_all  = [r.W  for r in results],
    H_all  = [r.H  for r in results],
)

@info "Saved → $(fname)"
