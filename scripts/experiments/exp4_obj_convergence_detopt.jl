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

# ── Parameters ───────────────────────────────────────────────────────────────
const METHODS = [
    :RelaxedL1_AD_LBFGS,
    :RelaxedL1_LBFGS,
    :L1_AD_LBFGS,
    :L1_LBFGS,
    :L1_ADMM,
    :L1_FISTA,
]

const METHOD_LABELS = [
    "rL1 AD-LBFGS",
    "rL1 LBFGS",
    "L1 AD-LBFGS",
    "L1 LBFGS",
    "L1 ADMM",
    "L1 FISTA",
]

params = Dict(
    :dataset       => :randn,
    :noc           => 15,
    :factor        => 1,
    :SNR           => 0,
    :inhibitindices => 0,
    :bias          => 0.1,
    :imgsz0        => (40, 20),
    :k             => 15,  # SVD rank
    :αₘ            => 1e-2,
    :αₙ            => 1e-2,
    :σ₀            => 2.0,
    :r             => 0.95,
    :maxiter       => 300,
    :inner_maxiter => 500,
    :tol           => 1e-12,
)

# ── Data ─────────────────────────────────────────────────────────────────────
@unpack dataset, noc, factor, SNR, inhibitindices, bias, imgsz0, k,
        αₘ, αₙ, σ₀, r, maxiter, inner_maxiter, tol = params

sqfactor = Int(floor(sqrt(factor)))
imgsz    = (sqfactor * imgsz0[1], sqfactor * imgsz0[2])
lengthT  = factor * 1000
sigma    = sqfactor * 5.0
if dataset == :fakecells
    @info "Dataset: $(dataset) → $(noc) cells, imgsz = $(imgsz0), factor = $(factor)"
    X, imsz, lhT, ncs, gtnoc, datadic = load_data(dataset;
        sigma=sigma, imgsz=imgsz, lengthT=lengthT, SNR=SNR, bias=bias,
        useCalciumT=true, inhibitindices=inhibitindices,
        issave=false, isload=false, gtincludebg=false,
        save_gtimg=false, save_maxSNR_X=false, save_X=false)
elseif dataset == :randn
    @info "Dataset: $(dataset)"
    X = randn(*(imgsz...), lengthT)
end

(m, n, p) = (size(X)..., noc)
gtW, gtH  = dataset == :fakecells ?
    (datadic["gtW"], datadic["gtH"]) :
    (Matrix{eltype(X)}(undef,0,0), Matrix{eltype(X)}(undef,0,0))

figdir = projectdir("scripts", "figures", "pcb")
mkpath(figdir)

results = map(METHODS) do method
    res = pcb(X, p, k;
        pcb_method    = method,
        αₘ            = αₘ,
        αₙ            = αₙ,
        σ₀            = σ₀,
        r             = r,
        maxiter       = maxiter,
        inner_maxiter = inner_maxiter,
        tol           = tol,
        track_history = true,
    )
    @info "$(rpad(string(method), 22)) → $(res.iterations) iters, " *
          "final fval = $(round(last(res.history.fvals); sigdigits=4))"
    W1, H1 = res.W, res.H
    LCSVD.normalizeW!(W1, H1)
    if dataset == :fakecells
        fv, ml, merrval, rerrs = LCSVD.matchedfitval(gtW, gtH, W1, H1; clamp=false)
        nodr = LCSVD.matchedorder(ml, noc)
        W1, H1 = W1[:, nodr], H1[nodr, :]
    else
        fv = LCSVD.fitd(X, W1 * H1)
    end
    (res=res, W=W1, H=H1, fv=fv)
end

# ── Save data ─────────────────────────────────────────────────────────────────
datadir_ = projectdir("scripts", "data")
mkpath(datadir_)

jldsave(joinpath(datadir_, "exp4_obj_convergence_$(dataset).jld2");
    params,
    m, n, p,
    method_names  = string.(METHODS),
    method_labels = METHOD_LABELS,
    times  = [r.res.history.times for r in results],
    fvals  = [r.res.history.fvals for r in results],
    iters  = [r.res.iterations    for r in results],
    W_all  = [r.W                 for r in results],
    H_all  = [r.H                 for r in results],
    fv_all = [r.fv                for r in results],
)

@info "Saved → scripts/data/exp4_obj_convergence_$(dataset).jld2"
