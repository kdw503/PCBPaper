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
    # # :L1_AD_LBFGS,
    # # :L1_LBFGS,
    # # :L1_ADMM,
    # # :L1_FISTA,
    :rL1_SC_AD_SVRG,
    :rL1_SC_AD_SAGA,
    :rL1_SC_AD_LBFGS,
    :rL1_SC_AD_SGD,
    :rL1_SC_AD_ADAM,
    :rL1_SC_SVRG,
    :rL1_SC_SAGA,
    :rL1_SC_LBFGS,
    :rL1_SC_SGD,
    :rL1_SC_ADAM,
]

const METHOD_LABELS = [
    "rL1 AD-LBFGS",
    "rL1 LBFGS",
    # # "L1 AD-LBFGS",
    # # "L1 LBFGS",
    # # "L1 ADMM",
    # # "L1 FISTA",
    "AD-SVRG",
    "AD-SAGA",
    "AD-LBFGS",
    "AD-SGD",
    "AD-ADAM",
    "Joint-SVRG",
    "Joint-SAGA",
    "Joint-LBFGS",
    "Joint-SGD",
    "Joint-ADAM",
]

params = Dict(
    :dataset       => :fakecells,
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
    :r             => 0.999,
    :maxiter       => 3000,
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
        issave=false, isload=true, gtincludebg=false,
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
initmethod = :isvd

results = map(METHODS) do method
    (; U, V, D, M, N) = pcb_init(X, p, k; initmethod=initmethod)
    res = pcb(U, V, D, M, N, p, k;
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
#     if method ∉ [:RelaxedL1_AD_LBFGS, :RelaxedL1_LBFGS]
#         @info "Run Relaxed L1 AD LCSVD again"
#         res2 = pcb(U, V, D, res.M, res.N, p, k;
#             pcb_method    = :rL1_SC_AD_SGD,
#             nαₘ_in        = res.nαₘ,
#             nαₙ_in        = res.nαₙ,
#             σ₀            = σ₀,
#             r             = r,
#             σ2ₘ           = res.σ2ₘ,
#             σ2ₙ           = res.σ2ₙ,
#             maxiter       = maxiter,
#             inner_maxiter = inner_maxiter,
#             tol           = tol,
#             track_history = true,
#         )
#         res2.history.times .+= res.history.times[end]
#         append!(res.history.times, res2.history.times)
#         append!(res.history.fvals, res2.history.fvals)
# #        append!(res.history.hist_inner_iters, res2.history.hist_inner_iters)
#         res3 = pcb(U, V, D, res2.M, res2.N, p, k;
#             pcb_method    = :rL1_SC_AD_SGD,
#             nαₘ_in        = res2.nαₘ,
#             nαₙ_in        = res2.nαₙ,
#             σ₀            = σ₀,
#             r             = r,
#             σ2ₘ           = res2.σ2ₘ,
#             σ2ₙ           = res2.σ2ₙ,
#             maxiter       = maxiter,
#             inner_maxiter = inner_maxiter,
#             tol           = tol,
#             track_history = true,
#         )
#         res3.history.times .+= res.history.times[end]
#         append!(res.history.times, res3.history.times)
#         append!(res.history.fvals, res3.history.fvals)
#         res4 = pcb(U, V, D, res3.M, res3.N, p, k;
#             pcb_method    = :rL1_SC_AD_SGD,
#             nαₘ_in        = res3.nαₘ,
#             nαₙ_in        = res3.nαₙ,
#             σ₀            = σ₀,
#             r             = r,
#             σ2ₘ           = res3.σ2ₘ,
#             σ2ₙ           = res3.σ2ₙ,
#             maxiter       = maxiter,
#             inner_maxiter = inner_maxiter,
#             tol           = tol,
#             track_history = true,
#         )
#         res4.history.times .+= res.history.times[end]
#         append!(res.history.times, res4.history.times)
#         append!(res.history.fvals, res4.history.fvals)
#     end

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

# ── Save W images ─────────────────────────────────────────────────────────────
figdir = projectdir("scripts", "figures", "pcb")
mkpath(figdir)

d = load(projectdir("scripts", "data", "exp4_obj_convergence_$(dataset).jld2"))
params        = d["params"]
method_labels = d["method_labels"]
times_all     = d["times"]
fvals_all     = d["fvals"]
iters_all     = d["iters"]
W_all         = d["W_all"]
H_all         = d["H_all"]
fv_all        = d["fv_all"]
m, n, p       = d["m"], d["n"], d["p"]

method_tags = METHOD_LABELS
for (i, (iters,times,fv,W,H,tag)) in enumerate(zip(iters_all,times_all,fv_all,W_all,H_all,method_tags))
    rt    = last(times)
    fname = joinpath(figdir, "exp4_$(tag)_am$(αₘ)_an$(αₙ)_f$(fv)_it$(iters)_rt$(round(rt;digits=1))")
    imsave_data(dataset, fname, W, H, imgsz, 100; saveH=false, verbose=false)
end
@info "Saved W images → scripts/figures/pcb/"

# ── plot ─────────────────────────────────────────────────────────────
include(projectdir("scripts", "experiments", "exp4_plot.jl"))
