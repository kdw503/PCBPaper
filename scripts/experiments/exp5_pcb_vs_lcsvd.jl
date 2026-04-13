# scripts/experiments/exp5_pcb_vs_lcsvd.jl
#
# Experiment 5: PCB RelaxedL1_LBFGS vs LCSVD LinearCombSVD
#
# Compares the new PCB implementation against the existing LCSVD implementation
# using the same data (fakecells) and matching parameters.
#
# Run interactively:
#   julia> include("scripts/experiments/exp5_pcb_vs_lcsvd.jl")
#
# Figures saved to: scripts/figures/exp5_pcb_vs_lcsvd.png / .pdf

using DrWatson
@quickactivate "PCBPaper"

using PenalizedComponentBlends
using LinearAlgebra, Random
using CairoMakie
using JLD2
using TestData
using LCSVD

# ── Parameters ───────────────────────────────────────────────────────────────
params = Dict(
    :dataset        => :fakecells,
    :noc            => 15,
    :factor         => 1,
    :SNR            => 0,
    :inhibitindices => 0,
    :bias           => 0.1,
    :imgsz0         => (40, 20),
    :k              => 15,   # SVD rank
    :αₘ             => 1e-2,
    :αₙ             => 1e-2,
    :σ₀             => 2.0,
    :r              => 0.95,
    :maxiter        => 300,
    :inner_maxiter  => 500,
    :tol            => 1e-6,
    :inner_tol      => 1e-6,
)

# ── Data ─────────────────────────────────────────────────────────────────────
@unpack dataset, noc, factor, SNR, inhibitindices, bias, imgsz0, k,
        αₘ, αₙ, σ₀, r, maxiter, inner_maxiter, tol, inner_tol = params

sqfactor = Int(floor(sqrt(factor)))
imgsz    = (sqfactor * imgsz0[1], sqfactor * imgsz0[2])
lengthT  = factor * 1000
sigma    = sqfactor * 5.0

X, imsz, lhT, ncs, gtnoc, datadic = load_data(dataset;
    sigma=sigma, imgsz=imgsz, lengthT=lengthT, SNR=SNR, bias=bias,
    useCalciumT=true, inhibitindices=inhibitindices,
    issave=false, isload=true, gtincludebg=false,
    save_gtimg=false, save_maxSNR_X=false, save_X=false)

(m, n, p) = (size(X)..., noc)
gtW, gtH  = dataset == :fakecells ?
    (datadic["gtW"], datadic["gtH"]) :
    (Matrix{eltype(X)}(undef,0,0), Matrix{eltype(X)}(undef,0,0))

# ── LCSVD wrappers ────────────────────────────────────────────────────────────
function _lcsvd_base_alg(imgsz, optim_method; αₘ, αₙ, σ₀, r, maxiter, inner_maxiter, tol, inner_tol)
    LCSVD.LinearCombSVD(;
        α1=αₘ, α2=αₙ, β1=0.0, β2=0.0,
        σ0=σ₀, r=r,
        optim_method      = optim_method,
        useprecond        = false,
        usedenoiseUVt     = false,
        uselv             = false,
        imgsz             = imgsz,
        maxiter           = maxiter,
        inner_maxiter     = inner_maxiter,
        store_trace       = false,
        store_inner_trace = false,
        show_trace        = false,
        allow_f_increases = true,
        f_abstol          = tol,
        f_reltol          = tol,
        f_inctol          = 1e2,
        x_abstol          = tol,
        x_reltol          = tol,
        inner_tol         = inner_tol,
        successive_f_converge = 0,
    )
end

function _run_lcsvd(X, p, imgsz, optim_method; αₘ, αₙ, σ₀, r, maxiter, inner_maxiter, tol, inner_tol)
    U, Vt, M0, N0, _, _, D = LCSVD.initpcb(X, p, 0; initmethod=:isvd, svdmethod=:isvd)
    V   = copy(Vt')
    N0t = copy(N0')
    alg = _lcsvd_base_alg(imgsz, optim_method; αₘ, αₙ, σ₀, r, maxiter, inner_maxiter, tol, inner_tol)
    M1, N1t = copy(M0), copy(N0t)
    rst = LCSVD.solve!(alg, X, U, V, D, M1, N1t)
    times   = rst.laps[1:end] .- rst.laps[1]
    fvals   = rst.exact_objvalues[1:end]
    history = (times=times, fvals=fvals)
    return PCBResult(rst.W, rst.Ht', rst.converged, rst.niters, history)
end

my_rL1_LBFGS(X, p, imgsz; kw...) =
    _run_lcsvd(X, p, imgsz, :lbfgs; kw...)
my_rL1_ADMM_LBFGS(X, p, imgsz; kw...) =
    _run_lcsvd(X, p, imgsz, :lbfgs_admm; kw...)

# ── Warmup (trigger JIT compilation before timed runs) ───────────────────────
let
    Xw = randn(20, 25)   # Float64, same types as main run
    pw, kw = 3, 3
    wkw = (αₘ=1e-2, αₙ=1e-2, σ₀=2.0, r=0.95, maxiter=2, inner_maxiter=10, tol=1e-6, inner_tol=1e-6)
    @info "Warming up PCB rL1-LBFGS..."
    pcb(Xw, pw, kw; pcb_method=:RelaxedL1_LBFGS, wkw..., track_history=true)
    @info "Warming up PCB rL1-AD-LBFGS..."
    pcb(Xw, pw, kw; pcb_method=:RelaxedL1_AD_LBFGS, wkw..., track_history=true)
    @info "Warming up LCSVD rL1-LBFGS..."
    my_rL1_LBFGS(Xw, pw, (4,5); wkw...)
    @info "Warming up LCSVD rL1-AD-LBFGS..."
    my_rL1_ADMM_LBFGS(Xw, pw, (4,5); wkw...)
    @info "Warmup done."
end

# ── Run ───────────────────────────────────────────────────────────────────────
method_labels = ["PCB rL1-LBFGS", "PCB rL1-AD-LBFGS", "LCSVD rL1-LBFGS", "LCSVD rL1-AD-LBFGS"]

function run_and_postprocess(label, res)
    @info "$(rpad(label, 22)) → $(res.iterations) iters, " *
          "final fval = $(round(last(res.history.fvals); sigdigits=4))"
    W1, H1 = copy(res.W), copy(res.H)
    LCSVD.normalizeW!(W1, H1)
    if dataset == :fakecells
        fv, ml, _, _ = LCSVD.matchedfitval(gtW, gtH, W1, H1; clamp=false)
        nodr = LCSVD.matchedorder(ml, noc)
        W1, H1 = W1[:, nodr], H1[nodr, :]
    else
        fv = LCSVD.fitd(X, W1 * H1)
    end
    (res=res, W=W1, H=H1, fv=fv)
end

run_kw = (αₘ=αₘ, αₙ=αₙ, σ₀=σ₀, r=r, maxiter=maxiter, inner_maxiter=inner_maxiter, tol=tol, inner_tol=inner_tol)

res_pcb_lbfgs    = pcb(X, p, k; pcb_method=:RelaxedL1_LBFGS,    run_kw..., track_history=true)
res_pcb_adlbfgs  = pcb(X, p, k; pcb_method=:RelaxedL1_AD_LBFGS, run_kw..., track_history=true)
res_lcsvd_lbfgs  = my_rL1_LBFGS(X, p, imgsz;      run_kw...)
res_lcsvd_admm   = my_rL1_ADMM_LBFGS(X, p, imgsz; run_kw...)

results = [
    run_and_postprocess("PCB rL1-LBFGS",        res_pcb_lbfgs),
    run_and_postprocess("PCB rL1-AD-LBFGS",     res_pcb_adlbfgs),
    run_and_postprocess("LCSVD rL1-LBFGS",      res_lcsvd_lbfgs),
    run_and_postprocess("LCSVD rL1-AD-LBFGS", res_lcsvd_admm),
]

# ── Save data ─────────────────────────────────────────────────────────────────
datadir_ = projectdir("scripts", "data")
mkpath(datadir_)

jldsave(joinpath(datadir_, "exp5_pcb_vs_lcsvd.jld2");
    params, m, n, p,
    method_labels,
    times  = [r.res.history.times  for r in results],
    fvals  = [r.res.history.fvals  for r in results],
    iters  = [r.res.iterations     for r in results],
    W_all  = [r.W                  for r in results],
    H_all  = [r.H                  for r in results],
    fv_all = [r.fv                 for r in results],
)
@info "Saved → scripts/data/exp5_pcb_vs_lcsvd.jld2"

# ── Check initial objective value ─────────────────────────────────────────────
# αₘ = 1e-2; αₙ = 1e-2
# U, Vt, M, N, W₀, H₀, D = LCSVD.initpcb(X, p, 0; initmethod=:isvd, svdmethod=:isvd)
# nαₘ = αₘ * norm(D, 2)^2 / norm(W₀, 1) / norm(N, 2)
# nαₙ = αₙ * norm(D, 2)^2 / norm(H₀, 1) / norm(M, 2)
# lcsvd_obj = PenalizedComponentBlends.objective_L1(M, N, U, D, Vt', nαₘ, nαₙ)
#     alg = LCSVD.LinearCombSVD(;
#         α1=αₘ, α2=αₙ,
#         maxiter           = 1,
#         inner_maxiter     = 1
#     )
# rst = LCSVD.solve!(alg, X, U, V, D, M, N')

# (; U, V, D, M, N, W₀, H₀) = PenalizedComponentBlends._pcb_init(X, p, k; initmethod = :isvd)
# nαₘ = αₘ * norm(D, 2)^2 / norm(W₀, 1) / norm(N, 2)
# nαₙ = αₙ * norm(D, 2)^2 / norm(H₀, 1) / norm(M, 2)
# pcb_obj = PenalizedComponentBlends.objective_L1(M, N, U, D, V, nαₘ, nαₙ)
# @show sum(abs, U*M), sum(abs, N*V'), norm(M), norm(N), nαₘ, nαₙ
