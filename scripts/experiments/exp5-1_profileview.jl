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
    :tol            => 1e-12,
)

# ── Data ─────────────────────────────────────────────────────────────────────
@unpack dataset, noc, factor, SNR, inhibitindices, bias, imgsz0, k,
        αₘ, αₙ, σ₀, r, maxiter, inner_maxiter, tol = params

sqfactor = Int(floor(sqrt(factor)))
imgsz    = (sqfactor * imgsz0[1], sqfactor * imgsz0[2])
lengthT  = factor * 1000
sigma    = sqfactor * 5.0

X, imsz, lhT, ncs, gtnoc, datadic = load_data(dataset;
    sigma=sigma, imgsz=imgsz, lengthT=lengthT, SNR=SNR, bias=bias,
    useCalciumT=true, inhibitindices=inhibitindices,
    issave=false, isload=false, gtincludebg=false,
    save_gtimg=false, save_maxSNR_X=false, save_X=false)

(m, n, p) = (size(X)..., noc)
gtW, gtH  = dataset == :fakecells ?
    (datadic["gtW"], datadic["gtH"]) :
    (Matrix{eltype(X)}(undef,0,0), Matrix{eltype(X)}(undef,0,0))

# ── LCSVD wrapper ─────────────────────────────────────────────────────────────
function my_rL1_LBFGS(X, p, imgsz; αₘ, αₙ, σ₀, r, maxiter, inner_maxiter, tol)
    U, Vt, M0, N0, _, _, D = LCSVD.initpcb(X, p, 0; initmethod=:isvd, svdmethod=:isvd)
    V   = copy(Vt')       # n × k
    N0t = copy(N0')       # k × p

    alg = LCSVD.LinearCombSVD(;
        α1=αₘ, α2=αₙ, β1=0.0, β2=0.0,
        σ0=σ₀, r=r,
        optim_method      = :lbfgs,
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
        inner_tol         = tol,
        successive_f_converge = 0,
    )

    M1, N1t = copy(M0), copy(N0t)
    rst = LCSVD.solve!(alg, X, U, V, D, M1, N1t)

    # Convert LCSVD history to PCBResult-compatible format
    # rst.laps[1] is the timestamp before iteration 1
    # rst.exact_objvalues[1] is the initial exact L1 (before any iteration)
    times   = rst.laps[1:end] .- rst.laps[1]
    fvals   = rst.exact_objvalues[1:end]
    history = (times=times, fvals=fvals)

    return PCBResult(rst.W, rst.Ht', rst.converged, rst.niters, history)
end

# ── Warmup (trigger JIT compilation before timed runs) ───────────────────────
let
    Xw = randn(20, 25)   # Float64, same types as main run
    pw, kw = 3, 3
    @info "Warming up PCB..."
    pcb(Xw, pw, kw; pcb_method=:RelaxedL1_LBFGS,
        αₘ=1e-2, αₙ=1e-2, σ₀=2.0, r=0.95,
        maxiter=2, inner_maxiter=10, tol=1e-6, track_history=true)
    @info "Warming up LCSVD..."
    my_rL1_LBFGS(Xw, pw, (4,5); αₘ=1e-2, αₙ=1e-2, σ₀=2.0, r=0.95,
        maxiter=2, inner_maxiter=10, tol=1e-6)
    @info "Warmup done."
end

# ── Profileview ───────────────────────────────────────────────────────────────────────
using ProfileView

U, Vt, M0, N0, _, _, D = LCSVD.initpcb(X, p, 0; initmethod=:isvd, svdmethod=:isvd)
V   = copy(Vt')       # n × k
N0t = copy(N0')       # k × p

alg = LCSVD.LinearCombSVD(;
    α1=αₘ, α2=αₙ, β1=0.0, β2=0.0,
    σ0=σ₀, r=r,
    optim_method      = :lbfgs,
    useprecond        = false,
    usedenoiseUVt     = false,
    uselv             = false,
    imgsz             = imgsz,
    maxiter           = 300,
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
    inner_tol         = tol,
    successive_f_converge = 0,
)

M1, N1t = copy(M0), copy(N0t)
@profview LCSVD.solve!(alg, X, U, V, D, M1, N1t)

@profview res_pcb = pcb(X, p, k;
    pcb_method    = :RelaxedL1_LBFGS,
    αₘ, αₙ, σ₀, r,
    maxiter, inner_maxiter, tol,
    track_history = true,
)

