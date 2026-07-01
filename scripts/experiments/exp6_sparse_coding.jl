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

# ── Parameters ───────────────────────────────────────────────────────────────
const METHODS = [
    :RelaxedL1_AD_LBFGS,
    :RelaxedL1_LBFGS,
    :rL1_SC_AD_SGD,
    :rL1_SC_AD_ADAM,
    :rL1_SC_SGD,
    :rL1_SC_ADAM,
]

const METHOD_LABELS = [
    "rL1 AD-LBFGS",
    "rL1 LBFGS",
    "AD-SGD",
    "AD-ADAM",
    "Joint-SGD",
    "Joint-ADAM",
]

params = Dict(
    :dataset       => :natural,
    :imgsz         => (12, 12),
    :lengthT       => 100000,
    :noc           => 72,
    :k             => 72,  # SVD rank
    :αₘ            => 0,
    :αₙ            => 5e-1, # 1e-2,
    :σ₀            => 1.0, # 1.0,
    :r             => 0.3, # 0.95,
    :maxiter       => 300, # 300,
    :inner_maxiter => 500, # 500,
    :tol           => 1e-10,
    :inner_tol     => 1e-10,
)

# ── Data ─────────────────────────────────────────────────────────────────────
@unpack dataset, imgsz, lengthT, noc, k, αₘ, αₙ, σ₀, r, maxiter, inner_maxiter, tol, inner_tol = params

patch_size = imgsz[1]

dd = load(joinpath(datapath, "X_whitened_Hspar","natural_SC_l3.0_iter50.jld2"))
sD = dd["D"]; αs = dd["αs"]; X_whitened = dd["X_whitened"]
X = X_whitened

(m, n, p, k) = (size(X)..., noc, noc)
gtW, gtH  = dataset == :fakecells ?
    (datadic["gtW"], datadic["gtH"]) :
    (Matrix{eltype(X)}(undef,0,0), Matrix{eltype(X)}(undef,0,0))

figdir = projectdir("scripts", "figures", "pcb")
mkpath(figdir)
initmethod = :isvd

results = map(METHODS) do method
    @info "$(rpad(string(method), 22))"
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
        inner_tol     = inner_tol,
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

fname="exp6_sparse_coding_$(dataset)_ah$(αₙ)_imiter$(inner_maxiter)_long_nstep.jld2"
jldsave(joinpath(datadir_, fname);
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

@info "Saved → $(fname)"

# ── plot ─────────────────────────────────────────────────────────────────
d = load(projectdir("scripts", "data", fname))
include(projectdir("scripts", "experiments", "exp6_plot.jl"))
