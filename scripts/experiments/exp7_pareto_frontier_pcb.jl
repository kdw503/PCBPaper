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
figdir = projectdir("scripts", "figures", "pcb")
mkpath(figdir)

# ── Parameters ───────────────────────────────────────────────────────────────
params = Dict(
    :method        => :PCB,
    :dataset       => :natural,
    :imgsz         => (12, 12),
    :lengthT       => 100000,
    :noc           => 72,
    :k             => 72,  # SVD rank
    :αₘ            => 0,
    :αₙ            => 5e-1, # 1e-2,
    :σ₀            => 1.0, # 1.0,
    :r             => 0.3, # 0.95,
    :maxiter       => 500, # 500,
    :inner_maxiter => 10, # 10,
    :tol           => 1e-10,
    :inner_tol     => 1e-10,
    :apowrng       => -4:0.1:1, # -4:0.1:1
)

# ── Data ─────────────────────────────────────────────────────────────────────
@unpack method, dataset, imgsz, lengthT, noc, k, αₘ, αₙ, σ₀, r, maxiter, inner_maxiter, tol, inner_tol, apowrng = params

dd = load(joinpath(datapath,"allinit.jld2"))
X = dd["X_whitened"][1]
U, Vt, D = dd["SVD"]; V = Vt'
initmethod = :randcolX # :BPDN, :isvd, :randH
Winit, Hinit, M, N0t, _ = dd[String(initmethod)]
N = copy(N0t')

# ── PCB ─────────────────────────────────────────────────────────────────
results = map(apowrng) do apow
    αₙ = 10.0^apow
    @info "$(rpad(string(apow), 22))"
    (; U, V, D, M, N) = pcb_init(X, noc, k; initmethod=:isvd)
    res = pcb(U, V, D, M, N, noc, k;
        pcb_method    = :RelaxedL1_AD_LBFGS,
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
    fname="exp7_pareto_frontier_ahpow$(apow)_imiter$(inner_maxiter)"
    imsave_data(dataset, joinpath(figdir, fname*".png"), W1, H1, imgsz, 100; saveH=false, verbose=false)
    fv = LCSVD.fitd(X, W1 * H1)
    LCSVD.normalizeWH!(W1,H1); sparH = norm(H1,1)
    jldsave(joinpath(datadir_, fname*".jld2"); αₙ, fv, sparH)
    (res=res, W=W1, H=H1, fv=fv, sparH=sparH)
end

# ── Save data ─────────────────────────────────────────────────────────────────

fname="exp7_pareto_frontier_pcb.jld2"
jldsave(joinpath(datadir_, fname);
    params,
    method_name  = string(method),
    apowrng = apowrng,
    fvs    = [r.fv  for r in results],
    sparHs = [r.sparH  for r in results],
    W_all  = [r.W  for r in results],
    H_all  = [r.H  for r in results],
)

@info "Saved → $(fname)"
