# scripts/experiments/exp8_naomi_test.jl
#
# Experiment 8: Generate synthetic neuron data via NAOMiSim
#
# Run interactively:
#   julia> include("scripts/experiments/exp8_naomi_test.jl")
#
# Data saved to: scripts/data/exp8_naomi_*.jld2
# Figures saved to: scripts/figures/naomi/

using DrWatson
@quickactivate "PCBPaper"

using NAOMiSim
using LinearAlgebra, Random, Statistics
using JLD2
using CairoMakie

# ── Helper functions ──────────────────────────────────────────────────────────
function add_awgn(X::AbstractMatrix, snr_db::Real)
    P = mean(abs2, X)
    X .+ sqrt(P / 10^(snr_db / 10)) .* randn(eltype(X), size(X))
end

"""Double-exponential calcium transient kernel (GCaMP-like)."""
function make_soma_activity(K, nt, n_spikes;
                             tau_rise=3f0, tau_decay=60f0, amp=4f0, seed=18)
    Random.seed!(seed)
    ker = Float32.([exp(-t / tau_decay) - exp(-t / tau_rise) for t in 0:149])
    ker ./= maximum(ker)
    soma = fill(1f0, K, nt)
    for k in 1:K
        times = sort(randperm(nt - 80)[1:n_spikes] .+ 40)
        for t0 in times
            len = min(length(ker), nt - t0 + 1)
            soma[k, t0:t0+len-1] .+= amp .* ker[1:len]
        end
    end
    soma
end

"""Set nucleus voxels to nuc_val in gp_vals (makes dark nuclear hole)."""
function apply_nucleus_darkening!(vol_out, nuc_val=0f0)
    K = size(vol_out.locs, 1)
    for kk in 1:K
        gp_nuc_kk = vol_out.gp_nuc[kk]
        isnothing(gp_nuc_kk) && continue
        nuc_idxs = gp_nuc_kk[1]
        isempty(nuc_idxs) && continue
        nuc_set  = Set(nuc_idxs)
        soma_idxs = vol_out.gp_vals[kk][1]
        soma_vals = vol_out.gp_vals[kk][2]
        for ii in eachindex(soma_idxs)
            soma_idxs[ii] in nuc_set && (soma_vals[ii] = nuc_val)
        end
    end
end

"""Set all soma voxel fluorescence to soma_val (uniform cytoplasm)."""
function make_uniform_soma!(vol_out, soma_val=1f0, nuc_val=0f0)
    for kk in 1:size(vol_out.locs, 1)
        vol_out.gp_vals[kk][2] .= soma_val
    end
    apply_nucleus_darkening!(vol_out, nuc_val)
end

"""Add a white border around each panel image."""
function add_border(img::AbstractMatrix, bw::Int=1, val=maximum(img))
    h, w = size(img)
    out  = fill(Float64(val), h + 2bw, w + 2bw)
    out[bw+1:bw+h, bw+1:bw+w] .= img
    out
end

# ── Parameters ────────────────────────────────────────────────────────────────
prefix = "naomi_small"
_avg_rad = 8.0  # soma radius (μm); drives min_dist and nuc_rad below
params = Dict(
    :seed        => 3,               # RNG seed for reproducibility
    :N_neur      => 10,              # number of neurons simulated (~7 visible near focal plane)
    :vol_sz      => [50.0, 50.0, 60.0],  # FOV [x, y, z] in μm; at vres=1 px/μm → 50×50px image; z≥50μm required by PSF
    :vol_depth   => 150.0,           # focal plane depth below the surface (μm)
    :avg_rad     => _avg_rad,        # mean soma radius (μm); at vres=1 px/μm → ~$(round(Int, 2*_avg_rad))px diameter
    :min_dist    => 2 * _avg_rad,    # minimum inter-neuron distance (μm); = soma diameter (hard lower bound)
    :nuc_rad     => round.([_avg_rad * 3.5/6, _avg_rad * 2.0/6]; digits=1),  # nucleus [equatorial, polar] radius (μm)
    :vres        => 1.0,             # volume resolution (px/μm); image size = vol_sz[1:2] .* vres
    :nt          => 5000,            # number of time frames
    :dt          => 1/30,            # frame interval (s); 30 Hz acquisition
    :prot        => "GCaMP6f",       # fluorophore protocol (determines photon yield and kinetics)
    :vasc_flag   => false,           # include vasculature absorption artifacts
    :psf_type    => "gaussian",      # PSF model: "gaussian" (fast) or "vector" (physically accurate)
    :sigma0      => 2.7,             # readout noise std (ADU); independent of photon count
    :pavg        => 5.0,            # average laser power (mW); 1mW→~2dB SNR, 5mW→~16dB SNR
    :scan_buff   => 0,               # extra scan margin around FOV (pixels)
    :sfrac       => 2,               # spatial downsampling factor; output image = vol_sz*vres / sfrac
    :n_spikes    => 12,              # number of spikes per neuron over the recording
    :tau_rise    => 3,               # calcium transient rise time (frames)
    :tau_decay   => 60,              # calcium transient decay time (frames; ~2s at 30Hz)
    :spike_amp   => 4.0,             # spike amplitude (ΔF/F)
    :scale       => 4,               # display upscale factor for saved figures (px per data pixel)
)

@unpack seed, N_neur, vol_sz, vol_depth, min_dist, avg_rad, nuc_rad, vres,
        nt, dt, prot, vasc_flag, psf_type, sigma0, pavg, scan_buff, sfrac,
        n_spikes, tau_rise, tau_decay, spike_amp, scale = params

Random.seed!(seed)

datadir_ = projectdir("scripts", "data")
mkpath(datadir_)

# ── Pre-scan cache (Volume + Optics + Activity) ───────────────────────────────
# Filename encodes geometry (N_neur, vol_sz, vol_depth, avg_rad, vres).
# _prescan_params covers the remaining params that affect pre-scan steps.
_prescan_params = Dict(k => params[k] for k in [
    :seed, :min_dist, :nuc_rad, :vasc_flag, :vol_depth, :sfrac,
    :psf_type, :n_spikes, :tau_rise, :tau_decay, :spike_amp, :nt, :dt, :prot,
])
_sz_str        = join(round.(Int, vol_sz[1:2]), "x")
_prescan_fname = "$(prefix)_prescan_N$(N_neur)_sz$(_sz_str)_nt$(nt)_r$(avg_rad)_vres$(vres).jld2"
_prescan_path  = joinpath(datadir_, _prescan_fname)

tpm_params   = check_tpm_params(TPMParams(; pavg))
noise_params = NoiseParams(; darkcount=0.0, sigma=0.0, sigma0, mu0=0.0, bleedp=0.0)
N1 = round(Int, vol_sz[1] * vres) ÷ sfrac   # image rows
N2 = round(Int, vol_sz[2] * vres) ÷ sfrac   # image cols

if isfile(_prescan_path) && load(_prescan_path, "prescan_params") == _prescan_params
    @info "Loading pre-scan cache → $(_prescan_fname)"
    _c         = load(_prescan_path)
    vol_out    = _c["vol_out"]
    vol_params = _c["vol_params"]
    PSF_struct = _c["PSF_struct"]
    K          = _c["K"]
    soma_act   = _c["soma_act"]
    neur_act   = _c["neur_act"]
    spike_opts = _c["spike_opts"]
    W_gt       = _c["W_gt"]
    H_gt       = _c["H_gt"]
    W_gt_n     = _c["W_gt_n"]
    H_gt_n     = _c["H_gt_n"]
    powers     = [norm(W_gt[:, k]) for k in 1:K]
else
    # ── Volume ────────────────────────────────────────────────────────────────
    neur_params = NeurParams(;
        avg_rad     = avg_rad,
        nuc_rad     = nuc_rad,
        nuc_fluorsc = 0.0,       # nucleus has no fluorescence (GCaMP is cytoplasmic)
    )
    dend_params = DendParams(    # minimize dendrites to avoid grid artifacts
        dtParams  = [0., 2., 2., 1., 0.],
        atParams  = [0., 0., 0., 0., 0.],
        atParams2 = [0., 0., 0., 0., 0.],
        dweight   = 0.0,
    )
    vol_params  = VolumeParams(; vol_sz, vol_depth, N_neur, min_dist, vres,
                                 AD_density=0., verbose=1)
    psf_params  = PSFParams(; type=psf_type)
    vasc_params = VascParams(; flag=vasc_flag)
    bg_params   = BgParams(; flag=false)
    axon_params = AxonParams(; flag=false)

    @info "Simulating neural volume ($(N_neur) neurons, $(vol_sz) μm)…"
    vol_out, vol_params = simulate_neural_volume(
        vol_params, neur_params, vasc_params, dend_params, bg_params, axon_params)[1:2]

    K = size(vol_out.locs, 1)
    @info "Done — $K neurons  z=$(round.(vol_out.locs[:,3]; digits=1)) μm"

    # Uniform cytoplasm fluorescence + dark nucleus
    make_uniform_soma!(vol_out, 1f0, 0f0)

    # ── Optics ────────────────────────────────────────────────────────────────
    @info "Computing optical propagation…"
    PSF_struct = simulate_optical_propagation(vol_params, psf_params, vol_out)

    # ── Synthetic calcium transient activity ──────────────────────────────────
    @info "Generating synthetic activity ($n_spikes spikes/neuron, τ_decay=$(tau_decay) fr)…"
    soma_act   = make_soma_activity(K, nt, n_spikes;
                                    tau_rise=Float32(tau_rise),
                                    tau_decay=Float32(tau_decay),
                                    amp=Float32(spike_amp), seed=seed)
    neur_act   = (soma=soma_act, dend=fill(1f0, K, nt), bg=fill(1f0, K, nt))
    spike_opts = SpikeOpts(; K, nt, dt, prot, N_bg=0, axonflag=false)

    # ── Ground-truth spatial footprints (W_gt) ────────────────────────────────────
    @info "Computing W_gt (individual neuron scans)…"
    sp1  = SpikeOpts(; K, nt=1, dt, prot, N_bg=0, axonflag=false)
    act_base = (soma=fill(1f0,K,1), dend=fill(1f0,K,1), bg=fill(1f0,K,1))
    _, F_base = scan_volume(vol_out, PSF_struct, act_base,
                            ScanParams(; motion=false, verbose=0, scan_buff, sfrac),
                            noise_params, sp1, tpm_params)

    W_gt = zeros(Float64, N1*N2, K)
    for k in 1:K
        soma_k       = fill(1f0, K, 1); soma_k[k,1] = 1f0 + Float32(spike_amp)
        act_k        = (soma=soma_k, dend=fill(1f0,K,1), bg=fill(1f0,K,1))
        _, F_k       = scan_volume(vol_out, PSF_struct, act_k,
                                ScanParams(; motion=false, verbose=0, scan_buff, sfrac),
                                noise_params, sp1, tpm_params)
        W_gt[:, k]   = vec(Float64.(F_k[:,:,1]) .- Float64.(F_base[:,:,1]))
        print("$k ")
    end
    println()

    H_gt = Float64.(soma_act)   # K × nt  (calcium transient traces)

    # ── Normalize W_gt, scale H_gt by spatial power ───────────────────────────────
    powers  = [norm(W_gt[:, k]) for k in 1:K]
    W_gt_n  = W_gt ./ powers'
    H_gt_n  = H_gt .* powers

    @info "W_gt: $(size(W_gt))  H_gt: $(size(H_gt))"
    @info "Powers: $(round.(powers; sigdigits=3))"

    # ── Save jld ──────────────────────────────────────────────────────────────────
    jldsave(_prescan_path; prescan_params=_prescan_params,
            vol_out, vol_params, PSF_struct, K, soma_act, neur_act, spike_opts,
            W_gt,   H_gt, W_gt_n, H_gt_n)
    @info "Saved pre-scan cache → $(_prescan_fname)"

    # ── Figures ───────────────────────────────────────────────────────────────────
    figdir = projectdir("scripts", "figures", "$prefix")
    mkpath(figdir)

    # W_gt: neuron footprints tiled horizontally with 1-pixel white border
    vmax_w  = maximum(max.(W_gt_n, 0.))
    panels  = [add_border(repeat(reshape(W_gt_n[:,k], N1, N2), inner=(scale,scale)), 1, vmax_w)
            for k in 1:K]
    W_canvas = vcat(panels...)
    fig_wgt = Figure(size=(size(W_canvas,1), size(W_canvas,2)), figure_padding=0)
    ax_wgt  = Axis(fig_wgt[1,1]); hidedecorations!(ax_wgt); hidespines!(ax_wgt)
    colsize!(fig_wgt.layout,1,Fixed(size(W_canvas,1)))
    rowsize!(fig_wgt.layout,1,Fixed(size(W_canvas,2)))
    heatmap!(ax_wgt, W_canvas; colormap=:grays, colorrange=(0., vmax_w))
    save(joinpath(figdir, "$(prefix)_N$(N_neur)_sz$(_sz_str)_nt$(nt)_r$(avg_rad)_vres$(vres)_W_gt.png"), fig_wgt)
    @info "Saved → $(prefix)_N$(N_neur)_sz$(_sz_str)_nt$(nt)_r$(avg_rad)_vres$(vres)_W_gt.png"

    # H_gt_n: power-scaled activity traces, one per neuron
    t_sec   = (0:nt-1) .* dt
    offset  = maximum(H_gt_n) * 1.1 / K
    colors  = Makie.wong_colors()
    fig_hgt = Figure(size=(900, 600))
    ax_hgt  = Axis(fig_hgt[1,1];
        xlabel="Time (s)", ylabel="Neuron",
        title ="Ground-truth activity (H_gt × spatial power)")
    ytick_pos = Float64[]; ytick_lbl = String[]
    for k in 1:K
        y_off = (K - k) * offset
        lines!(ax_hgt, t_sec, H_gt_n[k,:] .- powers[k] .+ y_off;
            color=colors[mod1(k, length(colors))], linewidth=0.8)
        push!(ytick_pos, y_off)
        push!(ytick_lbl, "N$k (p=$(round(Int, powers[k])))")
    end
    ax_hgt.yticks = (ytick_pos, ytick_lbl)
    save(joinpath(figdir, "$(prefix)_N$(N_neur)_sz$(_sz_str)_nt$(nt)_r$(avg_rad)_vres$(vres)_H_gt.png"), fig_hgt)
    @info "Saved → $(prefix)_N$(N_neur)_sz$(_sz_str)_nt$(nt)_r$(avg_rad)_vres$(vres)_H_gt.png"
end

# ── Scan (Poisson shot noise + Gaussian readout noise) ───────────────────────
scan_params  = ScanParams(; motion=false, verbose=1, scan_buff, sfrac)

@info "Scanning volume ($nt frames, pavg=$(pavg) mW, sigma0=$(sigma0))…"
Fnoisy, Fclean = scan_volume(vol_out, PSF_struct, neur_act,
                               scan_params, noise_params, spike_opts, tpm_params)

_, _, Nt = size(Fclean)
X_clean = reshape(Float64.(Fclean), N1*N2, Nt)
X_noisy = reshape(Float64.(Fnoisy), N1*N2, Nt)

# SNR: normalize noisy by PMT gain to compare in photon domain
mu_pmt     = 100.0
X_noisy_ph = X_noisy ./ mu_pmt
actual_snr     = 10 * log10(mean(abs2, X_clean) / mean(abs2, X_noisy_ph .- X_clean))
actual_snr_str = "$(round(actual_snr; digits=1))dB"
@info "Actual SNR: $(actual_snr_str)  (pavg=$(pavg) mW)"

# ── Correlation image ─────────────────────────────────────────────────────────
mean_tr = vec(mean(X_noisy; dims=1))
mean_c  = mean_tr .- mean(mean_tr); mean_c ./= norm(mean_c)
corr_vals = map(1:N1*N2) do i
    x = @view X_noisy[i,:]; xc = x .- mean(x); n = norm(xc)
    n < eps(eltype(xc)) ? 0.0 : dot(xc ./ n, mean_c)
end
corr_img = reshape(corr_vals, N1, N2)

# ── Save JLD2 ─────────────────────────────────────────────────────────────────
fname = "$(prefix)_N$(N_neur)_sz$(_sz_str)_r$(avg_rad)_vres$(vres)_pavg$(pavg).jld2"
jldsave(joinpath(datadir_, fname);
    params,
    N1, N2, Nt, K,
    X_clean, X_noisy, mu_pmt,
    powers,
    actual_snr,
    corr_img,
    locs = vol_out.locs,
)
@info "Saved → $(fname)"

# ── Figures ───────────────────────────────────────────────────────────────────
figdir = projectdir("scripts", "figures", "$(prefix)", "N$(N_neur)_sz$(_sz_str)_r$(avg_rad)_vres$(vres)_pavg$(pavg)")
mkpath(figdir)

F_noisy3d = reshape(X_noisy, N1, N2, Nt)
p_lo = quantile(vec(X_noisy), 0.005)
p_hi = quantile(vec(X_noisy), 0.999)

# # peak frame (raw noisy)
# pk = argmax(vec(maximum(reshape(X_clean, N1, N2, Nt); dims=(1,2))))
# fig_frame = Figure(size=(N1*scale, N2*scale), figure_padding=0)
# ax_fr = Axis(fig_frame[1,1]); hidedecorations!(ax_fr); hidespines!(ax_fr)
# colsize!(fig_frame.layout,1,Fixed(N1*scale)); rowsize!(fig_frame.layout,1,Fixed(N2*scale))
# heatmap!(ax_fr, F_noisy3d[:,:,pk]; colormap=:grays, colorrange=(p_lo, p_hi))
# save(joinpath(figdir, "exp8_peak_frame_pavg$(pavg)_$(actual_snr_str).png"), fig_frame)
# @info "Saved → exp8_peak_frame_pavg$(pavg)_$(actual_snr_str).png"

# Correlation image
c_lo, c_hi = extrema(corr_img)
fig_corr = Figure(size=(N1*scale, N2*scale), figure_padding=0)
ax_corr  = Axis(fig_corr[1,1]); hidedecorations!(ax_corr); hidespines!(ax_corr)
colsize!(fig_corr.layout,1,Fixed(N1*scale)); rowsize!(fig_corr.layout,1,Fixed(N2*scale))
heatmap!(ax_corr, corr_img; colormap=:grays, colorrange=(c_lo, c_hi))
save(joinpath(figdir, "$(prefix)_corr_image_pavg$(pavg)_$(actual_snr_str).png"), fig_corr)
@info "Saved → $(prefix)_corr_image_pavg$(pavg)_$(actual_snr_str).png"

# ── GIF: axes-free grayscale movie around peak activity ──────────────────────
F_noisy3d  = reshape(X_noisy_ph, N1, N2, Nt)   # photon-domain noisy movie
mean_act   = vec(mean(max.(reshape(X_clean, N1, N2, Nt), 0.); dims=(1,2)))
pk_gif     = argmax(mean_act)
win_s      = max(1, pk_gif - 50)
win_e      = min(Nt, win_s + 700)
frames_gif = win_s:5:win_e

gif_lo = quantile(vec(X_noisy_ph), 0.005)
gif_hi = quantile(vec(X_noisy_ph), 0.999)

fig_gif = Figure(size=(150, 150), figure_padding=0)
ax_gif  = Axis(fig_gif[1,1]); hidedecorations!(ax_gif); hidespines!(ax_gif)
colsize!(fig_gif.layout, 1, Fixed(150)); rowsize!(fig_gif.layout, 1, Fixed(150))
hm_gif  = heatmap!(ax_gif, F_noisy3d[:,:,win_s]; colormap=:grays,
                    colorrange=(gif_lo, gif_hi))
record(fig_gif, joinpath(figdir, "$(prefix)_movie_pavg$(pavg)_$(actual_snr_str).gif"), frames_gif; framerate=8) do t
    hm_gif[1] = F_noisy3d[:,:,t]
end
@info "Saved → $(prefix)_movie_pavg$(pavg)_$(actual_snr_str).gif  ($(length(frames_gif)) frames @ 8fps)"
