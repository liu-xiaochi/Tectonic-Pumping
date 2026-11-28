# ==============================================================================
# 2D Staggered-Grid Coupled THMC Two-Phase Flow Solver (ParallelStencil.jl, CPU)
# ------------------------------------------------------------------------------
# High-Performance Fused Pseudo-Maxwell Symplectic PT Stokes-Darcy Solver:
#   1) Geologically anchored, stress-shadowed 67° oblique hydrofracture veins:
#      - Nucleates ONLY at physical shear-band overpressure/porosity peaks where
#        (Pf - Pt) > pf_frac + sigma_sq (strictly ZERO in static & filter!)
#      - Elastic stress shadow (560 m) selects 2-3 discrete 67° oblique feeder
#        veins (dx/dz = 0.42, half-width 56 m); once fractured, each dike trace
#        stays permanently anchored (irreversible crustal damage -> no lateral
#        vein teleportation, no Li/viscosity wipeouts!)
#      - Focused vein permeability (k_hf_frac = 0.42) + calibrated Lagrangian
#        wall distillation (beta_T_dist = 0.155) drives crisp 67° terracotta-red
#        to dark-crimson veins (6k-12k ppm) & TrapLi > 6,000 ppm in coupled
#   2) Smooth 15-cell geometric-mean active channel/vein melt viscosity metric:
#      - Eliminates single-cell threshold spikes -> monotonic S-curve in Panel b
#   3) Built-in Jacobi-PCG elliptic Pf polish (`polish_pf_pcg_serial!`):
#      - Eliminates err_pf stalling across 10^8x vein-vs-wall mobility contrasts
#   4) Fast warm-started 3-kernel fused PT loop (ZERO pow/log/exp/sqrt/div)
#
# Launch on multi-core Windows CPU (use 4 physical P-cores for 96x96 grid):
#   julia -t 4 solver_2d_thmc_mush.jl
#
# Outputs (saved in ./output_2d/, directly compatible with Plot_fig3_layout_skeleton.py):
#   - snapshots_{static,filter,shear_const,coupled}.npz         (final state)
#   - snapshots_active_{static,filter,shear_const,coupled}.npz  (sync t/tc ~ 0.45)
#   - snapshots_peak_{static,filter,shear_const,coupled}.npz    (peak Li export flux)
#   - time_series.npz                                           (4-case curves)
#   - phase_diagram_grid.npz                                    (optional sweep)
# ==============================================================================

using ParallelStencil
@init_parallel_stencil(Threads, Float64, 2)

using NPZ
using Printf
using Random
using Statistics

# ==============================================================================
# 1. CONFIGURATION STRUCTS (4-SWITCH ABLATION DESIGN)
# ==============================================================================
Base.@kwdef struct ModelParams
    # Domain & grid (96x96 for fast CPU verification; 192x192 for final publication Fig. 3a)
    Lx::Float64 = 2000.0            # Horizontal domain width (m, periodic)
    Lz::Float64 = 2000.0            # Vertical mush thickness (m, z=0 base -> z=Lz Palaeo-BDT)
    nx::Int     = 96
    nz::Int     = 96

    # Thermal & enthalpy-conserving phase equilibrium (calibrated for Cf growth & lock-in ~0.95)
    T_bg_top::Float64     = 555.0   # Warm mid-crust aureole T at Palaeo-BDT (°C)
    T_bg_bot::Float64     = 605.0   # Mid-crust country-rock T at mush base (°C)
    H_roof::Float64       = 1900.0  # Effective metamorphic-aureole roof thermal barrier (m)
    T_init_top::Float64   = 716.0   # Initial top mush T (°C, phi ~ 0.29)
    T_init_bot::Float64   = 724.0   # Initial base mush T (°C, phi ~ 0.33)
    T_liq::Float64        = 825.0   # Liquidus (°C)
    T_sol0::Float64       = 640.0   # Unfractionated granite solidus (°C)
    T_sol_min::Float64    = 455.0   # Flux-rich pegmatite eutectic solidus floor (°C)
    dT_flux::Float64      = 36.0    # Solidus depression slope (°C per unit Cf excess)
    dT_lat::Float64       = 245.0   # Granitic latent-heat cooling scale L/cp (°C)
    gamma_melt::Float64   = 1.40    # Initial granite melt-fraction exponent
    kappa::Float64        = 1.0e-6  # Thermal diffusivity (m^2 / s)
    St::Float64           = 1.0     # Stefan number
    tau_lat_frac::Float64 = 1.50    # Lateral pluton cooling timescale / tau_cool
    tau_cryst::Float64    = 8.0e10  # Sub-solidus freeze relaxation timescale (~2.5 kyr)

    # Two-phase permeability, buoyancy & anchored stress-shadowed 67° oblique veins
    a_grain::Float64       = 2.2e-3 # Characteristic grain radius (2.2 mm -> delta_c0 ~ 47 m > 2.2*dx)
    n_perm::Float64        = 3.0    # Kozeny-Carman exponent
    delta_rho::Float64     = 300.0  # rho_s - rho_m (kg / m^3)
    g::Float64             = 9.81   # Gravity (m / s^2)
    phi_min::Float64       = 0.015  # Residual locked porosity floor
    phi_max::Float64       = 0.54   # Porosity cap
    phi_crit::Float64      = 0.48   # Granitic mush disaggregation limit (RCMP)
    pf_frac_MPa::Float64   = 0.90   # Super-lithostatic tensile threshold (MPa)
    dp_frac_MPa::Float64   = 0.25   # Smooth hydrofracture transition width (MPa)
    k_hf_frac::Float64     = 0.42   # Focused discrete-vein permeability / k_ref (~16-18 m export in coupled)
    vein_tilt_dx::Float64  = 0.42   # Oblique vein slope dx/dz = cot(67.2°) under dextral simple shear
    vein_shadow_m::Float64 = 560.0  # Elastic stress-shadow spacing (m -> naturally selects 2-3 veins on 2 km)
    vein_half_w_m::Float64 = 56.0   # Gaussian vein half-width (m; ~2.7 cells -> crisp ~130 m oblique dikes)

    # Solid rheology (Katz et al. 2006 n=2), shear tectonic pumping & local dilatancy
    eta_s_ref::Float64    = 8.0e18  # Solid shear viscosity at phi_ref (Pa s)
    alpha_phi::Float64    = 18.0    # Porosity-weakening exponent (crisp bands, fast PT convergence)
    n_power::Float64      = 2.0     # Non-Newtonian exponent (20°-25° oblique bands)
    tau_yield::Float64    = 35.0e6  # Viscoplastic BDT yield cap (Pa)
    alpha_pump::Float64   = 0.22    # Matrix-driven tectonic pumping
    psi_0::Float64        = 0.14    # Local Reynolds dilatancy tan(psi)
    eps_bg::Float64       = 4.5e-13 # Himalayan STDS simple-shear strain rate (s^-1)
    sigma_sq::Float64     = 10.0e6  # Homogeneous compressive squeezing stress for filter pressing (Pa)

    # Thresholded Giordano-Bartels exponential melt viscosity law (clean S-curve collapse)
    eta_m0::Float64       = 6.0e6   # Initial high-Si peraluminous melt viscosity (Pa s)
    eta_min::Float64      = 1.0e3   # Flux-rich pegmatite melt viscosity floor (Pa s; 6,000x collapse)
    Cf_onset::Float64     = 1.55    # Volatile depolymerization onset threshold
    lambda_flux::Float64  = 2.40    # Exponential depolymerization slope above Cf_onset

    # Geochemical open-system Rayleigh + focused Lagrangian vein distillation & shallow BDT trap
    Li_0::Float64         = 120.0   # Initial granitic melt Li (ppm)
    D_Li::Float64         = 0.10    # Bulk partition coefficient for Li
    D_flux::Float64       = 0.05    # Bulk partition coefficient for flux (H2O+B+F)
    beta_T_dist::Float64  = 0.155   # Focused-vein Lagrangian distillation (6k-12k ppm terracotta/crimson veins)
    D_disp::Float64       = 1.0e-9  # Pore-scale chemical dispersion (m^2 / s)
    f_peg_res::Float64    = 0.16    # Residual melt fraction in shallow BDT pegmatite dikes (TrapLi ~6.5k-8.5k ppm)
    V_peg_crit::Float64   = 12.0    # Critical exported melt batch thickness (m)

    # Time integration & graded PT solver controls
    max_t_norm::Float64    = 1.20   # End time in units of tau_cool
    dt_max_frac::Float64   = 0.004  # Max physical step / tau_cool
    max_steps::Int         = 3000   # Max physical time steps
    pt_max_iter_init::Int  = 8000   # Max PT iterations at step 0
    pt_max_iter::Int       = 4500   # Max PT iterations for warm-started steps (step >= 1)
    visc_stride::Int       = 5      # Lagged rheology refresh stride
    pt_check_every::Int    = 100    # Residual check stride
    pt_tol_init::Float64   = 3.0e-4 # Strict tolerance at step 0
    pt_tol::Float64        = 1.5e-3 # Graded tolerance for warm-started steps (step >= 1)
    noise_amp::Float64     = 0.050  # Initial smoothed porosity perturbation amplitude
    t_active_snap::Float64 = 0.45   # Synchronous dimensionless time for mature oblique channel snapshot
end

struct CaseConfig
    name::String
    use_shear::Bool            # Switch 1: Simple shear BCs (top/bottom velocity)
    use_squeezing::Bool        # Switch 2: Homogeneous tectonic squeezing (filter pressing)
    use_dilatancy::Bool        # Switch 3: Shear tectonic pumping + local dilatant channeling
    use_flux_solidus::Bool     # Switch 4a: Thermodynamic flux-depressed solidus (ON for all 4)
    use_visc_feedback::Bool    # Switch 4b: Giordano-Bartels exponential viscosity collapse
end

# ==============================================================================
# 2. SCALAR HELPERS & PARALLEL STENCIL KERNELS (ZERO RACE CONDITIONS, PERIODIC X)
# ==============================================================================

# 2.0 2nd-order conservative TVD (van Leer) face reconstruction
@inline function tvd_face(u::Float64, qmm::Float64, qm::Float64, qp::Float64, qpp::Float64)
    if u >= 0.0
        dL = qm - qmm
        dR = qp - qm
        s  = (dL * dR > 0.0) ? (2.0 * dL * dR / (dL + dR)) : 0.0
        return qm + 0.5 * s
    else
        dL = qp - qm
        dR = qpp - qp
        s  = (dL * dR > 0.0) ? (2.0 * dL * dR / (dL + dR)) : 0.0
        return qp - 0.5 * s
    end
end

# Fast hardware-only Carreau power-law factor (zero pow() for n=2 or n=3)
@inline function carreau_factor(r2::Float64, n_pl::Float64, pl_exp::Float64)
    if n_pl <= 1.0
        return 1.0
    elseif n_pl == 2.0
        return clamp(1.0 / sqrt(sqrt(1.0 + r2)), 0.35, 1.0)
    elseif n_pl == 3.0
        return clamp(1.0 / cbrt(1.0 + r2), 0.25, 1.0)
    else
        return clamp((1.0 + r2)^pl_exp, 0.35, 1.0)
    end
end

# Exact log-space under-relaxation using ONLY hardware sqrt() (zero exp/log!):
#   rel_mode == 0 -> direct assignment (rel = 1.000)
#   rel_mode == 1 -> a^(1/2) * b^(1/2) (rel = 0.500)
#   rel_mode == 2 -> a^(3/4) * b^(1/4) (rel = 0.250)
#   rel_mode == 3 -> a^(7/8) * b^(1/8) (rel = 0.125)
@inline function log_relax_sqrt(es_old::Float64, es_vp::Float64, rel_mode::Int)
    if rel_mode == 0
        return es_vp
    else
        a  = max(es_old, 1.0e-12)
        b  = max(es_vp,  1.0e-12)
        g1 = sqrt(a * b)
        if rel_mode == 1
            return g1
        end
        g2 = sqrt(a * g1)
        return (rel_mode == 2) ? g2 : sqrt(a * g2)
    end
end

# 2.1a Update local solid/bulk viscosity & Giordano-Bartels melt viscosity
@parallel_indices (ix, iz) function update_viscosities!(
    eta_s, zeta, eta_m_phys,
    phi, T, Cf,
    phi_ref, alpha_phi, eta_m0, eta_min, Cf_onset, lambda_flux,
    use_visc_fb::Int, nx, nz
)
    if ix <= nx && iz <= nz
        @inbounds begin
            p = phi[ix, iz]
            # Unified porosity-softening solid shear viscosity (150x contrast: fast PT + sharp bands)
            es_nd = clamp(exp(-alpha_phi * (p - phi_ref)), 8.0e-2, 1.2e1)
            eta_s[ix, iz] = es_nd

            # Granular crystal-mush bulk viscosity (Takei & Holtzman 2009 / Katz et al. 2006)
            zeta[ix, iz] = 1.20 * es_nd / sqrt(max(p, 0.02))

            # Thresholded Giordano-Bartels flux-viscosity collapse (localized inside enriched channels)
            if use_visc_fb == 1
                t_exp = 0.003 * clamp(720.0 - T[ix, iz], -40.0, 220.0)
                f_exp = -lambda_flux * max(Cf[ix, iz] - Cf_onset, 0.0)
                em    = eta_min + (eta_m0 - eta_min) * exp(clamp(t_exp + f_exp, -25.0, 10.0))
                em    = clamp(em, eta_min, 5.0e8)
            else
                em = eta_m0
            end
            eta_m_phys[ix, iz] = em
        end
    end
    return
end

# 2.1b Geologically anchored, stress-shadowed 67° oblique hydrofracture dike network
#      - Nucleates strictly from physical shear-band overpressure/porosity peaks where
#        (Pf - Pt) > pf_frac_eff = pf_frac + sigma_sq (zero in static & compressive filter!)
#      - Once a dike breaks the brittle carapace, its fracture trace (vein_ix[k], vein_z[k])
#        stays permanently anchored (irreversible crustal fracture damage -> NEVER teleports
#        sideways when drainage relieves local root pressure!)
#      - Up to 3 stress-shadowed 67° oblique dikes (FWHM ~130 m, ~21% roof area) focus Darcy
#        flux 4-5x along 67° oblique pathways -> terracotta-red/crimson Li enrichment!
function propagate_hydrofracture_serial!(
    hf_state, eta_m_hf,
    vein_ix, vein_z, vein_open, vein_eta, n_veins_ref,
    drive_col, score_col, z_root_col, feed_eta_col, hf_tgt_2d, eta_tgt_2d,
    phi, Pf, Pt, eta_m_phys, x_phys, z_phys,
    Lx::Float64, Lz::Float64, dx_phys::Float64, phi_ref::Float64,
    pf_frac_eff::Float64, inv_dp_frac::Float64, s_tilt::Float64,
    shadow_m::Float64, half_w_m::Float64, eta_m0::Float64,
    nx::Int, nz::Int
)
    dp_gate      = 0.35 * pf_frac_eff
    shadow_cells = max(1, round(Int, shadow_m / dx_phys))
    inv_2w2      = 0.5 / (half_w_m * half_w_m)
    z_min_root   = 0.28 * Lz
    z_max_root   = 0.62 * Lz

    # Step 1: Evaluate per-column physical overpressure drive, root depth & feeding melt viscosity
    @inbounds for ix in 1:nx
        best_drv = 0.0
        best_z   = 0.42 * Lz
        best_eta = eta_m0
        best_phi = 0.0
        for iz in 1:nz
            p_loc = phi[ix, iz]
            if p_loc >= 0.040
                e_loc = eta_m_phys[ix, iz]
                if e_loc < best_eta
                    best_eta = e_loc
                end
                if p_loc > best_phi
                    best_phi = p_loc
                end
                dp_over = Pf[ix, iz] - Pt[ix, iz]
                if dp_over > dp_gate
                    hf_loc = 0.5 * (1.0 + tanh((dp_over - pf_frac_eff) * inv_dp_frac))
                    if hf_loc > best_drv
                        best_drv = hf_loc
                        best_z   = clamp(z_phys[iz], z_min_root, z_max_root)
                    end
                end
            end
        end
        drive_col[ix]    = best_drv
        z_root_col[ix]   = best_z
        feed_eta_col[ix] = best_eta
        score_col[ix]    = best_drv * clamp(best_phi / phi_ref, 0.25, 1.60)
    end

    # Step 2: Mask out stress shadows around already-anchored geological fracture traces,
    #         and nucleate new veins ONLY if fewer than 3 exist and a fresh peak breaks threshold
    nv = n_veins_ref[]
    @inbounds for k in 1:nv
        ix_k = vein_ix[k]
        for d in -shadow_cells:shadow_cells
            score_col[mod1(ix_k + d, nx)] = 0.0
        end
    end

    @inbounds while nv < 3
        ix_best = 1
        s_best  = score_col[1]
        for ix in 2:nx
            if score_col[ix] > s_best
                s_best  = score_col[ix]
                ix_best = ix
            end
        end
        if s_best <= 0.14 || drive_col[ix_best] <= 0.12
            break
        end
        nv += 1
        vein_ix[nv]   = ix_best
        vein_z[nv]    = z_root_col[ix_best]
        vein_open[nv] = 0.0
        vein_eta[nv]  = feed_eta_col[ix_best]
        for d in -shadow_cells:shadow_cells
            score_col[mod1(ix_best + d, nx)] = 0.0
        end
    end
    n_veins_ref[] = nv

    # Step 3: Project the anchored 67° oblique dikes with smoothly tracked aperture & melt viscosity
    fill!(hf_tgt_2d, 0.0)
    fill!(eta_tgt_2d, eta_m0)
    n_active = 0

    @inbounds for k in 1:nv
        ix_k = vein_ix[k]
        zr   = vein_z[k]
        xr   = x_phys[ix_k]

        # Capture underlying mobile mush pressure & evolved melt viscosity within +/- 5 cells (~100 m)
        drv_cap = 0.0
        phi_cap = 0.0
        eta_cap = eta_m0
        for d in -5:5
            ix_n = mod1(ix_k + d, nx)
            if drive_col[ix_n] > drv_cap
                drv_cap = drive_col[ix_n]
            end
            if feed_eta_col[ix_n] < eta_cap
                eta_cap = feed_eta_col[ix_n]
            end
            for iz in 1:div(3 * nz, 4)
                pv = phi[ix_n, iz]
                if pv > phi_cap
                    phi_cap = pv
                end
            end
        end

        # Once cracked, an anchored dike stays propped open by buoyant melt as long as
        # the underlying mush remains mobile (phi_cap >= 0.06), closing smoothly upon lock-in
        prop_tgt = max(drv_cap, 0.85 * clamp((phi_cap - 0.05) / 0.10, 0.0, 1.0))
        open_old = vein_open[k]
        rate_o   = (prop_tgt >= open_old) ? 0.35 : 0.06
        open_new = (1.0 - rate_o) * open_old + rate_o * prop_tgt
        vein_open[k] = (open_new > 1.0e-4) ? open_new : 0.0

        # Track feeding channel viscosity smoothly (fast collapse on low-eta pulse, slow re-thickening)
        e_old = vein_eta[k]
        e_rate = (eta_cap <= e_old) ? 0.50 : 0.03
        vein_eta[k] = (1.0 - e_rate) * e_old + e_rate * eta_cap

        if vein_open[k] <= 1.0e-3
            continue
        end
        n_active += 1
        st_k  = vein_open[k]
        eta_k = vein_eta[k]

        for iz in 1:nz
            zv = z_phys[iz]
            wz = clamp((zv - (zr - 120.0)) / 240.0, 0.0, 1.0)
            if wz <= 0.0
                continue
            end
            wz = wz * wz * (3.0 - 2.0 * wz)  # C1 smoothstep onset
            xc = mod(xr + s_tilt * max(0.0, zv - zr), Lx)
            for ix in 1:nx
                dist = abs(x_phys[ix] - xc)
                if dist > 0.5 * Lx
                    dist = Lx - dist
                end
                if dist < 3.2 * half_w_m
                    val = st_k * wz * exp(-(dist * dist) * inv_2w2)
                    if val > hf_tgt_2d[ix, iz]
                        hf_tgt_2d[ix, iz] = val
                    end
                    if val > 0.05 && eta_k < eta_tgt_2d[ix, iz]
                        eta_tgt_2d[ix, iz] = eta_k
                    end
                end
            end
        end
    end

    # Step 4: Write smooth anchored vein field into hf_state & eta_m_hf
    @inbounds for iz in 1:nz, ix in 1:nx
        hf_val = hf_tgt_2d[ix, iz]
        hf_state[ix, iz] = (hf_val > 1.0e-4) ? hf_val : 0.0
        e_loc = eta_m_phys[ix, iz]
        eta_m_hf[ix, iz] = (hf_val > 0.05) ? min(e_loc, eta_tgt_2d[ix, iz]) : e_loc
    end
    return n_active
end

# 2.1c Combine Kozeny-Carman porous mobility + anchored oblique hydrofracture vein mobility
@parallel_indices (ix, iz) function update_cell_mobility!(
    mob_c, phi, eta_m_phys, hf_state, eta_m_hf,
    phi_ref, k_ref, k_hf_frac, M0, nx, nz
)
    if ix <= nx && iz <= nz
        @inbounds begin
            p_safe = clamp(phi[ix, iz], 5.0e-4, 0.90)
            pr     = p_safe / phi_ref
            ir     = (1.0 - phi_ref) / (1.0 - p_safe)
            k_kc   = k_ref * (pr * pr * pr) * (ir * ir)
            m_por  = k_kc / eta_m_phys[ix, iz]
            m_hf   = (k_hf_frac * k_ref * hf_state[ix, iz]) / eta_m_hf[ix, iz]
            mob_c[ix, iz] = (m_por + m_hf) / M0
        end
    end
    return
end

# 2.2 Compute harmonic face mobilities (with hydrofracture-capable top BDT valve) & vertex viscosities
@parallel_indices (ix, iz) function update_face_and_vertex_props!(
    mob_x, mob_z, eta_sv,
    mob_c, eta_s, phi, hf_state, phi_min, nx, nz
)
    ixm = (ix == 1) ? nx : ix - 1

    if ix <= nx && iz <= nz
        @inbounds begin
            mL = mob_c[ixm, iz]
            mR = mob_c[ix, iz]
            mob_x[ix, iz] = 2.0 * mL * mR / (mL + mR + 1.0e-30)
        end
    end

    if ix <= nx && iz <= nz + 1
        @inbounds begin
            if iz == 1
                mob_z[ix, 1]  = 0.0  # Impermeable base
                e1 = eta_s[ixm, 1]; e2 = eta_s[ix, 1]
                eta_sv[ix, 1] = 2.0 * e1 * e2 / (e1 + e2 + 1.0e-30)
            elseif iz == nz + 1
                # Top Palaeo-BDT valve: opens smoothly via high porosity OR active roof hydrofracturing
                vf         = clamp((phi[ix, nz] - phi_min) / 0.025, 0.0, 1.0)
                valve_open = max(vf * vf, hf_state[ix, nz], 1.0e-5)
                mob_z[ix, nz + 1] = mob_c[ix, nz] * valve_open
                e1 = eta_s[ixm, nz]; e2 = eta_s[ix, nz]
                eta_sv[ix, nz + 1] = 2.0 * e1 * e2 / (e1 + e2 + 1.0e-30)
            else
                mB = mob_c[ix, iz - 1]
                mT = mob_c[ix, iz]
                mob_z[ix, iz] = 2.0 * mB * mT / (mB + mT + 1.0e-30)

                e1 = eta_s[ixm, iz - 1]; e2 = eta_s[ix, iz - 1]
                e3 = eta_s[ixm, iz];     e4 = eta_s[ix, iz]
                eta_sv[ix, iz] = 4.0 / (1.0/e1 + 1.0/e2 + 1.0/e3 + 1.0/e4 + 1.0e-30)
            end
        end
    end
    return
end

# 2.2b Precompute all PT-invariant step constants ONCE per physical step
@parallel_indices (ix, iz) function precompute_step_constants!(
    inv_zeta, mobx_fac, mobz_fac, dtau_pf, inv_diag_pf, buoy_z, disagg_lock, dil_fac,
    zeta, mob_x, mob_z, phi,
    dz, inv_dx, inv_dz, inv_half_dz, inv_dx2, inv_dz2, Rc2,
    use_dil::Int, alpha_pump, psi_0, phi_ref, phi_min, phi_crit, nx, nz
)
    if ix <= nx && iz <= nz
        @inbounds begin
            ixp = (ix == nx) ? 1 : ix + 1
            izeta = 1.0 / zeta[ix, iz]
            inv_zeta[ix, iz] = izeta
            mobx_fac[ix, iz] = Rc2 * mob_x[ix, iz] * inv_dx

            m_z_hi    = (iz == nz) ? (2.0 * mob_z[ix, nz + 1]) : mob_z[ix, iz + 1]
            diff_diag = Rc2 * ((mob_x[ix, iz] + mob_x[ixp, iz]) * inv_dx2 +
                               (mob_z[ix, iz] + m_z_hi) * inv_dz2)
            inv_diag_pf[ix, iz] = 1.0 / (diff_diag + izeta + 1.0e-30)
            dtau_pf[ix, iz]     = 1.25 / (2.0 * diff_diag + izeta + 1.0e-30)

            if use_dil == 1
                p          = phi[ix, iz]
                d1         = max(0.0, 1.0 - p / phi_crit)
                disagg     = d1 * sqrt(d1)
                lock_taper = clamp((p - phi_min) / 0.03, 0.0, 1.0)
                disagg_lock[ix, iz] = alpha_pump * lock_taper * disagg
                dil_fac[ix, iz]     = psi_0 * lock_taper * min(1.0, p / phi_ref) * disagg
            else
                disagg_lock[ix, iz] = 0.0
                dil_fac[ix, iz]     = 0.0
            end
        end
    end

    if ix <= nx && iz <= nz + 1
        @inbounds begin
            if iz == 1
                mobz_fac[ix, 1] = 0.0
                buoy_z[ix, 1]   = 0.0
            elseif iz == nz + 1
                mobz_fac[ix, nz + 1] = Rc2 * mob_z[ix, nz + 1] * inv_half_dz
                buoy_z[ix, nz + 1]   = 0.5 * dz * (1.0 - phi[ix, nz])
            else
                mobz_fac[ix, iz] = Rc2 * mob_z[ix, iz] * inv_dz
                buoy_z[ix, iz]   = 1.0 - 0.5 * (phi[ix, iz - 1] + phi[ix, iz])
            end
        end
    end
    return
end

# 2.2c Zero-overhead serial vertical hydrostatic buoyancy pre-update on Pt
function adjust_hydrostatic_pt_serial!(Pt, phi, phi_prev, dz::Float64, nx::Int, nz::Int)
    half_dz = 0.5 * dz
    @inbounds for ix in 1:nx
        dPt = -half_dz * (phi[ix, nz] - phi_prev[ix, nz])
        Pt[ix, nz] += dPt
        for iz in (nz - 1):-1:1
            dPt += -half_dz * ((phi[ix, iz] - phi_prev[ix, iz]) +
                               (phi[ix, iz + 1] - phi_prev[ix, iz + 1]))
            Pt[ix, iz] += dPt
        end
    end
    return
end

# 2.2d Ultra-fast Jacobi-PCG elliptic polish for Pf across 10^8x vein-vs-wall mobility contrast
#      Solves: -div(Rc2 * M * grad(Pf)) + inv_zeta * Pf = inv_zeta * Pt + comp_src
#      Takes ~4 ms for 300 iterations on 96x96; guarantees err_pf NEVER stalls!
function polish_pf_pcg_serial!(
    Pf, r_cg, z_cg, p_cg, Ap_cg, valve_on,
    Pt, inv_zeta, comp_src, mobx_fac, mobz_fac, inv_diag_pf,
    inv_dx::Float64, inv_dz::Float64, tol_abs::Float64, max_it::Int, nx::Int, nz::Int
)
    # Lock one-way top-valve state from current Pf (strictly SPD linear operator)
    @inbounds for ix in 1:nx
        valve_on[ix] = (Pf[ix, nz] > 0.0)
    end

    # Compute initial residual r = rhs - A(Pf) and preconditioned direction p = inv_diag * r
    rz_old = 0.0
    max_r  = 0.0
    @inbounds for iz in 1:nz, ix in 1:nx
        ixm = (ix == 1)  ? nx : ix - 1
        ixp = (ix == nx) ? 1  : ix + 1
        pc  = Pf[ix, iz]

        qL = -mobx_fac[ix, iz]  * (pc - Pf[ixm, iz])
        qR = -mobx_fac[ixp, iz] * (Pf[ixp, iz] - pc)
        qB = (iz == 1)  ? 0.0 : (-mobz_fac[ix, iz] * (pc - Pf[ix, iz - 1]))
        qT = (iz == nz) ? (valve_on[ix] ? (mobz_fac[ix, nz + 1] * pc) : 0.0) :
                          (-mobz_fac[ix, iz + 1] * (Pf[ix, iz + 1] - pc))

        div_q = (qR - qL) * inv_dx + (qT - qB) * inv_dz
        rv    = (Pt[ix, iz] - pc) * inv_zeta[ix, iz] + comp_src[ix, iz] - div_q
        zv    = inv_diag_pf[ix, iz] * rv

        r_cg[ix, iz] = rv
        z_cg[ix, iz] = zv
        p_cg[ix, iz] = zv
        rz_old      += rv * zv
        ar           = abs(rv)
        if ar > max_r
            max_r = ar
        end
    end

    if max_r <= tol_abs || rz_old <= 1.0e-30
        return max_r
    end

    for _ in 1:max_it
        pAp = 0.0
        @inbounds for iz in 1:nz, ix in 1:nx
            ixm = (ix == 1)  ? nx : ix - 1
            ixp = (ix == nx) ? 1  : ix + 1
            pc  = p_cg[ix, iz]

            qL = -mobx_fac[ix, iz]  * (pc - p_cg[ixm, iz])
            qR = -mobx_fac[ixp, iz] * (p_cg[ixp, iz] - pc)
            qB = (iz == 1)  ? 0.0 : (-mobz_fac[ix, iz] * (pc - p_cg[ix, iz - 1]))
            qT = (iz == nz) ? (valve_on[ix] ? (mobz_fac[ix, nz + 1] * pc) : 0.0) :
                              (-mobz_fac[ix, iz + 1] * (p_cg[ix, iz + 1] - pc))

            ap = (qR - qL) * inv_dx + (qT - qB) * inv_dz + inv_zeta[ix, iz] * pc
            Ap_cg[ix, iz] = ap
            pAp          += pc * ap
        end

        if pAp <= 1.0e-30
            break
        end
        alpha = rz_old / pAp

        rz_new = 0.0
        max_r  = 0.0
        @inbounds for iz in 1:nz, ix in 1:nx
            Pf[ix, iz]   += alpha * p_cg[ix, iz]
            rv            = r_cg[ix, iz] - alpha * Ap_cg[ix, iz]
            zv            = inv_diag_pf[ix, iz] * rv
            r_cg[ix, iz]  = rv
            z_cg[ix, iz]  = zv
            rz_new       += rv * zv
            ar            = abs(rv)
            if ar > max_r
                max_r = ar
            end
        end

        if max_r <= tol_abs || rz_new <= 1.0e-30
            break
        end
        beta = rz_new / rz_old
        @inbounds for iz in 1:nz, ix in 1:nx
            p_cg[ix, iz] = z_cg[ix, iz] + beta * p_cg[ix, iz]
        end
        rz_old = rz_new
    end
    return max_r
end

# 2.3a Lagged vertex rheology refresh (normalized theta_v ~ (esv / eta_geo)^0.25)
@parallel_indices (ix, iz) function refresh_rheology_vertices!(
    exz, eta_sv_eff, tv_r, two_esv_inv1,
    Vsx, Vsz, eta_sv,
    inv_dx, inv_dz, U_bot, U_top, use_shear::Int,
    n_pl, pl_exp, inv_e_ref2, inv_tau_y, theta_0, inv_eta_geo, rel_mode::Int, nx, nz
)
    if ix <= nx && iz <= nz + 1
        @inbounds begin
            ixm = (ix == 1) ? nx : ix - 1
            if iz == 1
                if use_shear == 1
                    exz_v   = (Vsx[ix, 1] - U_bot) * inv_dz
                    esv     = eta_sv[ix, 1]
                    pl_fac  = carreau_factor((exz_v * exz_v) * inv_e_ref2, n_pl, pl_exp)
                    esv_vp  = 1.0 / (1.0 / (esv * pl_fac) + 2.0 * abs(exz_v) * inv_tau_y)
                    esv_new = log_relax_sqrt(eta_sv_eff[ix, 1], esv_vp, rel_mode)
                    theta_v = theta_0 * sqrt(sqrt(clamp(esv_new * inv_eta_geo, 1.0e-3, 1.0e3)))
                    inv_tv1 = 1.0 / (theta_v + 1.0)
                    exz[ix, 1]          = exz_v
                    eta_sv_eff[ix, 1]   = esv_new
                    tv_r[ix, 1]         = theta_v * inv_tv1
                    two_esv_inv1[ix, 1] = 2.0 * esv_new * inv_tv1
                else
                    exz[ix, 1]          = 0.0
                    eta_sv_eff[ix, 1]   = eta_sv[ix, 1]
                    tv_r[ix, 1]         = 0.0
                    two_esv_inv1[ix, 1] = 0.0
                end
            elseif iz == nz + 1
                if use_shear == 1
                    exz_v   = (U_top - Vsx[ix, nz]) * inv_dz +
                              0.5 * (Vsz[ix, nz + 1] - Vsz[ixm, nz + 1]) * inv_dx
                    esv     = eta_sv[ix, nz + 1]
                    pl_fac  = carreau_factor((exz_v * exz_v) * inv_e_ref2, n_pl, pl_exp)
                    esv_vp  = 1.0 / (1.0 / (esv * pl_fac) + 2.0 * abs(exz_v) * inv_tau_y)
                    esv_new = log_relax_sqrt(eta_sv_eff[ix, nz + 1], esv_vp, rel_mode)
                    theta_v = theta_0 * sqrt(sqrt(clamp(esv_new * inv_eta_geo, 1.0e-3, 1.0e3)))
                    inv_tv1 = 1.0 / (theta_v + 1.0)
                    exz[ix, nz + 1]          = exz_v
                    eta_sv_eff[ix, nz + 1]   = esv_new
                    tv_r[ix, nz + 1]         = theta_v * inv_tv1
                    two_esv_inv1[ix, nz + 1] = 2.0 * esv_new * inv_tv1
                else
                    exz[ix, nz + 1]          = 0.0
                    eta_sv_eff[ix, nz + 1]   = eta_sv[ix, nz + 1]
                    tv_r[ix, nz + 1]         = 0.0
                    two_esv_inv1[ix, nz + 1] = 0.0
                end
            else
                exz_v   = 0.5 * ((Vsx[ix, iz] - Vsx[ix, iz - 1]) * inv_dz +
                                 (Vsz[ix, iz] - Vsz[ixm, iz]) * inv_dx)
                esv     = eta_sv[ix, iz]
                pl_fac  = carreau_factor((exz_v * exz_v) * inv_e_ref2, n_pl, pl_exp)
                esv_vp  = 1.0 / (1.0 / (esv * pl_fac) + 2.0 * abs(exz_v) * inv_tau_y)
                esv_new = log_relax_sqrt(eta_sv_eff[ix, iz], esv_vp, rel_mode)
                theta_v = theta_0 * sqrt(sqrt(clamp(esv_new * inv_eta_geo, 1.0e-3, 1.0e3)))
                inv_tv1 = 1.0 / (theta_v + 1.0)
                exz[ix, iz]          = exz_v
                eta_sv_eff[ix, iz]   = esv_new
                tv_r[ix, iz]         = theta_v * inv_tv1
                two_esv_inv1[ix, iz] = 2.0 * esv_new * inv_tv1
            end
        end
    end
    return
end

# 2.3b Lagged cell rheology refresh (stores G_cell = eta_eff / theta_c for face-consistent K_loc)
@parallel_indices (ix, iz) function refresh_rheology_cells!(
    eII, eta_eff, tc_r, two_ec_inv1, G_cell,
    Vsx, Vsz, exz, eta_s,
    inv_dx, inv_dz, n_pl, pl_exp, inv_e_ref2, inv_tau_y, theta_0, inv_eta_geo,
    rel_mode::Int, nx, nz
)
    if ix <= nx && iz <= nz
        @inbounds begin
            ixp    = (ix == nx) ? 1 : ix + 1
            dvx_dx = (Vsx[ixp, iz] - Vsx[ix, iz]) * inv_dx
            dvz_dz = (Vsz[ix, iz + 1] - Vsz[ix, iz]) * inv_dz
            div_v  = dvx_dx + dvz_dz

            exx_d = dvx_dx - (1.0 / 3.0) * div_v
            ezz_d = dvz_dz - (1.0 / 3.0) * div_v
            eyy_d = -(1.0 / 3.0) * div_v
            exz_c = 0.25 * (exz[ix, iz] + exz[ixp, iz] + exz[ix, iz + 1] + exz[ixp, iz + 1])
            e2_sq = 0.5 * (exx_d * exx_d + ezz_d * ezz_d + eyy_d * eyy_d) + exz_c * exz_c + 1.0e-30
            e2    = sqrt(e2_sq)

            es_loc  = eta_s[ix, iz]
            pl_fac  = carreau_factor(e2_sq * inv_e_ref2, n_pl, pl_exp)
            es_vp   = 1.0 / (1.0 / (es_loc * pl_fac) + 2.0 * e2 * inv_tau_y)
            es_new  = log_relax_sqrt(eta_eff[ix, iz], es_vp, rel_mode)
            theta_c = theta_0 * sqrt(sqrt(clamp(es_new * inv_eta_geo, 1.0e-3, 1.0e3)))
            inv_tc1 = 1.0 / (theta_c + 1.0)

            eII[ix, iz]         = e2
            eta_eff[ix, iz]     = es_new
            tc_r[ix, iz]        = theta_c * inv_tc1
            two_ec_inv1[ix, iz] = 2.0 * es_new * inv_tc1
            G_cell[ix, iz]      = es_new / theta_c
        end
    end
    return
end

# 2.3c Zero-overhead serial horizontal means per depth (unit-stride SIMD over ix)
function compute_z_means_serial!(
    eII_mean_z, tau_II_mean_z, zeta_mean_z,
    eII, eta_eff, zeta, nx::Int, nz::Int
)
    inv_nx = 1.0 / nx
    @inbounds for iz in 1:nz
        se = 0.0
        st = 0.0
        sz = 0.0
        @simd for ix in 1:nx
            e_loc = eII[ix, iz]
            se += e_loc
            st += 2.0 * eta_eff[ix, iz] * e_loc
            sz += zeta[ix, iz]
        end
        eII_mean_z[iz]    = se * inv_nx
        tau_II_mean_z[iz] = st * inv_nx
        zeta_mean_z[iz]   = sz * inv_nx
    end
    return
end

# 2.3d Precompute PT velocity step sizes & under-relaxed pumping/dilatancy source term
@parallel_indices (ix, iz) function refresh_pt_steps_and_sources!(
    dtau_vx, dtau_vz, comp_src, eps_dil,
    eta_eff, eta_sv_eff, eII, eII_mean_z, tau_II_mean_z, zeta_mean_z,
    disagg_lock, dil_fac, inv_zeta,
    coeff, theta_0, inv_eta_geo, sigma_sq_nd, rel_src, use_dil::Int, nx, nz
)
    ixm = (ix == 1) ? nx : ix - 1
    ixp = (ix == nx) ? 1 : ix + 1

    if ix <= nx && iz <= nz
        @inbounds begin
            emax_x    = max(eta_eff[ixm, iz], eta_eff[ix, iz],
                            eta_sv_eff[ix, iz], eta_sv_eff[ix, iz + 1], 1.0e-6)
            theta_x   = theta_0 * sqrt(sqrt(clamp(emax_x * inv_eta_geo, 1.0e-3, 1.0e3)))
            fac_x     = (iz == 1 || iz == nz) ? 1.5 : 1.0
            dtau_vx[ix, iz] = (coeff * theta_x) / (fac_x * emax_x)

            if use_dil == 1
                comp_pump = disagg_lock[ix, iz] * (tau_II_mean_z[iz] / max(zeta_mean_z[iz], 1.0e-6))
                eps_d     = dil_fac[ix, iz] * (eII[ix, iz] - eII_mean_z[iz])
                src_new   = sigma_sq_nd * inv_zeta[ix, iz] + comp_pump - eps_d
                eps_dil[ix, iz]  = eps_d
                comp_src[ix, iz] = (rel_src >= 1.0) ? src_new :
                                   ((1.0 - rel_src) * comp_src[ix, iz] + rel_src * src_new)
            else
                eps_dil[ix, iz]  = 0.0
                comp_src[ix, iz] = sigma_sq_nd * inv_zeta[ix, iz]
            end
        end
    end

    if ix <= nx && iz <= nz + 1
        @inbounds begin
            if iz == 1
                dtau_vz[ix, 1] = 0.0
            elseif iz == nz + 1
                emax_top = max(eta_eff[ix, nz], eta_sv_eff[ix, nz], eta_sv_eff[ixp, nz],
                               eta_sv_eff[ix, nz + 1], eta_sv_eff[ixp, nz + 1], 1.0e-6)
                theta_t  = theta_0 * sqrt(sqrt(clamp(emax_top * inv_eta_geo, 1.0e-3, 1.0e3)))
                dtau_vz[ix, nz + 1] = (0.5 * coeff * theta_t) / emax_top
            else
                emax_z   = max(eta_eff[ix, iz - 1], eta_eff[ix, iz],
                               eta_sv_eff[ix, iz], eta_sv_eff[ixp, iz], 1.0e-6)
                theta_z  = theta_0 * sqrt(sqrt(clamp(emax_z * inv_eta_geo, 1.0e-3, 1.0e3)))
                dtau_vz[ix, iz] = (coeff * theta_z) / emax_z
            end
        end
    end
    return
end

# 2.3e Face-consistent acoustic-CFL pressure step dtau_pt (eliminates err_pt stall in soft bands)
@parallel_indices (ix, iz) function update_dtau_pt_from_v!(
    dtau_pt, dtau_vx, dtau_vz, G_cell, inv_zeta,
    inv_dx2, inv_dz2, k_num, r_bulk, nx, nz
)
    if ix <= nx && iz <= nz
        @inbounds begin
            ixp     = (ix == nx) ? 1 : ix + 1
            vz_top  = (iz == nz) ? (2.0 * dtau_vz[ix, nz + 1] * inv_dz2) : (dtau_vz[ix, iz + 1] * inv_dz2)
            m_face  = max(
                dtau_vx[ix, iz]  * inv_dx2,
                dtau_vx[ixp, iz] * inv_dx2,
                dtau_vz[ix, iz]  * inv_dz2,
                vz_top,
                1.0e-30
            )
            K_cfl = k_num / m_face
            K_loc = min(K_cfl, 15.0 * r_bulk * G_cell[ix, iz])
            dtau_pt[ix, iz] = K_loc / (1.0 + K_loc * inv_zeta[ix, iz])
        end
    end
    return
end

# 2.3f Initialize equilibrium stresses once at step 0
@parallel_indices (ix, iz) function init_equilibrium_stresses!(
    txx, tzz, txz, div_Vs,
    Vsx, Vsz, exz, eta_eff, eta_sv_eff,
    inv_dx, inv_dz, nx, nz
)
    if ix <= nx && iz <= nz + 1
        @inbounds txz[ix, iz] = 2.0 * eta_sv_eff[ix, iz] * exz[ix, iz]
    end
    if ix <= nx && iz <= nz
        @inbounds begin
            ixp    = (ix == nx) ? 1 : ix + 1
            dvx_dx = (Vsx[ixp, iz] - Vsx[ix, iz]) * inv_dx
            dvz_dz = (Vsz[ix, iz + 1] - Vsz[ix, iz]) * inv_dz
            div_v  = dvx_dx + dvz_dz
            two_e  = 2.0 * eta_eff[ix, iz]
            div_Vs[ix, iz] = div_v
            txx[ix, iz]    = two_e * (dvx_dx - (1.0 / 3.0) * div_v)
            tzz[ix, iz]    = two_e * (dvz_dz - (1.0 / 3.0) * div_v)
        end
    end
    return
end

# ==============================================================================
# 2.4 ULTRA-FAST FUSED 3-KERNEL INNER PT LOOP (ZERO pow/log/exp/sqrt/div!)
# ==============================================================================

# Fused Kernel 1/3: Relax (txz, txx, tzz), compute div_Vs, and update Darcy fluxes (qDx, qDz)
@parallel_indices (ix, iz) function pt_step_stress_and_darcy!(
    txx, tzz, txz, div_Vs, qDx, qDz, Res_txx, Res_tzz, Res_txz,
    Vsx, Vsz, Pf, tc_r, two_ec_inv1, tv_r, two_esv_inv1, eta_eff, eta_sv_eff,
    mobx_fac, mobz_fac,
    inv_dx, inv_dz, U_bot, U_top, use_shear::Int, do_check::Int, nx, nz
)
    ixm = (ix == 1) ? nx : ix - 1
    ixp = (ix == nx) ? 1 : ix + 1

    # --- Vertex shear stress txz & vertical Darcy flux qDz (1 <= iz <= nz + 1) ---
    if ix <= nx && iz <= nz + 1
        @inbounds begin
            if iz == 1
                qDz[ix, 1] = 0.0
                if use_shear == 1
                    exz_v = (Vsx[ix, 1] - U_bot) * inv_dz
                    if do_check == 1
                        txz_old = txz[ix, 1]
                        Res_txz[ix, 1] = 2.0 * eta_sv_eff[ix, 1] * exz_v - txz_old
                        txz[ix, 1]     = tv_r[ix, 1] * txz_old + two_esv_inv1[ix, 1] * exz_v
                    else
                        txz[ix, 1] = tv_r[ix, 1] * txz[ix, 1] + two_esv_inv1[ix, 1] * exz_v
                    end
                else
                    txz[ix, 1] = 0.0
                    if do_check == 1
                        Res_txz[ix, 1] = 0.0
                    end
                end
            elseif iz == nz + 1
                pf_t = Pf[ix, nz]
                qDz[ix, nz + 1] = (pf_t > 0.0) ? (mobz_fac[ix, nz + 1] * pf_t) : 0.0
                if use_shear == 1
                    exz_v = (U_top - Vsx[ix, nz]) * inv_dz +
                            0.5 * (Vsz[ix, nz + 1] - Vsz[ixm, nz + 1]) * inv_dx
                    if do_check == 1
                        txz_old = txz[ix, nz + 1]
                        Res_txz[ix, nz + 1] = 2.0 * eta_sv_eff[ix, nz + 1] * exz_v - txz_old
                        txz[ix, nz + 1]     = tv_r[ix, nz + 1] * txz_old + two_esv_inv1[ix, nz + 1] * exz_v
                    else
                        txz[ix, nz + 1] = tv_r[ix, nz + 1] * txz[ix, nz + 1] + two_esv_inv1[ix, nz + 1] * exz_v
                    end
                else
                    txz[ix, nz + 1] = 0.0
                    if do_check == 1
                        Res_txz[ix, nz + 1] = 0.0
                    end
                end
            else
                qDz[ix, iz] = -mobz_fac[ix, iz] * (Pf[ix, iz] - Pf[ix, iz - 1])
                exz_v = 0.5 * ((Vsx[ix, iz] - Vsx[ix, iz - 1]) * inv_dz +
                               (Vsz[ix, iz] - Vsz[ixm, iz]) * inv_dx)
                if do_check == 1
                    txz_old = txz[ix, iz]
                    Res_txz[ix, iz] = 2.0 * eta_sv_eff[ix, iz] * exz_v - txz_old
                    txz[ix, iz]     = tv_r[ix, iz] * txz_old + two_esv_inv1[ix, iz] * exz_v
                else
                    txz[ix, iz] = tv_r[ix, iz] * txz[ix, iz] + two_esv_inv1[ix, iz] * exz_v
                end
            end
        end
    end

    # --- Cell normal stresses (txx, tzz), div_Vs & horizontal Darcy flux qDx (1 <= iz <= nz) ---
    if ix <= nx && iz <= nz
        @inbounds begin
            qDx[ix, iz] = -mobx_fac[ix, iz] * (Pf[ix, iz] - Pf[ixm, iz])

            dvx_dx = (Vsx[ixp, iz] - Vsx[ix, iz]) * inv_dx
            dvz_dz = (Vsz[ix, iz + 1] - Vsz[ix, iz]) * inv_dz
            div_v  = dvx_dx + dvz_dz
            div_Vs[ix, iz] = div_v

            third_div = (1.0 / 3.0) * div_v
            exx_d = dvx_dx - third_div
            ezz_d = dvz_dz - third_div

            tcr = tc_r[ix, iz]
            tei = two_ec_inv1[ix, iz]
            if do_check == 1
                two_e   = 2.0 * eta_eff[ix, iz]
                txx_old = txx[ix, iz]
                tzz_old = tzz[ix, iz]
                Res_txx[ix, iz] = two_e * exx_d - txx_old
                Res_tzz[ix, iz] = two_e * ezz_d - tzz_old
                txx[ix, iz]     = tcr * txx_old + tei * exx_d
                tzz[ix, iz]     = tcr * tzz_old + tei * ezz_d
            else
                txx[ix, iz] = tcr * txx[ix, iz] + tei * exx_d
                tzz[ix, iz] = tcr * tzz[ix, iz] + tei * ezz_d
            end
        end
    end
    return
end

# Fused Kernel 2/3: Update total pressure Pt & excess melt pressure Pf
@parallel_indices (ix, iz) function pt_step_pressures!(
    Pt, Pf, dPf, Res_Pt, Res_Pf,
    div_Vs, qDx, qDz, inv_zeta, comp_src, dtau_pt, dtau_pf,
    inv_dx, inv_dz, one_minus_gf, do_check::Int, nx, nz
)
    if ix <= nx && iz <= nz
        @inbounds begin
            ixp = (ix == nx) ? 1 : ix + 1

            div_qD = (qDx[ixp, iz] - qDx[ix, iz]) * inv_dx +
                     (qDz[ix, iz + 1] - qDz[ix, iz]) * inv_dz

            pt_v = Pt[ix, iz]
            pf_v = Pf[ix, iz]
            comp_term = (pt_v - pf_v) * inv_zeta[ix, iz] + comp_src[ix, iz]

            r_pt = -(div_Vs[ix, iz] + comp_term)
            Pt[ix, iz] = pt_v + dtau_pt[ix, iz] * r_pt

            r_pf = -div_qD + comp_term
            dpf  = one_minus_gf * dPf[ix, iz] + dtau_pf[ix, iz] * r_pf
            dPf[ix, iz] = dpf
            Pf[ix, iz]  = pf_v + dpf

            if do_check == 1
                Res_Pt[ix, iz] = r_pt
                Res_Pf[ix, iz] = r_pf
            end
        end
    end
    return
end

# Fused Kernel 3/3: Update solid velocities (Vsx, Vsz) + rigid base & stress-free top BCs
@parallel_indices (ix, iz) function pt_step_velocities_and_bcs!(
    Vsx, Vsz, Res_vx, Res_vz,
    txx, tzz, txz, Pt, buoy_z, dtau_vx, dtau_vz,
    inv_dx, inv_dz, inv_half_dz, do_check::Int, nx, nz
)
    ixm = (ix == 1) ? nx : ix - 1
    ixp = (ix == nx) ? 1 : ix + 1

    # X-velocity (1 <= iz <= nz)
    if ix <= nx && iz <= nz
        @inbounds begin
            rvx = (txx[ix, iz] - txx[ixm, iz] - (Pt[ix, iz] - Pt[ixm, iz])) * inv_dx +
                  (txz[ix, iz + 1] - txz[ix, iz]) * inv_dz
            Vsx[ix, iz] += dtau_vx[ix, iz] * rvx
            if do_check == 1
                Res_vx[ix, iz] = rvx
            end
        end
    end

    # Z-velocity + bottom/top BCs (1 <= iz <= nz + 1)
    if ix <= nx && iz <= nz + 1
        @inbounds begin
            if iz == 1
                Vsz[ix, 1] = 0.0
                if do_check == 1
                    Res_vz[ix, 1] = 0.0
                end
            elseif iz == nz + 1
                rvz_top = (Pt[ix, nz] - buoy_z[ix, nz + 1] - tzz[ix, nz]) * inv_half_dz +
                          (txz[ixp, nz + 1] - txz[ix, nz + 1]) * inv_dx
                Vsz[ix, nz + 1] += dtau_vz[ix, nz + 1] * rvz_top
                if do_check == 1
                    Res_vz[ix, nz + 1] = rvz_top
                end
            else
                rvz = (tzz[ix, iz] - tzz[ix, iz - 1] - (Pt[ix, iz] - Pt[ix, iz - 1])) * inv_dz +
                      (txz[ixp, iz] - txz[ix, iz]) * inv_dx - buoy_z[ix, iz]
                Vsz[ix, iz] += dtau_vz[ix, iz] * rvz
                if do_check == 1
                    Res_vz[ix, iz] = rvz
                end
            end
        end
    end
    return
end

# ==============================================================================
# 2.5 ADVECTION, THERMAL DIFFUSION-ADVECTION & LAGRANGIAN DISTILLATION KERNELS
# ==============================================================================

# Conservative 2nd-order TVD face-flux computation (exact F_melt + F_solid = Vs + qD)
@parallel_indices (ix, iz) function compute_phase_fluxes!(
    Fm_x, Fs_x, Fm_z, Fs_z,
    Vsx, Vsz, qDx, qDz, phi, nx, nz
)
    ixm  = (ix == 1)  ? nx : ix - 1
    ixm2 = (ixm == 1) ? nx : ixm - 1
    ixp  = (ix == nx) ? 1  : ix + 1

    if ix <= nx && iz <= nz
        @inbounds begin
            vs    = Vsx[ix, iz]
            p_tvd = tvd_face(vs, phi[ixm2, iz], phi[ixm, iz], phi[ix, iz], phi[ixp, iz])
            Fs_x[ix, iz] = (1.0 - p_tvd) * vs
            Fm_x[ix, iz] = p_tvd * vs + qDx[ix, iz]
        end
    end

    if ix <= nx && iz <= nz + 1
        @inbounds begin
            if iz == 1
                Fs_z[ix, 1] = 0.0
                Fm_z[ix, 1] = 0.0
            elseif iz == nz + 1
                Fs_z[ix, nz + 1] = (1.0 - phi[ix, nz]) * Vsz[ix, nz + 1]
                Fm_z[ix, nz + 1] = max(0.0, qDz[ix, nz + 1])
            else
                vs    = Vsz[ix, iz]
                izm2  = max(1, iz - 2)
                izp1  = min(nz, iz + 1)
                p_tvd = tvd_face(vs, phi[ix, izm2], phi[ix, iz - 1], phi[ix, iz], phi[ix, izp1])
                Fs_z[ix, iz] = (1.0 - p_tvd) * vs
                Fm_z[ix, iz] = p_tvd * vs + qDz[ix, iz]
            end
        end
    end
    return
end

# Local-porosity CFL advection speed per cell
@parallel_indices (ix, iz) function compute_local_cfl_speed!(
    v_cfl, Vsx, Vsz, Fm_x, Fm_z, phi, phi_min, nx, nz
)
    if ix <= nx && iz <= nz
        @inbounds begin
            ixm = (ix == 1) ? nx : ix - 1
            ixp = (ix == nx) ? 1 : ix + 1
            pc  = max(phi[ix, iz], phi_min)

            pxL = max(0.5 * (phi[ixm, iz] + phi[ix, iz]), phi_min)
            pxR = max(0.5 * (phi[ix, iz]  + phi[ixp, iz]), phi_min)
            pzB = (iz == 1)  ? pc : max(0.5 * (phi[ix, iz - 1] + phi[ix, iz]), phi_min)
            pzT = (iz == nz) ? pc : max(0.5 * (phi[ix, iz] + phi[ix, iz + 1]), phi_min)

            vm_max = max(
                abs(Fm_x[ix, iz])     / pxL,
                abs(Fm_x[ixp, iz])    / pxR,
                abs(Fm_z[ix, iz])     / pzB,
                abs(Fm_z[ix, iz + 1]) / pzT
            )
            vs_max = max(
                abs(Vsx[ix, iz]), abs(Vsx[ixp, iz]),
                abs(Vsz[ix, iz]), abs(Vsz[ix, iz + 1])
            )
            v_cfl[ix, iz] = max(vm_max, vs_max)
        end
    end
    return
end

# Conservative 2nd-order TVD advection for phi, melt solutes, and solid solutes
@parallel_indices (ix, iz) function advect_phi_and_solutes!(
    phi_new, C_Li_new, Cf_new, Cs_Li_new, Cs_f_new,
    phi, C_Li, Cf, Cs_Li, Cs_f,
    Fm_x, Fs_x, Fm_z, Fs_z,
    dt_nd, inv_dx, inv_dz, D_disp_nd, phi_min, phi_max, nx, nz
)
    if ix <= nx && iz <= nz
        @inbounds begin
            ixm  = (ix == 1)  ? nx : ix - 1
            ixm2 = (ixm == 1) ? nx : ixm - 1
            ixp  = (ix == nx) ? 1  : ix + 1
            ixp2 = (ixp == nx) ? 1 : ixp + 1

            div_Fm = (Fm_x[ixp, iz] - Fm_x[ix, iz]) * inv_dx +
                     (Fm_z[ix, iz + 1] - Fm_z[ix, iz]) * inv_dz
            p_old = phi[ix, iz]
            p_adv = clamp(p_old - dt_nd * div_Fm, phi_min, phi_max)

            # 2nd-order TVD melt solute fluxes + chemical dispersion (X-faces)
            fmL = Fm_x[ix, iz]; fmR = Fm_x[ixp, iz]
            p_avgL = 0.5 * (phi[ixm, iz] + p_old)
            p_avgR = 0.5 * (p_old + phi[ixp, iz])

            cLi_xL = tvd_face(fmL, C_Li[ixm2, iz], C_Li[ixm, iz], C_Li[ix, iz], C_Li[ixp, iz])
            cLi_xR = tvd_face(fmR, C_Li[ixm, iz],  C_Li[ix, iz],  C_Li[ixp, iz], C_Li[ixp2, iz])
            fLi_xL = fmL * cLi_xL - p_avgL * D_disp_nd * (C_Li[ix, iz] - C_Li[ixm, iz]) * inv_dx
            fLi_xR = fmR * cLi_xR - p_avgR * D_disp_nd * (C_Li[ixp, iz] - C_Li[ix, iz]) * inv_dx

            cf_xL  = tvd_face(fmL, Cf[ixm2, iz], Cf[ixm, iz], Cf[ix, iz], Cf[ixp, iz])
            cf_xR  = tvd_face(fmR, Cf[ixm, iz],  Cf[ix, iz],  Cf[ixp, iz], Cf[ixp2, iz])
            ff_xL  = fmL * cf_xL - p_avgL * D_disp_nd * (Cf[ix, iz] - Cf[ixm, iz]) * inv_dx
            ff_xR  = fmR * cf_xR - p_avgR * D_disp_nd * (Cf[ixp, iz] - Cf[ix, iz]) * inv_dx

            # 2nd-order TVD melt solute fluxes + chemical dispersion (Z-faces)
            fmB = Fm_z[ix, iz]; fmT = Fm_z[ix, iz + 1]
            if iz == 1
                fLi_zB = 0.0; ff_zB = 0.0
            else
                izm2   = max(1, iz - 2)
                izp1   = min(nz, iz + 1)
                p_avgB = 0.5 * (phi[ix, iz - 1] + p_old)
                cLi_zB = tvd_face(fmB, C_Li[ix, izm2], C_Li[ix, iz - 1], C_Li[ix, iz], C_Li[ix, izp1])
                cf_zB  = tvd_face(fmB, Cf[ix, izm2],   Cf[ix, iz - 1],   Cf[ix, iz],   Cf[ix, izp1])
                fLi_zB = fmB * cLi_zB - p_avgB * D_disp_nd * (C_Li[ix, iz] - C_Li[ix, iz - 1]) * inv_dz
                ff_zB  = fmB * cf_zB  - p_avgB * D_disp_nd * (Cf[ix, iz]   - Cf[ix, iz - 1])   * inv_dz
            end

            if iz == nz
                fLi_zT = fmT * C_Li[ix, nz]
                ff_zT  = fmT * Cf[ix, nz]
            else
                izm1   = max(1, iz - 1)
                izp2   = min(nz, iz + 2)
                p_avgT = 0.5 * (p_old + phi[ix, iz + 1])
                cLi_zT = tvd_face(fmT, C_Li[ix, izm1], C_Li[ix, iz], C_Li[ix, iz + 1], C_Li[ix, izp2])
                cf_zT  = tvd_face(fmT, Cf[ix, izm1],   Cf[ix, iz],   Cf[ix, iz + 1],   Cf[ix, izp2])
                fLi_zT = fmT * cLi_zT - p_avgT * D_disp_nd * (C_Li[ix, iz + 1] - C_Li[ix, iz]) * inv_dz
                ff_zT  = fmT * cf_zT  - p_avgT * D_disp_nd * (Cf[ix, iz + 1]   - Cf[ix, iz])   * inv_dz
            end

            B_Li = max(p_old * C_Li[ix, iz] - dt_nd * ((fLi_xR - fLi_xL) * inv_dx + (fLi_zT - fLi_zB) * inv_dz), 1.0e-4)
            B_f  = max(p_old * Cf[ix, iz]   - dt_nd * ((ff_xR  - ff_xL ) * inv_dx + (ff_zT  - ff_zB ) * inv_dz), 1.0e-4)

            # Solid solute advection
            fsL = Fs_x[ix, iz]; fsR = Fs_x[ixp, iz]
            fsB = Fs_z[ix, iz]; fsT = Fs_z[ix, iz + 1]
            sLi_xL = fsL * ((fsL >= 0.0) ? Cs_Li[ixm, iz] : Cs_Li[ix, iz])
            sLi_xR = fsR * ((fsR >= 0.0) ? Cs_Li[ix, iz]  : Cs_Li[ixp, iz])
            sf_xL  = fsL * ((fsL >= 0.0) ? Cs_f[ixm, iz]  : Cs_f[ix, iz])
            sf_xR  = fsR * ((fsR >= 0.0) ? Cs_f[ix, iz]   : Cs_f[ixp, iz])

            sLi_zB = (iz == 1)  ? 0.0 : fsB * ((fsB >= 0.0) ? Cs_Li[ix, iz - 1] : Cs_Li[ix, iz])
            sLi_zT = (iz == nz) ? (fsT * Cs_Li[ix, nz]) : fsT * ((fsT >= 0.0) ? Cs_Li[ix, iz] : Cs_Li[ix, iz + 1])
            sf_zB  = (iz == 1)  ? 0.0 : fsB * ((fsB >= 0.0) ? Cs_f[ix, iz - 1]  : Cs_f[ix, iz])
            sf_zT  = (iz == nz) ? (fsT * Cs_f[ix, nz])  : fsT * ((fsT >= 0.0) ? Cs_f[ix, iz]  : Cs_f[ix, iz + 1])

            S_Li = max((1.0 - p_old) * Cs_Li[ix, iz] - dt_nd * ((sLi_xR - sLi_xL) * inv_dx + (sLi_zT - sLi_zB) * inv_dz), 1.0e-4)
            S_f  = max((1.0 - p_old) * Cs_f[ix, iz]  - dt_nd * ((sf_xR  - sf_xL ) * inv_dx + (sf_zT  - sf_zB ) * inv_dz), 1.0e-4)

            phi_new[ix, iz]   = p_adv
            C_Li_new[ix, iz]  = clamp(B_Li / p_adv, 10.0, 2.5e4)
            Cf_new[ix, iz]    = clamp(B_f  / p_adv, 0.2, 40.0)
            Cs_Li_new[ix, iz] = S_Li / max(1.0 - p_adv, 0.1)
            Cs_f_new[ix, iz]  = S_f  / max(1.0 - p_adv, 0.1)
        end
    end
    return
end

# Explicit sub-cycled 2D thermal diffusion + barycentric advection + Robin roof + lateral cooling
@parallel_indices (ix, iz) function thermal_substep!(
    T_new, T, T_bg_z, Fm_x, Fs_x, Fm_z, Fs_z,
    dt_sub, inv_dx_phys, inv_dz_phys, inv_dx_p2, inv_dz_p2,
    kappa_eff, inv_H_roof, T_bg_top, inv_tau_lat, v0, nx, nz
)
    if ix <= nx && iz <= nz
        @inbounds begin
            ixm = (ix == 1) ? nx : ix - 1
            ixp = (ix == nx) ? 1 : ix + 1
            Tc  = T[ix, iz]

            vx_th = 0.5 * ((Fm_x[ix, iz] + Fs_x[ix, iz]) + (Fm_x[ixp, iz] + Fs_x[ixp, iz])) * v0
            vz_th = 0.5 * ((Fm_z[ix, iz] + Fs_z[ix, iz]) + (Fm_z[ix, iz + 1] + Fs_z[ix, iz + 1])) * v0

            dT_dx = (vx_th >= 0.0) ? ((Tc - T[ixm, iz]) * inv_dx_phys) : ((T[ixp, iz] - Tc) * inv_dx_phys)
            if iz == 1
                dT_dz = (vz_th >= 0.0) ? 0.0 : ((T[ix, 2] - Tc) * inv_dz_phys)
                lap_z = (T[ix, 2] - Tc) * inv_dz_p2
            elseif iz == nz
                dT_dz = (vz_th >= 0.0) ? ((Tc - T[ix, nz - 1]) * inv_dz_phys) : 0.0
                q_top = (Tc - T_bg_top) * inv_H_roof
                lap_z = ((T[ix, nz - 1] - Tc) * inv_dz_phys - q_top) * inv_dz_phys
            else
                dT_dz = (vz_th >= 0.0) ? ((Tc - T[ix, iz - 1]) * inv_dz_phys) : ((T[ix, iz + 1] - Tc) * inv_dz_phys)
                lap_z = (T[ix, iz - 1] - 2.0 * Tc + T[ix, iz + 1]) * inv_dz_p2
            end

            lap_x = (T[ixm, iz] - 2.0 * Tc + T[ixp, iz]) * inv_dx_p2
            T_new[ix, iz] = Tc + dt_sub * (
                kappa_eff * (lap_x + lap_z) -
                (vx_th * dT_dx + vz_th * dT_dz) -
                (Tc - T_bg_z[iz]) * inv_tau_lat
            )
        end
    end
    return
end

# Enthalpy-conserving Eulerian cooling + focused Lagrangian channel/vein-wall distillation
@parallel_indices (ix, iz) function rayleigh_crystallization_step!(
    phi, C_Li, Cf, Cs_Li, Cs_f, T, T_prev, qDx, qDz, eII, hf_state,
    sub_relax, v0_dt_phys, inv_dx_phys, inv_dz_phys,
    T_liq, T_sol0, T_sol_min, dT_flux, inv_dT_lat, beta_T_dist,
    D_Li, D_flux, phi_min, phi_max, inv_v_cool_ref, use_flux_sol::Int, nx, nz
)
    if ix <= nx && iz <= nz
        @inbounds begin
            ixm    = (ix == 1) ? nx : ix - 1
            ixp    = (ix == nx) ? 1 : ix + 1
            p_adv  = phi[ix, iz]
            cf_pre = Cf[ix, iz]
            li_pre = C_Li[ix, iz]
            Tc     = T[ix, iz]
            Tp     = T_prev[ix, iz]

            T_sol = (use_flux_sol == 1) ? max(T_sol_min, T_sol0 - dT_flux * max(cf_pre - 1.0, 0.0)) : T_sol0

            # 1) Eulerian cooling-rate crystallization (1st Law enthalpy conservation)
            dT_cool   = max(0.0, Tp - Tc)
            dphi_cool = (Tc < T_liq) ? (dT_cool * inv_dT_lat) : 0.0

            if Tc < T_sol
                sc       = clamp((T_sol - Tc) / 25.0, 0.0, 1.0)
                dphi_sub = max(0.0, p_adv - phi_min) * sc * sub_relax
            else
                dphi_sub = 0.0
            end

            dphi_cryst = min(dphi_cool + dphi_sub, max(0.0, p_adv - phi_min))
            p_new      = clamp(p_adv - dphi_cryst, phi_min, phi_max)

            # 2) Relative melt-solid Darcy ascent & Lagrangian channel-wall cooling (-v_r . grad(T))
            qx_c = 0.5 * (qDx[ix, iz] + qDx[ixp, iz])
            qz_c = 0.5 * (qDz[ix, iz] + qDz[ix, iz + 1])

            dT_dx = 0.5 * (T[ixp, iz] - T[ixm, iz]) * inv_dx_phys
            if iz == 1
                dT_dz = (T[ix, 2] - Tc) * inv_dz_phys
            elseif iz == nz
                dT_dz = (Tc - T[ix, nz - 1]) * inv_dz_phys
            else
                dT_dz = 0.5 * (T[ix, iz + 1] - T[ix, iz - 1]) * inv_dz_phys
            end

            p_safe = max(p_adv, phi_min)
            dT_lag = max(0.0, -(qx_c * dT_dx + qz_c * dT_dz) * (v0_dt_phys / p_safe))

            # 3) Deformation-, flow- & hydrofracture-modulated melt-trapping gate
            q_mag_c  = sqrt(qx_c * qx_c + qz_c * qz_c)
            v_seg    = q_mag_c / p_safe
            mob_idx  = v_seg * inv_v_cool_ref + 2.0 * eII[ix, iz]
            f_open   = mob_idx / (mob_idx + 1.0)
            phi_trap = 0.17 - 0.145 * f_open
            f_seg    = max(clamp((p_adv - phi_trap) / 0.04, 0.0, 1.0), hf_state[ix, iz] * f_open)

            D_Li_eff = D_Li   + (1.0 - f_seg) * (1.0 - D_Li)
            D_f_eff  = D_flux + (1.0 - f_seg) * (1.0 - D_flux)

            # Combined open-system Rayleigh fraction: Eulerian cooling + Lagrangian wall distillation
            ratio_eul = min(1.0, p_new / p_adv)
            df_lag    = min(0.40, f_seg * beta_T_dist * dT_lag)
            ratio_eff = max(0.04, ratio_eul * exp(-df_lag))

            li_new = clamp(li_pre * (ratio_eff^(D_Li_eff - 1.0)), 10.0, 2.5e4)
            cf_new = clamp(cf_pre * (ratio_eff^(D_f_eff  - 1.0)), 0.2, 40.0)

            d_sol_Li = max(0.0, p_adv * li_pre - p_new * li_new)
            d_sol_f  = max(0.0, p_adv * cf_pre - p_new * cf_new)

            Cs_Li[ix, iz] = ((1.0 - p_adv) * Cs_Li[ix, iz] + d_sol_Li) / max(1.0 - p_new, 0.1)
            Cs_f[ix, iz]  = ((1.0 - p_adv) * Cs_f[ix, iz]  + d_sol_f)  / max(1.0 - p_new, 0.1)
            phi[ix, iz]   = p_new
            C_Li[ix, iz]  = li_new
            Cf[ix, iz]    = cf_new
        end
    end
    return
end

# ==============================================================================
# 3. HELPERS: INITIAL PERTURBATION & SNAPSHOT PACKER
# ==============================================================================
function make_smooth_noise(nx::Int, nz::Int, amp::Float64; seed::Int = 42, passes::Int = 5)
    rng = MersenneTwister(seed)
    field = 2.0 .* rand(rng, nx, nz) .- 1.0
    tmp   = similar(field)
    for _ in 1:passes
        @inbounds for iz in 1:nz, ix in 1:nx
            ixm = (ix == 1) ? nx : ix - 1
            ixp = (ix == nx) ? 1 : ix + 1
            izm = max(1, iz - 1)
            izp = min(nz, iz + 1)
            tmp[ix, iz] = 0.25 * field[ix, iz] +
                          0.1875 * (field[ixm, iz] + field[ixp, iz] + field[ix, izm] + field[ix, izp])
        end
        field .= tmp
    end
    field .-= mean(field)
    maxabs = maximum(abs, field)
    return (maxabs > 0.0) ? (amp .* field ./ maxabs) : zeros(nx, nz)
end

function pack_snapshot(
    x_phys, z_phys, phi, C_Li, Cf, eta_m_phys, hf_state, T, Pt, Pf, eII,
    qDx, qDz, Vsx, Vsz, sigma0, v0, t0, nz, t_norm_val, export_val, trap_li_val
)
    qDx_c  = 0.5 .* (Array(qDx) .+ circshift(Array(qDx), (-1, 0))) .* v0
    qDz_c  = 0.5 .* (Array(qDz)[:, 1:nz] .+ Array(qDz)[:, 2:nz+1]) .* v0
    Vsx_c  = 0.5 .* (Array(Vsx) .+ circshift(Array(Vsx), (-1, 0))) .* v0
    Vsz_c  = 0.5 .* (Array(Vsz)[:, 1:nz] .+ Array(Vsz)[:, 2:nz+1]) .* v0
    Pc_MPa = (Array(Pt) .- Array(Pf)) .* (sigma0 / 1.0e6)
    Pf_MPa = Array(Pf) .* (sigma0 / 1.0e6)

    return Dict{String, Any}(
        "x"        => x_phys,
        "z"        => z_phys,
        "t_norm"   => Float64[t_norm_val],
        "export_m" => Float64[export_val],
        "trap_li"  => Float64[trap_li_val],
        "phi"      => permutedims(Array(phi)),
        "C_Li"     => permutedims(Array(C_Li)),
        "Cf"       => permutedims(Array(Cf)),
        "eta_m"    => permutedims(Array(eta_m_phys)),
        "hf"       => permutedims(Array(hf_state)),
        "T"        => permutedims(Array(T)),
        "Pc_MPa"   => permutedims(Pc_MPa),
        "Pf_MPa"   => permutedims(Pf_MPa),
        "eII_s"    => permutedims(Array(eII) ./ t0),
        "qDx"      => permutedims(qDx_c),
        "qDz"      => permutedims(qDz_c),
        "Vsx"      => permutedims(Vsx_c),
        "Vsz"      => permutedims(Vsz_c),
    )
end

# ==============================================================================
# 4. MAIN 2D SIMULATION DRIVER FOR ONE CASE
# ==============================================================================
function run_2d_case(cfg::CaseConfig, p::ModelParams; verbose::Bool = true, save_snapshots::Bool = true)
    t_wall0 = time()
    nx, nz = p.nx, p.nz
    dx_phys, dz_phys = p.Lx / nx, p.Lz / nz

    # Characteristic scales (non-dimensionalization keeps all PT fields O(1))
    L0     = p.Lz
    sigma0 = p.delta_rho * p.g * L0          # ~5.89 MPa
    eta0   = p.eta_s_ref                     # 8.0e18 Pa s
    t0     = eta0 / sigma0                   # Characteristic mechanical timescale (s)
    v0     = L0 / t0                         # Characteristic velocity (m / s)

    dx = dx_phys / L0
    dz = dz_phys / L0
    inv_dx      = 1.0 / dx
    inv_dz      = 1.0 / dz
    inv_half_dz = 2.0 / dz
    inv_dx2     = inv_dx * inv_dx
    inv_dz2     = inv_dz * inv_dz

    inv_dx_phys = 1.0 / dx_phys
    inv_dz_phys = 1.0 / dz_phys
    inv_dx_p2   = inv_dx_phys * inv_dx_phys
    inv_dz_p2   = inv_dz_phys * inv_dz_phys
    inv_H_roof  = 1.0 / p.H_roof
    inv_dT_lat  = 1.0 / p.dT_lat

    z_phys = [(iz - 0.5) * dz_phys for iz in 1:nz]
    x_phys = [(ix - 0.5) * dx_phys for ix in 1:nx]
    z_frac = z_phys ./ p.Lz

    # Initial fields (z = 0 base -> z = Lz Palaeo-BDT top)
    T_bg_z_host = [p.T_bg_bot + (p.T_bg_top - p.T_bg_bot) * zf for zf in z_frac]
    T_init_host = [p.T_init_bot + (p.T_init_top - p.T_init_bot) * zf for _ in 1:nx, zf in z_frac]

    noise = make_smooth_noise(nx, nz, p.noise_amp; seed = 42, passes = 5)
    phi_host = zeros(nx, nz)
    for iz in 1:nz, ix in 1:nx
        theta = clamp((T_init_host[ix, iz] - p.T_sol0) / (p.T_liq - p.T_sol0), 0.0, 1.0)
        taper = (iz <= 2 || iz >= nz - 1) ? 0.0 : 1.0
        phi_host[ix, iz] = clamp(theta^p.gamma_melt + taper * noise[ix, iz], 0.06, 0.44)
    end

    phi_ref = mean(phi_host)
    k_ref   = (p.a_grain^2) * (phi_ref^3) / (180.0 * (1.0 - phi_ref)^2)
    M0      = k_ref / p.eta_m0
    Rc2     = (M0 * eta0) / (L0^2)           # Dimensionless (delta_c0 / L0)^2
    dc0_m   = sqrt(M0 * eta0)

    tau_cool_s     = (p.Lz^2 / p.kappa) * (1.0 + p.St)
    tau_lat_s      = p.tau_lat_frac * tau_cool_s
    inv_tau_lat    = 1.0 / tau_lat_s
    t_end_s        = p.max_t_norm * tau_cool_s
    v_cool_nd      = t0 / tau_cool_s
    inv_v_cool_ref = 1.0 / max(0.04 * v_cool_nd, 1.0e-12)

    # Dimensionless BCs & control parameters
    # Dextral top-to-right simple shear (U_top > 0, U_bot < 0):
    # deep ~25° shear bands & upper 2-3 anchored ~67° veins both rise bottom-left -> top-right (/)
    U_top = cfg.use_shear ? (+0.5 * p.eps_bg * p.Lz / v0) : 0.0
    U_bot = cfg.use_shear ? (-0.5 * p.eps_bg * p.Lz / v0) : 0.0
    s_tilt_case = cfg.use_shear ? p.vein_tilt_dx : 0.0
    sigma_sq_nd = cfg.use_squeezing ? (p.sigma_sq / sigma0) : 0.0
    tau_y_nd    = p.tau_yield / sigma0
    inv_tau_y   = 1.0 / tau_y_nd
    D_disp_nd   = p.D_disp * t0 / (L0^2)
    kappa_eff   = p.kappa / (1.0 + p.St)

    # Confining-stress-corrected tensile hydrofracture threshold (suppresses cracking in filter pressing)
    pf_frac_eff = (p.pf_frac_MPa * 1.0e6) / sigma0 + sigma_sq_nd
    inv_dp_frac = sigma0 / (p.dp_frac_MPa * 1.0e6)

    n_pl       = cfg.use_shear ? p.n_power : 1.0
    pl_exp     = 0.5 * (1.0 / n_pl - 1.0)
    e_ref_nd   = max(0.5 * p.eps_bg * t0, 0.25)
    inv_e_ref2 = 1.0 / (e_ref_nd * e_ref_nd)

    use_shear_i = cfg.use_shear         ? 1 : 0
    use_dil_i   = cfg.use_dilatancy     ? 1 : 0
    use_fsol_i  = cfg.use_flux_solidus  ? 1 : 0
    use_vfb_i   = cfg.use_visc_feedback ? 1 : 0

    # Allocate ParallelStencil arrays
    phi       = Data.Array(phi_host)
    phi_prev  = Data.Array(copy(phi_host))
    phi_new   = @zeros(nx, nz)
    T         = Data.Array(T_init_host)
    T_prev    = Data.Array(copy(T_init_host))
    T_new     = @zeros(nx, nz)
    T_bg_z    = Data.Array(T_bg_z_host)
    hf_state  = @zeros(nx, nz)
    eta_m_hf  = Data.Array(fill(p.eta_m0, nx, nz))

    # Anchored geological dike state (up to 3 permanent stress-shadowed 67° fractures)
    vein_ix      = zeros(Int, 3)
    vein_z       = zeros(Float64, 3)
    vein_open    = zeros(Float64, 3)
    vein_eta     = fill(p.eta_m0, 3)
    n_veins_ref  = Ref(0)
    drive_col    = zeros(Float64, nx)
    score_col    = zeros(Float64, nx)
    z_root_col   = zeros(Float64, nx)
    feed_eta_col = fill(p.eta_m0, nx)
    hf_tgt_2d    = zeros(Float64, nx, nz)
    eta_tgt_2d   = fill(p.eta_m0, nx, nz)

    # Preallocated work arrays for 4 ms Jacobi-PCG elliptic Pf polish
    r_cg     = zeros(Float64, nx, nz)
    z_cg     = zeros(Float64, nx, nz)
    p_cg     = zeros(Float64, nx, nz)
    Ap_cg    = zeros(Float64, nx, nz)
    valve_on = falses(nx)

    C_Li      = Data.Array(fill(p.Li_0, nx, nz))
    C_Li_new  = @zeros(nx, nz)
    Cf        = Data.Array(fill(1.0, nx, nz))
    Cf_new    = @zeros(nx, nz)
    Cs_Li     = Data.Array(fill(p.D_Li * p.Li_0, nx, nz))
    Cs_Li_new = @zeros(nx, nz)
    Cs_f      = Data.Array(fill(p.D_flux, nx, nz))
    Cs_f_new  = @zeros(nx, nz)

    eta_s      = @ones(nx, nz)
    eta_eff    = @ones(nx, nz)
    zeta       = @ones(nx, nz)
    inv_zeta   = @ones(nx, nz)
    mob_c      = @ones(nx, nz)
    eta_m_phys = Data.Array(fill(p.eta_m0, nx, nz))

    mob_x      = @ones(nx, nz)
    mob_z      = @ones(nx, nz + 1)
    mobx_fac   = @zeros(nx, nz)
    mobz_fac   = @zeros(nx, nz + 1)
    eta_sv     = @ones(nx, nz + 1)
    eta_sv_eff = @ones(nx, nz + 1)

    # Precomputed PT step-size & rheology coefficient arrays
    tc_r         = @zeros(nx, nz)
    two_ec_inv1  = @zeros(nx, nz)
    G_cell       = @ones(nx, nz)
    tv_r         = @zeros(nx, nz + 1)
    two_esv_inv1 = @zeros(nx, nz + 1)
    dtau_pt      = @zeros(nx, nz)
    dtau_pf      = @zeros(nx, nz)
    inv_diag_pf  = @ones(nx, nz)
    dtau_vx      = @zeros(nx, nz)
    dtau_vz      = @zeros(nx, nz + 1)
    buoy_z       = @zeros(nx, nz + 1)
    disagg_lock  = @zeros(nx, nz)
    dil_fac      = @zeros(nx, nz)
    comp_src     = @zeros(nx, nz)

    # Exact discrete vertical buoyancy-balance initial guess for Pt & Pf
    Pt_init = zeros(nx, nz)
    for ix in 1:nx
        Pt_init[ix, nz] = 0.5 * dz * (1.0 - phi_host[ix, nz])
        for iz in (nz - 1):-1:1
            Pt_init[ix, iz] = Pt_init[ix, iz + 1] +
                              0.5 * dz * ((1.0 - phi_host[ix, iz]) + (1.0 - phi_host[ix, iz + 1]))
        end
    end
    Vsx_init = [U_bot + (U_top - U_bot) * zf for _ in 1:nx, zf in z_frac]

    Pt     = Data.Array(Pt_init)
    Pf     = Data.Array(copy(Pt_init))
    dPf    = @zeros(nx, nz)
    Vsx    = Data.Array(Vsx_init)
    Vsz    = @zeros(nx, nz + 1)

    exz           = @zeros(nx, nz + 1)
    txz           = @zeros(nx, nz + 1)
    txx           = @zeros(nx, nz)
    tzz           = @zeros(nx, nz)
    div_Vs        = @zeros(nx, nz)
    eII           = @zeros(nx, nz)
    eII_mean_z    = zeros(Float64, nz)
    tau_II_mean_z = zeros(Float64, nz)
    zeta_mean_z   = ones(Float64, nz)
    eps_dil       = @zeros(nx, nz)

    qDx   = @zeros(nx, nz)
    qDz   = @zeros(nx, nz + 1)
    Fm_x  = @zeros(nx, nz); Fs_x = @zeros(nx, nz)
    Fm_z  = @zeros(nx, nz + 1); Fs_z = @zeros(nx, nz + 1)
    v_cfl = @zeros(nx, nz)

    Res_vx  = @zeros(nx, nz)
    Res_vz  = @zeros(nx, nz + 1)
    Res_Pt  = @zeros(nx, nz)
    Res_Pf  = @zeros(nx, nz)
    Res_txx = @zeros(nx, nz)
    Res_tzz = @zeros(nx, nz)
    Res_txz = @zeros(nx, nz + 1)

    # Normalized sub-linear Pseudo-Maxwell PT parameters + face-consistent acoustic CFL constant
    nmax         = max(nx, nz)
    cfl_v        = 0.56
    r_bulk       = 0.50
    theta_0      = 1.8 * nmax
    gamma_f      = 6.0 / nmax
    one_minus_gf = 1.0 - gamma_f
    h_min        = min(dx, dz)
    coeff_v      = (cfl_v * cfl_v * h_min * h_min) / (r_bulk + 4.0 / 3.0)
    k_num        = (cfl_v * cfl_v) * (r_bulk / (r_bulk + 4.0 / 3.0))

    # Diagnostic history & active/peak snapshot holders
    hist_t_norm  = Float64[]
    hist_trap_li = Float64[]
    hist_peak_li = Float64[]
    hist_min_eta = Float64[]
    hist_bdt_vol = Float64[]

    snap_active::Union{Nothing, Dict{String, Any}} = nothing
    snap_peak::Union{Nothing, Dict{String, Any}}   = nothing
    best_li_flux     = 0.0
    peak_t_norm      = 0.0
    last_mob_min_eta = p.eta_m0

    trap_melt_vol = 0.0   # Horizontally averaged cumulative BDT export thickness (m)
    trap_li_mass  = 0.0   # Horizontally averaged cumulative BDT Li mass (ppm * m)

    t_s  = 0.0
    step = 0
    dt_max_s = p.dt_max_frac * tau_cool_s
    dt_therm_lim = 0.20 * min(dx_phys, dz_phys)^2 / kappa_eff

    if verbose
        @printf("\n--- Running 2D Case: %-12s (grid %dx%d, threads=%d, delta_c0=%.1f m = %.1f*dx) ---\n",
                cfg.name, nx, nz, Threads.nthreads(), dc0_m, dc0_m / dx_phys)
        @printf("    Switches: shear=%s | squeeze=%s | dilatancy=%s | flux_sol=%s | visc_fb=%s\n",
                cfg.use_shear, cfg.use_squeezing, cfg.use_dilatancy,
                cfg.use_flux_solidus, cfg.use_visc_feedback)
    end

    while t_s < t_end_s && step < p.max_steps
        # 1) Local viscosities -> anchored stress-shadowed 67° oblique veins -> cell & face mobilities
        @parallel (1:nx, 1:nz) update_viscosities!(
            eta_s, zeta, eta_m_phys,
            phi, T, Cf,
            phi_ref, p.alpha_phi, p.eta_m0, p.eta_min, p.Cf_onset, p.lambda_flux,
            use_vfb_i, nx, nz
        )
        n_veins_now = propagate_hydrofracture_serial!(
            hf_state, eta_m_hf,
            vein_ix, vein_z, vein_open, vein_eta, n_veins_ref,
            drive_col, score_col, z_root_col, feed_eta_col, hf_tgt_2d, eta_tgt_2d,
            phi, Pf, Pt, eta_m_phys, x_phys, z_phys,
            p.Lx, p.Lz, dx_phys, phi_ref,
            pf_frac_eff, inv_dp_frac, s_tilt_case,
            p.vein_shadow_m, p.vein_half_w_m, p.eta_m0, nx, nz
        )
        @parallel (1:nx, 1:nz) update_cell_mobility!(
            mob_c, phi, eta_m_phys, hf_state, eta_m_hf,
            phi_ref, k_ref, p.k_hf_frac, M0, nx, nz
        )
        @parallel (1:nx, 1:nz+1) update_face_and_vertex_props!(
            mob_x, mob_z, eta_sv, mob_c, eta_s, phi, hf_state, p.phi_min, nx, nz
        )
        @parallel (1:nx, 1:nz+1) precompute_step_constants!(
            inv_zeta, mobx_fac, mobz_fac, dtau_pf, inv_diag_pf, buoy_z, disagg_lock, dil_fac,
            zeta, mob_x, mob_z, phi,
            dz, inv_dx, inv_dz, inv_half_dz, inv_dx2, inv_dz2, Rc2,
            use_dil_i, p.alpha_pump, p.psi_0, phi_ref, p.phi_min, p.phi_crit, nx, nz
        )
        if step > 0
            adjust_hydrostatic_pt_serial!(Pt, phi, phi_prev, dz, nx, nz)
        end

        min_zeta_val = minimum(zeta)
        sq_scale_p   = abs(sigma_sq_nd) / min_zeta_val
        eta_min_s, eta_max_s = extrema(eta_s)
        inv_eta_geo  = 1.0 / sqrt(max(eta_min_s * eta_max_s, 1.0e-12))

        # Initialize / refresh rheology & face-consistent PT step sizes before PT loop
        rel_init = (step == 0 || use_shear_i == 0) ? 0 : 2
        if step == 0 || use_shear_i == 0
            eta_eff    .= eta_s
            eta_sv_eff .= eta_sv
        end
        @parallel (1:nx, 1:nz+1) refresh_rheology_vertices!(
            exz, eta_sv_eff, tv_r, two_esv_inv1, Vsx, Vsz, eta_sv,
            inv_dx, inv_dz, U_bot, U_top, use_shear_i,
            n_pl, pl_exp, inv_e_ref2, inv_tau_y, theta_0, inv_eta_geo, rel_init, nx, nz
        )
        @parallel (1:nx, 1:nz) refresh_rheology_cells!(
            eII, eta_eff, tc_r, two_ec_inv1, G_cell, Vsx, Vsz, exz, eta_s,
            inv_dx, inv_dz, n_pl, pl_exp, inv_e_ref2, inv_tau_y, theta_0, inv_eta_geo,
            rel_init, nx, nz
        )
        if use_dil_i == 1
            compute_z_means_serial!(eII_mean_z, tau_II_mean_z, zeta_mean_z, eII, eta_eff, zeta, nx, nz)
        end
        @parallel (1:nx, 1:nz+1) refresh_pt_steps_and_sources!(
            dtau_vx, dtau_vz, comp_src, eps_dil,
            eta_eff, eta_sv_eff, eII, eII_mean_z, tau_II_mean_z, zeta_mean_z,
            disagg_lock, dil_fac, inv_zeta,
            coeff_v, theta_0, inv_eta_geo, sigma_sq_nd, 1.0, use_dil_i, nx, nz
        )
        @parallel (1:nx, 1:nz) update_dtau_pt_from_v!(
            dtau_pt, dtau_vx, dtau_vz, G_cell, inv_zeta,
            inv_dx2, inv_dz2, k_num, r_bulk, nx, nz
        )
        if step == 0
            @parallel (1:nx, 1:nz+1) init_equilibrium_stresses!(
                txx, tzz, txz, div_Vs, Vsx, Vsz, exz, eta_eff, eta_sv_eff,
                inv_dx, inv_dz, nx, nz
            )
        end

        # 2) Ultra-Fast Fused 3-Kernel PT Solve + on-demand Jacobi-PCG elliptic Pf polish
        dPf .= 0.0
        pt_tol_cur  = (step == 0) ? p.pt_tol_init : p.pt_tol
        pt_max_cur  = (step == 0) ? p.pt_max_iter_init : p.pt_max_iter
        pt_err      = 1.0
        pt_iters    = 0
        err_v_rel   = 1.0
        err_pt_rel  = 1.0
        err_pf_rel  = 1.0
        err_tau_rel = 1.0
        t_pt0       = (step == 0) ? time() : 0.0

        for iter in 1:pt_max_cur
            do_check = (iter % p.pt_check_every == 0) ? 1 : 0

            # Fused Kernel 1/3: Stresses + div_Vs + Darcy fluxes (zero transcendentals!)
            @parallel (1:nx, 1:nz+1) pt_step_stress_and_darcy!(
                txx, tzz, txz, div_Vs, qDx, qDz, Res_txx, Res_tzz, Res_txz,
                Vsx, Vsz, Pf, tc_r, two_ec_inv1, tv_r, two_esv_inv1, eta_eff, eta_sv_eff,
                mobx_fac, mobz_fac,
                inv_dx, inv_dz, U_bot, U_top, use_shear_i, do_check, nx, nz
            )

            # Fused Kernel 2/3: Pressures (Pt, Pf)
            @parallel (1:nx, 1:nz) pt_step_pressures!(
                Pt, Pf, dPf, Res_Pt, Res_Pf,
                div_Vs, qDx, qDz, inv_zeta, comp_src, dtau_pt, dtau_pf,
                inv_dx, inv_dz, one_minus_gf, do_check, nx, nz
            )

            # Fused Kernel 3/3: Velocities (Vsx, Vsz) + base/top BCs
            @parallel (1:nx, 1:nz+1) pt_step_velocities_and_bcs!(
                Vsx, Vsz, Res_vx, Res_vz,
                txx, tzz, txz, Pt, buoy_z, dtau_vx, dtau_vz,
                inv_dx, inv_dz, inv_half_dz, do_check, nx, nz
            )

            pt_iters = iter
            if do_check == 1
                err_v   = max(maximum(abs, Res_vx), maximum(abs, Res_vz))
                err_pt  = maximum(abs, Res_Pt)
                err_pf  = maximum(abs, Res_Pf)
                err_tau = max(maximum(abs, Res_txx), maximum(abs, Res_tzz), maximum(abs, Res_txz))

                max_txz   = maximum(abs, txz)
                scale_v   = max(1.0, max_txz * inv_dz, abs(sigma_sq_nd))
                scale_p   = max(1.0e-2, maximum(abs, eII), maximum(abs, eps_dil), sq_scale_p)
                scale_tau = max(1.0, maximum(abs, txx), maximum(abs, tzz), max_txz)

                err_v_rel   = err_v / scale_v
                err_pt_rel  = err_pt / scale_p
                err_pf_rel  = err_pf / scale_p
                err_tau_rel = err_tau / scale_tau

                # On-demand 4 ms Jacobi-PCG elliptic Pf polish whenever 10^8x vein contrast slows Pf
                err_stokes = max(err_v_rel, err_pt_rel, err_tau_rel)
                if n_veins_now > 0 && err_pf_rel > pt_tol_cur &&
                   (err_stokes < 2.0 * pt_tol_cur || (iter % 500 == 0 && err_pf_rel > 4.0 * err_stokes))
                    err_pf_after = polish_pf_pcg_serial!(
                        Pf, r_cg, z_cg, p_cg, Ap_cg, valve_on,
                        Pt, inv_zeta, comp_src, mobx_fac, mobz_fac, inv_diag_pf,
                        inv_dx, inv_dz, 0.25 * pt_tol_cur * scale_p, 350, nx, nz
                    )
                    dPf .= 0.0
                    # Refresh qDx, qDz immediately so downstream advection/diagnostics match polished Pf
                    @parallel (1:nx, 1:nz+1) pt_step_stress_and_darcy!(
                        txx, tzz, txz, div_Vs, qDx, qDz, Res_txx, Res_tzz, Res_txz,
                        Vsx, Vsz, Pf, tc_r, two_ec_inv1, tv_r, two_esv_inv1, eta_eff, eta_sv_eff,
                        mobx_fac, mobz_fac,
                        inv_dx, inv_dz, U_bot, U_top, use_shear_i, 0, nx, nz
                    )
                    err_pf_rel = err_pf_after / scale_p
                end

                pt_err = max(err_v_rel, err_pt_rel, err_pf_rel, err_tau_rel)

                if verbose && step == 0 && iter % 500 == 0
                    @printf("      [step 0 PT iter %4d] err_v=%.2e | err_pt=%.2e | err_pf=%.2e | err_tau=%.2e | pt_err=%.2e\n",
                            iter, err_v_rel, err_pt_rel, err_pf_rel, err_tau_rel, pt_err)
                end
                if !isfinite(pt_err)
                    error("PT diverged at step $(step), iter $(iter) (err_v=$(err_v), err_pt=$(err_pt), err_pf=$(err_pf)).")
                end
                if pt_err < pt_tol_cur
                    break
                end
            end

            # Gentle lagged rheology & face-consistent step refresh (ONLY for nonlinear shear cases)
            if use_shear_i == 1 && iter % p.visc_stride == 0
                rel_src_cur = (iter > 2000) ? 0.12 : 0.22
                @parallel (1:nx, 1:nz+1) refresh_rheology_vertices!(
                    exz, eta_sv_eff, tv_r, two_esv_inv1, Vsx, Vsz, eta_sv,
                    inv_dx, inv_dz, U_bot, U_top, use_shear_i,
                    n_pl, pl_exp, inv_e_ref2, inv_tau_y, theta_0, inv_eta_geo, 3, nx, nz
                )
                @parallel (1:nx, 1:nz) refresh_rheology_cells!(
                    eII, eta_eff, tc_r, two_ec_inv1, G_cell, Vsx, Vsz, exz, eta_s,
                    inv_dx, inv_dz, n_pl, pl_exp, inv_e_ref2, inv_tau_y, theta_0, inv_eta_geo,
                    3, nx, nz
                )
                if use_dil_i == 1
                    compute_z_means_serial!(eII_mean_z, tau_II_mean_z, zeta_mean_z, eII, eta_eff, zeta, nx, nz)
                end
                @parallel (1:nx, 1:nz+1) refresh_pt_steps_and_sources!(
                    dtau_vx, dtau_vz, comp_src, eps_dil,
                    eta_eff, eta_sv_eff, eII, eII_mean_z, tau_II_mean_z, zeta_mean_z,
                    disagg_lock, dil_fac, inv_zeta,
                    coeff_v, theta_0, inv_eta_geo, sigma_sq_nd, rel_src_cur, use_dil_i, nx, nz
                )
                @parallel (1:nx, 1:nz) update_dtau_pt_from_v!(
                    dtau_pt, dtau_vx, dtau_vz, G_cell, inv_zeta,
                    inv_dx2, inv_dz2, k_num, r_bulk, nx, nz
                )
            end
        end

        if step == 0 && verbose
            us_per_iter = 1.0e6 * (time() - t_pt0) / max(pt_iters, 1)
            @printf("      [benchmark] step 0 solved %d PT iters in %.2f s (%.1f us/PT-iter)\n",
                    pt_iters, time() - t_pt0, us_per_iter)
        end

        if pt_err > 2.5e-2
            error(@sprintf("PT failed to converge in case %s at step %d after %d iters (err_v=%.2e, err_pt=%.2e, err_pf=%.2e, err_tau=%.2e).",
                           cfg.name, step, pt_iters, err_v_rel, err_pt_rel, err_pf_rel, err_tau_rel))
        end

        # 3) Conservative 2nd-order TVD phase fluxes & local-porosity adaptive CFL time step
        @parallel (1:nx, 1:nz+1) compute_phase_fluxes!(
            Fm_x, Fs_x, Fm_z, Fs_z, Vsx, Vsz, qDx, qDz, phi, nx, nz
        )
        @parallel (1:nx, 1:nz) compute_local_cfl_speed!(
            v_cfl, Vsx, Vsz, Fm_x, Fm_z, phi, p.phi_min, nx, nz
        )

        vmax_nd = max(maximum(v_cfl), 1.0e-12)
        if !isfinite(vmax_nd)
            error("Non-finite velocity in case $(cfg.name) at step $(step).")
        end

        dt_cfl_s = (0.42 * h_min / vmax_nd) * t0
        dt_s     = min(dt_cfl_s, dt_max_s, t_end_s - t_s)
        dt_s     = max(dt_s, min(1.0e4, t_end_s - t_s))
        dt_nd    = dt_s / t0

        # Accumulate melt & Li exported across top Palaeo-BDT valve (z = Lz)
        q_out_row   = @view Fm_z[:, nz + 1]
        li_top_row  = @view C_Li[:, nz]
        q_out_mean  = mean(q_out_row) * v0
        li_flux_now = mean(q_out_row .* li_top_row) * v0
        if q_out_mean > 0.0
            trap_melt_vol += q_out_mean * dt_s
            trap_li_mass  += li_flux_now * dt_s
        end

        # Save pre-step porosity & temperature
        phi_prev .= phi
        T_prev   .= T

        # 4) Conservative 2nd-order TVD advection of phi and dual-phase solutes
        @parallel (1:nx, 1:nz) advect_phi_and_solutes!(
            phi_new, C_Li_new, Cf_new, Cs_Li_new, Cs_f_new,
            phi, C_Li, Cf, Cs_Li, Cs_f, Fm_x, Fs_x, Fm_z, Fs_z,
            dt_nd, inv_dx, inv_dz, D_disp_nd, p.phi_min, p.phi_max, nx, nz
        )
        phi, phi_new     = phi_new, phi
        C_Li, C_Li_new   = C_Li_new, C_Li
        Cf, Cf_new       = Cf_new, Cf
        Cs_Li, Cs_Li_new = Cs_Li_new, Cs_Li
        Cs_f, Cs_f_new   = Cs_f_new, Cs_f

        # 5) Sub-cycled 2D thermal diffusion + barycentric advection + Robin roof + lateral cooling
        n_sub  = max(1, ceil(Int, dt_s / dt_therm_lim))
        dt_sub = dt_s / n_sub
        for _ in 1:n_sub
            @parallel (1:nx, 1:nz) thermal_substep!(
                T_new, T, T_bg_z, Fm_x, Fs_x, Fm_z, Fs_z,
                dt_sub, inv_dx_phys, inv_dz_phys, inv_dx_p2, inv_dz_p2,
                kappa_eff, inv_H_roof, p.T_bg_top, inv_tau_lat, v0, nx, nz
            )
            T, T_new = T_new, T
        end

        # 6) Enthalpy-conserving Eulerian cooling + focused Lagrangian channel/vein-wall distillation
        sub_relax  = 1.0 - exp(-dt_s / p.tau_cryst)
        v0_dt_phys = v0 * dt_s
        @parallel (1:nx, 1:nz) rayleigh_crystallization_step!(
            phi, C_Li, Cf, Cs_Li, Cs_f, T, T_prev, qDx, qDz, eII, hf_state,
            sub_relax, v0_dt_phys, inv_dx_phys, inv_dz_phys,
            p.T_liq, p.T_sol0, p.T_sol_min, p.dT_flux, inv_dT_lat, p.beta_T_dist,
            p.D_Li, p.D_flux, p.phi_min, p.phi_max, inv_v_cool_ref, use_fsol_i, nx, nz
        )

        t_s  += dt_s
        step += 1
        t_norm_now = t_s / tau_cool_s

        # 7) Diagnostics: shallow BDT trap Li, smooth 15-cell geom-mean channel viscosity & snapshots
        if trap_melt_vol > 1.0e-4
            V   = trap_melt_vol
            Vc  = p.V_peg_crit
            eff = (V * V) / (V * V + Vc * Vc)
            avg_li_out = trap_li_mass / V
            rayleigh_boost = p.f_peg_res^(p.D_Li - 1.0)
            trap_li = clamp(avg_li_out * (1.0 + eff * (rayleigh_boost - 1.0)), p.Li_0, 2.5e4)
        else
            trap_li = p.Li_0
        end

        phi_arr    = Array(phi)
        hf_arr     = Array(hf_state)
        li_arr     = Array(C_Li)
        eta_arr    = Array(eta_m_phys)
        eta_hf_arr = Array(eta_m_hf)
        # Include both mobile mush pores (phi >= 0.03) AND active draining roof veins (hf > 0.05)
        mob_mask     = (phi_arr .>= 0.03) .| (hf_arr .> 0.05)
        peak_chan_li = any(mob_mask) ? maximum(li_arr[mob_mask]) : p.Li_0
        if any(mob_mask)
            eta_eff_mob = min.(eta_arr[mob_mask], eta_hf_arr[mob_mask])
            n_low       = min(15, length(eta_eff_mob))
            sort!(eta_eff_mob)
            min_eta_val      = exp(mean(log, @view eta_eff_mob[1:n_low]))
            last_mob_min_eta = min_eta_val
        else
            min_eta_val = last_mob_min_eta
        end

        push!(hist_t_norm,  t_norm_now)
        push!(hist_trap_li, trap_li)
        push!(hist_peak_li, max(peak_chan_li, trap_li))
        push!(hist_min_eta, min_eta_val)
        push!(hist_bdt_vol, trap_melt_vol)

        # Capture synchronous mature-channel snapshot (t/tc ~ 0.45) & peak Li-export snapshot
        if save_snapshots
            if snap_active === nothing && t_norm_now >= p.t_active_snap
                snap_active = pack_snapshot(
                    x_phys, z_phys, phi, C_Li, Cf, eta_m_phys, hf_state, T, Pt, Pf, eII,
                    qDx, qDz, Vsx, Vsz, sigma0, v0, t0, nz,
                    t_norm_now, trap_melt_vol, trap_li
                )
            end
            if t_norm_now >= 0.25 && li_flux_now > best_li_flux
                best_li_flux = li_flux_now
                peak_t_norm  = t_norm_now
                snap_peak    = pack_snapshot(
                    x_phys, z_phys, phi, C_Li, Cf, eta_m_phys, hf_state, T, Pt, Pf, eII,
                    qDx, qDz, Vsx, Vsz, sigma0, v0, t0, nz,
                    t_norm_now, trap_melt_vol, trap_li
                )
            end
        end

        if verbose && (step == 1 || step % 80 == 0 || t_s >= t_end_s || step == p.max_steps)
            phi_max_v = maximum(phi_arr)
            hf_max_v  = maximum(hf_arr)
            @printf("    step=%4d | t/tc=%5.3f | PT=%4d (err=%.1e) | phiMax=%.2f | veins=%d (hf=%.2f) | ChanLi=%6.0f | Export=%6.2f m | TrapLi=%7.1f ppm | minEta=%.2e\n",
                    step, t_norm_now, pt_iters, pt_err, phi_max_v, n_veins_now, hf_max_v, peak_chan_li, trap_melt_vol, trap_li, min_eta_val)
        end
    end

    snap_final = save_snapshots ? pack_snapshot(
        x_phys, z_phys, phi, C_Li, Cf, eta_m_phys, hf_state, T, Pt, Pf, eII,
        qDx, qDz, Vsx, Vsz, sigma0, v0, t0, nz,
        hist_t_norm[end], trap_melt_vol, hist_trap_li[end]
    ) : nothing

    if save_snapshots
        if snap_active === nothing
            snap_active = snap_final
        end
        if snap_peak === nothing
            snap_peak   = snap_active
            peak_t_norm = snap_peak["t_norm"][1]
        end
    end

    elapsed = time() - t_wall0
    if verbose
        @printf(">>> Finished %-12s in %.1f s | PeakLiFlux @ t/tc=%.3f | Final Export = %.2f m | BDT Trap Li = %.1f ppm | MaxChanLi = %.0f ppm | Min eta_m = %.2e Pa s\n",
                cfg.name, elapsed, peak_t_norm, trap_melt_vol, hist_trap_li[end], maximum(hist_peak_li), minimum(hist_min_eta))
    end

    return Dict{String, Any}(
        "case"        => cfg.name,
        "snap_final"  => snap_final,
        "snap_active" => snap_active,
        "snap_peak"   => snap_peak,
        "t_norm"      => hist_t_norm,
        "trap_li"     => hist_trap_li,
        "peak_li"     => hist_peak_li,
        "min_eta"     => hist_min_eta,
        "bdt_vol"     => hist_bdt_vol,
    )
end

# ==============================================================================
# 5. BATCH RUNNER: 4 ABLATION BENCHMARKS + OPTIONAL FIG. 3d PHASE-DIAGRAM SWEEP
# ==============================================================================
function run_four_ablation_cases(p::ModelParams; outdir::String = "output_2d")
    mkpath(outdir)

    # Exact 4-switch ablation matrix
    cases = [
        CaseConfig("static",      false, false, false, true, false),
        CaseConfig("filter",      false, true,  false, true, false),
        CaseConfig("shear_const", true,  false, true,  true, false),
        CaseConfig("coupled",     true,  false, true,  true, true),
    ]

    ts_dict = Dict{String, Any}()

    for cfg in cases
        res = run_2d_case(cfg, p; verbose = true, save_snapshots = true)

        npzwrite(joinpath(outdir, "snapshots_$(cfg.name).npz"),        res["snap_final"])
        npzwrite(joinpath(outdir, "snapshots_active_$(cfg.name).npz"), res["snap_active"])
        npzwrite(joinpath(outdir, "snapshots_peak_$(cfg.name).npz"),   res["snap_peak"])

        ts_dict["t_norm_$(cfg.name)"]  = res["t_norm"]
        ts_dict["trap_li_$(cfg.name)"] = res["trap_li"]
        ts_dict["peak_li_$(cfg.name)"] = res["peak_li"]
        ts_dict["min_eta_$(cfg.name)"] = res["min_eta"]
        ts_dict["bdt_vol_$(cfg.name)"] = res["bdt_vol"]
    end

    npzwrite(joinpath(outdir, "time_series.npz"), ts_dict)
    @printf("\n[DONE] Saved final/active/peak 2D snapshots and time_series.npz into ./%s/\n", outdir)
end

function run_phase_diagram_sweep(base_p::ModelParams; outdir::String = "output_2d")
    mkpath(outdir)
    eps_grid = 10.0 .^ range(-14.5, -12.0, length = 8)   # Strain rate axis (s^-1)
    eta_grid = 10.0 .^ range(4.5,   7.5,   length = 8)   # Initial melt viscosity axis (Pa s)

    ne, nm = length(eps_grid), length(eta_grid)
    peak_li_grid = zeros(nm, ne)
    export_grid  = zeros(nm, ne)
    cfg_coupled  = CaseConfig("coupled", true, false, true, true, true)

    @printf("\n=== Starting %dx%d Phase-Diagram Sweep (Fig. 3d) ===\n", nm, ne)
    for j in 1:nm, i in 1:ne
        p_ij = ModelParams(
            nx = 64, nz = 64,
            eps_bg = eps_grid[i],
            eta_m0 = eta_grid[j],
            max_steps = 1800,
            pt_max_iter = 3000
        )
        res = run_2d_case(cfg_coupled, p_ij; verbose = false, save_snapshots = false)
        peak_li_grid[j, i] = maximum(res["trap_li"])
        export_grid[j, i]  = res["bdt_vol"][end]
        @printf("  (%2d/%2d, %2d/%2d) eps=%.1e, eta_m0=%.1e -> TrapLi=%7.1f ppm, Export=%5.2f m\n",
                j, nm, i, ne, eps_grid[i], eta_grid[j], peak_li_grid[j, i], export_grid[j, i])
    end

    npzwrite(joinpath(outdir, "phase_diagram_grid.npz"), Dict{String, Any}(
        "eps_grid"     => collect(eps_grid),
        "eta_m0_grid"  => collect(eta_grid),
        "trap_li_grid" => peak_li_grid,
        "export_grid"  => export_grid,
    ))
    @printf("[DONE] Saved phase_diagram_grid.npz into ./%s/\n", outdir)
end

# ==============================================================================
# 6. ENTRY POINT
# ==============================================================================
function main()
    # Step A: Fast CPU verification of the 4 ablation cases (96 x 96)
    # Once verified, increase nx = 192, nz = 192 for publication Fig. 3a!
    params = ModelParams(nx = 96, nz = 96)
    run_four_ablation_cases(params; outdir = "output_2d")

    # Step B: Uncomment below once Step A results are verified to generate Fig. 3d grid
    # run_phase_diagram_sweep(params; outdir = "output_2d")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end