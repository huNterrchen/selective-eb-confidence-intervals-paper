"""
   Abstract type representing F-Localizations.
"""
abstract type FLocalization end

"""
   Abstract type representing a fitted F-Localization
   (i.e., wherein the F-localization has already been determined by data).
"""
abstract type FittedFLocalization end

# Holy trait
abstract type FLocalizationVexity end
struct LinearVexity <: FLocalizationVexity end
struct ConvexVexity <: FLocalizationVexity end



StatsBase.fit(floc::FittedFLocalization, args...; kwargs...) = floc

function nominal_alpha(floc::FLocalization)
    floc.α
end



function flocalization_constraint!(model, floc, prior::PriorVariable)
    model
end

"""
    DvoretzkyKieferWolfowitz(;α = 0.05, max_constraints = 1000) <: FLocalization

The Dvoretzky-Kiefer-Wolfowitz band (based on the Kolmogorov-Smirnov distance)
at confidence level `1-α` that bounds the distance of the true distribution function
to the ECDF ``\\widehat{F}_n`` based on ``n`` samples. The constant of the band is the sharp
constant derived by Massart:

```math
F \\text{ distribution}:  \\sup_{t \\in \\mathbb R}\\lvert F(t) - \\widehat{F}_n(t) \\rvert  \\leq  \\sqrt{\\log(2/\\alpha)/(2n)}
```
The supremum above is enforced discretely on at most `max_constraints` number of points.
"""
Base.@kwdef struct DvoretzkyKieferWolfowitz{T,N} <: FLocalization
    α::T = 0.05
    side::Symbol = :both
    max_constraints::N = 1000
end

# for backwards compatibility
DvoretzkyKieferWolfowitz(α) = DvoretzkyKieferWolfowitz(;α=α)

vexity(::DvoretzkyKieferWolfowitz) = LinearVexity()

struct FittedDvoretzkyKieferWolfowitz{T,S,D<:StatsDiscretizations.Dictionary{T,S},DKW} <:
        FittedFLocalization
    summary::D
    band::S
    side::Symbol
    dkw::DKW
    homoskedastic::Bool
end

vexity(dkw::FittedDvoretzkyKieferWolfowitz) = vexity(dkw.dkw)


function nominal_alpha(dkw::FittedDvoretzkyKieferWolfowitz)
    nominal_alpha(dkw.dkw)
end

# TODO: Allow this to work more broadly.
function StatsBase.fit(dkw::DvoretzkyKieferWolfowitz, Zs::AbstractVector{<:EBayesSample})
    StatsBase.fit(dkw, summarize(Zs))
end

function StatsBase.fit(dkw::DvoretzkyKieferWolfowitz, Zs_summary::MultinomialSummary{T}) where T
    α = nominal_alpha(dkw)
    n = nobs(Zs_summary)
    side = dkw.side

    if skedasticity(Zs_summary) === Heteroskedastic()
        homoskedastic = false
        multiplier = exp(1)
        Zs_summary = compound(Zs_summary)
    elseif skedasticity(Zs_summary) === Homoskedastic()
        if T<:CompoundSample || n != sum(values(Zs_summary))
            homoskedastic = false
            multiplier = exp(1)
        else
            homoskedastic = true
            multiplier = 1
        end
    end
    bonferroni_correction = side == :both ? 2 : 1
    band = sqrt(log(bonferroni_correction * multiplier / α) / (2n))

    n_constraints = length(Zs_summary)


    cdf_probs = cumsum(Zs_summary.store) #SORTED important here
    cdf_probs /= cdf_probs[end]
    _Zs = collect(keys(Zs_summary.store))

    issorted(_Zs, by=response, lt=StatsDiscretizations._isless) || error("MultinomialSummary not sorted.")

    max_constraints = dkw.max_constraints

    if max_constraints < n_constraints - 10
        _step = div(n_constraints-2, max_constraints)
        idxs = [1; 2:_step:(n_constraints-1); n_constraints]
        _Zs = _Zs[idxs]
        cdf_probs = cdf_probs[idxs]
    end
    # TODO: report issue with SortedDict upstream
    _dict = StatsDiscretizations.Dictionary(_Zs, cdf_probs)

    FittedDvoretzkyKieferWolfowitz(_dict, band, side, dkw, homoskedastic)
end
#=
function refit_dkw_with_floor(fit0::FittedDvoretzkyKieferWolfowitz;
                              Zs_trunc::Vector{<:Empirikos.TruncatedSample},
                              t_min::Float64 = 3.0,
                              max_constraints::Int = 1000)

    # Pull out the original discrete grid and ECDF values
    xs_all = collect(keys(fit0.summary))      # these are FoldedNormalSample "grid" points
    Fs_all = collect(values(fit0.summary))    # ECDF at those grid points

    # Convert FoldedNormalSample -> numeric |z| via `response`
    xnum_all = response.(xs_all)

    # Keep only grid points at/above t_min
    keep = xnum_all .>= t_min
    xs = xs_all[keep]
    Fs = Fs_all[keep]
    xnum = xnum_all[keep]

    # Make sure we *include* an explicit constraint exactly at t_min:
    # ECDF at t_min is simply the share of truncated data with |Z| <= t_min
    x_trunc = response.(getfield.(Zs_trunc, :Z))  # numeric |Z| values from the truncated samples
    F_at_tmin = mean(x_trunc .<= t_min)

    # If t_min is not already the first grid point, prepend it
    if isempty(xs) || response(xs[1]) > t_min + eps()
        pushfirst!(xs, Empirikos.TruncatedSample(FoldedNormalSample(t_min, 1.0), Interval(2.1, Inf)))
        pushfirst!(Fs, F_at_tmin)
        pushfirst!(xnum, t_min)
    elseif abs(response(xs[1]) - t_min) ≤ 1e-10
        Fs[1] = F_at_tmin  # replace ECDF at that exact grid point for cleanliness
    end

    # If too many constraints, thin them (same scheme as library)
    n_constraints = length(xs)
    if max_constraints < n_constraints - 10
        _step = div(n_constraints - 2, max_constraints)
        idxs = vcat(1, collect(2:_step:(n_constraints-1)), n_constraints)
        xs = xs[idxs]
        Fs = Fs[idxs]
    end

    # Rebuild the discrete Dictionary for DKW
    dict = StatsDiscretizations.Dictionary(xs, Fs)

    # Recreate a fitted object with the *same* band but the edited summary
    FittedDvoretzkyKieferWolfowitz(dict, fit0.band, fit0.side, fit0.dkw, fit0.homoskedastic)
end

# --- build the "floored" DKW fit with the first constraint at t=3.0 ---
function refit_dkw_with_custom_gri(Zs_trunc; trunc_set, α=0.05, side=:both,
                                    grid_len=1000, t_min=3.0)
    # numeric |Z| from truncated samples
    x = response.(getfield.(Zs_trunc, :Z))
    x = x[x .>= t_min]
    n = length(x)
    @assert n > 0 "No points ≥ t_min"

    # ECDF on arbitrary t
    ecdf_at = t -> mean(x .<= t)
    ts = range(minimum(x), maximum(x); length=min(grid_len, n))
    Fs = [ecdf_at(t) for t in ts]

    bonf = (side == :both) ? 2 : 1
    band = sqrt(log(bonf / α) / (2n))

    keys_ts = [Empirikos.TruncatedSample(FoldedNormalSample(t,1.0), trunc_set) for t in ts]
    dict = StatsDiscretizations.Dictionary(keys_ts, Fs)

    DKW = DvoretzkyKieferWolfowitz(; α=α, side=side, max_constraints=grid_len)
    return FittedDvoretzkyKieferWolfowitz(dict, band, side, DKW, true)
end


function refit_dkw_with_window(
    fit0::FittedDvoretzkyKieferWolfowitz;
    Zs_trunc::Vector{<:Empirikos.TruncatedSample},
    t_min::Float64 = 3.0,
    t_max::Union{Nothing,Float64} = nothing,
    a::Float64 = 2.1,                # truncation threshold used to build TruncatedSample keys
    max_constraints::Int = 1000
)
    # Original discrete grid and ECDF values
    xs_all = collect(keys(fit0.summary))    # FoldedNormalSample grid points
    Fs_all = collect(values(fit0.summary))  # ECDF at those grid points
    xnum_all = response.(xs_all)            # numeric |z|

    # Window filter
    keep_left = xnum_all .>= t_min
    keep = keep_left
    if t_max !== nothing
        keep .= keep .& (xnum_all .<= t_max)
    end

    xs   = xs_all[keep]
    Fs   = Fs_all[keep]
    xnum = xnum_all[keep]

    # Numeric |Z| from truncated samples for endpoint ECDFs
    x_trunc = response.(getfield.(Zs_trunc, :Z))

    # ---- inject/replace left endpoint at t_min ----
    F_at_tmin = mean(x_trunc .<= t_min)
    if isempty(xs) || response(xs[1]) > t_min + eps()
        pushfirst!(xs, Empirikos.TruncatedSample(FoldedNormalSample(t_min, 1.0),
                                                 Interval(a, Inf)))
        pushfirst!(Fs, F_at_tmin)
        pushfirst!(xnum, t_min)
    else
        # If grid already has t_min (within tiny tol), overwrite ECDF there
        if abs(response(xs[1]) - t_min) ≤ 1e-10
            Fs[1] = F_at_tmin
        end
    end

    # ---- inject/replace right endpoint at t_max (if provided) ----
    if t_max !== nothing
        F_at_tmax = mean(x_trunc .<= t_max)
        need_right =
            isempty(xs) || response(xs[end]) < (t_max - eps())

        if need_right
            push!(xs, Empirikos.TruncatedSample(FoldedNormalSample(t_max, 1.0),
                                                Interval(a, Inf)))
            push!(Fs, F_at_tmax)
            push!(xnum, t_max)
        else
            # If the last grid point is effectively t_max, overwrite ECDF there
            if abs(response(xs[end]) - t_max) ≤ 1e-10
                Fs[end] = F_at_tmax
            end
        end
    end

    # Ensure everything is sorted by x
    perm = sortperm(xnum)
    xs, Fs, xnum = xs[perm], Fs[perm], xnum[perm]

    # Optional thinning (similar to your scheme)
    n_constraints = length(xs)
    if max_constraints < n_constraints - 10
        step = max(1, div(n_constraints - 2, max_constraints))
        idxs = vcat(1, collect(2:step:(n_constraints-1)), n_constraints)
        xs = xs[idxs]
        Fs = Fs[idxs]
    end

    dict = StatsDiscretizations.Dictionary(xs, Fs)
    return FittedDvoretzkyKieferWolfowitz(
        dict,
        fit0.band,         # keep same band width
        fit0.side,         # one-/two-sided choice preserved
        fit0.dkw,
        fit0.homoskedastic
    )
end
function refit_dkw_drop_middle(
    fit0::FittedDvoretzkyKieferWolfowitz;
    middle_drop::Tuple{Float64,Float64},
    closed::Bool = false
)
    m_lo, m_hi = middle_drop
    @assert m_lo < m_hi "Require middle_drop = (m_lo, m_hi) with m_lo < m_hi."

    # Original grid and ECDF
    xs_all   = collect(keys(fit0.summary))         # FoldedNormalSample points
    Fs_all   = collect(values(fit0.summary))       # ECDF at those points
    xnum_all = response.(xs_all)                   # numeric |z|

    # Keep everything except the middle interval
    keep = if closed
        .!((xnum_all .>= m_lo) .& (xnum_all .<= m_hi))   # drop [m_lo, m_hi]
    else
        .!((xnum_all .>  m_lo) .& (xnum_all .<  m_hi))   # drop (m_lo, m_hi)
    end

    xs = xs_all[keep]
    Fs = Fs_all[keep]
    # Order is preserved by boolean indexing, no need to sort.

    dict = StatsDiscretizations.Dictionary(xs, Fs)
    return FittedDvoretzkyKieferWolfowitz(
        dict,
        fit0.band,          # preserve band width
        fit0.side,          # preserve sidedness
        fit0.dkw,
        fit0.homoskedastic
    )
end
function refit_dkw_drop(fit0::FittedDvoretzkyKieferWolfowitz;
                        drop_idxs::Vector{Int})
    xs_all = collect(keys(fit0.summary))
    Fs_all = collect(values(fit0.summary))

    keep_mask = trues(length(xs_all))
    keep_mask[drop_idxs] .= false

    xs = xs_all[keep_mask]
    Fs = Fs_all[keep_mask]

    dict = StatsDiscretizations.Dictionary(xs, Fs)
    # Reuse same band/side/settings
    return FittedDvoretzkyKieferWolfowitz(dict, fit0.band, fit0.side, fit0.dkw, fit0.homoskedastic)
end

function refit_dkw_multi(fit0::FittedDvoretzkyKieferWolfowitz;
                        multi::Float64)
    xs = collect(keys(fit0.summary))
    Fs = collect(values(fit0.summary))
    band = fit0.band*multi

    dict = StatsDiscretizations.Dictionary(xs, Fs)
    # Reuse same band/side/settings
    return FittedDvoretzkyKieferWolfowitz(dict, band, fit0.side, fit0.dkw, fit0.homoskedastic)
end
=#
function flocalization_constraint!(
    model,
    dkw::FittedDvoretzkyKieferWolfowitz,
    prior::PriorVariable,
)
    band = dkw.band
    side = dkw.side

    bound_upper = (side == :both) || (side == :upper)
    bound_lower = (side == :both) || (side == :lower)

    for (Z, cdf_value) in zip(keys(dkw.summary), values(dkw.summary))
        marginal_cdf = cdf(prior, Z::EBayesSample)
        if bound_upper
            if cdf_value + band < 1
                @constraint(model, marginal_cdf <= cdf_value + band)
            end
        end
        if bound_lower
            if cdf_value - band > 0
                @constraint(model, marginal_cdf >= cdf_value - band)
            end
        end
    end
    model
end
#=
struct DKWCache
    upper_refs::Vector{Any}
    lower_refs::Vector{Any}
    Z_points::Vector{Any}
    ecdf_vals::Vector{Float64}
    band::Float64
end

function Empirikos.flocalization_constraint!(model,
    dkw::FittedDvoretzkyKieferWolfowitz,
    prior::PriorVariable)

    band = dkw.band
    side = dkw.side
    upper_refs = Any[]
    lower_refs = Any[]
    Z_points   = Any[]
    ecdf_vals  = Float64[]

    for (Z, F̂) in zip(keys(dkw.summary), values(dkw.summary))
        # model-implied CDF at Z
        marginal_cdf = cdf(prior, Z::EBayesSample)

        # store points + ecdf
        push!(Z_points, Z); push!(ecdf_vals, F̂)

        # add constraints and KEEP their ConstraintRef in our arrays
        if (side == :both || side == :upper) && (F̂ + band < 1)
            c = @constraint(model, marginal_cdf <= F̂ + band)
            push!(upper_refs, c)
        else
            push!(upper_refs, nothing)
        end
        if (side == :both || side == :lower) && (F̂ - band > 0)
            c = @constraint(model, marginal_cdf >= F̂ - band)
            push!(lower_refs, c)
        else
            push!(lower_refs, nothing)
        end
    end

    return DKWCache(upper_refs, lower_refs, Z_points, ecdf_vals, band)
end
=#
# add discrete version of this?
@recipe function f(fitted_dkw::FittedDvoretzkyKieferWolfowitz)
    x_dkw = response.(collect(keys(fitted_dkw.summary)))
    F_dkw = collect(values(fitted_dkw.summary))

    band = fitted_dkw.band
    lower = max.(F_dkw .- band, 0.0)
    upper = min.(F_dkw .+ band, 1.0)
    cis_ribbon  = F_dkw .- lower, upper .- F_dkw
    fillalpha --> 0.36
    legend --> :topleft
    seriescolor --> "#018AC4"
    ribbon --> cis_ribbon

    x_dkw, F_dkw
end
