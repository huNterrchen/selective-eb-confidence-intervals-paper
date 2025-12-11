
"""
    TruncatedSample{T, EB<:ContinuousEBayesSample{T}, S} <: ContinuousEBayesSample{T}

A truncated empirical Bayes sample wrapping an underlying sample and truncation set.

# Fields
- `Z::EB`: Original continuous empirical Bayes sample
- `truncation_set::S`: Set defining the truncation region (must be left-closed, right-unbounded interval)

"""

struct TruncatedSample{T, EB<:ContinuousEBayesSample{T}, S} <: ContinuousEBayesSample{T}
    Z::EB
    truncation_set::S
end

"""
    nuisance_parameter(Z::TruncatedSample)

Return the truncation set defining the truncated region of the sample space.
"""

nuisance_parameter(Z::TruncatedSample) = Z.truncation_set

"""
    primary_parameter(Z::TruncatedSample)

Return the parameter of Interest of the underlying sample. This is the parameter that is estimated or inferred from the sample.
"""
#=
function Empirikos.TruncatedSample(sample::EBayesSample{Missing}, trunc_set)
    TruncatedSample{Missing, typeof(trunc_set)}(sample, trunc_set)
end
=#
primary_parameter(Z::TruncatedSample) = primary_parameter(Z.Z)


response(Z::TruncatedSample) = response(Z.Z)


function Base.show(io::IO, Z::TruncatedSample)
    print(io, "Trunc{", Z.Z," to ", string(Z.truncation_set),"}")
end

"""
    _truncated(d::Distribution, interval::AbstractInterval)

Helper function to create a left-truncated distribution. 

# Arguments
- `d`: Untruncated distribution
- `interval`: Must be left-closed and right-unbounded (`[a, ∞)`)

# Throws
- `ArgumentError` if interval is not left-closed and right-bounded
"""
#=
function _truncated(d::Distribution, interval::AbstractInterval)
    (isleftclosed(interval) && isinf(rightendpoint(interval))) || throw(ArgumentError("Only [a, ∞) {left-closed, right-unbounded} intervals allowed currently."))
    Distributions.truncated(d, leftendpoint(interval), nothing)
end
=#
function _truncated(d::Distribution, interval::AbstractInterval)
    Distributions.truncated(d, leftendpoint(interval), rightendpoint(interval))
end

#=
@inline function logdiffexp(α::Float64, β::Float64)
    δ = β - α  # ≤ 0
    if δ > -0.6931471805599453  # -log(2)
        return α + log(-expm1(δ))   # stable when δ close to 0
    else
        return α + log1p(-exp(δ))   # stable when δ is very negative
    end
end

# Robust log mass over an interval (x_lo, x_hi] for any univariate dist
@inline function log1mexp(x::Float64)
    # threshold ~ -log(2)
    if x > -0.6931471805599453
        return log(-expm1(x))
    else
        return log1p(-exp(x))
    end
end
@inline function log1mexp(x::Float64)
    # threshold ~ -log(2)
    if x > -0.6931471805599453
        return log(-expm1(x))
    else
        return log1p(-exp(x))
    end
end

function logcdf_uniformnormal(a::Float64, b::Float64, σ::Float64, x::Float64)
    zL = (x - a)/σ
    zR = (x - b)/σ
    # Use Normal() tails in log-space (Distributions.jl provides logcdf/logccdf)
    logΦL = logcdf(Normal(), zL)
    logΦR = logcdf(Normal(), zR)

    # log(Φ(zL) - Φ(zR)) - log(b-a)
    return logdiffexp(logΦL, logΦR) - log(b - a)
end

function logcdf_folded_uniformnormal(a::Float64, b::Float64, σ::Float64, x::Float64)
    @assert x ≥ 0.0
    logFx_pos = logcdf_uniformnormal(a,b,σ, x)
    logFx_neg = logcdf_uniformnormal(a,b,σ,-x)
    # log(Fx(x) - Fx(-x))
    return logdiffexp(logFx_pos, logFx_neg)
end

function cdf_trunc_folded_uniformnormal(a::Float64, b::Float64, σ::Float64,
                                        A::Float64, B::Float64, x::Float64)
    # boundaries
    if x ≤ A
        return 0.0
    elseif x ≥ B
        return 1.0
    end
    @assert A ≥ 0.0  # fold domain is nonnegative

    # log F_fold at key points
    logF_A = logcdf_folded_uniformnormal(a,b,σ, A)
    logF_x = logcdf_folded_uniformnormal(a,b,σ, x)

    # numerator: F_fold(x) - F_fold(A)
    log_num = logdiffexp(logF_x, logF_A)

    # denominator:
    if isinf(B)  # B = +∞
        # 1 - F_fold(A)
        log_den = log1mexp(logF_A)
    else
        logF_B = logcdf_folded_uniformnormal(a,b,σ, B)
        log_den = logdiffexp(logF_B, logF_A)
    end

    r = exp(log_num - log_den)
    # Clamp to [0,1] with a tiny tolerance to enforce monotonicity
    return clamp(r, 0.0, 1.0)
end
function Distributions.cdf(d::Truncated{<:Folded{<:UniformNormal}}, x::Real)
    fd = d.untruncated                    # Folded{UniformNormal}
    A, B = float(d.lower), float(d.upper)
    a, b, σ = fd.dist.a, fd.dist.b, fd.dist.σ

    if x ≤ A
        return 0.0
    elseif x ≥ B
        return 1.0
    end
    return cdf_trunc_folded_uniformnormal(a,b,σ, A,B, float(x))
end
=#
"""
    likelihood_distribution(Z::TruncatedSample, μ:number)

Returns the distribution ``p(\\cdot \\mid \\mu, Z \\in S )`` of ``Z \\mid \\mu, Z \\in S``, 
where ``S`` is the truncation set.
"""
function likelihood_distribution(Z::TruncatedSample{<:Any,<:Any,<:AbstractInterval}, μ)
    untruncated_d = likelihood_distribution(Z.Z, μ)
    _truncated(untruncated_d, Z.truncation_set) # TODO: introduce truncated subject to more general constraints
end




"""
    SelectionTilted{D<:ContinuousUnivariateDistribution, F1, T, EB, S} <: ContinuousUnivariateDistribution

The tilting operation in theorem 5, which allows confidence interval to be constructed as in Ignatiadis and Wager (2022)

# Fields
- `untilted`: Original prior distribution G of μ before tilting
- `tilting_function`: Φ(S; μ) = ``p(Z \\in S \\mid \\mu)``
- `selection_probability`: ∫Φ(S; μ)G dμ
- `log_selection_probability`: log(∫Φ(S; μ)G dμ)
- `truncation_sample`: Observed truncated sample Z
"""
Base.@kwdef struct SelectionTilted{D<:Distributions.UnivariateDistribution, F1,  T, EB} <: Distributions.ContinuousUnivariateDistribution
    untilted::D      # the original distribution
    tilting_function::F1
    selection_probability::T
    log_selection_probability::T
    truncation_sample::EB
end

function Base.show(io::IO, d::SelectionTilted)
    print(io, "SelectionTilted{", string(d.untilted)," to ", string(d.truncation_sample),"}")
end

"""
    tilt(d, Z)

Create prior ``\textrm{Tilt}_S[G]`` from original prior ``G`` and truncated sample ``Z``.

# Arguments
- `d`: Base distribution to tilt
- `Z`: Truncated sample

# Returns
- `SelectionTilted`: new prior ``\textrm{Tilt}_S[G]``
"""
function tilt(d::Distribution, Z)
    selection_probability = pdf(d, Z)
    log_selection_probability = logpdf(d, Z)
    tilting_function(μ) = Empirikos.likelihood(Z, μ)
    SelectionTilted(;untilted = d,
        tilting_function = tilting_function,
        selection_probability = selection_probability,
        log_selection_probability = log_selection_probability,
        truncation_sample = Z
    )
end
#=
function tilt(d::Dirac, Z)
    μ = d.value 
    

    selection_probability = likelihood(Z, μ)
    log_selection_probability = log(selection_probability)
    

    tilting_function(μ_val) = μ_val == μ ? selection_probability : 0.0
    
    SelectionTilted(;
        untilted = d,
        tilting_function = tilting_function,
        selection_probability = selection_probability,
        log_selection_probability = log_selection_probability,
        truncation_sample = Z
    )
end
=#
"""
    untilt(d::SelectionTilted)

Returns the prior distribution G of μ before tilting

# Examples
```julia-repl
julia> d = Normal(0, 1)
Normal{Float64}(μ=0.0, σ=1.0)

julia> Z = Empirikos.FoldedNormalSample([2.0, 2.2], 2.0)
|N([2.0, 2.2]; μ, σ=2.0)|

julia> tilted = Empirikos.tilt(d, Z)
SelectionTilted{Normal{Float64}(μ=0.0, σ=1.0) to |N([2.0, 2.2]; μ, σ=2.0)|}

julia> Empirikos.untilt(tilted)
Normal{Float64}(μ=0.0, σ=1.0)
```
"""
function untilt(d::SelectionTilted)
    d.untilted
end

"""
    selection_probability(d::SelectionTilted)

Returns the selection_probability ∫Φ(S; μ)G dμ
"""

function selection_probability(d::SelectionTilted)
    d.selection_probability
end

"""
    set_response(Z::TruncatedSample, znew=missing) -> TruncatedSample

Update the response value of a truncated sample while preserving the truncation set.

# Examples
```jldoctest
julia> Empirikos.set_response(Empirikos.TruncatedSample(StandardNormalSample(1), Interval(1.96, Inf)), 2)
Trunc{N(2; μ, σ=1) to 1.96 .. Inf}
julia> Empirikos.set_response(Empirikos.TruncatedSample(StandardNormalSample(1), Interval(1.96, Inf)))
Trunc{N(missing; μ, σ=1.0) to 1.96 .. Inf}
```
"""

function set_response(Z::TruncatedSample, znew=missing)
    new_sample = set_response(Z.Z, znew)
    TruncatedSample(new_sample, Z.truncation_set)
end

"""
    tilt(𝒢::AbstractMixturePriorClass, Z_trunc_set)

Create prior ``\textrm{Tilt}_S[G]`` for each component G in 𝒢 and truncated sample Z_trunc_set.

# Arguments
- `𝒢`: MixturePriorClass
- `Z_trunc_set`: Truncated sample

# Returns
- `SelectionTilted`: new prior ``\textrm{Tilt}_S[G]`` for each component G in 𝒢
"""

function tilt(𝒢::AbstractMixturePriorClass, Z_trunc_set)
    MixturePriorClass(tilt.(components(𝒢), Z_trunc_set))
end


function Distributions.pdf(d::SelectionTilted, x::Real)
    Distributions.pdf(d.untilted, x) / d.selection_probability * d.tilting_function(x)
end     


"""
    marginalize(Z_trunc::TruncatedSample, prior::SelectionTilted)

Compute marginal distribution of Z under model (A) with prior G.

# Arguments
- `Z_trunc`: Truncated observed sample
- `prior`: Tilted prior distribution

# Returns
- `Truncated{Distribution}`: Marginal distribution of Z under model (A) with prior G.
# Throws
- Argument error if prior truncation sample and constraints mismatch
"""

function marginalize(Z_trunc::TruncatedSample, prior::SelectionTilted)
    Z_untrunc = Z_trunc.Z
    truncation_set = Z_trunc.truncation_set
    if set_response(Z_trunc.Z, truncation_set) != prior.truncation_sample
        throw(ArgumentError("selection tilt and truncated sample do not match"))
    end
    marginal_untrunc = marginalize(Z_untrunc, prior.untilted)
    _truncated(marginal_untrunc, truncation_set)
end


"""
    ExtendedMarginalDensity{T} <: LinearEBayesTarget

A target for computing the marginal density  ''f_G^A(z)'' of a truncated sample under a selection-tilted prior, 
assuming truncation in the end.

# Fields
- `Z`: A truncated sample containing:
  - The observed data `Z.Z` (original untruncated sample)
  - The truncation set `Z.truncation_set`

# Methods
- `(target::ExtendedMarginalDensity)(prior)`: Computes the marginal density ''f_G^A(z)''as:
 

where:
- `untilted_prior` is the original prior G
- `selection_probability` is the probability of observing data under marginal distribution of Z

Validates that the truncation set in `Z` matches the prior's assumed truncation.
Throws an error if there is a mismatch.
"""

struct ExtendedMarginalDensity{T} <: LinearEBayesTarget
    Z::T
end

location(target::ExtendedMarginalDensity) = target.Z
 

function (target::ExtendedMarginalDensity)(prior::SelectionTilted)
    # code duplication with marginalize.
    Z_trunc = target.Z
    Z_untrunc = Z_trunc.Z
    truncation_set = Z_trunc.truncation_set
    if set_response(Z_trunc.Z, truncation_set) != prior.truncation_sample
        throw(ArgumentError("selection tilt and truncated sample do not match"))
    end
    Distributions.pdf(prior.untilted, Z_untrunc) / prior.selection_probability
end

function (target::ExtendedMarginalDensity)(d::MixtureModel)
    sum( probs(d) .* target.(components(d)))
end

"""
    struct ExtendedMarginalDensity_without{T} <: BasicPosteriorTarget

Represents the normalized marginal density without tilting adjustment, defined as:
```math
f_G(z)/\\PP[G]{Z \\in \\selection}
```
"""
struct ExtendedMarginalDensity_without{T} <: BasicPosteriorTarget
    Z::T
end

struct SelectionProbability{T} <: LinearEBayesTarget
    Z::T
end
function (t::SelectionProbability)(prior::Distribution)
    Z_trunc = t.Z
    truncation_set = Z_trunc.truncation_set
    new_sample = set_response(Z_trunc.Z, truncation_set)
    tilted = tilt(prior, new_sample)
    tilted.selection_probability 
end

(t::SelectionProbability)(d::MixtureModel) = sum(probs(d) .* t.(components(d)))

location(target::ExtendedMarginalDensity_without) = target.Z
 
(t::ExtendedMarginalDensity_without)(d::MixtureModel) = numerator(t)(d)/denominator(t)(d)
Base.numerator(t::ExtendedMarginalDensity_without)  = Empirikos.MarginalDensity(location(t).Z)
Base.denominator(t::ExtendedMarginalDensity_without) = SelectionProbability(location(t))


struct ExtendedMarginalDistribution{T} <: LinearEBayesTarget
    Z::T
end

location(target::ExtendedMarginalDistribution) = target.Z
 

function (target::ExtendedMarginalDistribution)(prior::SelectionTilted)
    Z_trunc = target.Z
    Z_untrunc = Z_trunc.Z
    truncation_set = Z_trunc.truncation_set
    if set_response(Z_trunc.Z, truncation_set) != prior.truncation_sample
        throw(ArgumentError("selection tilt and truncated sample do not match"))
    end
    z_val = response(Z_trunc.Z)
    if z_val >= truncation_set.left
        dist = prior.untilted
        f = FoldedNormalSample(truncation_set.left, 1.0)     
        numerator = Distributions.cdf(dist, Z_untrunc) - Distributions.cdf(dist, f) 
    else
        numerator = Distributions.cdf(prior.untilted, Z_untrunc)
    end
    denominator = prior.selection_probability    
    
    return numerator / denominator
end

function (target::ExtendedMarginalDistribution)(d::MixtureModel)
    sum( probs(d) .* target.(components(d)))
end



"""
    UntiltNormalizationConstant <: LinearEBayesTarget

Computes the normalization constant required to untilt the tilting adjustment in selection-tilted distributions.
For a selection-tilted prior `d`, returns `1 / selection_probability(d)`. Handles mixture priors by summation.

# Fields
- None (singleton type).

# Methods
- `(::UntiltNormalizationConstant)(d::SelectionTilted)`: Returns `1 / d.selection_probability`.
- `(::UntiltNormalizationConstant)(d::MixtureModel)`: Weighted sum over mixture components.
"""
struct UntiltNormalizationConstant <: LinearEBayesTarget
end

(::UntiltNormalizationConstant)(d::SelectionTilted) = 1/selection_probability(d)
function (target::UntiltNormalizationConstant)(d::MixtureModel)
    sum(Distributions.probs(d) .*  target.(components(d)))
end



"""
    UntiltLinearFunctionalNumerator{T} <: LinearEBayesTarget

Rewrite a linear functional of G in terms of the ratio of selection-tilted distribution. Calculate the numerator of the
ratio functional, which is ∫ϕ(μ)G dμ/∫Φ(s;μ)G dμ.
"""
# probably need to say how we are pretilting to avoid bugs
# but for now let's assume pretilt is with respect to identical tilt of selection measure
struct UntiltLinearFunctionalNumerator{T} <: LinearEBayesTarget
    target::T
end


"""
(pretilt_target::UntiltLinearFunctionalNumerator)(d::SelectionTilted)

Evaluate numerator functional for selection-tilted distributions through:

1. Untilt to recover base prior G
2. Apply original target ϕ(μ)
3. Divide by total selection probability \\int \\Phi(s;μ) G(dμ)
"""
function (pretilt_target::UntiltLinearFunctionalNumerator)(d::SelectionTilted)
    target = pretilt_target.target
    target(untilt(d)) / selection_probability(d)
end

function (target::UntiltLinearFunctionalNumerator)(d::MixtureModel)
    sum(Distributions.probs(d) .*  target.(components(d)))
end

"""
    UntiltedLinearTarget{T<:LinearEBayesTarget} <: AbstractPosteriorTarget

A linear target of G the ratio of untilted linear functionals, calculated as ratio functionals of tilted distributions.
"""
struct UntiltedLinearTarget{T<:LinearEBayesTarget} <: AbstractPosteriorTarget
    target::T
end

function untilt(target::LinearEBayesTarget)
    UntiltedLinearTarget(target)
end


Base.denominator(::UntiltedLinearTarget) = UntiltNormalizationConstant()
function Base.numerator(target::UntiltedLinearTarget)
    UntiltLinearFunctionalNumerator(target.target)
end

function (target::UntiltedLinearTarget)(d)
    numerator(target)(d) / denominator(target)(d)
end

"""
    UntiltedPosteriorTarget{T<:BasicPosteriorTarget} <: AbstractPosteriorTarget

A posterior target of G, which is a ratio functional of G, rewritten in terms of the ratio 
functionals of tilted distributions.
R(G) = N(G)/D(G) = 
"""

struct UntiltedPosteriorTarget{T<:BasicPosteriorTarget} <: AbstractPosteriorTarget
    target::T
end

function untilt(target::BasicPosteriorTarget)
    UntiltedPosteriorTarget(target)
end



function (target::UntiltedPosteriorTarget)(d)
    numerator(target)(d) / denominator(target)(d)
end

function Base.denominator(target::UntiltedPosteriorTarget)
    Base.numerator(untilt(Base.denominator(target.target)))
end

function Base.numerator(target::UntiltedPosteriorTarget)
    Base.numerator(untilt(Base.numerator(target.target)))
end



#=
struct SelectionTilted{D<:Distributions.ContinuousUnivariateDistribution, EB, I,
         T<: Real, TL<:Union{T,Nothing}, TU<:Union{T,Nothing}} <: Distributions.ContinuousUnivariateDistribution
    untruncated::D      # the original distribution (untruncated)
    ebayes_sample::EB
    selection_event::I
    log_selection_probability::T

    lower::TL     # lower bound
    upper::TU     # upper bound
    loglcdf::T    # log-cdf of lower bound (exclusive): log P(X < lower)
    lcdf::T       # cdf of lower bound (exclusive): P(X < lower)
    ucdf::T       # cdf of upper bound (inclusive): P(X ≤ upper)

    tp::T         # the probability of the truncated part, i.e. ucdf - lcdf
    logtp::T      # log(tp), i.e. log(ucdf - lcdf)

    function Truncated(d::UnivariateDistribution, l::TL, u::TU, loglcdf::T, lcdf::T, ucdf::T, tp::T, logtp::T) where {T <: Real, TL <: Union{T,Nothing}, TU <: Union{T,Nothing}}
        new{typeof(d), value_support(typeof(d)), T, TL, TU}(d, l, u, loglcdf, lcdf, ucdf, tp, logtp)
    end
end 


function tilt(d::SymmetricLocationPair, Z::EBayesSample)
    selection_prob = pdf(d, Z)
    log_selection_prob = logpdf(d, Z)
    tilting_func(μ) = likelihood(Z, μ)
    
    SelectionTilted(;
        untilted = d,
        tilting_function = tilting_func,
        selection_probability = selection_prob,
        log_selection_probability = log_selection_prob,
        truncation_sample = Z
    )
end

# Extend untilt for SelectionTilted of SymmetricLocationPair
function untilt(d::SelectionTilted{SymmetricLocationPair})
    d.untilted
end

# Extend tilt for mixture prior class
function tilt(𝒢::SymmetricGaussianLocationScaleMixtureClass, Z_trunc_set)
    # Tilt each component individually
    tilted_comps = [
        tilt(SymmetricLocationPair(μ, 𝒢.std), Z_trunc_set) for μ in 𝒢.μs_pos
    ]
    tilted_zero_comps = [
        tilt(Normal(0, σ), Z_trunc_set) for σ in 𝒢.σs
    ]
    
    # Create new mixture prior class with tilted components
    TiltedMixturePriorClass(vcat(tilted_comps, tilted_zero_comps))
end
function marginalize(Z::EBayesSample, prior::SymmetricLocationPair)
    # Extract parameters
    μ = prior.μ
    σ = prior.σ
    
    # Compute marginal likelihood for both components
    marginal_minus = marginalize(Z, Normal(-μ, σ))
    marginal_plus = marginalize(Z, Normal(μ, σ))
    
    # Combine with 0.5 weights
    return 0.5 * marginal_minus + 0.5 * marginal_plus
end
=#