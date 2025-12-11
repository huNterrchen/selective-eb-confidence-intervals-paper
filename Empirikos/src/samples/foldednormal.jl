"""
    Folded{D<:ContinuousUnivariateDistribution} <: ContinuousUnivariateDistribution

A folded continuous univariate distribution, representing the absolute value of a random variable 
following the original distribution `D`. For a distribution `dist`, the folded version has:

- `pdf(d, x) = pdf(unfold(d), x) + pdf(unfold(d), -x)` for x ≥ 0  
- `cdf(d, x) =  P(unfold(d) ≤ x) − P(unfold(d) ≤ −x)` for x ≥ 0  
- analogously for `ccdf` and `quantile`

# Fields
- `dist::D`: The original (unfolded) distribution.
"""

struct Folded{D<:ContinuousUnivariateDistribution} <: ContinuousUnivariateDistribution
    dist::D
end

"""
    fold(dist::ContinuousUnivariateDistribution)

Create a `Folded` version of the distribution `dist`, representing `|X|` where `X ~ dist`.
"""
fold(dist::ContinuousUnivariateDistribution) = Folded(dist)

"""
    unfold(dist::Folded)

Return the original (unfolded) distribution wrapped in `dist`.
"""
unfold(dist::Folded) = dist.dist

"""
    minimum(::Folded)

Return `0.0`, the minimum value of a folded distribution (always non-negative).
# Examples
"""
Base.minimum(::Folded) = zero(Float64)

"""
    maximum(d::Folded)

Return the maximum value of the folded distribution, which returns Inf or max{max(unfold(d)), -min(unfold(d))} .
# Examples
"""
function Distributions.maximum(d::Folded)
    orig_max = maximum(unfold(d))
    if isinf(orig_max)
        return Inf
    else
        return Float64(max(orig_max, -minimum(unfold(d))))
    end
end

"""
    pdf(d::Folded, x::Real)

Compute the probability density function of the folded distribution at `x`, which is the sum 
of the PDF of the original distribution at  `x` and `-x` for `x ≥ 0`, and 0 otherwise.
"""
function Distributions.pdf(d::Folded, x::Real)
    d_pdf = pdf(unfold(d), x) + pdf(unfold(d), -x)
    return x >= 0 ? d_pdf : zero(d_pdf)
end

"""
    logpdf(d::Folded, x::Real)

Compute the log probability density function using log-sum-exp for numerical stability.
"""
function Distributions.logpdf(d::Folded, x::Real)
    d_logpdf_left = logpdf(unfold(d), -x)
    d_logpdf_right = logpdf(unfold(d), x)
    d_logpdf = LogExpFunctions.logaddexp(d_logpdf_left, d_logpdf_right)
    return x >= 0 ? d_logpdf : oftype(d_logpdf, -Inf)
end

"""
    cdf(d::Folded, x::Real)

Compute the cumulative distribution function as `cdf(dist, x) - cdf(dist, -x)` for `x > 0`, and 0 otherwise.
"""
function Distributions.cdf(d::Folded, x::Real)
    d_cdf = cdf(unfold(d), x) - cdf(unfold(d), -x)
    return x > 0 ? d_cdf : zero(d_cdf)
end

function Distributions.logcdf(d::Folded, x::Real)
    if x ≤ 0
        return oftype(float(x), -Inf)  
    elseif x == Inf
        return oftype(float(x), 0.0)    
    end

    lx  = logcdf(unfold(d),  x)
    lmx = logcdf(unfold(d), -x)
    LogExpFunctions.logsubexp(lx, lmx)
end
#=
function Distributions.logcdf(d::Folded, x::Real)
    if x ≤ 0
        return oftype(float(x), -Inf)
    elseif x == Inf
        return oftype(float(x), 0.0)
    end
    base = unfold(d)

    # Left form: log(F(x) - F(-x))  with known ordering: F(x) ≥ F(-x) ⇒ lx ≥ lmx
    lx  = logcdf(base,  x)
    lmx = logcdf(base, -x)
    δ1 = lx - lmx
    left = if δ1 ≤ 1e-9
        # First-order: log(exp(lx) - exp(lmx)) ≈ lmx + log(δ1), guard δ1>0
        (δ1 ≤ 0 ? oftype(lx, -Inf) : (lmx + log(δ1)))
    else
        logsubexp(lx, lmx)
    end

    # Right form: log(S(-x) - S(x)) with S(-x) ≥ S(x) ⇒ smx ≥ sx
    sx  = logccdf(base,  x)
    smx = logccdf(base, -x)
    δ2 = smx - sx
    right = if δ2 ≤ 1e-9
        (δ2 ≤ 0 ? oftype(sx, -Inf) : (sx + log(δ2)))
    else
        logsubexp(smx, sx)
    end

    return max(left, right)
end
=#
"""
    ccdf(d::Folded, x::Real)

Compute the complementary CDF as `ccdf(dist, x) + cdf(dist, -x)` for `x > 0`, and 1 otherwise.
"""
function Distributions.ccdf(d::Folded, x::Real)
    d_ccdf = ccdf(unfold(d), x) + cdf(unfold(d), -x)
    return x > 0 ? d_ccdf : one(d_ccdf)
end

function Distributions.logccdf(d::Folded, x::Real)
    if x ≤ 0
        return oftype(float(x), 0.0)     
    elseif x == Inf
        return oftype(float(x), -Inf)   
    end
     
    sx   = logccdf(unfold(d), x)              
    fmx  = logcdf(unfold(d), -x)             

    return LogExpFunctions.logsumexp(sx, fmx)
end

"""
    quantile(d::Folded, q::Real)

Compute the quantile of a folded normal by using the noncentral chi-squared distribution.
"""
function Distributions.quantile(d::Folded{<:Normal}, q::Real)
    orig_normal = d.dist
    μ = mean(orig_normal)  
    σ = std(orig_normal)  
    σ == 0 && return abs(μ)
    
    λ = (μ/σ)^2
    nc_chisq = NoncentralChisq(1, λ)
    
    σ * sqrt(quantile(nc_chisq, q))
end

"""
    quantile(d::Folded{<:TDist}, q::Real)

Compute the quantile for folded t-distributions, this maps to the 
`(1 + q)/2` quantile of the original symmetric distribution.
"""
function Distributions.quantile(d::Folded{<:TDist}, q::Real)
    Distributions.quantile(unfold(d), (1+q)/2)
 end

"""
    FoldedNormalSample(Z,σ)

An observed sample ``Z`` equal to the absolute value of a draw
from a Normal distribution with known variance ``\\sigma^2 > 0``.

```math
Z = |Y|, Y\\sim \\mathcal{N}(\\mu, \\sigma^2)
```

``\\mu`` is assumed unknown. The type above is used when the sample ``Z`` is to be used for estimation or inference of ``\\mu``.
"""
struct FoldedNormalSample{T,S} <: ContinuousEBayesSample{T}
    Z::T
    σ::S
end

function FoldedNormalSample(Z::Real, σ::Real)
    z = float(Z); s = float(σ)
    z ≥ 0      || throw(DomainError(Z, "Folded observation requires Z ≥ 0."))
    (isfinite(s) && s > 0) || throw(DomainError(σ, "σ must be finite and > 0."))
    return FoldedNormalSample{typeof(z), typeof(s)}(z, s)
end

"""
    FoldedNormalSample(Z)

Construct a `FoldedNormalSample` with default `σ = 1.0`.
"""
FoldedNormalSample(Z) = FoldedNormalSample(Z, 1.0)


"""
    FoldedNormalSample()

Construct a `FoldedNormalSample` with missing data and `σ = 1.0`.
# Examples
```julia-repl
julia> Empirikos.FoldedNormalSample()
|N(missing; μ, σ=1.0)|
```
"""
FoldedNormalSample() = FoldedNormalSample(missing)

"""
    FoldedNormalSample(Z::AbstractNormalSample)

Convert a `NormalSample` to a `FoldedNormalSample` by taking its absolute value.
"""
function FoldedNormalSample(Z::AbstractNormalSample)
    FoldedNormalSample(abs(response(Z)), std(Z))
end

"""
    response(Z::FoldedNormalSample)

Return the observed value `Z` (non-negative).
# Examples
"""
response(Z::FoldedNormalSample) = Z.Z

"""
    nuisance_parameter(Z::FoldedNormalSample)

Return the known standard deviation `σ` of the original Normal distribution.
"""
nuisance_parameter(Z::FoldedNormalSample) = Z.σ

std(Z::FoldedNormalSample) = Z.σ

"""
    var(Z::FoldedNormalSample)

Return the variance `σ²`.
"""
var(Z::FoldedNormalSample) = abs2(std(Z))

"""
    NormalSample(Z::FoldedNormalSample; positive_sign = true)

Convert a `FoldedNormalSample` back to a `NormalSample` by assigning a sign (default: positive).
"""
function NormalSample(Z::FoldedNormalSample; positive_sign = true)
    response_Z = positive_sign ? response(Z) : -response(Z)
    NormalSample(response_Z, nuisance_parameter(Z))
end


function Base.show(io::IO, Z::FoldedNormalSample)
    print(io, "|", NormalSample(Z), "|")
end

"""
    _symmetrize(Zs::AbstractVector{<:FoldedNormalSample})

Randomly assign signs to a vector of FoldedNormalSample to reconstruct a vector of `NormalSample`s. 
"""
function _symmetrize(Zs::AbstractVector{<:FoldedNormalSample})
   random_signs =  2 .* rand(Bernoulli(), length(Zs)) .-1
   NormalSample.(random_signs .* response.(Zs), std.(Zs))
end

"""
    likelihood_distribution(Z::FoldedNormalSample, μ)

Return the folded Normal distribution `fold(Normal(μ, σ))` for a given mean `μ`.
# Examples
"""
function likelihood_distribution(Z::FoldedNormalSample, μ)
    fold(Normal(μ, nuisance_parameter(Z)))
end



function default_target_computation(::BasicPosteriorTarget,
    ::FoldedNormalSample,
    ::Normal
)
    Conjugate()
end


"""
    marginalize(Z::FoldedNormalSample, prior::Normal) -> Folded{Normal}

Compute marginal distribution for folded normal observation with normal prior.

1. Unfold the observation to normal sample
2. Compute marginal distribution of the normal sample with the normal prior
3. Fold the resulting distribution
"""
# perhaps this can apply to more general folded samples.
function marginalize(Z::FoldedNormalSample, prior::Normal)
    Z_unfolded = NormalSample(Z)
    fold(marginalize(Z_unfolded, prior)) 
end

"""
marginalize(Z::FoldedNormalSample, prior::Folded{Normal}) -> Folded{Normal}

Compute marginal distribution for folded normal observation with folded normal prior by:
1. Unfolding the prior to its original normal distribution
2. Computing the conjugate marginal distribution
3. Re-folding the result
```
"""
function marginalize(Z::FoldedNormalSample, prior::Folded{<:Normal})
    marginalize(Z, unfold(prior))
end

"""
    posterior(Z::FoldedNormalSample, prior::Normal) -> MixtureModel{Normal}

Compute the posterior distribution for a folded normal observation by considering both possible 
signs of the original observation, weighted by their marginal probabilities under the prior.



# Examples
```julia-repl
julia> Empirikos.posterior(FoldedNormalSample(2.0), Normal(0,1))
MixtureModel{Normal{Float64}}(K = 2)
components[1] (prior = 0.5000): Normal{Float64}(μ=1.0, σ=0.7071067811865476)
components[2] (prior = 0.5000): Normal{Float64}(μ=-1.0, σ=0.7071067811865476)
```
"""
function posterior(Z::FoldedNormalSample, prior::Normal)
    Z_unfolded_positive = NormalSample(Z; positive_sign = true)
    Z_unfolded_negative = NormalSample(Z; positive_sign = false)

    marginal_prob_positive = pdf(prior, Z_unfolded_positive)
    marginal_prob_negative = pdf(prior, Z_unfolded_negative)
    marginal_prob_sum = marginal_prob_positive + marginal_prob_negative

    prob_positive = marginal_prob_positive / marginal_prob_sum
    prob_negative = marginal_prob_negative / marginal_prob_sum

    posterior_dbn_positive = posterior(Z_unfolded_positive, prior)
    posterior_dbn_negative = posterior(Z_unfolded_negative, prior)

    posterior_dbn = MixtureModel(
        [posterior_dbn_positive, posterior_dbn_negative],
        [prob_positive, prob_negative]
    )
    posterior_dbn
end


"""
    marginalize(Z::FoldedNormalSample, prior::Uniform) -> Folded{UniformNormal}

Compute the marginal distribution for a folded normal observation under a uniform prior.

# Arguments
- `Z::FoldedNormalSample`: A folded normal observation.
- `prior::Uniform`: Uniform prior distribution.

# Returns
- `Folded{UniformNormal}`: Folded UniformNormal distribution representing the marginal distribution.
"""
function marginalize(Z::FoldedNormalSample, prior::Uniform)
    Z_unfolded = NormalSample(Z)
    unif_normal = marginalize(Z_unfolded, prior)
    fold(unif_normal)
end

"""
    SignAgreementProbability(Z::FoldedNormalSample) <: AbstractPosteriorTarget

Type representing the probability that the observed z-score and the true stan-
dardized effect share the same sign, i.e.,

```math
P_{G}{\\mu \\cdot Z > 0 \\mid \\abs{Z}=z}
```
"""
struct SignAgreementProbability{T<:FoldedNormalSample} <: BasicPosteriorTarget
    Z::T
end

location(target::SignAgreementProbability) = target.Z

function (target::SignAgreementProbability)(prior::Distribution)
    num_val = numerator(target)(prior)
    den_val = denominator(target)(prior)
    return num_val / den_val
end

struct SignAgreementProbabilityNumerator{T<:FoldedNormalSample} <: LinearEBayesTarget
    Z::T
end

function (t::SignAgreementProbabilityNumerator)(prior::Distribution)
    z_val = t.Z.Z
    Z_plus = NormalSample(z_val, 1)
    Z_minus = NormalSample(-z_val, 1)
    
    positive_set = Interval(0.0, Inf)
    negative_set = Interval(-Inf, 0.0)
    prob_positive = numerator(PosteriorProbability(Z_plus, positive_set))(prior)
    prob_negative = numerator(PosteriorProbability(Z_minus, negative_set))(prior)
    
    prob_positive + prob_negative
end


Base.numerator(t::SignAgreementProbability) = SignAgreementProbabilityNumerator(location(t))


#=
"""
    SignAgreementProbability(Z::EBayesSample) <: AbstractPosteriorTarget

Type representing the probability that the observed z-score and the true stan-
dardized effect share the same sign, i.e.,

```math
P_{G}{\\mu \\cdot Z > 0 \\mid \\abs{Z}=z}
```
"""
struct SignAgreementProbability{T<:FoldedNormalSample} <: BasicPosteriorTarget
    Z::T
end

location(target::SignAgreementProbability) = target.Z
function compute_target(::Conjugate, target::SignAgreementProbability, Z::FoldedNormalSample, prior)
    mm = posterior(location(target), prior)
    post_plus,  post_minus  = mm.components
    w_plus,     w_minus     = mm.prior.p
    p_agree =
        w_plus  * (1 - cdf(post_plus,  0.0)) +
        w_minus * (      cdf(post_minus, 0.0))

    return p_agree
end

function (t::SignAgreementProbability)(μ::Number)
    Zfold = location(t)
    z = response(Zfold)
    σ = std(Zfold)

    ℓ_plus  = pdf(Normal(μ, σ),  +z)
    ℓ_minus = pdf(Normal(μ, σ),  -z)
    denom   = ℓ_plus + ℓ_minus          

    if denom == 0
        return zero(float(μ))
    end

    agree = (μ > 0 ? ℓ_plus : 0.0) + (μ < 0 ? ℓ_minus : 0.0)
    return agree / denom
end

function (target::SignAgreementProbability)(prior::Distribution)
    num_val = numerator(target)(prior)
    den_val = denominator(target)(prior)
    return num_val / den_val
end

struct SignAgreementProbability_num{T<:FoldedNormalSample} <: LinearEBayesTarget
    Z::T
end

function (t::SignAgreementProbability_num)(prior::Distribution)
    z_val = response(location(t))
    Z_plus = NormalSample(z_val, 1)
    Z_minus = NormalSample(-z_val, 1)
    
    positive_set = Interval(0.0, Inf)
    negative_set = Interval(-Inf, 0.0)
    prob_positive = numerator(PosteriorProbability(Z_plus, positive_set))(prior)
    prob_negative = numerator(PosteriorProbability(Z_minus, negative_set))(prior)
    
    prob_positive + prob_negative
end


Base.numerator(t::SignAgreementProbability) = SignAgreementProbability_num(location(t))
=#