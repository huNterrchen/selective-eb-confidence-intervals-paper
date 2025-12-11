abstract type AbstractNormalSample{T} <: ContinuousEBayesSample{T} end

"""
    NormalSample(Z,σ)

An observed sample ``Z`` drawn from a Normal distribution with known variance ``\\sigma^2 > 0``.

```math
Z \\sim \\mathcal{N}(\\mu, \\sigma^2)
```

``\\mu`` is assumed unknown.
The type above is used when the sample ``Z`` is to be used for estimation or inference of ``\\mu``.

```jldoctest
julia> NormalSample(0.5, 1.0)          #Z=0.5, σ=1
N(0.5; μ, σ=1.0)
```
"""
struct NormalSample{T,S} <: AbstractNormalSample{T}
    Z::T
    σ::S
end


function NormalSample(σ::S) where {S}
    NormalSample(missing, σ)
end



"""
    StandardNormalSample(Z)

An observed sample ``Z`` drawn from a Normal distribution with known variance ``\\sigma^2 =1``.

```math
Z \\sim \\mathcal{N}(\\mu, 1)
```

``\\mu`` is assumed unknown. The type above is used when the sample ``Z`` is to be used for estimation or inference of ``\\mu``.

```jldoctest
julia> StandardNormalSample(0.5)          #Z=0.5
N(0.5; μ, σ=1.0)
```
"""
struct StandardNormalSample{T} <: AbstractNormalSample{T}
    Z::T
end

StandardNormalSample() = StandardNormalSample(missing)

eltype(Z::AbstractNormalSample{T}) where {T} = T
support(Z::AbstractNormalSample) = RealInterval(-Inf, +Inf)

response(Z::AbstractNormalSample) = Z.Z
var(Z::AbstractNormalSample) = Z.σ^2
var(Z::StandardNormalSample) = one(eltype(response(Z)))
var(Z::StandardNormalSample{Missing}) = 1.0

std(Z::AbstractNormalSample) = Z.σ
std(Z::StandardNormalSample) = one(eltype(response(Z)))
std(Z::StandardNormalSample{Missing}) = 1.0

nuisance_parameter(Z::AbstractNormalSample) = std(Z)
primary_parameter(::AbstractNormalSample) = :μ

likelihood_distribution(Z::AbstractNormalSample, μ) = Normal(μ, std(Z))


function Base.show(io::IO, Z::AbstractNormalSample)
    Zz = response(Z)
    print(io, "N(", Zz, "; μ, σ=", std(Z),")")
end





# Targets

# TODO: Note this is not correct for intervals.
function cf(target::MarginalDensity{<:AbstractNormalSample}, t)
    error_dbn = likelihood_distribution(location(target))
    cf(error_dbn, t)
end


# Conjugate computations
function default_target_computation(::BasicPosteriorTarget,
    ::AbstractNormalSample,
    ::Normal
)
    Conjugate()
end

function marginalize(Z::AbstractNormalSample, prior::Normal)
    prior_var = var(prior)
    prior_μ = mean(prior)
    likelihood_var = var(Z)
    marginal_σ = sqrt(likelihood_var + prior_var)
    Normal(prior_μ, marginal_σ)
end


function posterior(Z::AbstractNormalSample, prior::Normal)
    z = response(Z)
    sigma_squared = var(Z)
    prior_mu = mean(prior)
    prior_A = var(prior)

    post_mean =
        (prior_A) / (prior_A + sigma_squared) * z +
        sigma_squared / (prior_A + sigma_squared) * prior_mu
    post_var = prior_A * sigma_squared / (prior_A + sigma_squared)
    Normal(post_mean, sqrt(post_var))
end

# Uniform-Normal

struct UniformNormal{T} <: Distributions.ContinuousUnivariateDistribution
    a::T 
    b::T
    σ::T
end

Distributions.@distr_support UniformNormal -Inf Inf

function Distributions.pdf(d::UniformNormal, x::Real)
    exp(Distributions.logpdf(d, x))
end


function Distributions.logpdf(d::UniformNormal, x::Real)
    a, b, σ = d.a, d.b, d.σ
    z1 = (b - x) / σ
    z2 = (a - x) / σ
    d = Normal(0,1)
    log_cdf_b = logcdf(d, z1)
    log_cdf_a = logcdf(d, z2)
    
    return LogExpFunctions.logsubexp(log_cdf_b, log_cdf_a) - log(b - a)
end


function Distributions.cdf(d::UniformNormal, x::Real)
    exp(Distributions.logcdf(d, x))
end


function Distributions.logcdf(d::UniformNormal, x::Real)
    σ = d.σ
    a, b = d.a, d.b

    if x == -Inf
        return -Inf
    elseif x == Inf
        return 0.0
    end

    N = Normal()
    zL = (x - a) / σ
    zR = (x - b) / σ
    logφL = logpdf(N, zL);  logΦL = logcdf(N, zL)
    logφR = logpdf(N, zR);  logΦR = logcdf(N, zR)
    if min(zL, zR) ≤ 5.0

        logIL = (zL ≥ 0) ? LogExpFunctions.logsumexp(logφL, log(zL) + logΦL) :
                       LogExpFunctions.logsubexp(logφL, log(-zL) + logΦL)
        logIR = (zR ≥ 0) ? LogExpFunctions.logsumexp(logφR, log(zR) + logΦR) :
                       LogExpFunctions.logsubexp(logφR, log(-zR) + logΦR)
        return log(σ) - log(b - a) + LogExpFunctions.logsubexp(logIL, logIR)
    end
    logSL = logccdf(N, zL)
    logSR = logccdf(N, zR)

    logDL = (zL ≥ 0) ? LogExpFunctions.logsubexp(logφL, log(zL) + logSL) :
                       LogExpFunctions.logsumexp(logφL, log(-zL) + logSL)
    logDR = (zR ≥ 0) ? LogExpFunctions.logsubexp(logφR, log(zR) + logSR) :
                       LogExpFunctions.logsumexp(logφR, log(-zR) + logSR) 
    log1mF = log(σ) - log(b - a) + LogExpFunctions.logsubexp(logDR, logDL)                   
    return LogExpFunctions.log1mexp(log1mF)
   
end

#=
function Distributions.logdiffcdf(d::UnivariateDistribution, x::Real, y::Real)
    _x, _y = promote(x, y)
    _x < _y && throw(ArgumentError("requires x ≥ y"))

    ux = logcdf(d, _x)
    uy = logcdf(d, _y)

    return LogExpFunctions.logsubexp(ux, uy)
end

=#
function marginalize(Z::AbstractNormalSample, prior::Uniform)
    UniformNormal(prior.a, prior.b, std(Z))
end


# Target specifics
function Base.extrema(density::MarginalDensity{<:AbstractNormalSample{<:Real}})
    (0.0, 1 / sqrt(2π * var(location(density))))
end

# Marginalize Distributions.AffineDistribution

function marginalize(Z::AbstractNormalSample, prior::Distributions.AffineDistribution)
    (;μ,σ,ρ) = prior
    iszero(σ) && throw(ArgumentError("σ must be non-zero"))
    Zprime = NormalSample(response(Z), std(Z)/σ)
    μ + σ * marginalize(Zprime, ρ)
end