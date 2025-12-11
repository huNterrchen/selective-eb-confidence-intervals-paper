using Empirikos
using Test


_target = MarginalDensity(StandardNormalSample(3.0))
@test Base.extrema(_target)[2] == pdf(Normal(),0)

@test Base.extrema(MarginalDensity(NormalSample(3.0, 0.5)))[2] == pdf(Normal(0,0.5),0.0)

n = Normal()
a = 1.0
b = 2.3

# Check if marginalization of a normal with an AffineDistribution is correct

prior = a + b * n
prior_locscale = Distributions.AffineDistribution(a, b, n)
prior_locscale_mixture = Distributions.AffineDistribution(a, b, MixtureModel([n],[1.0]))

Z = NormalSample(3.0, 0.5)
@test marginalize(Z, prior_locscale) == marginalize(Z, prior)

marg_mix = marginalize(Z, prior_locscale_mixture)
@test probs(marg_mix.ρ) == [1.0]
@test marg_mix.μ + marg_mix.σ * first(components(marg_mix.ρ)) == marginalize(Z, prior)


@testset "Folded Normal quantile" begin
    d1 = Empirikos.Folded(Normal(0,1))
    q05 = quantile(d1, 0.5)
    @test isapprox(q05, quantile(Normal(), (1+0.5)/2); atol=1e-12)
    # Edge case
    d3 = Empirikos.Folded(Normal(-3.0, 0.0))
    @test quantile(d3, 0.1) == 3.0
    @test quantile(d3, 0.9) == 3.0
end


@testset "Folded t quantile" begin
    d2 = Empirikos.Folded(TDist(5))
    for q in (0.1, 0.5, 0.9)
        val = quantile(d2, q)
        ref = quantile(TDist(5), (1+q)/2)
        @test isapprox(val, ref; atol=1e-12)
    end
end
# --- helper: safe comparison of exp(logpdf) and pdf ---
# Only assert equality where pdf is reasonably above underflow (e.g., ≥ 1e-300)
# and below overflow (≤ 1e300) to avoid trivial failures due to floating limits.
function assert_exp_logpdf_matches_pdf(d, xs; atol=1e-12, rtol=1e-10, floor=1e-300, ceil=1e300)
    @testset "exp(logpdf) ≈ pdf on safe range" begin
        for x in xs
            p = pdf(d, x)
            if isfinite(p) && floor ≤ p ≤ ceil
                lp = logpdf(d, x)
                @test isfinite(lp)
                @test exp(lp) ≈ p atol=atol rtol=rtol
            end
        end
    end
end

@testset "UniformNormal.logpdf" begin
    # A few parameter sets
    params = [
        (-1.0, 1.0, 1.0),     # symmetric, moderate σ
        (-0.1, 0.1, 0.01),    # symmetric, tiny σ (hard case)
        (-0.0266, 0.0266, 1.0), # symmetric, very narrow interval vs σ
        (0.0, 2.0, 0.1),      # asymmetric, small σ
        (-3.0, -1.0, 2.0)     # asymmetric, large σ
    ]

    for (a,b,σ) in params
        d = Empirikos.UniformNormal(a,b,σ)

        @testset "Basic finiteness & no throw: a=$a, b=$b, σ=$σ" begin
            xs = vcat(
                range(a-10σ, a-3σ; length=3),
                [a-σ, a-1e-9, a, (a+b)/2, b, b+1e-9, b+σ],
                range(b+3σ, b+10σ; length=3)
            )
            # Should not throw and should be finite (logpdf can be very negative but finite)
            for x in xs
                @test isfinite(logpdf(d, x))
                @test isfinite(pdf(d, x)) || isnan(pdf(d, x)) == false  # pdf can underflow to 0.0, that's OK
            end
        end

        @testset "exp(logpdf) ≈ pdf in central region: a=$a, b=$b, σ=$σ" begin
            xs = [
                (a+b)/2,
                a + 0.1*(b-a),
                a + 0.5*(b-a),
                a + 0.9*(b-a),
                a - 0.5σ,
                b + 0.5σ
            ]
            assert_exp_logpdf_matches_pdf(d, xs)
        end

        @testset "Extreme tails are handled correctly: a=$a, b=$b, σ=$σ" begin
            xs = [ (b-a)*100 + b, -(b-a)*100 + a, 10σ + b, -10σ + a, 50.0, -50.0 ]
            for x in xs
                lp = logpdf(d, x)
                @test !isnan(lp)
                @test lp ≤ 0.0   # should not be NaN or throw
            end
        end

        if isapprox(a, -b; atol=0.0, rtol=0.0)  # symmetric support case
            @testset "Symmetry: a=-b, σ=$σ" begin
                xs = [-5σ, -2σ, -σ, 0.0, σ, 2σ, 5σ, a, b, (a+b)/2]
                for x in xs
                    @test logpdf(d, x) ≈ logpdf(d, -x) atol=1e-12 rtol=0.0
                end
            end
        end
    end
end