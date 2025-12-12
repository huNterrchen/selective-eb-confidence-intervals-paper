using Pkg
using Revise
Pkg.develop("Empirikos")
using Empirikos
using Plots
using CSV
using IntervalSets
using StatsDiscretizations 
using Distributions
using PGFPlotsX
using MosekTools
using LaTeXStrings
using Roots
using QuadGK
using Random
using Hypatia
using Mosek
using JuMP
using Gurobi
using DataFrames, PrettyTables
using RCall
using ProgressMeter
using StatsBase
using StatsFuns
using LogExpFunctions
using CSV, DataFrames
begin
    pgfplotsx()
    empty!(PGFPlotsX.CUSTOM_PREAMBLE)
    push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{amssymb}")
    push!(
        PGFPlotsX.CUSTOM_PREAMBLE,
        raw"\newcommand{\PP}[2][]{\mathbb{P}_{#1}\left[#2\right]}",
    )
    push!(
        PGFPlotsX.CUSTOM_PREAMBLE,
        raw"\newcommand{\EE}[2][]{\mathbb{E}_{#1}\left[#2\right]}",
    )
end

theme(
    :default;
    background_color_legend = :transparent,
    foreground_color_legend = :transparent,
    grid = nothing,
    legendfonthalign = :left,
    thickness_scaling = 1.15,
    size = (420, 330),
)

tbl = CSV.File("cochrane_rct_effects_23551.csv")


zs = tbl.z
pgfplotsx()

bins = -10.04:0.16:10.04
histogram(zs[abs.(zs) .< 10]; bins=bins, xlim =(-10,10),
    label = "", linecolor=:gray, alpha=0.5, linewidth=0.25, size = (500, 270),
    xlabel = L"Z_i", ylabel = "Count",
    xticks = [-8, -5, -1.96, 0, 1.96, 5, 8])
vline!([-1.96; 1.96], label="", linestyle=:dash, color=:black)
savefig("cochrane_histogram.png")
trunc_set = Interval(2.1, Inf)

#for simulation
trunc_set = Interval(1.96, Inf)
#
abs_zs_init = abs.(tbl.z)
abs_zs = abs_zs_init[in.( abs_zs_init, Ref(trunc_set))]
abs_zs = abs_zs[isfinite.(abs_zs)]
#for simulation
Zs = [z >=6 ? FoldedNormalSample(6..Inf) : FoldedNormalSample(z) for z in abs_zs]
#
Zs = FoldedNormalSample.(abs_zs, 1.0)
Z_trunc_set = FoldedNormalSample(trunc_set, 1.0)

gcal_scalemix = Empirikos.autoconvexclass(Empirikos.GaussianScaleMixtureClass();
    grid_scaling = 1.2, σ_min = 0.001, σ_max=100.0)
tilted_scale_mix = Empirikos.tilt(gcal_scalemix, Z_trunc_set)

ucal_scalemix = Empirikos.autoconvexclass(Empirikos.UniformScaleMixtureClass();
             a_min=0.001, a_max=100.0, grid_scaling=1.2
         )
tilted_u_scale_mix = Empirikos.tilt(ucal_scalemix, Z_trunc_set)

support_points = vcat(0:1:6, 7:8)
support_points = 0:1:6
z_curvemix = Empirikos.autoconvexclass(Empirikos.DiscretePriorClass(support_points))
tilted_z_curvemix = Empirikos.tilt(z_curvemix, Z_trunc_set)
gcal_locationscale_mix = Empirikos.autoconvexclass(Empirikos.GaussianLocationScaleMixtureClass();
    grid_scaling = 1.2, μ_min=0, μ_max=12, std = 0.05, σ_min = 0.001, σ_max=100.0)
tilted_location_scale_mix = Empirikos.tilt(gcal_locationscale_mix, Z_trunc_set)

Zs_trunc = Empirikos.TruncatedSample.(Zs, Ref(trunc_set))

dkw = DvoretzkyKieferWolfowitz(α = 0.05)
fitted_dkw = fit(dkw, Zs_trunc)
quiet_Gurobi = optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => 0)
quiet_Mosek = optimizer_with_attributes(Mosek.Optimizer,"QUIET" => true,"MSK_IPAR_INTPNT_MAX_ITERATIONS" => 1000   
)
floc_dkw = FLocalizationInterval(convexclass=tilted_scale_mix,
                                     flocalization=fitted_dkw,
                                     solver = quiet_Gurobi)
floc_zcurve_dkw = FLocalizationInterval(convexclass=tilted_z_curvemix,
                                     flocalization=fitted_dkw,
                                     solver = quiet_Gurobi)                                     
floc_unif_dkw = FLocalizationInterval(
    convexclass = tilted_u_scale_mix,
    flocalization = fitted_dkw,
    solver = quiet_Gurobi)    

floc_locationscale_dkw = FLocalizationInterval(
    convexclass = tilted_location_scale_mix,
    flocalization = fitted_dkw,
    solver = quiet_Gurobi)      

_grid = 0.0:0.1:8.0
marginal_density_pretilt_unnormalized = Empirikos.ExtendedMarginalDensity.(Empirikos.TruncatedSample.(FoldedNormalSample.(_grid, 1.0), Ref(trunc_set)))



ci_marginal_density_unnormalized = confint.(floc_dkw, marginal_density_pretilt_unnormalized)
ci_unif_marginal_density_unnormalized = confint.(floc_unif_dkw, marginal_density_pretilt_unnormalized)
ci_ls_marginal_density_unnormalized = confint.(floc_locationscale_dkw, marginal_density_pretilt_unnormalized)
ci_zcurve_marginal_density_unnormalized = confint.(floc_zcurve_dkw, marginal_density_pretilt_unnormalized)

@rlibrary zcurve
@rput abs_zs_init
R"""
library(zcurve) 
m.EM <- zcurve(abs_zs_init, method = "EM", parallel = TRUE)
x <- m.EM

zdist_lpdf = function(x, mu, sigma, a) {
  l1    = dnorm( x, mu, sigma, log = FALSE);
  l2    = dnorm(-x, mu, sigma, log = FALSE);

  l1_2  = pnorm(a, mu, sigma, lower.tail=TRUE, log.p=FALSE);

  l2_1  = pnorm(-a, mu, sigma, lower.tail=TRUE, log.p=FALSE);

  L = log(l1 + l2) - log(1-l1_2+l2_1);
  return(L);
}
x_seq <- seq(0, x$control$b, .1)
y_den_boot <- sapply(1:nrow(x$boot$mu),function(b){
      y_den <- sapply(1:length(x$boot$mu[b,]), function(i){
        x$boot$weights[b,i]*exp(zdist_lpdf(x_seq, x$boot$mu[b,i], 1, x$control$a))
      })
      y_den <- apply(y_den, 1, sum)
    })
z_curve_CI_l_with <- apply(y_den_boot, 1, stats::quantile, prob = .025)
z_curve_CI_u_with<- apply(y_den_boot, 1, stats::quantile, prob = .975)
"""
@rget z_curve_CI_l_with
@rget z_curve_CI_u_with
@rget x_seq


R"""
plot(x,CI = T)
"""
function zdist_lpdf(x::AbstractVector, mu::Real, sigma::Real, a::Real)
    d = Normal(mu, sigma)
    l1 = pdf.(d, x)
    l2 = pdf.(d, -x)
    
    l1_2 = cdf(d, a)
    l2_1 = cdf(d, -a)
    norm_factor = 1 - l1_2 + l2_1
    

    
    return log.(l1 .+ l2) .- log(norm_factor)
end


#simulation
marginal_density_pretilt_normalized_0 = Empirikos.ExtendedMarginalDensity.(Empirikos.TruncatedSample.(FoldedNormalSample.(0, 1.0), Ref(trunc_set)))
ci_marginal_density_normalized_0= Empirikos.fit(floc_zcurve_dkw, marginal_density_pretilt_normalized_0)
mixture_model = ci_marginal_density_normalized_0.g1
weight = mixture_model.prior.p
non_zero_indices = findall(w -> w > 1e-6, weight)
non_zero_components = components(mixture_model)[non_zero_indices]
non_zero_weight = weight[non_zero_indices]
μ_values = [
    1,
    2,
    4,
    5,
    6,
]
default(
    grid = false,
    background_color_legend = :transparent,
    foreground_color_legend = :transparent,
    thickness_scaling = 1.15,
    size = (420, 330),
    legend = :topright,
)
plt = plot(
    μ_values,
    non_zero_weight,
    seriestype=:sticks,
    xlabel = L"\shortstack{$\mu$ values}",
    ylabel = L"\shortstack{Probability}",
    legend=false,
    xticks=[0,1,2,3,4,5,6],
    ylims=(0, 1),
    xlims=(0, 7),
    frame = :box
)

savefig("mixture_weights.pdf")


G = DiscreteNonParametric(μ_values, non_zero_weight)
Random.seed!(1234)
function simulate_data(G; n=500, selection_threshold=1.96)
    μ_samples = zeros(n)
    z_samples = zeros(n)
    observed_mask = falses(n)
    
    for i in 1:n
        μ = rand(G)
        z = rand(Normal(μ, 1.0))
        
        μ_samples[i] = μ
        z_samples[i] = z
        
        observed_mask[i] = abs(z) >= selection_threshold
    end
    observed_z = z_samples[observed_mask]
    
    return (μ_samples, z_samples, observed_z)
end   

trunc_set = Interval(1.96, 6)
_grid = 0.0:0.1:6.0
true_density = Empirikos.pdf.(G, FoldedNormalSample.(0.0:0.1:6.0, 1.0))/Empirikos.pdf(G, FoldedNormalSample(trunc_set, 1.0))


coverage_rate_plot = plot(
     _grid, true_density;
    xticks = [0, 2, 4, 6, 8],
    label      = "True density",
    linecolor  = :black,
    linewidth  = 1.5,   
    xlabel     = L"|z|",
    ylabel     =L"f_G(z)/\PP[G]{Z \in \mathcal{S}}"
  )
savefig("true_density.png")


const MASTER_SEED = 1234
n_sims = 100
n_grid = length(_grid)
n_batch = 1
batch_size = 100
start_sim = (n_batch - 1) * batch_size + 1
end_sim = n_batch * batch_size

# Initialize arrays for this batch
ci_zcurve_batch = Array{Tuple{Float64,Float64}}(undef, batch_size, n_grid)
ci_amari_batch = Array{Tuple{Float64,Float64}}(undef, batch_size, n_grid)
ci_floc_batch = Array{Tuple{Float64,Float64}}(undef, batch_size, n_grid)

for sim in start_sim:end_sim
    println("Running simulation $sim")
    sim_seed = MASTER_SEED + sim
    Random.seed!(sim_seed)
    μ_samples, z_samples, observed_z = simulate_data(G, n=10000)
    abs_zs = abs.(observed_z)
    @rlibrary zcurve
    @rput abs_zs
    R"""
    library(zcurve) 
    ctrl <- list(max_iter = 20000)
    m.EM <- zcurve(abs_zs, method = "EM", parallel = TRUE, control=ctrl)
    x <- m.EM

   zdist_lpdf = function(x, mu, sigma, a, b) {
    l1    = dnorm( x, mu, sigma, log = FALSE);
    l2    = dnorm(-x, mu, sigma, log = FALSE);

    l1_2  = pnorm(a, mu, sigma, lower.tail=TRUE, log.p=FALSE);
    l1_1 =  pnorm(b, mu, sigma, lower.tail=TRUE, log.p=FALSE)

    l2_1  = pnorm(-a, mu, sigma, lower.tail=TRUE, log.p=FALSE);
    l2_2 = pnorm(-b, mu, sigma, lower.tail=TRUE, log.p=FALSE)
    L = log(l1 + l2) -  log(l1_1-l1_2+l2_1-l2_2);
    return(L);
}
    x_seq <- seq(0, x$control$b, .1)
    y_den_boot <- sapply(1:nrow(x$boot$mu),function(b){
        y_den <- sapply(1:length(x$boot$mu[b,]), function(i){
            x$boot$weights[b,i]*exp(zdist_lpdf(x_seq, x$boot$mu[b,i], 1, x$control$a, x$control$b))
      })
        y_den <- apply(y_den, 1, sum)
    })
    z_curve_CI_l_with <- apply(y_den_boot, 1, stats::quantile, prob = .025)
    z_curve_CI_u_with<- apply(y_den_boot, 1, stats::quantile, prob = .975)
    """
    @rget z_curve_CI_l_with
    @rget z_curve_CI_u_with
    abs_zs = abs_zs[in.( abs_zs, Ref(trunc_set))]
    Zs = FoldedNormalSample.(abs_zs, 1.0)
    Zs_trunc = Empirikos.TruncatedSample.(Zs, Ref(trunc_set))
    dkw_amari = DvoretzkyKieferWolfowitz(α = 0.005)
    fitted_dkw = fit(dkw_amari, Zs_trunc)
    discr = RealLineDiscretizer{:closed,:open}(2.01:0.05:5.96)
    
    amari = Empirikos.AMARI(convexclass=tilted_z_curvemix,
                                     flocalization=fitted_dkw,
                                     discretizer=discr,
                                     solver = quiet_Mosek)
    marginal_density_pretilt_unnormalized = Empirikos.ExtendedMarginalDensity.(Empirikos.TruncatedSample.(FoldedNormalSample.(_grid, 1.0), Ref(trunc_set)))
    ci_amari_sim= confint.(amari, marginal_density_pretilt_unnormalized, Zs_trunc)
    dkw = DvoretzkyKieferWolfowitz(α = 0.05)
    fitted_dkw = fit(dkw, Zs_trunc)
    floc_zcurve_dkw = FLocalizationInterval(convexclass=tilted_z_curvemix,
                                     flocalization=fitted_dkw,
                                     solver = quiet_Gurobi)     
    ci_floc_sim = confint.(floc_zcurve_dkw, marginal_density_pretilt_unnormalized)
    # Store in batch arrays
    batch_idx = sim - start_sim + 1
    for i in 1:n_grid
        ci_zcurve_batch[batch_idx, i] = (z_curve_CI_l_with[i], z_curve_CI_u_with[i])
        ci_amari_batch[batch_idx, i] = (ci_amari_sim[i].lower, ci_amari_sim[i].upper)
        ci_floc_batch[batch_idx, i] = (ci_floc_sim[i].lower, ci_floc_sim[i].upper)
    end
end
lower_zcurve = [ci[1] for ci in ci_zcurve_batch]
upper_zcurve = [ci[2] for ci in ci_zcurve_batch]

lower_amari = [ci[1] for ci in ci_amari_batch]
upper_amari = [ci[2] for ci in ci_amari_batch]

lower_floc = [ci[1] for ci in ci_floc_batch]
upper_floc = [ci[2] for ci in ci_floc_batch]
lower_zcurve_M = getindex.(ci_zcurve_batch, 1)
upper_zcurve_M = getindex.(ci_zcurve_batch, 2)

lower_amari_M  = getindex.(ci_amari_batch, 1)
upper_amari_M  = getindex.(ci_amari_batch, 2)

lower_floc_M   = getindex.(ci_floc_batch, 1)
upper_floc_M   = getindex.(ci_floc_batch, 2)

lower_zcurve = reshape(lower_zcurve_M', :)
upper_zcurve = reshape(upper_zcurve_M', :)

lower_amari  = reshape(lower_amari_M',  :)
upper_amari  = reshape(upper_amari_M',  :)

lower_floc   = reshape(lower_floc_M',   :)
upper_floc   = reshape(upper_floc_M',   :)
zcurve_df = DataFrame(
    sim = repeat(start_sim:end_sim, inner=n_grid),
    grid_index = repeat(1:n_grid, outer=batch_size),
    lower = vec(lower_zcurve),
    upper = vec(upper_zcurve)
)

amari_df = DataFrame(
    sim = repeat(start_sim:end_sim, inner=n_grid),
    grid_index = repeat(1:n_grid, outer=batch_size),
    lower = vec(lower_amari),
    upper = vec(upper_amari)
)

floc_df = DataFrame(
    sim = repeat(start_sim:end_sim, inner=n_grid),
    grid_index = repeat(1:n_grid, outer=batch_size),
    lower = vec(lower_floc),
    upper = vec(upper_floc)
)


CSV.write("batch_1_zcurve_cis.csv", zcurve_df)
CSV.write("batch_1_amari_cis.csv", amari_df)
CSV.write("batch_1_floc_cis.csv", floc_df)




n_batch = 2
batch_size = 100
start_sim = (n_batch - 1) * batch_size + 1
end_sim = n_batch * batch_size

ci_zcurve_batch_2 = Array{Tuple{Float64,Float64}}(undef, batch_size, n_grid)
ci_amari_batch_2 = Array{Tuple{Float64,Float64}}(undef, batch_size, n_grid)
ci_floc_batch_2 = Array{Tuple{Float64,Float64}}(undef, batch_size, n_grid)

for sim in start_sim:end_sim
    println("Running simulation $sim")
    sim_seed = MASTER_SEED + sim
    Random.seed!(sim_seed)
    μ_samples, z_samples, observed_z = simulate_data(G, n=10000)
    abs_zs = abs.(observed_z)
    @rlibrary zcurve
    @rput abs_zs
    R"""
    library(zcurve) 
    ctrl <- list(max_iter = 20000)
    m.EM <- zcurve(abs_zs, method = "EM", parallel = TRUE, control=ctrl)
    x <- m.EM

   zdist_lpdf = function(x, mu, sigma, a, b) {
    l1    = dnorm( x, mu, sigma, log = FALSE);
    l2    = dnorm(-x, mu, sigma, log = FALSE);

    l1_2  = pnorm(a, mu, sigma, lower.tail=TRUE, log.p=FALSE);
    l1_1 =  pnorm(b, mu, sigma, lower.tail=TRUE, log.p=FALSE)

    l2_1  = pnorm(-a, mu, sigma, lower.tail=TRUE, log.p=FALSE);
    l2_2 = pnorm(-b, mu, sigma, lower.tail=TRUE, log.p=FALSE)
    L = log(l1 + l2) -  log(l1_1-l1_2+l2_1-l2_2);
    return(L);
}
    x_seq <- seq(0, x$control$b, .1)
    y_den_boot <- sapply(1:nrow(x$boot$mu),function(b){
        y_den <- sapply(1:length(x$boot$mu[b,]), function(i){
            x$boot$weights[b,i]*exp(zdist_lpdf(x_seq, x$boot$mu[b,i], 1, x$control$a, x$control$b))
      })
        y_den <- apply(y_den, 1, sum)
    })
    z_curve_CI_l_with <- apply(y_den_boot, 1, stats::quantile, prob = .025)
    z_curve_CI_u_with<- apply(y_den_boot, 1, stats::quantile, prob = .975)
    """
    @rget z_curve_CI_l_with
    @rget z_curve_CI_u_with
    abs_zs = abs_zs[in.( abs_zs, Ref(trunc_set))]
    Zs = FoldedNormalSample.(abs_zs, 1.0)
    Zs_trunc = Empirikos.TruncatedSample.(Zs, Ref(trunc_set))
    dkw_amari = DvoretzkyKieferWolfowitz(α = 0.005)
    fitted_dkw = fit(dkw_amari, Zs_trunc)
    discr = RealLineDiscretizer{:closed,:open}(2.01:0.05:5.96)
    
    amari = Empirikos.AMARI(convexclass=tilted_z_curvemix,
                                     flocalization=fitted_dkw,
                                     discretizer=discr,
                                     solver = quiet_Mosek)
    marginal_density_pretilt_unnormalized = Empirikos.ExtendedMarginalDensity.(Empirikos.TruncatedSample.(FoldedNormalSample.(_grid, 1.0), Ref(trunc_set)))
    ci_amari_sim= confint.(amari, marginal_density_pretilt_unnormalized, Zs_trunc)
    dkw = DvoretzkyKieferWolfowitz(α = 0.05)
    fitted_dkw = fit(dkw, Zs_trunc)
    floc_zcurve_dkw = FLocalizationInterval(convexclass=tilted_z_curvemix,
                                     flocalization=fitted_dkw,
                                     solver = quiet_Gurobi)     
    ci_floc_sim = confint.(floc_zcurve_dkw, marginal_density_pretilt_unnormalized)
    # Store in batch arrays
    batch_idx = sim - start_sim + 1
    for i in 1:n_grid
        ci_zcurve_batch_2[batch_idx, i] = (z_curve_CI_l_with[i], z_curve_CI_u_with[i])
        ci_amari_batch_2[batch_idx, i] = (ci_amari_sim[i].lower, ci_amari_sim[i].upper)
        ci_floc_batch_2[batch_idx, i] = (ci_floc_sim[i].lower, ci_floc_sim[i].upper)
    end
end
lower_zcurve_2 = [ci[1] for ci in ci_zcurve_batch_2]
upper_zcurve_2 = [ci[2] for ci in ci_zcurve_batch_2]

lower_amari_2 = [ci[1] for ci in ci_amari_batch_2]
upper_amari_2 = [ci[2] for ci in ci_amari_batch_2]

lower_floc_2 = [ci[1] for ci in ci_floc_batch_2]
upper_floc_2 = [ci[2] for ci in ci_floc_batch_2]
lower_zcurve_M = getindex.(ci_zcurve_batch_2, 1)
upper_zcurve_M = getindex.(ci_zcurve_batch_2, 2)

lower_amari_M  = getindex.(ci_amari_batch_2, 1)
upper_amari_M  = getindex.(ci_amari_batch_2, 2)

lower_floc_M   = getindex.(ci_floc_batch_2, 1)
upper_floc_M   = getindex.(ci_floc_batch_2, 2)

lower_zcurve_2 = reshape(lower_zcurve_M', :)
upper_zcurve_2 = reshape(upper_zcurve_M', :)

lower_amari_2  = reshape(lower_amari_M',  :)
upper_amari_2  = reshape(upper_amari_M',  :)

lower_floc_2   = reshape(lower_floc_M',   :)
upper_floc_2   = reshape(upper_floc_M',   :)
zcurve_df = DataFrame(
    sim = repeat(start_sim:end_sim, inner=n_grid),
    grid_index = repeat(1:n_grid, outer=batch_size),
    lower = vec(lower_zcurve_2),
    upper = vec(upper_zcurve_2)
)

amari_df = DataFrame(
    sim = repeat(start_sim:end_sim, inner=n_grid),
    grid_index = repeat(1:n_grid, outer=batch_size),
    lower = vec(lower_amari_2),
    upper = vec(upper_amari_2)
)

floc_df = DataFrame(
    sim = repeat(start_sim:end_sim, inner=n_grid),
    grid_index = repeat(1:n_grid, outer=batch_size),
    lower = vec(lower_floc_2),
    upper = vec(upper_floc_2)
)
CSV.write("batch_2_zcurve_cis.csv", zcurve_df)
CSV.write("batch_2_amari_cis.csv", amari_df)
CSV.write("batch_2_floc_cis.csv", floc_df)


n_batch = 3
batch_size = 100
start_sim = (n_batch - 1) * batch_size + 1
end_sim = n_batch * batch_size

ci_zcurve_batch_3= Array{Tuple{Float64,Float64}}(undef, batch_size, n_grid)
ci_amari_batch_3 = Array{Tuple{Float64,Float64}}(undef, batch_size, n_grid)
ci_floc_batch_3 = Array{Tuple{Float64,Float64}}(undef, batch_size, n_grid)

for sim in start_sim:end_sim
    println("Running simulation $sim")
    sim_seed = MASTER_SEED + sim
    Random.seed!(sim_seed)
    μ_samples, z_samples, observed_z = simulate_data(G, n=10000)
    abs_zs = abs.(observed_z)
    @rlibrary zcurve
    @rput abs_zs
    R"""
    library(zcurve) 
    ctrl <- list(max_iter = 20000)
    m.EM <- zcurve(abs_zs, method = "EM", parallel = TRUE, control=ctrl)
    x <- m.EM

   zdist_lpdf = function(x, mu, sigma, a, b) {
    l1    = dnorm( x, mu, sigma, log = FALSE);
    l2    = dnorm(-x, mu, sigma, log = FALSE);

    l1_2  = pnorm(a, mu, sigma, lower.tail=TRUE, log.p=FALSE);
    l1_1 =  pnorm(b, mu, sigma, lower.tail=TRUE, log.p=FALSE)

    l2_1  = pnorm(-a, mu, sigma, lower.tail=TRUE, log.p=FALSE);
    l2_2 = pnorm(-b, mu, sigma, lower.tail=TRUE, log.p=FALSE)
    L = log(l1 + l2) -  log(l1_1-l1_2+l2_1-l2_2);
    return(L);
}
    x_seq <- seq(0, x$control$b, .1)
    y_den_boot <- sapply(1:nrow(x$boot$mu),function(b){
        y_den <- sapply(1:length(x$boot$mu[b,]), function(i){
            x$boot$weights[b,i]*exp(zdist_lpdf(x_seq, x$boot$mu[b,i], 1, x$control$a, x$control$b))
      })
        y_den <- apply(y_den, 1, sum)
    })
    z_curve_CI_l_with <- apply(y_den_boot, 1, stats::quantile, prob = .025)
    z_curve_CI_u_with<- apply(y_den_boot, 1, stats::quantile, prob = .975)
    """
    @rget z_curve_CI_l_with
    @rget z_curve_CI_u_with
    abs_zs = abs_zs[in.( abs_zs, Ref(trunc_set))]
    Zs = FoldedNormalSample.(abs_zs, 1.0)
    Zs_trunc = Empirikos.TruncatedSample.(Zs, Ref(trunc_set))
    dkw_amari = DvoretzkyKieferWolfowitz(α = 0.005)
    fitted_dkw = fit(dkw_amari, Zs_trunc)
    discr = RealLineDiscretizer{:closed,:open}(2.01:0.05:5.96)
    
    amari = Empirikos.AMARI(convexclass=tilted_z_curvemix,
                                     flocalization=fitted_dkw,
                                     discretizer=discr,
                                     solver = quiet_Mosek)
    marginal_density_pretilt_unnormalized = Empirikos.ExtendedMarginalDensity.(Empirikos.TruncatedSample.(FoldedNormalSample.(_grid, 1.0), Ref(trunc_set)))
    ci_amari_sim= confint.(amari, marginal_density_pretilt_unnormalized, Zs_trunc)
    dkw = DvoretzkyKieferWolfowitz(α = 0.05)
    fitted_dkw = fit(dkw, Zs_trunc)
    floc_zcurve_dkw = FLocalizationInterval(convexclass=tilted_z_curvemix,
                                     flocalization=fitted_dkw,
                                     solver = quiet_Gurobi)     
    ci_floc_sim = confint.(floc_zcurve_dkw, marginal_density_pretilt_unnormalized)
    # Store in batch arrays
    batch_idx = sim - start_sim + 1
    for i in 1:n_grid
        ci_zcurve_batch_3[batch_idx, i] = (z_curve_CI_l_with[i], z_curve_CI_u_with[i])
        ci_amari_batch_3[batch_idx, i] = (ci_amari_sim[i].lower, ci_amari_sim[i].upper)
        ci_floc_batch_3[batch_idx, i] = (ci_floc_sim[i].lower, ci_floc_sim[i].upper)
    end
end
lower_zcurve_3 = [ci[1] for ci in ci_zcurve_batch_3]
upper_zcurve_3 = [ci[2] for ci in ci_zcurve_batch_3]

lower_amari_3 = [ci[1] for ci in ci_amari_batch_3]
upper_amari_3 = [ci[2] for ci in ci_amari_batch_3]

lower_floc_3 = [ci[1] for ci in ci_floc_batch_3]
upper_floc_3 = [ci[2] for ci in ci_floc_batch_3]

lower_zcurve_M = getindex.(ci_zcurve_batch_3, 1)
upper_zcurve_M = getindex.(ci_zcurve_batch_3, 2)

lower_amari_M  = getindex.(ci_amari_batch_3, 1)
upper_amari_M  = getindex.(ci_amari_batch_3, 2)

lower_floc_M   = getindex.(ci_floc_batch_3, 1)
upper_floc_M   = getindex.(ci_floc_batch_3, 2)

lower_zcurve_3 = reshape(lower_zcurve_M', :)
upper_zcurve_3 = reshape(upper_zcurve_M', :)

lower_amari_3  = reshape(lower_amari_M',  :)
upper_amari_3  = reshape(upper_amari_M',  :)

lower_floc_3   = reshape(lower_floc_M',   :)
upper_floc_3   = reshape(upper_floc_M',   :)
zcurve_df = DataFrame(
    sim = repeat(start_sim:end_sim, inner=n_grid),
    grid_index = repeat(1:n_grid, outer=batch_size),
    lower = vec(lower_zcurve_3),
    upper = vec(upper_zcurve_3)
)

amari_df = DataFrame(
    sim = repeat(start_sim:end_sim, inner=n_grid),
    grid_index = repeat(1:n_grid, outer=batch_size),
    lower = vec(lower_amari_3),
    upper = vec(upper_amari_3)
)

floc_df = DataFrame(
    sim = repeat(start_sim:end_sim, inner=n_grid),
    grid_index = repeat(1:n_grid, outer=batch_size),
    lower = vec(lower_floc_3),
    upper = vec(upper_floc_3)
)
CSV.write("batch_3_zcurve_cis.csv", zcurve_df)
CSV.write("batch_3_amari_cis.csv", amari_df)
CSV.write("batch_3_floc_cis.csv", floc_df)


n_batch = 4
batch_size = 100
start_sim = (n_batch - 1) * batch_size + 1
end_sim = n_batch * batch_size

ci_zcurve_batch_4 = Array{Tuple{Float64,Float64}}(undef, batch_size, n_grid)
ci_amari_batch_4 = Array{Tuple{Float64,Float64}}(undef, batch_size, n_grid)
ci_floc_batch_4 = Array{Tuple{Float64,Float64}}(undef, batch_size, n_grid)

for sim in start_sim:end_sim
    println("Running simulation $sim")
    sim_seed = MASTER_SEED + sim
    Random.seed!(sim_seed)
    μ_samples, z_samples, observed_z = simulate_data(G, n=10000)
    abs_zs = abs.(observed_z)
    @rlibrary zcurve
    @rput abs_zs
    R"""
    library(zcurve) 
    ctrl <- list(max_iter = 20000)
    m.EM <- zcurve(abs_zs, method = "EM", parallel = TRUE, control=ctrl)
    x <- m.EM

   zdist_lpdf = function(x, mu, sigma, a, b) {
    l1    = dnorm( x, mu, sigma, log = FALSE);
    l2    = dnorm(-x, mu, sigma, log = FALSE);

    l1_2  = pnorm(a, mu, sigma, lower.tail=TRUE, log.p=FALSE);
    l1_1 =  pnorm(b, mu, sigma, lower.tail=TRUE, log.p=FALSE)

    l2_1  = pnorm(-a, mu, sigma, lower.tail=TRUE, log.p=FALSE);
    l2_2 = pnorm(-b, mu, sigma, lower.tail=TRUE, log.p=FALSE)
    L = log(l1 + l2) -  log(l1_1-l1_2+l2_1-l2_2);
    return(L);
}
    x_seq <- seq(0, x$control$b, .1)
    y_den_boot <- sapply(1:nrow(x$boot$mu),function(b){
        y_den <- sapply(1:length(x$boot$mu[b,]), function(i){
            x$boot$weights[b,i]*exp(zdist_lpdf(x_seq, x$boot$mu[b,i], 1, x$control$a, x$control$b))
      })
        y_den <- apply(y_den, 1, sum)
    })
    z_curve_CI_l_with <- apply(y_den_boot, 1, stats::quantile, prob = .025)
    z_curve_CI_u_with<- apply(y_den_boot, 1, stats::quantile, prob = .975)
    """
    @rget z_curve_CI_l_with
    @rget z_curve_CI_u_with
    abs_zs = abs_zs[in.( abs_zs, Ref(trunc_set))]
    Zs = FoldedNormalSample.(abs_zs, 1.0)
    Zs_trunc = Empirikos.TruncatedSample.(Zs, Ref(trunc_set))
    dkw_amari = DvoretzkyKieferWolfowitz(α = 0.005)
    fitted_dkw = fit(dkw_amari, Zs_trunc)
    discr = RealLineDiscretizer{:closed,:open}(2.01:0.05:5.96)
    
    amari = Empirikos.AMARI(convexclass=tilted_z_curvemix,
                                     flocalization=fitted_dkw,
                                     discretizer=discr,
                                     solver = quiet_Mosek)
    marginal_density_pretilt_unnormalized = Empirikos.ExtendedMarginalDensity.(Empirikos.TruncatedSample.(FoldedNormalSample.(_grid, 1.0), Ref(trunc_set)))
    ci_amari_sim= confint.(amari, marginal_density_pretilt_unnormalized, Zs_trunc)
    dkw = DvoretzkyKieferWolfowitz(α = 0.05)
    fitted_dkw = fit(dkw, Zs_trunc)
    floc_zcurve_dkw = FLocalizationInterval(convexclass=tilted_z_curvemix,
                                     flocalization=fitted_dkw,
                                     solver = quiet_Gurobi)     
    ci_floc_sim = confint.(floc_zcurve_dkw, marginal_density_pretilt_unnormalized)
    # Store in batch arrays
    batch_idx = sim - start_sim + 1
    for i in 1:n_grid
        ci_zcurve_batch_4[batch_idx, i] = (z_curve_CI_l_with[i], z_curve_CI_u_with[i])
        ci_amari_batch_4[batch_idx, i] = (ci_amari_sim[i].lower, ci_amari_sim[i].upper)
        ci_floc_batch_4[batch_idx, i] = (ci_floc_sim[i].lower, ci_floc_sim[i].upper)
    end
end
lower_zcurve_4 = [ci[1] for ci in ci_zcurve_batch_4]
upper_zcurve_4 = [ci[2] for ci in ci_zcurve_batch_4]

lower_amari_4 = [ci[1] for ci in ci_amari_batch_4]
upper_amari_4 = [ci[2] for ci in ci_amari_batch_4]

lower_floc_4 = [ci[1] for ci in ci_floc_batch_4]
upper_floc_4 = [ci[2] for ci in ci_floc_batch_4]

lower_zcurve_M = getindex.(ci_zcurve_batch_4, 1)
upper_zcurve_M = getindex.(ci_zcurve_batch_4, 2)

lower_amari_M  = getindex.(ci_amari_batch_4, 1)
upper_amari_M  = getindex.(ci_amari_batch_4, 2)

lower_floc_M   = getindex.(ci_floc_batch_4, 1)
upper_floc_M   = getindex.(ci_floc_batch_4, 2)

lower_zcurve_4 = reshape(lower_zcurve_M', :)
upper_zcurve_4 = reshape(upper_zcurve_M', :)

lower_amari_4  = reshape(lower_amari_M',  :)
upper_amari_4  = reshape(upper_amari_M',  :)

lower_floc_4   = reshape(lower_floc_M',   :)
upper_floc_4   = reshape(upper_floc_M',   :)

zcurve_df = DataFrame(
    sim = repeat(start_sim:end_sim, inner=n_grid),
    grid_index = repeat(1:n_grid, outer=batch_size),
    lower = vec(lower_zcurve_4),
    upper = vec(upper_zcurve_4)
)

amari_df = DataFrame(
    sim = repeat(start_sim:end_sim, inner=n_grid),
    grid_index = repeat(1:n_grid, outer=batch_size),
    lower = vec(lower_amari_4),
    upper = vec(upper_amari_4)
)

floc_df = DataFrame(
    sim = repeat(start_sim:end_sim, inner=n_grid),
    grid_index = repeat(1:n_grid, outer=batch_size),
    lower = vec(lower_floc_4),
    upper = vec(upper_floc_4)
)
CSV.write("batch_4_zcurve_cis.csv", zcurve_df)
CSV.write("batch_4_amari_cis.csv", amari_df)
CSV.write("batch_4_floc_cis.csv", floc_df)


n_batch = 5
batch_size = 100
start_sim = (n_batch - 1) * batch_size + 1
end_sim = n_batch * batch_size

ci_zcurve_batch_5 = Array{Tuple{Float64,Float64}}(undef, batch_size, n_grid)
ci_amari_batch_5 = Array{Tuple{Float64,Float64}}(undef, batch_size, n_grid)
ci_floc_batch_5 = Array{Tuple{Float64,Float64}}(undef, batch_size, n_grid)

for sim in start_sim:end_sim
    println("Running simulation $sim")
    sim_seed = MASTER_SEED + sim
    Random.seed!(sim_seed)
    μ_samples, z_samples, observed_z = simulate_data(G, n=10000)
    abs_zs = abs.(observed_z)
    @rlibrary zcurve
    @rput abs_zs
    R"""
    library(zcurve) 
    ctrl <- list(max_iter = 20000)
    m.EM <- zcurve(abs_zs, method = "EM", parallel = TRUE, control=ctrl)
    x <- m.EM

   zdist_lpdf = function(x, mu, sigma, a, b) {
    l1    = dnorm( x, mu, sigma, log = FALSE);
    l2    = dnorm(-x, mu, sigma, log = FALSE);

    l1_2  = pnorm(a, mu, sigma, lower.tail=TRUE, log.p=FALSE);
    l1_1 =  pnorm(b, mu, sigma, lower.tail=TRUE, log.p=FALSE)

    l2_1  = pnorm(-a, mu, sigma, lower.tail=TRUE, log.p=FALSE);
    l2_2 = pnorm(-b, mu, sigma, lower.tail=TRUE, log.p=FALSE)
    L = log(l1 + l2) -  log(l1_1-l1_2+l2_1-l2_2);
    return(L);
}
    x_seq <- seq(0, x$control$b, .1)
    y_den_boot <- sapply(1:nrow(x$boot$mu),function(b){
        y_den <- sapply(1:length(x$boot$mu[b,]), function(i){
            x$boot$weights[b,i]*exp(zdist_lpdf(x_seq, x$boot$mu[b,i], 1, x$control$a, x$control$b))
      })
        y_den <- apply(y_den, 1, sum)
    })
    z_curve_CI_l_with <- apply(y_den_boot, 1, stats::quantile, prob = .025)
    z_curve_CI_u_with<- apply(y_den_boot, 1, stats::quantile, prob = .975)
    """
    @rget z_curve_CI_l_with
    @rget z_curve_CI_u_with
    abs_zs = abs_zs[in.( abs_zs, Ref(trunc_set))]
    Zs = FoldedNormalSample.(abs_zs, 1.0)
    Zs_trunc = Empirikos.TruncatedSample.(Zs, Ref(trunc_set))
    dkw_amari = DvoretzkyKieferWolfowitz(α = 0.005)
    fitted_dkw = fit(dkw_amari, Zs_trunc)
    discr = RealLineDiscretizer{:closed,:open}(2.01:0.05:5.96)
    amari = Empirikos.AMARI(convexclass=tilted_z_curvemix,
                                     flocalization=fitted_dkw,
                                     discretizer=discr,
                                     solver = quiet_Mosek)
    marginal_density_pretilt_unnormalized = Empirikos.ExtendedMarginalDensity.(Empirikos.TruncatedSample.(FoldedNormalSample.(_grid, 1.0), Ref(trunc_set)))
    ci_amari_sim= confint.(amari, marginal_density_pretilt_unnormalized, Zs_trunc)
    dkw = DvoretzkyKieferWolfowitz(α = 0.05)
    fitted_dkw = fit(dkw, Zs_trunc)
    floc_zcurve_dkw = FLocalizationInterval(convexclass=tilted_z_curvemix,
                                     flocalization=fitted_dkw,
                                     solver = quiet_Gurobi)     
    ci_floc_sim = confint.(floc_zcurve_dkw, marginal_density_pretilt_unnormalized)
    # Store in batch arrays
    batch_idx = sim - start_sim + 1
    for i in 1:n_grid
        ci_zcurve_batch_5[batch_idx, i] = (z_curve_CI_l_with[i], z_curve_CI_u_with[i])
        ci_amari_batch_5[batch_idx, i] = (ci_amari_sim[i].lower, ci_amari_sim[i].upper)
        ci_floc_batch_5[batch_idx, i] = (ci_floc_sim[i].lower, ci_floc_sim[i].upper)
    end
end
lower_zcurve_5 = [ci[1] for ci in ci_zcurve_batch_5]
upper_zcurve_5 = [ci[2] for ci in ci_zcurve_batch_5]

lower_amari_5 = [ci[1] for ci in ci_amari_batch_5]
upper_amari_5 = [ci[2] for ci in ci_amari_batch_5]

lower_floc_5 = [ci[1] for ci in ci_floc_batch_5]
upper_floc_5 = [ci[2] for ci in ci_floc_batch_5]

lower_zcurve_M = getindex.(ci_zcurve_batch_5, 1)
upper_zcurve_M = getindex.(ci_zcurve_batch_5, 2)

lower_amari_M  = getindex.(ci_amari_batch_5, 1)
upper_amari_M  = getindex.(ci_amari_batch_5, 2)

lower_floc_M   = getindex.(ci_floc_batch_5, 1)
upper_floc_M   = getindex.(ci_floc_batch_5, 2)

lower_zcurve_5 = reshape(lower_zcurve_M', :)
upper_zcurve_5 = reshape(upper_zcurve_M', :)

lower_amari_5  = reshape(lower_amari_M',  :)
upper_amari_5  = reshape(upper_amari_M',  :)

lower_floc_5   = reshape(lower_floc_M',   :)
upper_floc_5   = reshape(upper_floc_M',   :)


zcurve_df = DataFrame(
    sim = repeat(start_sim:end_sim, inner=n_grid),
    grid_index = repeat(1:n_grid, outer=batch_size),
    lower = vec(lower_zcurve_5),
    upper = vec(upper_zcurve_5)
)

amari_df = DataFrame(
    sim = repeat(start_sim:end_sim, inner=n_grid),
    grid_index = repeat(1:n_grid, outer=batch_size),
    lower = vec(lower_amari_5),
    upper = vec(upper_amari_5)
)

floc_df = DataFrame(
    sim = repeat(start_sim:end_sim, inner=n_grid),
    grid_index = repeat(1:n_grid, outer=batch_size),
    lower = vec(lower_floc_5),
    upper = vec(upper_floc_5)
)
CSV.write("batch_5_zcurve_cis.csv", zcurve_df)
CSV.write("batch_5_amari_cis.csv", amari_df)
CSV.write("batch_5_floc_cis.csv", floc_df)


function df_to_array(df::DataFrame, n_grid::Int)
    sort!(df, [:sim, :grid_index])

    ci = [(df.lower[i], df.upper[i]) for i in 1:nrow(df)]


    permutedims(reshape(ci, n_grid, :), (2, 1))
end

ci_zcurve_all = Array{Tuple{Float64,Float64}}(undef, 0, n_grid)
ci_amari_all  = Array{Tuple{Float64,Float64}}(undef, 0, n_grid)
ci_floc_all   = Array{Tuple{Float64,Float64}}(undef, 0, n_grid)

for batch in 1:5
    zcurve_batch = CSV.read("batch_$(batch)_zcurve_cis.csv", DataFrame)
    amari_batch  = CSV.read("batch_$(batch)_amari_cis.csv",  DataFrame)
    floc_batch   = CSV.read("batch_$(batch)_floc_cis.csv",   DataFrame)

    zc = df_to_array(zcurve_batch, n_grid)
    am = df_to_array(amari_batch,  n_grid)
    fl = df_to_array(floc_batch,   n_grid)

    ci_zcurve_all = vcat(ci_zcurve_all, zc)
    ci_amari_all  = vcat(ci_amari_all,  am)
    ci_floc_all   = vcat(ci_floc_all,   fl)
end

covered_per_point_zcurve = zeros(n_grid)
covered_per_point_amari = zeros(n_grid)
covered_per_point_floc = zeros(n_grid)



for sim in 1:500
    zcurve_lower = [ci_zcurve_all[sim, i][1] for i in 1:n_grid]
    zcurve_upper = [ci_zcurve_all[sim, i][2] for i in 1:n_grid]
    
    amari_lower = [ci_amari_all[sim, i][1] for i in 1:n_grid]
    amari_upper = [ci_amari_all[sim, i][2] for i in 1:n_grid]
    
    floc_lower = [ci_floc_all[sim, i][1] for i in 1:n_grid]
    floc_upper = [ci_floc_all[sim, i][2] for i in 1:n_grid]
    
    for i in 1:n_grid
        true_val = true_density[i]
        if zcurve_lower[i] <= true_val <= zcurve_upper[i]
            covered_per_point_zcurve[i] += 1
        end
        if amari_lower[i] <= true_val <= amari_upper[i]
            covered_per_point_amari[i] += 1
        end
        if floc_lower[i] <= true_val <= floc_upper[i]
            covered_per_point_floc[i] += 1
        end
    end
end

pointwise_coverage_zcurve = covered_per_point_zcurve ./ 500
pointwise_coverage_amari = covered_per_point_amari ./ 500
pointwise_coverage_floc = covered_per_point_floc ./ 500


plt = plot(
    μ_values,
    non_zero_weight,
    seriestype=:sticks,
    xlabel = L"\shortstack{$\mu$ values}",
    ylabel = L"\shortstack{Probability}",
    legend=false,
    xticks=[0,1,2,3,4,5,6],
    ylims=(0, 1),
    xlims=(0, 6),
    frame = :box
)
plt = plot(
    xlabel = L"|z|",
    ylabel = L"\shortstack{Coverage Rate}",
    ylims = (0, 1.0),
    yticks = vcat(0:0.1:1.0, 0.95),
    legend = :bottomleft,
    frame = :box
)
plot!(_grid, pointwise_coverage_zcurve, 
      label = L"\shortstack{Z-Curve}")

plot!(_grid, pointwise_coverage_amari, 
      label = L"\shortstack{AMARI}", 
      color = :black)

plot!(_grid, pointwise_coverage_floc, 
      label = L"\shortstack{FLOC}", 
      color = :orange)

hline!([0.95], 
       label = "Target (95%)", 
       linestyle = :dash, 
       color = :red,
       linewidth = 2)
plot!(
    plt;
    legend_columns       = 1,
    legend_font_pointsize= 8,
    legend_position     = :bottomright,
  )
savefig("pointwise_coverage_zcurve_cochrane_final.pdf")
n_sims = 500
zcurve_avg_lower = [mean([ci_zcurve_all[sim, i][1] for sim in 1:n_sims]) for i in 1:n_grid]
zcurve_avg_upper = [mean([ci_zcurve_all[sim, i][2] for sim in 1:n_sims]) for i in 1:n_grid]
amari_avg_lower = [mean([ci_amari_all[sim, i][1] for sim in 1:n_sims]) for i in 1:n_grid]
amari_avg_upper = [mean([ci_amari_all[sim, i][2] for sim in 1:n_sims]) for i in 1:n_grid]
floc_avg_lower = [mean([ci_floc_all[sim, i][1] for sim in 1:n_sims]) for i in 1:n_grid]
floc_avg_upper = [mean([ci_floc_all[sim, i][2] for sim in 1:n_sims]) for i in 1:n_grid]


average_ci_plot = plot(
     _grid, true_density;
    xticks = [0, 2, 4, 6, 8],
    label      = "True density",
    linecolor  = :red,
    linewidth  = 1.5,   
    xlabel     = L"|z|",
    ylabel     =L"f_G(z)/\PP[G]{Z \in \mathcal{S}}",
    frame = :box
  )
  plot!(
    average_ci_plot;
    legend_columns       = 1,
    legend_font_pointsize= 8,
    legend_position     = :topright
  )
savefig("true_density.png")
average_ci_plot = begin
  average_ci_plot = plot(
     _grid, true_density;
    xticks = [0, 2, 4, 6, 8],
    label      = "True density",
    linecolor  = :red,
    linewidth  = 1.5,   
    xlabel     = L"|z|",
    ylabel     =L"f_G(z)/\PP[G]{Z \in \mathcal{S}}",
    frame = :box
  )
  
  plot!(
        average_ci_plot,
        _grid,
        zcurve_avg_lower,
        linecolor = :purple,
        linestyle = :dot,
        linewidth = 1.5,
        label = ""
    )
    
  plot!(
        average_ci_plot,
        _grid,
        zcurve_avg_upper,
        linecolor = :purple,
        linestyle = :dot,
        linewidth = 1.5,
        label = L"\shortstack{Z-Curve}"
    )
  plot!(
        average_ci_plot,
        _grid,
        floc_avg_lower,
        linecolor = :orange,
        linestyle = :solid,
        linewidth = 1.5,
        label = ""
    )
  plot!(
        average_ci_plot,
        _grid,
        floc_avg_upper,
        linecolor = :orange,
        linestyle = :solid,
        linewidth = 1.5,
        label = L"\shortstack{FLOC}"
    )

   plot!(
        average_ci_plot,
        _grid,
        amari_avg_lower,
        linecolor = :black,
        linestyle = :dot,
        linewidth = 1.5,
        label = ""
    )
  plot!(
        average_ci_plot,
        _grid,
        amari_avg_upper,
        linecolor = :black,
        linestyle = :dot,
        linewidth = 1.5,
        label = L"\shortstack{AMARI}"
    )
   plot!(
    average_ci_plot;
    legend_columns       = 1,
    legend_font_pointsize= 8,
    legend_position     = :topright
  )
end
savefig("average_ci_zcurve_cochrane_final.pdf")



marginal_cdf_196_pretilt = Empirikos.untilt(Empirikos.MarginalDensity(FoldedNormalSample(Interval(0.0, 1.96), 1.0)))

dkw_975= DvoretzkyKieferWolfowitz(α = 0.025)
fitted_dkw_975 = fit(dkw_975, Zs_trunc)

floc_dkw_975 = FLocalizationInterval(convexclass=tilted_scale_mix,
                                     flocalization=fitted_dkw_975 ,
                                     solver = quiet_Gurobi)
floc_unif_dkw_975 = FLocalizationInterval(
    convexclass = tilted_u_scale_mix,
    flocalization = fitted_dkw_975 ,
    solver = quiet_Gurobi)    

floc_locationscale_dkw_975 = FLocalizationInterval(
    convexclass = tilted_location_scale_mix,
    flocalization = fitted_dkw_975 ,
    solver = quiet_Gurobi)


dkw_amari = DvoretzkyKieferWolfowitz(α = 0.01)
fitted_dkw_amari = fit(dkw_amari, Zs_trunc)
discr = RealLineDiscretizer{:closed,:open}(2.15:0.05:7.96)
amari_975 = Empirikos.AMARI(convexclass= tilted_scale_mix,
                                     flocalization=fitted_dkw_amari,
                                     discretizer=discr,
                                     solver = quiet_Mosek)
ci_marginal_cdf_196_pretilt = confint(amari_975, marginal_cdf_196_pretilt, Zs_trunc; level = 0.975)

ci_marginal_cdf_196_pretilt = confint(floc_dkw_975, marginal_cdf_196_pretilt)

ci_frac_left = ci_marginal_cdf_196_pretilt.lower / (1-ci_marginal_cdf_196_pretilt.lower)
ci_frac_right = ci_marginal_cdf_196_pretilt.upper / (1-ci_marginal_cdf_196_pretilt.upper)

n_published = length(abs_zs_init)
sum(abs_zs_init .>= 1.96)
frac_published_above_196 = sum(abs_zs_init .>= 1.96) / n_published
_se = sqrt(frac_published_above_196 * (1-frac_published_above_196) / n_published)


frac_published_above_196_left = frac_published_above_196 - 2.24 * _se
frac_published_above_196_right = frac_published_above_196 + 2.24 * _se

frac_published_above_196_left / (1-frac_published_above_196_left)
frac_published_above_196_right / (1-frac_published_above_196_right)


n_published_above_196 = sum(abs_zs_init .>= 1.96)
n_published_below_196 = n_published - n_published_above_196
ω₁ =  n_published_above_196/n_published_below_196
ω_left =  frac_published_above_196_left / (1-frac_published_above_196_left) * ci_frac_left
ω_right =  frac_published_above_196_right / (1-frac_published_above_196_right) * ci_frac_right

amari_975 = Empirikos.AMARI(convexclass= tilted_u_scale_mix,
                                     flocalization=fitted_dkw_amari,
                                     discretizer=discr,
                                     solver = quiet_Mosek)
ci_unif_marginal_cdf_196_pretilt = confint(amari_975, marginal_cdf_196_pretilt, Zs_trunc; level = 0.975)

ci_unif_marginal_cdf_196_pretilt = confint(floc_unif_dkw_975, marginal_cdf_196_pretilt)
ci_unif_frac_left =ci_unif_marginal_cdf_196_pretilt.lower / (1-ci_unif_marginal_cdf_196_pretilt.lower)
ci_unif_frac_right = ci_unif_marginal_cdf_196_pretilt.upper / (1-ci_unif_marginal_cdf_196_pretilt.upper)

ω_left =  frac_published_above_196_left / (1-frac_published_above_196_left) * ci_unif_frac_left
ω_right =  frac_published_above_196_right / (1-frac_published_above_196_right) * ci_unif_frac_right

amari_975 = Empirikos.AMARI(convexclass= tilted_location_scale_mix,
                                     flocalization=fitted_dkw_amari,
                                     discretizer=discr,
                                     solver = quiet_Mosek)
ci_ls_marginal_cdf_196_pretilt = confint(amari_975, marginal_cdf_196_pretilt, Zs_trunc; level = 0.975)
ci_ls_marginal_cdf_196_pretilt = confint(floc_locationscale_dkw_975, marginal_cdf_196_pretilt)
ci_ls_frac_left =ci_ls_marginal_cdf_196_pretilt.lower / (1-ci_ls_marginal_cdf_196_pretilt.lower)
ci_ls_frac_right = ci_ls_marginal_cdf_196_pretilt.upper / (1-ci_ls_marginal_cdf_196_pretilt.upper)

ω_left =  frac_published_above_196_left / (1-frac_published_above_196_left) * ci_ls_frac_left
ω_right =  frac_published_above_196_right / (1-frac_published_above_196_right) * ci_ls_frac_right

marginal_density_unnormalized_plot = begin
  marginal_density_unnormalized_plot = histogram(
    abs_zs;
    normalize=true, xlim=(0,8), ylim=(0, 8),
    bins=2.0999:0.2:8,
    label="Histogram of truncated data",
    alpha=0.5, linewidth=0.25,
  )

  plot!(
    marginal_density_unnormalized_plot, _grid, ci_marginal_density_unnormalized;
    xticks = [0, 2, 4, 6, 8],
    label      = L"\shortstack{Gaussian scale ($\mathcal{G}^{\mathrm{sN}}$)}",
    fillcolor  = :blue,
    fillalpha  = 0.4,
    linewidth  = 0,   
    xlabel     = L"|z|",
    ylabel     =L"f_G(z)/\PP[G]{|Z| \in \mathcal{S}}"
  )
  plot!(
    marginal_density_unnormalized_plot, _grid, ci_unif_marginal_density_unnormalized;
    label      = L"\shortstack{Unimodal($\mathcal{G}^{\mathrm{unm}}$)}",
    show_ribbon= false,
    linestyle  = :solid,
    linecolor  = :orange,
    linewidth  = 1.5
  )
  plot!(
    marginal_density_unnormalized_plot, _grid, ci_ls_marginal_density_unnormalized;
    label      = L"\shortstack{All ($\mathcal{G}^{\mathrm{all}}$)}",
    show_ribbon= false,
    linestyle  = :dash,
    linecolor  = :black,
    linewidth  = 1.5
  )

  vline!(
    marginal_density_unnormalized_plot, [2.1];
    label      = "Truncation cutoff",
    linestyle  = :dash,
    linecolor  = :red,
    linewidth  = 1.0
  )
  plot!(
    marginal_density_unnormalized_plot;
    legend_columns       = 1,
    legend_font_pointsize= 7,
    legend_position     = :topright
  )
end
savefig("marginal_density_normalized_plot_cochrane.pdf")


marginal_density_pretilt = Empirikos.untilt.(Empirikos.MarginalDensity.(FoldedNormalSample.(_grid, 1.0)))
ci_marginal_density_normalized = confint.(floc_dkw, marginal_density_pretilt)
ci_unif_marginal_density_normalized = confint.(floc_unif_dkw, marginal_density_pretilt)
ci_ls_marginal_density_normalized = confint.(floc_locationscale_dkw, marginal_density_pretilt)

marginal_density_normalized_plot = begin
    marginal_density_normalized_plot = histogram(
    abs_zs_init[abs_zs_init .< 8];
    normalize=true, xlim=(0,8),
    bins=0:0.2:8,
    label="Histogram of full data",
    alpha=0.5, linewidth=0.25,
  )

    plot!(
        _grid,
        ci_marginal_density_normalized;
        xlabel       = L"|z|",
        ylabel       = L"f_G(z)",
        label        = L"\shortstack{Gaussian scale ($\mathcal{G}^{\mathrm{sN}}$)}",
        fillcolor    = :blue,
        fillalpha    = 0.4,
        linewidth    = 0,  
        legend       = :topright
    )
    plot!(
        marginal_density_normalized_plot,
        _grid,
        ci_unif_marginal_density_normalized;
        show_ribbon  = false,
        linecolor    = :orange,
        linestyle    = :solid,
        linewidth    = 1.5,
        label        = L"\shortstack{Unimodal ($\mathcal{G}^{\mathrm{unm}}$)}"
    )
    plot!(
        marginal_density_normalized_plot,
        _grid,
        ci_ls_marginal_density_normalized;
        show_ribbon  = false,
        linecolor    = :black,
        linestyle    = :dash,
        linewidth    = 1.5,
        label        =  L"\shortstack{All ($\mathcal{G}^{\mathrm{all}}$)}"
    )

    plot!(
        marginal_density_normalized_plot;
        legend_columns        = 1,
        legend_font_pointsize = 7,
        legend_position       = :topright
    )
end

savefig("marginal_density_unnormalized_plot_cochrane.pdf")

post_means = Empirikos.untilt.(PosteriorMean.(NormalSample.(_grid, 1.0)))
ci_post_means = confint.(floc_dkw, post_means)

ci_unif_post_means = confint.(floc_unif_dkw, post_means)
s_post_means = Empirikos.untilt.(Empirikos.SymmetricPosteriorMean.(NormalSample.(_grid, 1.0)))
ci_s_post_means_2 = confint.(floc_locationscale_dkw, s_post_means)

posterior_mean_plot = begin
    posterior_mean_plot = plot(
        _grid,xlim=(0,8),
        xticks = [0, 2, 4, 6, 8],
        yticks = 0:2:12,
        ci_post_means;
        xlabel    = L"z",
        ylabel    = L"\EE[\mathrm{Symm}(G)]{\mu \mid Z = z}",
        label     = L"\shortstack{Gaussian scale ($\mathcal{G}^{\mathrm{sN}}$)}",
        fillcolor = :blue,
        fillalpha = 0.4,
        linewidth = 0,        
        legend    = :topright
    )
    plot!(
        posterior_mean_plot,
        _grid,
        ci_unif_post_means;
        show_ribbon = false,
        linecolor   = :orange,
        linestyle   = :solid,
        linewidth   =1.5,
        label       =  L"\shortstack{Unimodal ($\mathcal{G}^{\mathrm{unm}}$)}"
    )

    plot!(
        posterior_mean_plot,
        _grid,
        ci_s_post_means_2;
        show_ribbon = false,
        linecolor   = :black,
        linestyle   = :dash,
        linewidth   = 1.5,
        label       = L"\shortstack{All ($\mathcal{G}^{\mathrm{all}}$)}"
    )

    plot!(
        posterior_mean_plot,
        [0, 10],
        [0, 10];
        linestyle   = :dot,
        linecolor   = :red,
        linewidth   = 1.5,
        label       = L"\shortstack{Identity ($\mu = z$)}"
    )
    plot!(
        posterior_mean_plot;
        legend_columns        = 1,
        legend_font_pointsize = 7,
        legend_position       = :topleft
    )
end

savefig("posterior_mean_plot_cochrane.pdf")


same_sign_prob = Empirikos.untilt.(Empirikos.SignAgreementProbability.(FoldedNormalSample.(_grid, 1.0)))
ci_same_sign_prob = confint.(floc_dkw, same_sign_prob)
ci_unif_same_sign_prob = confint.(floc_unif_dkw, same_sign_prob)
ci_s_same_sign_prob = confint.(floc_locationscale_dkw, same_sign_prob)


same_sign_prob_plot = begin
    same_sign_prob_plot = plot(
        _grid,xlim=(0,8),
        xticks = [0, 2, 4, 6, 8],
        ylim  = (0.5, 1),
        ci_same_sign_prob;
        xlabel    = L"z",
        ylabel    = L"\PP[G]{\mu \cdot Z > 0 \mid |Z|=z}",
        label     = L"\shortstack{Gaussian scale ($\mathcal{G}^{\mathrm{sN}}$)}",
        fillcolor = :blue,
        fillalpha = 0.4,
        linewidth = 0,        
        legend    = :topright
    )
    plot!(
        same_sign_prob_plot,
        _grid,
        ci_unif_same_sign_prob;
        show_ribbon = false,
        linecolor   = :orange,
        linestyle   = :solid,
        linewidth   =1.5,
        label       =  L"\shortstack{Unimodal ($\mathcal{G}^{\mathrm{unm}}$)}"
    )

    plot!(
        same_sign_prob_plot,
        _grid,
        ci_s_same_sign_prob;
        show_ribbon = false,
        linecolor   = :black,
        linestyle   = :dash,
        linewidth   = 1.5,
        label       = L"\shortstack{All ($\mathcal{G}^{\mathrm{all}}$)}"
    )
    plot!(
        same_sign_prob_plot;
        legend_columns        = 1,
        legend_font_pointsize = 7,
        legend_position       = :bottomright
    )
end

savefig("same_sign_prob_plot_cochrane.pdf")

coverage_prob = Empirikos.untilt.(Empirikos.CoverageProbability.(FoldedNormalSample.(_grid, 1.0)))
ci_coverage_prob = confint.(floc_dkw, coverage_prob)
ci_unif_coverage_prob = confint.(floc_unif_dkw, coverage_prob)
ci_s_coverage_prob = confint.(floc_locationscale_dkw, coverage_prob)


coverage_prob_plot = begin
    coverage_prob_plot = plot(
        _grid,xlim=(0,8),
        xticks = [0, 2, 4, 6, 8],
        ylim  = (0, 1), yticks = vcat(0:0.2:1, 0.95),
        ci_coverage_prob;
        xlabel    = L"z",
        ylabel    = L"\PP[G]{Z-1.96 < \mu < Z+1.96\mid |Z|=z}",
        label     = L"\shortstack{Gaussian scale ($\mathcal{G}^{\mathrm{sN}}$)}",
        fillcolor = :blue,
        fillalpha = 0.4,
        linewidth = 0,        
        legend    = :topright
    )
    plot!(
        coverage_prob_plot,
        _grid,
        ci_unif_coverage_prob;
        show_ribbon = false,
        linecolor   = :orange,
        linestyle   = :solid,
        linewidth   =1.5,
        label       =  L"\shortstack{Unimodal ($\mathcal{G}^{\mathrm{unm}}$)}"
    )

    plot!(
        coverage_prob_plot,
        _grid,
        ci_s_coverage_prob;
        show_ribbon = false,
        linecolor   = :black,
        linestyle   = :dash,
        linewidth   = 1.5,
        label       = L"\shortstack{All ($\mathcal{G}^{\mathrm{all}}$)}"
    )
    hline!(
    coverage_prob_plot, [0.95];
    label      = "Target coverage (95%)",
    linestyle  = :dash,
    linecolor  = :red,
    linewidth  = 1.0
   )
    plot!(
        coverage_prob_plot;
        legend_columns        = 1,
        legend_font_pointsize = 7,
        legend_position = (0.55, 0.50)
    )
end
savefig("coverage_prob_plot_cochrane.pdf")

zs_locs = sort(vcat(0.0:0.1:8, [1.644854; 1.959964; 2.575829; 3.290527; 3.890592]))
repl_prob = Empirikos.untilt.(Empirikos.ConditionalReplicationProbability.(FoldedNormalSample.(zs_locs, 1.0)))
ci_repl_prob = confint.(floc_dkw, repl_prob)
ci_unif_repl_prob = confint.(floc_unif_dkw, repl_prob)
ci_s_repl_prob = confint.(floc_locationscale_dkw, repl_prob)

repl_prob_plot = begin
    repl_prob_plot = plot(
        zs_locs,xlim=(0,8),
        xticks = [0, 2, 4, 6, 8],
        ylim  = (0, 1),
        ci_repl_prob ;
        xlabel    = L"z",
        ylabel    = L"\PP[G]{|Z'| > 1.96, ZZ'>0\mid |Z|=z}",
        label     = L"\shortstack{Gaussian scale ($\mathcal{G}^{\mathrm{sN}}$)}",
        fillcolor = :blue,
        fillalpha = 0.4,
        linewidth = 0,        
        legend    = :topright
    )
    plot!( 
        repl_prob_plot,
        zs_locs,
        ci_unif_repl_prob;
        show_ribbon = false,
        linecolor   = :orange,
        linestyle   = :solid,
        linewidth   =1.5,
        label       =  L"\shortstack{Unimodal ($\mathcal{G}^{\mathrm{unm}}$)}"
    )
    plot!(
        repl_prob_plot,
        zs_locs,
        ci_s_repl_prob;
        show_ribbon = false,
        linecolor   = :black,
        linestyle   = :dash,
        linewidth   = 1.5,
        label       = L"\shortstack{All ($\mathcal{G}^{\mathrm{all}}$)}"
    )

    plot!(
        repl_prob_plot;
        legend_columns        = 1,
        legend_font_pointsize = 7,
        legend_position       = :bottomright
    )
end
savefig("repl_prob_plot_cochrane.pdf")

future_coverage_prob = Empirikos.untilt.(Empirikos.FutureCoverageProbability.(FoldedNormalSample.(_grid, 1.0)))
ci_future_coverage_prob = confint.(floc_dkw, future_coverage_prob)
ci_unif_future_coverage_prob = confint.(floc_unif_dkw, future_coverage_prob)
ci_s_future_coverage_prob = confint.(floc_locationscale_dkw, future_coverage_prob)
future_coverage_prob_plot = begin
    future_coverage_prob_plot = plot(
        _grid,xlim=(0,8),
        xticks = [0, 2, 4, 6, 8],
        ylim  = (0, 1), yticks = (0:0.2:1),
        ci_future_coverage_prob;
        xlabel    = L"z",
        ylabel    = L"\PP[G]{ Z \in Z' \pm 1.96  \mid |Z|=z}",
        label     = L"\shortstack{Gaussian scale ($\mathcal{G}^{\mathrm{sN}}$)}",
        fillcolor = :blue,
        fillalpha = 0.4,
        linewidth = 0,        
        legend    = :topright
    )
    plot!(
        future_coverage_prob_plot,
        _grid,
        ci_unif_future_coverage_prob;
        show_ribbon = false,
        linecolor   = :orange,
        linestyle   = :solid,
        linewidth   =1.5,
        label       =  L"\shortstack{Unimodal ($\mathcal{G}^{\mathrm{unm}}$)}"
    )

    plot!(
        future_coverage_prob_plot,
        _grid,
        ci_s_future_coverage_prob;
        show_ribbon = false,
        linecolor   = :black,
        linestyle   = :dash,
        linewidth   = 1.5,
        label       = L"\shortstack{All ($\mathcal{G}^{\mathrm{all}}$)}"
    )
    plot!(
        future_coverage_prob_plot;
        legend_columns        = 1,
        legend_font_pointsize = 7,
        legend_position = (0.45, 0.30)
    )
end
savefig("future_coverage_prob_plot_cochrane.pdf")


replication_dominant_prob = Empirikos.untilt.(Empirikos.ReplicationDominantProbability.(FoldedNormalSample.(_grid, 1.0)))
ci_replication_dominant_prob = confint.(floc_dkw, replication_dominant_prob)
ci_unif_replication_dominant_prob = confint.(floc_unif_dkw, replication_dominant_prob)
ci_s_replication_dominant_prob = confint.(floc_locationscale_dkw, replication_dominant_prob)
replication_dominant_plot = begin
    replication_dominant_prob_plot = plot(
        _grid,xlim=(0,8),
        xticks = [0, 2, 4, 6, 8],
        ylim  = (0, 1), yticks = (0:0.2:1),
        ci_replication_dominant_prob;
        xlabel    = L"z",
        ylabel    = L"\PP[G]{|Z'| \geq |Z|\mid |Z|=z}",
        label     = L"\shortstack{Gaussian scale ($\mathcal{G}^{\mathrm{sN}}$)}",
        fillcolor = :blue,
        fillalpha = 0.4,
        linewidth = 0,        
        legend    = :topright
    )
    plot!(
        replication_dominant_prob_plot,
        _grid,
        ci_unif_replication_dominant_prob;
        show_ribbon = false,
        linecolor   = :orange,
        linestyle   = :solid,
        linewidth   =1.5,
        label       =  L"\shortstack{Unimodal ($\mathcal{G}^{\mathrm{unm}}$)}"
    )

    plot!(
        replication_dominant_prob_plot,
        _grid,
        ci_s_replication_dominant_prob;
        show_ribbon = false,
        linecolor   = :black,
        linestyle   = :dash,
        linewidth   = 1.5,
        label       = L"\shortstack{All ($\mathcal{G}^{\mathrm{all}}$)}"
    )
    plot!(
        replication_dominant_prob_plot;
        legend_columns        = 1,
        legend_font_pointsize = 7,
        legend_position       = :topright
    )
end
savefig("replication_dominant_prob_plot_cochrane.pdf")

struct StandardNormalPowerDensity <: Empirikos.LinearEBayesTarget
    β::Float64
end

struct StandardNormalPowerDistribution <: Empirikos.LinearEBayesTarget
    β::Float64
end

struct StandardUnifPowerDensity <: Empirikos.LinearEBayesTarget
    β::Float64
end

struct StandardUnifPowerDistribution <: Empirikos.LinearEBayesTarget
    β::Float64
end

#Fig.5 a)
function (pow::StandardNormalPowerDensity)(d::Distribution)
    β = pow.β
    zq = quantile(Normal(), 0.975)
    μ₀ = find_zero( μ -> cdf(Normal(), -zq - μ) + ccdf(Normal(), zq - μ) - β, (0.0, 10.0))
    (pdf(d, μ₀) + pdf(d, -μ₀)) / (pdf(Normal(μ₀), zq) - pdf(Normal(μ₀), -zq))
end

function (pow::StandardNormalPowerDistribution)(d::Distribution)
    β = pow.β
    zq = quantile(Normal(), 0.975)
    μ₀ = find_zero( μ -> cdf(Normal(), -zq - μ) + ccdf(Normal(), zq - μ) - β, (0.0, 10.0))
    cdf(d, μ₀) - cdf(d, -μ₀)
end

function (pow::StandardUnifPowerDistribution)(d::Distribution) 
    β  = pow.β
    zq = quantile(Normal(), 0.975)
    μ₀ = find_zero( μ -> cdf(Normal(), -zq - μ) + ccdf(Normal(), zq - μ) - β, (0.0, 10.0))
    a  = d.b   
    return min(μ₀, a)/a
end


function (pow::StandardUnifPowerDensity)(d::Distribution) 
    β  = pow.β
    zq = quantile(Normal(), 0.975)
    μ₀ = find_zero( μ -> cdf(Normal(), -zq - μ) + ccdf(Normal(), zq - μ) - β, (0.0, 10.0))
    a  = d.b
    num = (abs(μ₀) ≤ a) ? 1/a : zero(a)
    den = pdf(Normal(μ₀), zq) - pdf(Normal(μ₀), -zq)

    return num/den
end
βs = 0.05:0.01:0.99

bin_edges = 0.05:0.05:1.0
bin_midpoints = (bin_edges[1:end-1] .+ bin_edges[2:end]) ./ 2

struct PowerBinProbability <: Empirikos.LinearEBayesTarget
    β::Float64
end

function (pow::PowerBinProbability)(d::Distribution)
    bin_low = pow.β - 0.025
    bin_high = pow.β + 0.025

    bin_low = max(0.05, bin_low)
    bin_high = min(1.0, bin_high)
    
    zq = quantile(Normal(), 0.975)
    power_fun(μ) = 1 - (cdf(Normal(), zq - abs(μ)) - cdf(Normal(), -zq - abs(μ)))
    
    if bin_low ≈ 0.05
        μ_low = 0.0 
    else
        μ_low = find_zero(μ -> power_fun(μ) - bin_low, (0.0, 10.0))
    end
    
    if bin_high ≈ 1.0
        μ_high = Inf 
    else
        μ_high = find_zero(μ -> power_fun(μ) - bin_high, (0.0, 10.0))
    end
    
    prob_left = cdf(d, -μ_low) - cdf(d, -μ_high)
    prob_right = cdf(d, μ_high) - cdf(d, μ_low)
    
    return prob_left + prob_right
end

struct Power_80_Probability <: Empirikos.LinearEBayesTarget
    β::Float64
end

function (pow::Power_80_Probability)(d::Distribution)
    bin_low = 0.8
    
    zq = quantile(Normal(), 0.975)
    power_fun(μ) = 1 - (cdf(Normal(), zq - abs(μ)) - cdf(Normal(), -zq - abs(μ)))
    

    μ_low = find_zero(μ -> power_fun(μ) - bin_low, (0.0, 10.0))
    
    
    μ_high = Inf 
   
    prob_left = cdf(d, -μ_low) - cdf(d, -μ_high)
    prob_right = cdf(d, μ_high) - cdf(d, μ_low)
    
    return prob_left + prob_right
end

β_centers = 0.075:0.05:0.975  
 

pow_bin_targets = PowerBinProbability.(β_centers)

pow_80_target = Empirikos.untilt(Power_80_Probability(0.8))
pow_80_target = Empirikos.untilt(StandardNormalPowerDistribution(0.8))

cis_power_80_gaussian = confint(floc_dkw, pow_80_target)
cis_power_80_unif = confint(floc_unif_dkw, pow_80_target)
cis_power_80_ls = confint(floc_locationscale_dkw, pow_80_target)
dkw_amari = DvoretzkyKieferWolfowitz(α = 0.01)
fitted_dkw_amari = fit(dkw_amari, Zs_trunc)
discr = RealLineDiscretizer{:closed,:open}(2.15:0.05:7.96)
amari_gaussian_975 = Empirikos.AMARI(convexclass= tilted_scale_mix,
                                     flocalization=fitted_dkw_amari,
                                     discretizer=discr,
                                     solver = quiet_Mosek)
ci_power_80 = confint(amari_gaussian_975, pow_80_target, Zs_trunc; level = 0.95)

amari_unif_975 = Empirikos.AMARI(convexclass= tilted_u_scale_mix,
                                     flocalization=fitted_dkw_amari,
                                     discretizer=discr,
                                     solver = quiet_Mosek)
ci_unif_power_80 = confint(amari_unif_975, pow_80_target, Zs_trunc; level = 0.95)

amari_ls_975 = Empirikos.AMARI(convexclass= tilted_location_scale_mix,
                                     flocalization=fitted_dkw_amari,
                                     discretizer=discr,
                                     solver = quiet_Mosek)
ci_ls_power_80 = confint(amari_ls_975, pow_80_target, Zs_trunc; level = 0.95)

bin_targets = Empirikos.untilt.(PowerBinProbability.(β_centers))
cis_power_bin_gaussian = confint.(floc_dkw, bin_targets)
cis_power_bin_unif = confint.(floc_unif_dkw, bin_targets)
cis_power_bin_ls = confint.(floc_locationscale_dkw, bin_targets)


bin_edges = 0.05:0.05:1.0
bin_width = 0.05
lower_gaussian = [ci.lower for ci in cis_power_bin_gaussian]
upper_gaussian = [ci.upper for ci in cis_power_bin_gaussian]

lower_unif = [ci.lower for ci in cis_power_bin_unif]
upper_unif = [ci.upper for ci in cis_power_bin_unif]

lower_ls = [ci.lower for ci in cis_power_bin_ls]
upper_ls = [ci.upper for ci in cis_power_bin_ls]
power_distribution_plot = begin
    p = plot(
        xlabel = "Power",
        ylabel = "Probability",
        ylim = (0, 1),
        xticks = 0.05:0.05:1.0,
        xrotation = 45,
        legend = :topright,
        framestyle = :box
    )
    n_bins = length(β_centers)
    group_spacing = 0.01
    positions = β_centers

    function plot_confidence_bounds!(positions, lowers, uppers, color, label, marker, offset, linestyle=:solid)

        scatter!(
            positions .+ offset,
            lowers,
            color = color,
            marker = marker,
            markersize = 3,
            label = label,
            alpha = 0.8
        )
        
  
        scatter!(
            positions .+ offset,
            uppers,
            color = color,
            marker = marker,
            markersize = 3,
            label = "",
            alpha = 0.8
        )
        
    
        for i in 1:length(positions)
            plot!(
                [positions[i] + offset, positions[i] + offset],
                [lowers[i], uppers[i]],
                color = color,
                linewidth = 1,
                label = "",
                linestyle = linestyle, 
                alpha = 0.4
            )
        end
    end

    plot_confidence_bounds!(
        positions,
        lower_gaussian,
        upper_gaussian,
        :blue,
        L"\shortstack{Gaussian scale ($\mathcal{G}^{\mathrm{sN}}$)}",
        :circle,
        -group_spacing,
        :solid
    )

    plot_confidence_bounds!(
        positions,
        lower_unif,
        upper_unif,
        :orange,
        L"\shortstack{Unimodal ($\mathcal{G}^{\mathrm{unm}}$)}",
        :square,
        0,
        :solid
    )
    
    plot_confidence_bounds!(
        positions,
        lower_ls,
        upper_ls,
        :black,
         L"\shortstack{All ($\mathcal{G}^{\mathrm{all}}$)}",
        :diamond,
        group_spacing,
        :dash
    )
    

    
    plot!(
        p,
        legend_columns = 1,
        legend_font_pointsize = 9,
        legend_position = :topright
    )
end

power_probability_plot = begin
plot(β_centers, upper_gaussian, fillrange=lower_gaussian ,seriestype=:sticks,
            frame=:box,
            grid=nothing,
            xlabel = "Power",
            ylabel = "Probability",
            legend = :topright,
            linewidth=2,
            linecolor=:blue,
            alpha = 0.4,
            background_color_legend = :transparent,
            legendfonthalign = :left,
            foreground_color_legend = :transparent, xticks = vcat(0.05:0.1:0.95, 1.0),
        xrotation = 45,ylim = (0, 1), thickness_scaling=1.3,
            label=L"\shortstack{Gaussian scale ($\mathcal{G}^{\mathrm{sN}}$)}")

plot!(β_centers, [lower_unif upper_unif], seriestype=:scatter,  markershape=:hline,
            label=[L"\shortstack{Unimodal ($\mathcal{G}^{\mathrm{unm}}$)}" nothing], markerstrokecolor= :darkorange, markersize=4.5)

plot!(β_centers, [lower_ls upper_ls], seriestype=:scatter,  markershape=:circle,
             label=[L"\shortstack{All ($\mathcal{G}^{\mathrm{all}}$)}" nothing], color=:black, alpha=0.9, markersize=2.0, markerstrokealpha=0)

end
savefig("power_probability_plot_cochrane.pdf")




pow_distribution_targets = Empirikos.untilt.(StandardNormalPowerDistribution.(βs))
cis_power_distribution = confint.(floc_dkw, pow_distribution_targets)

pow_unif_distribution_targets = Empirikos.untilt.(StandardUnifPowerDistribution.(βs))

cis_unif_power_distribution = confint.(floc_unif_dkw,pow_distribution_targets)

cis_ls_power_distribution = confint.(floc_locationscale_dkw,pow_distribution_targets)

power_distribution_plot = begin
    power_distribution_plot = plot(
        βs,
        cis_power_distribution;
        xlabel       = "Power",
        ylabel       = "Distribution",
        ylim         = (0, 1),
        xticks       = [0.05, 0.2, 0.4, 0.6, 0.8, 1.0],
        label        = L"\shortstack{Gaussian scale ($\mathcal{G}^{\mathrm{sN}}$)}",
        fillcolor    = :blue,
        fillalpha    = 0.4,
        linewidth    = 0,        # ribbon only
        legend       = :topright
    )
    plot!(
        power_distribution_plot,
        βs,
        cis_unif_power_distribution;
        show_ribbon = false,
        linecolor   = :orange,
        linestyle   = :solid,
        linewidth   = 1.5,
        label       =  L"\shortstack{Unimodal ($\mathcal{G}^{\mathrm{unm}}$)}"
    )

    plot!(
        power_distribution_plot,
        βs,
        cis_ls_power_distribution;
        show_ribbon = false,
        linecolor   = :black,
        linestyle   = :dash,
        linewidth   = 1.5,
        label       = L"\shortstack{All ($\mathcal{G}^{\mathrm{all}}$)}"
    )

    plot!(
        power_distribution_plot;
        legend_columns        = 1,
        legend_font_pointsize = 5,
        legend_position       = :bottomright
    )
end

savefig("power_distribution_plot_cohorane.pdf")