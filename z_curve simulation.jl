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


trunc_set = Interval(1.96, Inf)
abs_zs_init = abs.(tbl.z)
abs_zs = abs_zs_init[in.( abs_zs_init, Ref(trunc_set))]
abs_zs = abs_zs[isfinite.(abs_zs)]
Zs = [z >=6 ? FoldedNormalSample(6..Inf) : FoldedNormalSample(z) for z in abs_zs]
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