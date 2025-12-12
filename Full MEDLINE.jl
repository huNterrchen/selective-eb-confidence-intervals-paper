using Pkg
using MathOptInterface
using Plots
using CSV
using IntervalSets
using StatsDiscretizations 
using Distributions
using PGFPlotsX
using MosekTools
using LaTeXStrings
push!(PGFPlotsX.CUSTOM_PREAMBLE, "\\usepackage{amsmath}")

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
using Revise
Pkg.develop("Empirikos")
using Empirikos


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

example_se = (log(0.96)-log(0.52))/(2*1.96)
example_z = (log(0.96)+log(0.52))/(2*example_se)
abs_example_z = abs(example_z)
tbl = CSV.File("medline_2000_2018_random.csv")

grid = -6:0.1:6
G = Normal(0,2)
trunc_set = Interval(2.1, Inf)
G_density = Empirikos.pdf.(G, grid)
norm = cdf.(Normal.(grid, 1), -2.1) .+ (1 .- cdf.(Normal.(grid, 1), 2.1))
tilted_density = (norm.*G_density)/Empirikos.pdf(G, FoldedNormalSample(trunc_set, 1.0))

p = plot(grid, G_density, label=L"G = \mathrm{N}(0,2)", lw=2, color=:black,
         xlabel="μ", ylabel="density", legend=:topleft)
plot!(p, grid, tilted_density, label=L"\operatorname{Tilt}_{\mathcal{S}}\left[G\right]", lw=2, color=:red)
savefig("tilted_prior.pdf")
plot(grid, norm , lw=2, color=:black,label="",
         xlabel="μ", ylabel=L"\Phi(\mathcal{S};\mu)")
savefig("normalizer.pdf")
zs = tbl.z

pgfplotsx()

bins = -8.12:0.14:8.12
histogram(zs[abs.(zs) .< 8]; bins=bins, xlim =(-8,8),
    label = "", linecolor=:gray, alpha=0.5, linewidth=0.25, size = (480, 270),
    xlabel = L"Z_i", ylabel = "Count",
    xticks = [-8, -5, -1.96, 0, 1.96, 5, 8],
    yticks=(5000:5000:25000, string.(5000:5000:25000))
    )
vline!([-1.96; 1.96], label="", linestyle=:dash, color=:black)
savefig("z_dist_medline.pdf")

bins = 0:0.14:8.12
histogram(abs.(zs)[abs.(zs) .< 8];
    bins=bins,
    xlim=(0,8),
    label=L"|Z| \mid D=1",
    linecolor=:gray,
    alpha=0.5,
    linewidth=0.25,
    size=(350,230),
    xlabel=L"|Z_i|",
    ylabel="Count",
    xticks=[0,  2.1, 5, 8],
    yticks=(5000:5000:40000, string.(5000:5000:40000))
)

histogram!(abs.(zs)[(abs.(zs) .< 8) .& (abs.(zs) .>= 2.1)];
    bins=bins,
    label=L"|Z| \mid D=1, |Z| \in \mathcal{S}",
    color=:red,
    alpha=0.2,
    linecolor=:red
)
vline!([2.1], label="", linestyle=:dash, color=:black)
histogram!(
    legend_columns       = 1,
    legend_font_pointsize= 5,
    legend_position     = :topright
  )
savefig("z_dist_medline_overlap.pdf")

trunc_set = Interval(2.1, Inf)
abs_zs_init = abs.(tbl.z)
abs_zs_init = abs_zs_init[isfinite.(abs_zs_init)]
abs_zs = abs_zs_init[in.( abs_zs_init, Ref(trunc_set))]
maximum(abs_zs)
Zs = FoldedNormalSample.(abs_zs, 1.0)
Z_trunc_set = FoldedNormalSample(trunc_set, 1.0)

gcal_scalemix = Empirikos.autoconvexclass(Empirikos.GaussianScaleMixtureClass();
    grid_scaling = 1.2, σ_min = 0.001, σ_max=100.0)
tilted_scale_mix = Empirikos.tilt(gcal_scalemix, Z_trunc_set)

ucal_scalemix = Empirikos.autoconvexclass(Empirikos.UniformScaleMixtureClass();
             a_min=0.001, a_max=100.0, grid_scaling=1.2
         )
tilted_u_scale_mix = Empirikos.tilt(ucal_scalemix, Z_trunc_set)
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
floc_unif_dkw = FLocalizationInterval(
    convexclass = tilted_u_scale_mix,
    flocalization = fitted_dkw ,
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

#with uniform prior

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
    normalize=true, xlim=(0,8), ylim=(0, 6),
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
    ylabel     =L"f_G(z)/\PP[G]{Z \in \mathcal{S}}"
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
        marginal_density_unnormalized_plot,
        x_seq,
        z_curve_CI_l_with,
        linecolor = :purple,
        linestyle = :dot,
        linewidth = 1.5,
        label = ""
    )
    
  plot!(
        marginal_density_unnormalized_plot,
        x_seq,
        z_curve_CI_u_with,
        linecolor = :purple,
        linestyle = :dot,
        linewidth = 1.5,
        label = L"\shortstack{Z-Curve}"
    )

  plot!(
    marginal_density_unnormalized_plot;
    legend_columns       = 1,
    legend_font_pointsize= 7,
    legend_position     = :topright
  )
end
marginal_density_unnormalized_plot = begin
  marginal_density_unnormalized_plot = histogram(
    abs_zs;
    normalize=true, xlim=(0,8), ylim=(0, 4),
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

savefig("marginal_density_normalized_plot_medline_full.pdf")


marginal_density_pretilt = Empirikos.untilt.(Empirikos.MarginalDensity.(FoldedNormalSample.(_grid, 1.0)))
ci_marginal_density_normalized = confint.(floc_dkw, marginal_density_pretilt)
ci_unif_marginal_density_normalized = confint.(floc_unif_dkw, marginal_density_pretilt)
ci_ls_marginal_density_normalized = confint.(floc_locationscale_dkw, marginal_density_pretilt)

marginal_density_normalized_plot = begin
    marginal_density_normalized_plot = plot(
        _grid,
        ylims = (0, 0.8),
        yticks = 0:0.2:0.8,
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
savefig("marginal_density_unnormalized_plot_medline_full.pdf")


same_sign_prob_example = Empirikos.untilt(Empirikos.SignAgreementProbability(FoldedNormalSample(abs_example_z, 1.0)))
ci_same_sign_prob_example = confint(floc_dkw, same_sign_prob_example)
ci_unif_same_sign_prob_example = confint(floc_unif_dkw, same_sign_prob_example)
ci_s_same_sign_prob_example = confint(floc_locationscale_dkw, same_sign_prob_example)

dkw_amari = DvoretzkyKieferWolfowitz(α = 0.01)
fitted_dkw_amari = fit(dkw_amari, Zs_trunc)
discr = RealLineDiscretizer{:closed,:open}(2.15:0.05:7.96)
amari_gaussian_975 = Empirikos.AMARI(convexclass= tilted_scale_mix,
                                     flocalization=fitted_dkw_amari,
                                     discretizer=discr,
                                     solver = quiet_Mosek)
ci_same_sign_prob_example = confint(amari_gaussian_975, same_sign_prob_example, Zs_trunc; level = 0.95)

amari_unif_975 = Empirikos.AMARI(convexclass= tilted_u_scale_mix,
                                     flocalization=fitted_dkw_amari,
                                     discretizer=discr,
                                     solver = quiet_Mosek)
ci_unif_same_sign_prob_example = confint(amari_unif_975, same_sign_prob_example, Zs_trunc; level = 0.95)

amari_ls_975 = Empirikos.AMARI(convexclass= tilted_location_scale_mix,
                                     flocalization=fitted_dkw_amari,
                                     discretizer=discr,
                                     solver = quiet_Mosek)
ci_ls_same_sign_prob_example = confint(amari_ls_975, same_sign_prob_example, Zs_trunc; level = 0.95)

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

savefig("same_sign_prob_plot_full.pdf")


coverage_prob_example = Empirikos.untilt(Empirikos.CoverageProbability(FoldedNormalSample(abs_example_z, 1.0)))
ci_coverage_prob_example = confint(floc_dkw, coverage_prob_example)
ci_unif_coverage_prob_example = confint(floc_unif_dkw, coverage_prob_example)
ci_s_coverage_prob_example = confint(floc_locationscale_dkw, coverage_prob_example)

dkw_amari = DvoretzkyKieferWolfowitz(α = 0.01)
fitted_dkw_amari = fit(dkw_amari, Zs_trunc)
discr = RealLineDiscretizer{:closed,:open}(2.15:0.05:7.96)
amari_gaussian_975 = Empirikos.AMARI(convexclass= tilted_scale_mix,
                                     flocalization=fitted_dkw_amari,
                                     discretizer=discr,
                                     solver = quiet_Mosek)
ci_coverage_prob_example = confint(amari_gaussian_975, coverage_prob_example, Zs_trunc; level = 0.95)

amari_unif_975 = Empirikos.AMARI(convexclass= tilted_u_scale_mix,
                                     flocalization=fitted_dkw_amari,
                                     discretizer=discr,
                                     solver = quiet_Mosek)
ci_unif_coverage_prob_example = confint(amari_unif_975, coverage_prob_example, Zs_trunc; level = 0.95)

amari_ls_975 = Empirikos.AMARI(convexclass= tilted_location_scale_mix,
                                     flocalization=fitted_dkw_amari,
                                     discretizer=discr,
                                     solver = quiet_Mosek)
ci_ls_coverage_prob_example = confint(amari_ls_975, coverage_prob_example, Zs_trunc; level = 0.95)


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
        legend_position = (0.45, 0.30)
    )
end
savefig("coverage_prob_plot_full.pdf")

repl_prob_example = Empirikos.untilt(Empirikos.ConditionalReplicationProbability(FoldedNormalSample(abs_example_z, 1.0)))
ci_repl_prob_example = confint(floc_dkw, repl_prob_example)
ci_unif_repl_prob_example = confint(floc_unif_dkw, repl_prob_example)
ci_s_repl_prob_example = confint(floc_locationscale_dkw, repl_prob_example)

zs_locs = sort(vcat(0.0:0.1:8, [1.644854; 1.959964; 2.575829; 3.290527; 3.890592]))
repl_prob = Empirikos.untilt.(Empirikos.ConditionalReplicationProbability.(FoldedNormalSample.(zs_locs, 1.0)))
ci_repl_prob = confint.(floc_dkw, repl_prob)
ci_unif_repl_prob = confint.(floc_unif_dkw, repl_prob)
ci_s_repl_prob = confint.(floc_locationscale_dkw, repl_prob)


dkw_amari = DvoretzkyKieferWolfowitz(α = 0.01)
fitted_dkw_amari = fit(dkw_amari, Zs_trunc)
discr = RealLineDiscretizer{:closed,:open}(2.15:0.05:7.96)
amari_gaussian_975 = Empirikos.AMARI(convexclass= tilted_scale_mix,
                                     flocalization=fitted_dkw_amari,
                                     discretizer=discr,
                                     solver = quiet_Mosek)
ci_repl_prob_example = confint(amari_gaussian_975, repl_prob_example, Zs_trunc; level = 0.95)

amari_unif_975 = Empirikos.AMARI(convexclass= tilted_u_scale_mix,
                                     flocalization=fitted_dkw_amari,
                                     discretizer=discr,
                                     solver = quiet_Mosek)
ci_unif_repl_prob_example = confint(amari_unif_975,repl_prob_example, Zs_trunc; level = 0.95)

amari_ls_975 = Empirikos.AMARI(convexclass= tilted_location_scale_mix,
                                     flocalization=fitted_dkw_amari,
                                     discretizer=discr,
                                     solver = quiet_Mosek)
ci_ls_repl_prob_example = confint(amari_ls_975, repl_prob_example, Zs_trunc; level = 0.95)

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
savefig("repl_prob_plot_full.pdf")

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
savefig("future_coverage_prob_plot_full.pdf")


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
savefig("replication_dominant_prob_plot_full.pdf")


post_means = Empirikos.untilt.(PosteriorMean.(NormalSample.(_grid, 1.0)))
ci_post_means = confint.(floc_dkw, post_means)

ci_unif_post_means = confint.(floc_unif_dkw, post_means)
s_post_means = Empirikos.untilt.(Empirikos.SymmetricPosteriorMean.(NormalSample.(_grid, 1.0)))
ci_s_post_means_2 = confint.(floc_locationscale_dkw, s_post_means)


post_means_example = Empirikos.untilt(Empirikos.PosteriorMean(NormalSample(abs_example_z, 1.0)))
ci_post_means_example = confint(floc_dkw, post_means_example)
ci_unif_post_means_example = confint(floc_unif_dkw, post_means_example)
s_post_means_example = Empirikos.untilt(Empirikos.SymmetricPosteriorMean(NormalSample.(abs_example_z, 1.0)))
ci_s_post_means_example = confint(floc_locationscale_dkw, s_post_means_example)

dkw_amari = DvoretzkyKieferWolfowitz(α = 0.01)
fitted_dkw_amari = fit(dkw_amari, Zs_trunc)
discr = RealLineDiscretizer{:closed,:open}(2.15:0.05:7.96)
amari_gaussian_975 = Empirikos.AMARI(convexclass= tilted_scale_mix,
                                     flocalization=fitted_dkw_amari,
                                     discretizer=discr,
                                     solver = quiet_Mosek)
ci_post_means_example = confint(amari_gaussian_975, post_means_example, Zs_trunc; level = 0.95)

amari_unif_975 = Empirikos.AMARI(convexclass= tilted_u_scale_mix,
                                     flocalization=fitted_dkw_amari,
                                     discretizer=discr,
                                     solver = quiet_Mosek)
ci_post_means_example = confint(amari_unif_975,post_means_example, Zs_trunc; level = 0.95)

amari_ls_975 = Empirikos.AMARI(convexclass= tilted_location_scale_mix,
                                     flocalization=fitted_dkw_amari,
                                     discretizer=discr,
                                     solver = quiet_Mosek)
ci_ls_s_post_means_example = confint(amari_ls_975, s_post_means_example, Zs_trunc; level = 0.95)

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

savefig("posterior_mean_plot_medline_full.pdf")

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


β_centers = 0.075:0.05:0.975  
 

pow_bin_targets = PowerBinProbability.(β_centers)

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
savefig("power_probability_plot_medline_full.pdf")




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

savefig("power_distribution_plot_medline_full.pdf")


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