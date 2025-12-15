# selective-eb-confidence-intervals-paper
## Reproducible code for Empirical Bayes learning from selectively reported confidence intervals

Our analysis relies on the [Empirikos.jl](https://github.com/nignatiadis/Empirikos.jl) package. While we are integrating everything into the package, please use the Empirikos package here to reproduce the analysis in our paper.

1. Full MEDLINE.jl: contains the code for inference on the full MEDLINE data set, including Figure.4, Table 2 and Table 3 in Section 5 of our paper.
2. MEDLINE 2018.jl: contains the code for inference on studies published in 2018 from the MEDLINE data in Section 6.1 of our paper, responsible for Figure.S1, Table S2 and Table S3 in Supplement Section D.2.
3. Cochrane (with truncation).jl,  Cochrane (without truncation).jl: contains the code for inference on the Cochrane data set under two settings as described in Section 6.1 of our paper, responsible for Figure.S2, Figure.S3, Table S4 and Table S5 in Supplement Section D.3.
   - Cochrane (with truncation).jl: applying our selective tilting procedure
   - Cochrane (without truncation).jl: without truncation adjustment
4. z_curve simulation.jl: contains the code for the simulation study comparing with Z-Curve 2.0 in Section 6.3 of our paper, including Figure.5.
