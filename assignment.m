%% ========================================================================
%  Empirical Methods in Financial Markets
%  ========================================================================

clear; close all; clc;

%% === Figure saving setup =========================

outdir = '/Users/louis/Documents/MATLAB/EMFM-Assignment/figs';
if ~exist(outdir,'dir'), mkdir(outdir); end

% Hide all figures created in this session
set(0, 'DefaultFigureVisible', 'off');
cleanupObj = onCleanup(@() set(0,'DefaultFigureVisible','on'));  % auto-restore on exit

pngdpi = '-r300';   % 300 DPI

%% === 1. Trend treatment =========================

% --- Step 1a : Load data and visual inspection -----

% Load dataset
load('/Users/louis/Documents/MATLAB/EMFM-Assignment/M-files/datagroup1.mat');
X = x;

% Create time index
t = (1:length(X))';

% Plot X_t over time
f1 = figure;
plot(t, X, 'b');
title('X_t over time');
xlabel('t'); ylabel('X_t');
print(f1, fullfile(outdir,'step1a_Xt_over_time'), '-dpng', pngdpi);
close(f1);

% Scatter plot of X_t vs time
f2 = figure;
scatter(t, X, 'filled');
title('Scatter(X_t, t)');
xlabel('t'); ylabel('X_t');
print(f2, fullfile(outdir,'step1a_scatter_Xt_vs_t'), '-dpng', pngdpi);
close(f2);

% --- Step 1b : OLS on raw X_t to test a cubic deterministic trend -----
alpha = 0.05; 
N  = length(X);
t0 = (t - mean(t)) / std(t);
Z  = [ones(N,1), t0, t0.^2, t0.^3];   % [Intercept, t, t^2, t^3]
pZ = size(Z,2);

% regress returns: stats = [R2, F, p_overall, sigma2_hat]
[beta,bint,r,~,stats] = regress(X, Z);
R2        = stats(1);
F_overall = stats(2);
p_overall = stats(3);
sigma2    = stats(4);

% Coefficient standard errors, t-stats, and p-values
covB = sigma2 * ((Z' * Z) \ eye(pZ));
se   = sqrt(diag(covB));
tstat = beta ./ se;
dof = N - pZ;

% Two-sided p-values for t-stats:
if exist('tcdf','file') == 2   % Stats TB available
    pval = 2 * (1 - tcdf(abs(tstat), dof));
else
    % Normal approximation fallback (good when dof is large)
    pval = 2 * (1 - 0.5*(1 + erf(abs(tstat)/sqrt(2))));
end

coef_names = {'Intercept','t','t^2','t^3'}';
fprintf('\n[1b.] OLS on X_t ~ [1, t, t^2, t^3]:\n');
for i = 1:pZ
    fprintf('%-9s: beta=% .4g, SE=% .4g, t=% .4g, p=% .4g\n', ...
        coef_names{i}, beta(i), se(i), tstat(i), pval(i));
end
fprintf('[1b.] Model: R^2=%.4f, F(%d,%d)=%.3f, p=%.4g\n', R2, pZ-1, dof, F_overall, p_overall);

% Candidate differencing order from polynomial significance
p_t   = pval(2);      % t
p_t2  = pval(3);      % t^2
p_t3  = pval(4);      % t^3
d_poly = 0;
if p_t  < alpha, d_poly = 1; end
if p_t2 < alpha, d_poly = 2; end
if p_t3 < alpha, d_poly = 3; end
fprintf('[1b.] Heuristic from OLS: d_poly = %d (highest significant degree)\n', d_poly);

% --- Step 1c : Automatic selection of differencing order using lagmatrix -
candD = 0:3;
sel   = [];   % [d, p_slope, max|rho|, cb, slope_ok, sacf_ok]
fprintf('\n[1c.] Differencing order check with lagmatrix (drift + SACF band):\n');

for dtry = candD
    % Build dtry-th difference using helper
    Xd  = diff_lagmatrix(X, dtry);
    t_d = t(dtry+1:end);

    % Drift check: Xd ~ [1, t_d]
    Z2   = [ones(length(t_d),1), t_d];
    pZ2  = size(Z2,2);
    [beta2,~,~,~,stats2] = regress(Xd, Z2);
    sigma2_2 = stats2(4);
    covB2 = sigma2_2 * ((Z2' * Z2) \ eye(pZ2));
    se2   = sqrt(diag(covB2));
    tstat2 = beta2 ./ se2;
    dof2 = length(Xd) - pZ2;

    slope_idx = 2;
    if exist('tcdf','file') == 2
        p_slope = 2 * (1 - tcdf(abs(tstat2(slope_idx)), dof2));
    else
        p_slope = 2 * (1 - 0.5*(1 + erf(abs(tstat2(slope_idx))/sqrt(2))));
    end
    slope_ok = (p_slope > alpha);

    % SACF band check
    Lacf = min(20, floor(length(Xd)/3));
    if Lacf < 1
        maxAbsRho = Inf; cb = Inf; sacf_ok = false;
    else
        [rho, ~] = sacf(Xd, Lacf, 0, 0);
        N_curr = length(Xd);
        cb = 1.96 / sqrt(N_curr);      
        maxAbsRho = max(abs(rho));
        sacf_ok   = all(abs(rho) <= cb);
    end

    sel = [sel; dtry, p_slope, maxAbsRho, cb, slope_ok, sacf_ok];
    fprintf('d=%d | p_slope=%.4g | max|rho|=%.4f vs cb=%.4f | slope_ok=%d | sacf_ok=%d\n',...
        dtry, p_slope, maxAbsRho, cb, slope_ok, sacf_ok);
end

idx = find(sel(:,5)==1 & sel(:,6)==1, 1, 'first');
if ~isempty(idx)
    d = sel(idx,1);
    reason = 'smallest d with insignificant drift and SACF within band';
else
    d = d_poly;
    reason = 'fallback to highest significant polynomial degree (OLS)';
end
fprintf('[1c.] Selected trend differencing order: d=%d (%s)\n', d, reason);

% --- Step 1d : Apply chosen differencing with lagmatrix + plots -----
% d = 2 already yields negligible practical drift
d=2;
fprintf('\n[1d.] Forced trend differencing order: d=%d (already yields negligible practical drift)\n', d);

Xd  = diff_lagmatrix(X, d);
t_d = t(d+1:end);

% Loop over d = 1,...,d and plot each differenced series
for dplot = 1:d
    Xd_plot  = diff_lagmatrix(X, dplot);
    t_d_plot = t(dplot+1:end);

    % Time-series plot of differenced series
    f3 = figure;
    plot(t_d_plot, Xd_plot, 'b', 'LineWidth', 1); grid on; box on;
    title(sprintf('Differenced series \\nabla^{%d} X_t (lagmatrix)', dplot));
    xlabel('t'); ylabel(sprintf('\\nabla^{%d} X_t', dplot));
    print(f3, fullfile(outdir, sprintf('step1d_diff%d_series', dplot)), '-dpng', pngdpi);
    close(f3);

    % Scatter plot with OLS fit
    f4 = figure;
    scatter(t_d_plot, Xd_plot, 12, 'filled'); grid on; box on; hold on;
    p = polyfit(t_d_plot, Xd_plot, 1);
    plot(t_d_plot, polyval(p, t_d_plot), 'r--', 'LineWidth', 1.2);
    xlabel('t'); ylabel(sprintf('\\nabla^{%d} X_t', dplot));
    title(sprintf('Scatter(\\nabla^{%d} X_t, t) with linear fit', dplot));
    legend('Data','OLS fit','Location','best'); hold off;
    print(f4, fullfile(outdir, sprintf('step1d_diff%d_scatter_with_OLS', dplot)), '-dpng', pngdpi);
    close(f4);
end

% Final drift check report on chosen d
Z2   = [ones(length(t_d),1), t_d];
pZ2  = size(Z2,2);
[beta2,~,~,~,stats2] = regress(Xd, Z2);
sigma2_2 = stats2(4);
covB2 = sigma2_2 * ((Z2' * Z2) \ eye(pZ2));
se2   = sqrt(diag(covB2));
tstat2 = beta2 ./ se2;
dof2 = length(Xd) - pZ2;
if exist('tcdf','file') == 2
    p_slope = 2 * (1 - tcdf(abs(tstat2(2)), dof2));
else
    p_slope = 2 * (1 - 0.5*(1 + erf(abs(tstat2(2))/sqrt(2))));
end
fprintf('\n[1d.] Drift check on nabla^%d X_t ~ [1, t]:\n', d);
fprintf('Slope (t): beta=% .4g, SE=% .4g, t=% .4g, p=% .4g\n', ...
    beta2(2), se2(2), tstat2(2), p_slope);

% Detrended series
r1 = Xd;

%% === 2. Seasonality treatment =========================

% --- Step 2a : ACF + peak detection -----
maxLag = min(30, floor(length(r1)/3));

[xrho, ~] = sacf(r1, maxLag, 0, 0); 
N1 = length(r1);
cb1 = 1.96 / sqrt(N1); % The 95% confidence bound for white noise

% simple peak detection in |rho|
absrho = abs(xrho);
cand = find( absrho(2:end-1) >= absrho(1:end-2) & ...
             absrho(2:end-1) >= absrho(3:end)     & ...
             absrho(2:end-1) >= cb1 ) + 1;

% enforce minimal spacing >= 2
if ~isempty(cand)
    keep = [true; diff(cand) >= 2];
    cand = cand(keep);
end

% fundamental period guess from peak spacing
S = NaN;
if numel(cand) >= 2
    dLag = diff(cand);
    base = mode(round(dLag));
    
    % --- sign-aware correction ---
    sgn = sign(xrho(cand));
    if numel(sgn) >= 2
        % fraction of consecutive peaks with opposite sign
        altFrac = mean(sgn(1:end-1) .* sgn(2:end) < 0);

        % if they alternate a lot (like + - + - + -), treat base as S/2
        if altFrac > 0.7           % threshold, (0.7?0.8)
            S = 2*base;        
        else
            S = base;
        end
    else
        S = base;
    end
elseif numel(cand) == 1
    S = cand(1);
else
    [~, idxMax] = max(absrho(2:end));
    S = idxMax + 1;
end
S = max(2, min(S, floor(length(r1)/2)));

fprintf('\n[2a.] SACF-based seasonal guess S = %d\n', S);

% Plot SACF (pre), enforce minimal spacing >= 2
sacf(r1, maxLag);
title('Sample SACF of r_1(t)'); ylabel('SACF'); xlabel('Lags');
print(gcf, fullfile(outdir,'step2a_acf_before_seasonal'), '-dpng', pngdpi);
close(gcf);

% --- Step 2b : Seasonal differencing once at S -----
% Force S=4 for this run
%S=4;
%fprintf('\n=> Forced seasonal differencing order: S=%d\n', S);

r2 = seasdiff_lagmatrix(r1, S, 1);
t_s = t_d(S+1:end);

% Plot seasonally differenced series
f5 = figure;
plot(t_s, r2, 'b', 'LineWidth', 1); hold on; grid on; box on;
plot([min(t_s) max(t_s)], [0 0], 'k--');  % horizontal zero line
title(sprintf('Seasonally differenced series  \\nabla_{%d} r_1(t)', S));
xlabel('t'); ylabel(sprintf('\\nabla_{%d} r_1(t)', S));
print(f5, fullfile(outdir, sprintf('step2b_seasdiff_S%d_series', S)), '-dpng', pngdpi);
close(f5);

% Scatter vs time with OLS fit
f6 = figure;
scatter(t_s, r2, 12, 'filled'); hold on; grid on; box on;
p2 = polyfit(t_s, r2, 1);
plot(t_s, polyval(p2, t_s), 'r--', 'LineWidth', 1.2);
xlabel('t'); ylabel(sprintf('\\nabla_{%d} r_1(t)', S));
title(sprintf('Scatter(\\nabla_{%d} r_1(t), t) with linear fit', S));
legend('Data','OLS fit','Location','best');
print(f6, fullfile(outdir, sprintf('step2b_seasdiff_S%d_scatter_with_OLS', S)), '-dpng', pngdpi);
close(f6);

% --- Step 2c : SACF/LB diagnostics after differencing -----
maxLag2 = min(30, floor(length(r2)/3));
sacf(r2, maxLag2);
title(sprintf('SACF of \\nabla_{%d} r_1(t)', S)); ylabel('SACF'); xlabel('Lags');
print(gcf, fullfile(outdir, sprintf('step2c_acf_after_seasonal_S%d', S)), '-dpng', pngdpi);
close(gcf);

% Ljung-Box diagnostics (using some lags including seasonal multiples)
LB_lags = unique([S, 2*S, 3*S, 12, 20]);
LB_lags_pre  = LB_lags(LB_lags <= floor(length(r1)/4));
LB_lags_post = LB_lags(LB_lags <= floor(length(r2)/4));

if ~isempty(LB_lags_pre)
    Lmax_pre = max(LB_lags_pre);
    [Qpre, Ppre] = ljungbox(r1, Lmax_pre);
    fprintf('\n[2c.] Ljung-Box on r_1(t) (before seasonal differencing):\n');
    for L = LB_lags_pre, fprintf('Q(%d) = %.3f, p = %.4g\n', L, Qpre(L), Ppre(L)); end
end
if ~isempty(LB_lags_post)
    Lmax_post = max(LB_lags_post);
    [Qpost, Ppost] = ljungbox(r2, Lmax_post);
    fprintf('\n[2c.] Ljung-Box on y(t) = nabla_%d r_1(t) (after seasonal differencing):\n', S);
    for L = LB_lags_post, fprintf('Q(%d) = %.3f, p = %.4g\n', L, Qpost(L), Ppost(L)); end
end

% deseasonalized series
r3 = r2;   

%% === 3. Stationarity analysis =========================

% --- Step 3a : Is a pure random process suitable  -----
% We check iid-ness via SACF/PACF, Ljung-Box, and normality diagnostics.

Ny = length(r3);
maxLagPre = min(30, floor(Ny/4));

% Basic summary stats (with fallback for skew/kurt)
muY = mean(r3); sdY = std(r3);
skY = skewness(r3); 
kuY = kurtosis(r3);

fprintf('\n[3a.] Summary(Y_t): N=%d, mean=%.4g, sd=%.4g, skew=%.4g, kurt=%.4g\n', ...
        Ny, muY, sdY, skY, kuY);

% SACF
sacf(r3, maxLagPre);
title('SACF of Y_t'); ylabel('SACF'); xlabel('Lags');
print(gcf, fullfile(outdir,'step3a_prelim_sacf_Yt'), '-dpng', pngdpi);
close(gcf);

% SPACF
spacf(r3, maxLagPre);
title('SPACF of Y_t'); ylabel('SPACF'); xlabel('Lags');
print(gcf, fullfile(outdir,'step3a_prelim_spacf_Yt'), '-dpng', pngdpi);
close(gcf);

% Ljung-Box tests at multiple lags
LB_lags = [5 10 15 20];
LB_lags = LB_lags(LB_lags <= floor(Ny/4));
if ~isempty(LB_lags)
    Lmax = max(LB_lags);
    [Qvec, Pvec] = ljungbox(r3, Lmax);
    fprintf('\n[3a.] Ljung-Box on Y_t:\n');
    for L = LB_lags
        fprintf('  Q(%2d) = %.3f, p = %.4g\n', L, Qvec(L), Pvec(L));
    end
end

% Normality diagnostics
% Histogram (with ~30 bins)
fH = figure; histogram(r3, 30); grid on; box on;
title('Histogram of Y_t'); xlabel('Y_t'); ylabel('Count');
print(fH, fullfile(outdir,'step3a_prelim_hist_Yt'), '-dpng', pngdpi); close(fH);

% QQ-plot vs Normal(0,1)
fQ = figure; qqplot((r3 - muY)/sdY); grid on; box on;
title('QQ-plot of standardized Y_t vs N(0,1)');
print(fQ, fullfile(outdir,'step3a_prelim_qqplot_Yt'), '-dpng', pngdpi); close(fQ);

% Kolmogorov-Smirnov
[ksstat, pKS, hKS] = kolmogorov(r3, [], 'normcdf', muY, sdY);
fprintf('[3a.] Kolmogorov (MFE, vs N(muY,sdY)): stat=%.4f, p=%.4g, reject=%d\n', ksstat, pKS, hKS);

% Unit-root check for stationarity 
% Augmented Dickey-Fuller with automatic lag
p_det  = 1;                      % 0: none, 1: constant, 2: trend, 3: const + trending DGP
lagsAD = 0;                      % we can try 0 (DF); or a small integer

[adfStat, adfP, adfCV] = augdf(r3, p_det, lagsAD);   % output order is [stat, pval, crit]

fprintf('[3a.] ADF (p=%d, lags=%d): stat=%.4f, p=%.4g (H0: unit root)\n', ...
        p_det, lagsAD, adfStat, adfP);

% --- Step 3b : What is the best stationary model -----
% Methodology: Detect significant ACF lags in the first 30 lags.
% Estimate a sparse model containing only those specific lags.
c = 0; % Constant (drift)
maxLag_Ident = 30;

fprintf('\n[3b.] Identifying Sparse MA Model (checking first %d lags)...\n', maxLag_Ident);

[rho_ident, std_ident] = sacf(r3, maxLag_Ident, 1, 0);

% Detect Significant Lags
sig_lags = find(abs(rho_ident) > 2 * std_ident);

% Filter: Use detected lags
if isempty(sig_lags)
    fprintf('    No significant lags found. Fallback to White Noise (MA(0)).\n');
    Qv = [];
else
    Qv = sig_lags(:); % Force column vector
end
Pv = []; % No AR terms (p=0)

fprintf('    Identified Significant Lags: %s\n', mat2str(Qv'));

% Forced Qv after residuals analysis of MA(1,4,9) and MA(1,2,3,4,9)
Qv = [1; 2; 3; 4];
fprintf('    Forced Significant Lags: %s\n', mat2str(Qv'));

HOLD_BACK = max(0, d + S); 
opts = optimset('lsqnonlin'); opts.Display = 'off';

try
    % Estimate parameters using Maximum Likelihood (via armaxfilter)
    % theta vector structure: [constant, AR_coeffs, MA_coeffs]
    [theta, LL, eps, SEreg, diagnostics] = armaxfilter(r3, c, Pv, Qv, [], [], opts, HOLD_BACK);
    
    % Display parameters
    fprintf('    Model Estimated Successfully.\n');
    fprintf('    Log-Likelihood: %.4f\n', LL);
    fprintf('    Sigma (SEreg):  %.4f\n', SEreg);
    fprintf('    AIC Score:      %.4f\n', diagnostics.AIC);
    fprintf('    BIC (SBIC) Score: %.4f\n', diagnostics.SBIC);
    
    if c
        fprintf('    Constant (mu):  %.4f\n', theta(1));
        ma_start_idx = 2;
    else
        ma_start_idx = 1;
    end
    
    theta_MA = theta(ma_start_idx:end);
    fprintf('    Estimated MA Coefficients:\n');
    for i = 1:length(Qv)
        fprintf('       MA(%d): %.4f\n', Qv(i), theta_MA(i));
    end

    % --- Invertibility Check (Sparse MA) ---
    % Polynomial: 1 + theta_q1 * L^q1 + theta_q2 * L^q2 + ...
    % We must construct the full coefficient vector including zeros for missing lags.
    if isempty(Qv)
         fprintf('    Invertibility:  N/A (White Noise)\n');
    else
        q_order = max(Qv);
        
        % Matlab 'roots' expects: c_n x^n + ... + c_1 x + c_0
        % Our polynomial is in L (backward shift). 
        % Roots of 1 + th_1*z + ... + th_q*z^q = 0 must be outside unit circle.
        % Vector for roots: [theta_q, theta_{q-1}, ..., theta_1, 1]
        
        poly_vec = zeros(q_order + 1, 1);
        poly_vec(end) = 1;              % Constant 1
        poly_vec(end - Qv) = theta_MA;  % Map sparse coeffs to correct powers
        
        ma_roots = roots(poly_vec);
        min_root_mod = min(abs(ma_roots));
        
        if min_root_mod > 1
            fprintf('    Invertibility:  YES (Min Root Modulus %.4f > 1)\n', min_root_mod);
        else
            fprintf('    Invertibility:  NO  (Min Root Modulus %.4f <= 1) -> Non-invertible\n', min_root_mod);
        end
    end

catch ME
    error('Estimation failed: %s', ME.message);
end

% --- Step 3c : Residual analysis -----
model_str = ['MA(' mat2str(Qv') ')'];

% Plot Residuals
f_res = figure; 
plot(eps); grid on; hold on;
plot(xlim, [0 0], 'r--');
title(sprintf('Residuals of %s', model_str));
ylabel('Error \epsilon_t'); xlabel('Time');
print(f_res, fullfile(outdir, sprintf('step3c_resid_series_MA')), '-dpng', pngdpi);
close(f_res);

% Residual ACF/PACF
Lacf_e = min(30, floor(length(eps)/4));

sacf(eps, Lacf_e);
title(sprintf('Residual SACF - %s', model_str)); ylabel('SACF'); xlabel('Lags');
print(gcf, fullfile(outdir, sprintf('step3c_resid_sacf_MA')), '-dpng', pngdpi);
close(gcf);

spacf(eps, Lacf_e);
title(sprintf('Residual SPACF - %s', model_str)); ylabel('SPACF'); xlabel('Lags');
print(gcf, fullfile(outdir, sprintf('step3c_resid_spacf_MA')), '-dpng', pngdpi);
close(gcf);

% Ljung-Box Test on Residuals
% H0: No serial correlation up to lag L
LB_lags = [5, 8, 10, 12, 15, 20, 25, 30];
fprintf('\n    Ljung-Box Test on Residuals (Testing for Linear Independence):\n');
[Qe, Pe] = ljungbox(eps, max(LB_lags));
num_params = numel(Qv) + c;

for L = LB_lags
    df = max(1, L - num_params); % Degrees of freedom adjustment
    p_adj = 1 - chi2cdf(Qe(L), df);
    fprintf('    Lag %2d: Q=%.3f, p_val=%.4f (df=%d) -> %s\n', ...
        L, Qe(L), p_adj, df, ifelse(p_adj>0.05, 'Fail to Reject H0 (Good)', 'Reject H0 (Bad)'));
end

% Normality Checks (Gaussian Assumption)
fprintf('\n    Normality Checks:\n');
mu_e = mean(eps); 
sd_e = std(eps);
skew_e = skewness(eps); 
kurt_e = kurtosis(eps);
fprintf('    Mean: %.4g, Std: %.4g, Skewness: %.4f, Kurtosis: %.4f\n', mu_e, sd_e, skew_e, kurt_e);

% Histogram
%f_hist = figure; histogram(eps, 30, 'Normalization', 'pdf'); grid on; hold on;
%x_rng = linspace(min(eps), max(eps), 100);
%plot(x_rng, normpdf(x_rng, mu_e, sd_e), 'r', 'LineWidth', 2);
%title(sprintf('Histogram of Residuals vs Normal Fit - %s)', model_str));
%print(f_hist, fullfile(outdir, sprintf('step3c_resid_hist_MA')), '-dpng', pngdpi);
%close(f_hist);

% Histogram (with ~30 bins)
fH = figure; histogram(eps, 30); grid on; box on;
title(sprintf('Histogram of residuals -  %s', model_str)); xlabel('Y_t'); ylabel('Count');
print(fH, fullfile(outdir,'step3c_resid_hist_MA'), '-dpng', pngdpi); close(fH);

% QQ-plot vs Normal(0,1)
fQ = figure; qqplot((eps - mu_e)/sd_e); grid on; box on;
title(sprintf('QQ-plot of standardized residuals vs N(0,1)- %s', model_str));
print(fQ, fullfile(outdir,'step3c_resid_qqplot_MA'), '-dpng', pngdpi); close(fQ);

% Kolmogorov-Smirnov Test
% H0: Data is distributed as Normal(mu_e, sd_e)
[ksstat_e, pKS_e, hKS_e] = kolmogorov(eps, [], 'normcdf', mu_e, sd_e);
fprintf('    Kolmogorov-Smirnov: Stat=%.4f, p=%.4g -> %s\n', ...
    ksstat_e, pKS_e, ifelse(hKS_e==0, 'Normality Not Rejected', 'Normality Rejected'));

% Heteroscedasticity Checks (ARCH Effects)
% Check ACF of Squared Residuals
eps2 = eps.^2;

% SACF
sacf(eps2, Lacf_e);
title(sprintf('SACF of squared residuals - %s', model_str)); ylabel('SACF'); xlabel('Lags');
print(gcf, fullfile(outdir, sprintf('step3c_resid2_sacf_MA')), '-dpng', pngdpi);
close(gcf);

% SPACF
spacf(eps2, Lacf_e);
title(sprintf('SPACF of squared residuals - %s', model_str)); ylabel('SPACF'); xlabel('Lags');
print(gcf, fullfile(outdir, sprintf('step3c_resid2_spacf_MA')), '-dpng', pngdpi);
close(gcf);

% Ljung-Box on Squared Residuals
fprintf('\n    Ljung-Box on Squared Residuals (Testing for ARCH Effects):\n');
[Qe2, ~] = ljungbox(eps2, max(LB_lags));
for L = LB_lags
    p_val2 = 1 - chi2cdf(Qe2(L), L);
    fprintf('    Lag %2d: Q=%.3f, p_val=%.4f -> %s\n', ...
        L, Qe2(L), p_val2, ifelse(p_val2>0.05, 'No ARCH Effects', 'ARCH Effects Present'));
end

%% === 4. One-step-ahead forecasting =========================
% Forecasts of an MA model can easily be obtained. Because the model has finite 
%  memory, its point forecasts go to the mean of the series quickly.
%
% We perform 1-step ahead forecasting for the last 20 observations (Test Set).
% The model parameters are re-estimated using only the Training Set.

% --- 4a. Train/Test Split ---
N_test = 20;
T_total = length(r3);
T_train = T_total - N_test;

r_train = r3(1:T_train);
r_test  = r3(T_train+1:end);

fprintf('\n[4.] Forecasting (Horizon h=1) for last %d observations...\n', N_test);
fprintf('     Training Set: %d observations\n', length(r_train));
fprintf('     Test Set:     %d observations\n', length(r_test));

% --- 4b. Re-estimate Model on Training Data ---
% We use the Qv (lags) identified in Step 3b, but re-fit parameters on r_train.

% Ensure c and Qv are defined from previous steps
if ~exist('Qv','var'), Qv = [1]; end % Fallback if Step 3b skipped
if ~exist('c','var'), c = 0; end

fprintf('     Re-estimating MA(%s) on Training Set...\n', mat2str(Qv'));

opts = optimset('lsqnonlin'); opts.Display = 'off';
HOLD_BACK_TRAIN = max(30, d + S); 

% Estimate on Training Data
try
    [theta_train, ~, eps_train, sigma_train, ~] = armaxfilter(r_train, c, [], Qv, [], [], opts, HOLD_BACK_TRAIN);
    
    % Extract parameters
    if c
        mu_train = theta_train(1);
        ma_coeffs = theta_train(2:end);
    else
        mu_train = 0;
        ma_coeffs = theta_train;
    end
    
    fprintf('     Training Sigma: %.4f\n', sigma_train);
    
catch ME
    error('Training Estimation failed: %s', ME.message);
end

% --- 4c. Forecasting Loop ---
% To forecast MA models, we need the history of shocks (epsilon).
% We start with the shocks recovered from the training set.
% As we step through the test set, we compute the *new* shock:
% a_t = r_t (actual) - r_hat_t (predicted) [cite: 1037]

forecasts = zeros(N_test, 1);
% Full history of errors: [Training Errors; Zeros for Test]
E_history = [eps_train; zeros(N_test, 1)];

fprintf('     Computing 1-step ahead predictions...\n');

for i = 1:N_test
    % Current time index in the full series (Training Length + i)
    t_idx = T_train + i;
    
    % Calculate MA component: Sum( theta_j * a_{t-j} )
    ma_term = 0;
    for k = 1:length(Qv)
        lag_k = Qv(k);
        % We look back into E_history
        if (t_idx - lag_k) > 0
            ma_term = ma_term + ma_coeffs(k) * E_history(t_idx - lag_k);
        end
    end
    
    % 1-step forecast: Mean + MA_component
    % Note: MFE armaxfilter convention is usually y_t = c + eps_t + theta*eps_{t-1}
    % (Checking signs: Syllabus uses minus[cite: 921], but toolboxes usually use plus. 
    %  The estimated theta implies the sign convention of the tool is handled automatically).
    pred_val = mu_train + ma_term;
    forecasts(i) = pred_val;
    
    % Compute the actual forecast error (shock) for this step
    % This shock is needed for the *next* step's prediction (MA recursion)
    actual_val = r_test(i);
    shock = actual_val - pred_val;
    
    % Store shock in history
    E_history(t_idx) = shock;
end

% --- 4d. Evaluation (RMSE) ---
forecast_errors = r_test - forecasts;
RMSE = sqrt(mean(forecast_errors.^2));

fprintf('     Evaluation Results:\n');
fprintf('     RMSE: %.5f\n', RMSE);

% --- 4e. Plotting ---
f_fc = figure;

% Plot a portion of training data for context (e.g., last 50 points)
context_win = 30;
idx_train_plot = (T_train - context_win + 1):T_train;
idx_test_plot  = (T_train + 1):T_total;

% Plot History (Train) with Black Dots
plot(idx_train_plot, r_train(end-context_win+1:end), 'k.-', 'LineWidth', 1.2, 'MarkerSize', 15); hold on;
% Plot Actual (Test) - Blue Dots
plot(idx_test_plot, r_test, 'b.-', 'LineWidth', 1.2, 'MarkerSize', 15);
% Plot Forecast - Hollow Red Circles
plot(idx_test_plot, forecasts, 'r.--', 'LineWidth', 1.2, 'MarkerSize', 15);

% Confidence Intervals (+/- 1.96 * sigma) 
% For 1-step ahead, variance is sigma_a^2
ci_upper = forecasts + 1.96 * sigma_train;
ci_lower = forecasts - 1.96 * sigma_train;

plot(idx_test_plot, ci_upper, 'r:', 'LineWidth', 0.8);
plot(idx_test_plot, ci_lower, 'r:', 'LineWidth', 0.8);

% Define coordinates for the last training point and first test/forecast points
x_last_train = T_train;
y_last_train = r_train(end);
x_first_test = idx_test_plot(1);
y_first_actual = r_test(1);
y_first_forecast = forecasts(1);
% Blue linking line: Last History -> First Actual
plot([x_last_train, x_first_test], [y_last_train, y_first_actual], 'b-', 'LineWidth', 1.2);
% Red dashed linking line: Last History -> First Forecast
plot([x_last_train, x_first_test], [y_last_train, y_first_forecast], 'r--', 'LineWidth', 1.2);

grid on; box on;
xlabel('Time Index'); ylabel('Returns');
title(sprintf('1-Step Ahead Forecast - MA(%s)', mat2str(Qv')));
legend('History (Train)', 'Actual (Test)', 'Forecast', '95% CI', 'Location', 'best');

print(f_fc, fullfile(outdir, 'step4e_forecast'), '-dpng', pngdpi);

% --- 4f. Forecast reconstruction on original scale ---
% We now transform the stationary forecasts (r3) back to the original scale (X).
% To do this, we essentially "undo" the differencing operations:
% r3_t = (1-L^S)(1-L)^d X_t
% X_t = r3_t - [terms involving past X]

fprintf('\n[5.] Reconstructing forecasts on original scale X...\n');

% Construct the full differencing polynomial
% Trend diff: (1-L)^d
poly_d = 1;
for k = 1:d
    poly_d = conv(poly_d, [1 -1]);
end

% Seasonal diff: (1-L^S) -> [1, 0, ..., 0, -1]
poly_s = zeros(1, S+1);
poly_s(1) = 1; 
poly_s(end) = -1;

% Full difference polynomial: P(L) = (1-L^S)(1-L)^d
% X_t * P(L) = r3_t
poly_full = conv(poly_d, poly_s);

% Check: poly_full(1) should be 1. 
% X_t + c_1*X_{t-1} + ... + c_k*X_{t-k} = r3_t
% X_t = r3_t - (c_1*X_{t-1} + ... + c_k*X_{t-k})

% Define Indices on Original Scale
% The test set corresponds to the LAST N_test observations of X.
N_X = length(X);
idx_test_start = N_X - N_test + 1;
idx_test_range = idx_test_start:N_X;

% Extract Actuals (Force Column Vector)
X_test_actual = X(idx_test_range);
X_test_actual = X_test_actual(:);

% Reconstruction Loop 
forecasts_X = zeros(N_test, 1);

for i = 1:N_test
    % Index of the current point in the original series X
    curr_X_idx = idx_test_range(i);
    
    % Get the stationary forecast for this step (from Step 4)
    r3_hat = forecasts(i);
    
    % Reconstruct X_hat using the polynomial and past actual X values
    % Equation: X_t = r3_t - sum_{j=2..K} ( poly_full(j) * X_{t-(j-1)} )
    % We use 'X' directly for past values (1-step ahead assumption)
    
    past_val_sum = 0;
    for j = 2:length(poly_full)
        lag = j - 1;
        coeff = poly_full(j);
        
        % Retrieve past X from the original series
        val_lagged = X(curr_X_idx - lag);
        past_val_sum = past_val_sum + coeff * val_lagged;
    end
    
    X_hat = r3_hat - past_val_sum;
    forecasts_X(i) = X_hat;
end

% Evaluation (Original Scale) 
errors_X = X_test_actual - forecasts_X;
RMSE_X = sqrt(mean(errors_X.^2));

fprintf('     Original Scale Evaluation:\n');
fprintf('     RMSE: %.4f\n', RMSE_X);

% Plotting
f_orig = figure;

% Plot context: Last 30 points of training data
idx_train_end = idx_test_start - 1;
idx_plot_train = (idx_train_end - 30 + 1) : idx_train_end;

plot(idx_plot_train, X(idx_plot_train), 'k.-', 'LineWidth', 1.2, 'MarkerSize', 15); hold on;
plot(idx_test_range, X_test_actual, 'b-o', 'LineWidth', 1.2, 'MarkerSize', 10);
plot(idx_test_range, forecasts_X, 'r.--', 'LineWidth', 1.2, 'MarkerSize', 15);

% Confidence Intervals on Original Scale
% For 1-step ahead, the variance is the same as the stationary series: sigma_a^2
ci_upper_X = forecasts_X + 1.96 * sigma_train;
ci_lower_X = forecasts_X - 1.96 * sigma_train;

plot(idx_test_range, ci_upper_X, 'r:', 'LineWidth', 0.8);
plot(idx_test_range, ci_lower_X, 'r:', 'LineWidth', 0.8);

% Visual Links
plot([idx_plot_train(end), idx_test_range(1)], ...
     [X(idx_plot_train(end)), X_test_actual(1)], 'b-', 'LineWidth', 1.2);
plot([idx_plot_train(end), idx_test_range(1)], ...
     [X(idx_plot_train(end)), forecasts_X(1)], 'r--', 'LineWidth', 1.2);

grid on; box on;
title(sprintf('1-Step Ahead Forecast (Original Scale) - MA(%s)', mat2str(Qv')));
xlabel('Time'); ylabel('Original Series X_t');
legend('History', 'Actual', 'Forecast', '95% CI', 'Location', 'best');

print(f_orig, fullfile(outdir, 'step4f_forecast_original_scale'), '-dpng', pngdpi);
close(f_orig);

%% === Helpers =========================
function z = ifelse(cond,a,b), if cond, z=a; else, z=b; end, end

function y = diff_lagmatrix(x, d)
% Trend differencing via lagmatrix, order d > 0.
    if nargin<2, d = 1; end
    y = x(:);
    for k = 1:d
        L1 = lagmatrix(y, 1);
        y  = y(2:end) - L1(2:end);
    end
end

function y = seasdiff_lagmatrix(x, S, D)
% Seasonal differencing via lagmatrix, period S, order D > 0.
    if nargin<3, D = 1; end
    y = x(:);
    for k = 1:D
        LS = lagmatrix(y, S);
        y  = y(S+1:end) - LS(S+1:end); 
    end
end

%% ========================================================================
% End of script
% ========================================================================