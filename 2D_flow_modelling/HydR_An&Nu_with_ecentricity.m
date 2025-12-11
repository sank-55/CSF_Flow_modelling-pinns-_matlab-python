%%---------------------------------------------------------------
%      NUMERICAL VS ANALYTICAL Dimention less R vs Normalized Ecentricty plot 
%      also contain R_norm varing with ecentricty plot with comparision of the Analytical and Numerical Values
%
%-----------------------------------------------------------------------------

clear; close all; clc;

mu = 1e-3;           % viscosity [Pa.s]
G  = 1.0;            % magnitude of -dp/dz (set to 1 for dimensionless R=1/Q)

a_phys = 10e-6;      % inner radius [m] (choose any positive scale)
K_values = [0.2, 0.5, 1, 2, 3];    % area ratio values to plot (like paper)
Ne = 200;            % number of normalized eccentricity samples (0..~1)
eps_max_plot = 0.98; % max normalized eccentricity to evaluate (avoid exact 1)
Nterms_max = 900000;    % maximum number of series terms (increase if needed)
tol_term = 1e-14;    % adaptive truncation tolerance for series

doNumeric = true;    % overlay numeric FD markers for circular-outer case (slower)
Nx = 400; Ny = 400;  % numeric grid (if doNumeric)

%% ---------------- Prepare ecc grid ----------------
e_hat = linspace(0, eps_max_plot, Ne);   % e_hat = c/(b-a) from 0..~1

%% Preallocate containers
nK = numel(K_values);
R_analytic_dimless = nan(nK, Ne);  % stores a^4 R / mu
R_numeric_dimless  = nan(nK, Ne);  % numeric overlay (if computed)
% Storing the R_norm Value for both
R_norm_an = nan(1,Ne);
R_norm_nu = nan(1,Ne);
fprintf('Computing analytic curves for K = [%s]\n', num2str(K_values));

%% ---------------- Analytic evaluation (White series) ----------------
for ik = 1:nK
    K = K_values(ik);
    a = a_phys;
    b = a * sqrt(K + 1);      % b consistent with area ratio K
    cmax = b - a;             % tangent offset
    alpha = b / a;
    for ie = 1:Ne
        ehat = e_hat(ie);
        c = min(ehat * cmax, 0.9999*cmax);  % dimensional offset
        % compute Q using White's A,B,M series with adaptive truncation
        Q = analytic_Q_white_adaptive(a, b, c, G, mu, Nterms_max, tol_term);
        fprintf(' value of resistance is %.3e \n',G/Q);
        R = G / Q;
        R_analytic_dimless(ik, ie) = (a^4 / mu) * R;  % dimensionless as in paper
    end
    fprintf('  analytic done for K=%.3g (b/a=%.3g)\n', K, alpha);
end

%% ---------------- Optional numeric FD overlay (circular outer) ----------------
if doNumeric
    fprintf('Running numeric FD (circular outer) overlay — this may take some time...\n');
    % prepare common grid sized to largest b
    b_max = a_phys * sqrt(max(K_values) + 1);
    xpad = 1.05;
    x = linspace(-b_max*xpad, b_max*xpad, Nx);
    y = linspace(-b_max*xpad, b_max*xpad, Ny);
    [X,Y] = meshgrid(x,y);
    dx = x(2)-x(1); dy = y(2)-y(1);
    idx = reshape(1:Nx*Ny, Ny, Nx);

    for ik = 1:nK
        K = K_values(ik);
        a = a_phys;
        b = a * sqrt(K + 1);
        cmax = b - a;
        for ie = 1:Ne
            ehat = e_hat(ie);
            c = min(ehat * cmax, 0.9999*cmax);
            inner_mask = ((X - c).^2 + Y.^2) <= a^2;
            outer_mask = (X.^2 + Y.^2) <= b^2;
            fluid_mask = outer_mask & (~inner_mask);
            nF = nnz(fluid_mask);
            if nF < 30
                R_numeric_dimless(ik,ie) = NaN;
                continue;
            end
            % map fluid nodes
            map = zeros(size(idx)); map(fluid_mask) = 1:nF;
            % assemble sparse Laplacian (5-point)
            maxE = nF*5; I = zeros(maxE,1); J = zeros(maxE,1); V = zeros(maxE,1); cnt = 0;
            bvec = (1/mu)*(-G) * ones(nF,1);
            [NyA,NxA] = size(map);
            for jj = 1:NyA
                for ii = 1:NxA
                    nid = map(jj,ii);
                    if nid==0, continue; end
                    cnt = cnt+1; I(cnt)=nid; J(cnt)=nid; V(cnt)= -2/dx^2 - 2/dy^2;
                    if ii>1, nL = map(jj,ii-1); if nL>0, cnt=cnt+1; I(cnt)=nid; J(cnt)=nL; V(cnt)=1/dx^2; end; end
                    if ii<NxA, nR = map(jj,ii+1); if nR>0, cnt=cnt+1; I(cnt)=nid; J(cnt)=nR; V(cnt)=1/dx^2; end; end
                    if jj>1, nD = map(jj-1,ii); if nD>0, cnt=cnt+1; I(cnt)=nid; J(cnt)=nD; V(cnt)=1/dy^2; end; end
                    if jj<NyA, nU = map(jj+1,ii); if nU>0, cnt=cnt+1; I(cnt)=nid; J(cnt)=nU; V(cnt)=1/dy^2; end; end
                end
            end
            I = I(1:cnt); J = J(1:cnt); V = V(1:cnt);
            A = sparse(I,J,V,nF,nF); A_pos = -A; b_pos = -bvec;
            uvec = A_pos \ b_pos;
            U = zeros(size(X)); U(fluid_mask) = uvec;
            Qnum = sum(U(fluid_mask),'all') * dx * dy;
            Rnum = G / Qnum;
            R_numeric_dimless(ik,ie) = (a^4 / mu) * Rnum;
        end
        fprintf('  numeric K=%.3g done\n', K);
    end
end

%% ---------------- Plotting ----------------
% figure('Color','w','Position',[200 150 860 620]); hold on; box on;
% colors = parula(nK);
% for ik = 1:nK
%     plot(e_hat, R_analytic_dimless(ik,:), 'LineWidth', 1.8, 'Color', colors(ik,:));
%     if doNumeric
%         valid = ~isnan(R_numeric_dimless(ik,:));
%         plot(e_hat(valid), R_numeric_dimless(ik,valid), 'o', ...
%             'Color', colors(ik,:), 'MarkerFaceColor', colors(ik,:), 'MarkerSize', 4);
%     end
% end
% xlabel('Normalized eccentricity c/(b-a)','FontSize',13);
% ylabel('a^4 R / \mu  (dimensionless)','FontSize',13);
% title('Analytic eccentric annulus (White) — a^4 R/\mu vs normalized eccentricity','FontSize',14);
% legend(arrayfun(@(K)sprintf('K=%.2g',K),K_values,'UniformOutput',false),'Location','northeast');
% set(gca,'FontSize',12,'YScale','log','XLim',[0 1]);
% grid on;



figure('Color','w','Position',[200 150 860 620]); 
hold on; box on; grid on;

colors = parula(nK);

hAnalytic = gobjects(nK,1);
hNumeric  = gobjects(nK,1);

for ik = 1:nK
    % ----- Plot analytic curve (store handle for legend) -----
    hAnalytic(ik) = plot(e_hat, R_analytic_dimless(ik,:), ...
        'LineWidth', 1.8, 'Color', colors(ik,:));

    % ----- Plot numeric markers -----
    if doNumeric
        valid = ~isnan(R_numeric_dimless(ik,:));
        hNumeric(ik) = plot(e_hat(valid), R_numeric_dimless(ik,valid), 'o', ...
            'Color', colors(ik,:), ...
            'MarkerFaceColor', colors(ik,:), 'MarkerSize', 4);
    end
end

xlabel('Normalized eccentricity c/(b-a)');
ylabel('a^4 R / \mu  (dimensionless)');
title('Analytic (line) vs Numerical (o) — Hydraulic Resistance');

set(gca,'FontSize',12,'YScale','log','XLim',[0 1]);

% ---------------- Legend ----------------
legLabels = arrayfun(@(K)sprintf('K = %.2g',K), K_values, 'UniformOutput', false);

legendObjs = [hAnalytic(1); hNumeric(1)];   % representative styles
legendTxt  = {'Analytic (line)','Numerical (circle)'};

lgd1 = legend(legendObjs, legendTxt, 'Location','southwest');
set(lgd1,'FontSize',11);

% Second legend showing K values (color-coded)
lgd2 = legend(hAnalytic, legLabels, 'Location','northeast');
set(lgd2,'FontSize',11);
legend boxoff



figure(2);
% PLot the R_norm for both vs ecentricity case 
R0_nu = R_numeric_dimless(2,1);
R0_an = R_analytic_dimless(2,1);
plot(e_hat, R_numeric_dimless(2,:)/R0_nu,'-o','LineWidth',1.5); hold on;
plot(e_hat, R_analytic_dimless(2,:)/R0_an,'k-','LineWidth',1.5);

xlabel('Normalized eccentric shift(c/b-a)'); ylabel('Normalized R (R/R_{ecc=0})');
legend('Numerical','Analytical','Location','northeast');
title('Hydraulic resistance vs eccentricity (normalized)');
grid on;



fprintf('Done. Figure produced.\n');
%% ---------------- Supporting function: analytic_Q_white_adaptive ----------------
function Q = analytic_Q_white_adaptive(r1, r2, c, G, mu, Nmax, tol)
% Returns dimensional Q (m^2/s) for eccentric circular annulus using White form.
% Adaptive summation of series until term < tol (or reach Nmax).
    if c == 0
        alpha = r2 / r1;
        Q = (pi * G) / (8 * mu) * ( r2^4 - r1^4 - ( (r2^2 - r1^2)^2 ) / log(alpha) );
        return;
    end
    alpha = r2 / r1;
    eps = c / r1;   % dimensionless eccentricity
    F = (alpha^2 - 1 + eps^2) / (2 * eps);
    M = sqrt(F.^2 - alpha^2);
    A = 0.5 * log((F + M) / (F - M));
    B = 0.5 * log((F - eps + M) / (F - eps - M));
    % bracket base
    A0 = alpha^4 - 1;
    % adaptive series (White's form): S = sum_{n=1..inf} n * exp(-n*(B+A)) / sinh(n*(B-A))
    % prefactor = 8*eps^2*M^2
    pref = 8 * eps^2 * M^2;
    S = 0;
    % iterate terms until small
    for n = 1:Nmax
        denom = sinh(n*(B-A));
        if denom == 0
            term = 0;
        else
            term = n * exp(-n*(B + A)) / denom;
        end
        S = S + term;
        if abs(term) < tol * max(1, abs(S))
            break;
        end
    end
    % T1 = (4 eps^2 M^2) / (B-A)
    T1 = (4 * eps^2 * M^2) / (B - A);
    % full bracket
    bracket = A0 - T1 - pref * S;
    % dimensional Q (note factor r1^4)
    Q = (pi * G) / (8 * mu) * ( r1^4 * bracket );
    
end
