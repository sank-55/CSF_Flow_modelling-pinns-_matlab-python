%% ================================================================
%          P vs Z plots & Do Vs Z at different criteria , with amplitude 
%         look in the cheb and parameters for better tuning 
%% ================================================================

clear all; close all; clc;
format long;
set(0, 'DefaultAxesFontSize', 12, 'DefaultLineLineWidth', 1.5);
set(groot, 'DefaultFigureColor', 'w');

%% ========== PART 1: PHYSIOLOGICAL PARAMETERS ==========
fprintf('=== PHYSIOLOGICAL PARAMETERS ===\n');

% Dimensional parameters (from paper)
param.rho = 1000;           % Fluid density [kg/m³]
param.mu = 0.0009;          % Dynamic viscosity [Pa·s]
param.omega = 2*pi*1.0;     % Angular frequency [rad/s]
param.b = 10e-6;            % Film thickness [m]
param.lambda = 0.1;         % Wavelength [m]

% Non-dimensional parameters (Table 1 ranges)
param.epsilon_val = param.b / param.lambda;
param.Re_val = param.epsilon_val * param.rho * param.omega * param.b^2 / param.mu;
param.R1_val = 10;          % From Figure 6: R1 = 10
param.L = 20;               % From Figure 6: L = 20

% Standard values from paper
param.Pa = 1e-3;            % Inlet pressure
param.Pb = 0;               % Outlet pressure
param.Pe = 0;               % Reference pressure

fprintf('Non-dimensional parameters:\n');
fprintf('ε = %.3e, Re = %.3e\n', param.epsilon_val, param.Re_val);
fprintf('R1 = %.1f, L = %d\n', param.R1_val, param.L);

%% ========== PART 2: ANALYTICAL SOLUTION FOR VALIDATION ==========
function P_analytical = analytical_solution_exact(Z, param, Pa, Pb, Pe)
    % Analytical solution for H=0, Ee→∞ case (Equation 16)
    
    R1 = param.R1_val;
    Me = param.Me_val;
    L = max(Z);
    
    % Compute B0 from equation (38a)
    if R1 > 0
        alpha_inf = (R1^2 - (R1+1)^2) / (4*(log(R1+1) - log(R1)));
        beta_inf = -R1^2/4 - alpha_inf * log(R1);
        
        term1 = alpha_inf/4 * ((R1+1)*(2*log(R1+1)-1) - R1^2/(R1+1)*(2*log(R1)-1));
        term2 = beta_inf/2 * ((R1+1) - R1^2/(R1+1));
        term3 = 1/16 * ((R1+1)^3 - R1^4/(R1+1));
        
        B0 = term1 + term2 + term3;
        B = Me / B0;
        
        % Ensure B is negative (|B| = -B)
        if B > 0
            B = -B;
        end
        
        sqrtB = sqrt(-B);
        
        % Solve for coefficients from boundary conditions
        A = [1, 1; exp(sqrtB*L), exp(-sqrtB*L)];
        b = [Pa - Pe; Pb - Pe];
        gamma = A \ b;
        
        P_analytical = Pe + gamma(1)*exp(sqrtB*Z) + gamma(2)*exp(-sqrtB*Z);
    else
        P_analytical = zeros(size(Z));
    end
end

%% ========== PART 3: CHEBYSHEV DIFFERENTIATION ==========
function [D, D2, x] = cheb(N, a, b)
    % Compute Chebyshev differentiation matrices
    if N == 0
        D = 0; D2 = 0; x = (a+b)/2;
        return
    end
    
    % Chebyshev points in [-1,1]
    x_cheb = -cos(pi*(0:N)/N)';
    
    % Differentiation matrix
    c = [2; ones(N-1,1); 2] .* (-1).^(0:N)';
    X = repmat(x_cheb, 1, N+1);
    dX = X - X';
    D = (c*(1./c)') ./ (dX + eye(N+1));
    D = D - diag(sum(D,2));
    
    % Map to [a,b]
    x = (b-a)/2 * x_cheb + (a+b)/2;
    D = 2/(b-a) * D;
    
    % Second derivative
    D2 = D^2;
end

%% ========== PART 4: NUMERICAL SOLVER FOR EQN (10) ==========
function [P0, D0, Ue, Z, T] = solve_perivascular_flow(param, Pa, Pb, Pe, H_amp, Ee, Me, L, T_final)
    % Solve equation (10) numerically
    
    % Discretization parameters
    N = 800;                 % Number of Chebyshev points
    dt = 0.001;             % Time step
    
    % Spatial discretization
    [Dz, D2z, Z] = cheb(N, 0, L);
    n = length(Z);
    
    % Time discretization
    T = 0:dt:T_final;
    nt = length(T);
    
    % Initialize
    P0 = zeros(n, nt);
    D0 = zeros(n, nt);
    Ue = zeros(n, nt);
    
    % Initial condition (linear pressure)
    P0(:,1) = Pa + (Pb - Pa) * Z / L;
    D0(:,1) = (P0(:,1) - Pe) / Ee;
    Ue(:,1) = Me * P0(:,1);
    
    % Main time loop
    for k = 1:nt-1
        t = T(k);
        
        % Current state
        P = P0(:,k);
        D = D0(:,k);
        
        % Peristaltic wave
        H = H_amp * sin(2*pi*(Z - t));
        dH_dt = -2*pi * H_amp * cos(2*pi*(Z - t));
        
        % Compute geometric parameters
        R1 = param.R1_val;
        R_inner = R1 + H;
        R_outer = R1 + 1 + D;
        
        % Avoid division by zero
        R_inner(R_inner <= 0) = 1e-10;
        R_outer(R_outer <= 0) = 1e-10;
        
        % Compute alpha and beta (Equations 37d,e)
        alpha = ((R_inner).^2 - (R_outer).^2) ./ (4 * (log(R_outer) - log(R_inner)));
        beta = -(R_inner).^2/4 - alpha .* log(R_inner);
        
        % Compute A0, A1, A2 coefficients (Equations 37a-c)
        term1 = (R_outer .* (2*log(R_outer) - 1) - (R_inner).^2 ./ R_outer .* (2*log(R_inner) - 1));
        term2 = (R_outer - (R_inner).^2 ./ R_outer);
        term3 = ((R_outer).^3 - (R_inner).^4 ./ R_outer)/16;
        
        A0_coeff = Ee/4 * alpha .* term1 + Ee/2 * beta .* term2 + Ee * term3;
        
        % Compute derivatives for A1
        dalpha_dZ = Dz * alpha;
        dbeta_dZ = Dz * beta;
        A1_coeff = Ee/4 * dalpha_dZ .* term1 + Ee/2 * dbeta_dZ .* term2;
        
        A2_coeff = Ee * Me * ones(size(A0_coeff));
        
        % Source term
        S = Ee * (R1 + H) ./ (R1 + 1 + D) .* dH_dt + Ee * Me * Pe;
        
        % Build linear system for implicit Euler
        I = speye(n);
        A0_diag = spdiags(A0_coeff, 0, n, n);
        A1_diag = spdiags(A1_coeff, 0, n, n);
        A2_diag = spdiags(A2_coeff, 0, n, n);
        
        M = I/dt + A0_diag*D2z + A1_diag*Dz + A2_diag;
        rhs = P/dt + S;
        
        % Boundary conditions
        M(1,:) = 0; M(1,1) = 1; rhs(1) = Pa;
        M(end,:) = 0; M(end,end) = 1; rhs(end) = Pb;
        
        % Solve
        P0(:,k+1) = M \ rhs;
        
        % Update deformation and velocity
        D0(:,k+1) = (P0(:,k+1) - Pe) / Ee;
        Ue(:,k+1) = Me * P0(:,k+1);
    end
end

%% ========== PART 5: FIGURE 6 - A0 AND A1 COEFFICIENTS ==========
fprintf('\n=== FIGURE 6: Computing A0 and A1 coefficients ===\n');

% Parameters from Figure 6
R1 = 10;
L = 20;
T_final = 100;  % Run for 100 time units

% Parameter ranges
Me_vals = [0.1, 0.2, 0.5, 1, 2, 5];
H_vals = [0, 0.05, 0.1, 0.15, 0.2];
Ee_cases = [0.01, 0.1, 1];

% Markers for different Me values
markers = {'o', 's', 'd', '^', 'v', '>'};
colors = lines(length(Me_vals));

% Store results
A0_results = cell(length(Ee_cases), 1);
A1_results = cell(length(Ee_cases), 1);

% Create figure
figure('Position', [100, 100, 1400, 1000]);

for e_idx = 1:length(Ee_cases)
    Ee = Ee_cases(e_idx);
    
    fprintf('  Processing Ee = %.2f...\n', Ee);
    
    A0_mat = zeros(length(Me_vals), length(H_vals));
    A1_mat = zeros(length(Me_vals), length(H_vals));
    
    for m_idx = 1:length(Me_vals)
        Me = Me_vals(m_idx);
        
        for h_idx = 1:length(H_vals)
            H_amp = H_vals(h_idx);
            
            fprintf('    Me = %.1f, H = %.2f...\n', Me, H_amp);
            
            % Solve for this parameter set
            param.R1_val = R1;
            [P0, D0, ~, Z, T] = solve_perivascular_flow(...
                param, param.Pa, param.Pb, param.Pe, H_amp, Ee, Me, L, T_final);
            
            % Time average over last half
            n_avg = floor(0.5 * length(T));
            D0_avg = mean(D0(:, end-n_avg+1:end), 2);
            
            % Linear fit in bulk region (Z in [5, L-5])
            bulk_idx = (Z >= 5) & (Z <= L-5);
            if sum(bulk_idx) > 1
                % Fit: D0_avg = A0 + A1*Z
                p = polyfit(Z(bulk_idx), D0_avg(bulk_idx), 1);
                A0_mat(m_idx, h_idx) = p(2);  % Intercept
                A1_mat(m_idx, h_idx) = p(1);  % Slope
            else
                A0_mat(m_idx, h_idx) = 0;
                A1_mat(m_idx, h_idx) = 0;
            end
        end
    end
    
    A0_results{e_idx} = A0_mat;
    A1_results{e_idx} = A1_mat;
    
    % Plot A0 (left column)
    subplot(3, 2, 2*e_idx-1);
    hold on; grid on;
    
    for m_idx = 1:length(Me_vals)
        plot(H_vals, A0_mat(m_idx, :), 'Marker', markers{m_idx}, ...
            'LineWidth', 1.5, 'MarkerSize', 8, 'Color', colors(m_idx,:));
    end
    
    title(sprintf('A_0 for %s Brain Tissue', get_tissue_name(e_idx)));
    xlabel('Peristaltic wave amplitude, H');
    ylabel('A_0 coefficient');
    set(gca, 'FontSize', 10, 'FontWeight', 'bold');
    
    % Plot A1 (right column)
    subplot(3, 2, 2*e_idx);
    hold on; grid on;
    
    for m_idx = 1:length(Me_vals)
        plot(H_vals, A1_mat(m_idx, :), 'Marker', markers{m_idx}, ...
            'LineWidth', 1.5, 'MarkerSize', 8, 'Color', colors(m_idx,:));
    end
    
    title(sprintf('A_1 for %s Brain Tissue', get_tissue_name(e_idx)));
    xlabel('Peristaltic wave amplitude, H');
    ylabel('A_1 coefficient');
    set(gca, 'FontSize', 10, 'FontWeight', 'bold');
end

% Add legend to first subplot
subplot(3,2,1);
legend(cellstr(num2str(Me_vals', 'M_e=%.1f')), 'Location', 'best', 'FontSize', 8);

% Helper function for tissue names
function name = get_tissue_name(idx)
    switch idx
        case 1
            name = 'Soft (E_e=0.01)';
        case 2
            name = 'Medium-Stiff (E_e=0.1)';
        case 3
            name = 'Rigid (E_e=1)';
        otherwise
            name = 'Unknown';
    end
end

%% ========== PART 6: THROUGH-FLOW VELOCITIES (FIGURES 7-9) ==========
fprintf('\n=== FIGURES 7-9: Through-flow velocities ===\n');

% Parameters
Me_selected = [0.1, 0.5, 1, 2, 5];
H_vals_flow = [0, 0.05, 0.1, 0.15, 0.2];
H_colors = {'k', 'b', 'r', 'g', 'c'};

% Create figure
figure('Position', [50, 50, 1500, 1000]);

for e_idx = 1:length(Ee_cases)
    Ee = Ee_cases(e_idx);
    
    for m_idx = 1:length(Me_selected)
        Me = Me_selected(m_idx);
        
        subplot(3, length(Me_selected), (e_idx-1)*length(Me_selected) + m_idx);
        hold on;
        
        for h_idx = 1:length(H_vals_flow)
            H_amp = H_vals_flow(h_idx);
            
            % Solve
            param.R1_val = R1;
            [P0, ~, Ue, Z, T] = solve_perivascular_flow(...
                param, param.Pa, param.Pb, param.Pe, H_amp, Ee, Me, L, T_final);
            
            % Time average
            n_avg = floor(0.5 * length(T));
            Ue_avg = mean(Ue(:, end-n_avg+1:end), 2);
            
            % Plot
            plot(Z, Ue_avg, '-', 'Color', H_colors{h_idx}, ...
                'LineWidth', 1.5, 'DisplayName', sprintf('H=%.2f', H_amp));
        end
        
        % Format subplot
        if e_idx == 1
            title(sprintf('M_e = %.1f', Me), 'FontWeight', 'bold');
        end
        
        if m_idx == 1
            ylabel('⟨U_e⟩', 'FontWeight', 'bold');
            text(-0.3, 0.5, get_tissue_name(e_idx), 'Units', 'normalized', ...
                'Rotation', 90, 'VerticalAlignment', 'middle', ...
                'FontSize', 11, 'FontWeight', 'bold');
        end
        
        if e_idx == length(Ee_cases)
            xlabel('Z', 'FontWeight', 'bold');
        end
        
        grid on;
        set(gca, 'FontSize', 9, 'FontWeight', 'bold');
        
        % Add legend to first subplot
        if m_idx == 1 && e_idx == 1
            legend('Location', 'best', 'FontSize', 8);
        end
    end
end

%% ========== PART 7: FIGURE 5 - LENGTH AND CURVATURE EFFECTS ==========
fprintf('\n=== FIGURE 5: Length and curvature effects ===\n');

% Fixed parameters for Figure 5
param_fig5 = param;
param_fig5.Me_val = 0.5;
param_fig5.Ee_val = 0.1;
param_fig5.H_amp = 0.0;

figure('Position', [100, 100, 1000, 800]);

% Subplot 1: Effect of length
subplot(2,1,1);
L_vals = [2, 5, 10, 20];
line_styles = {':', '-.', '--', '-'};
colors_L = {'r', 'g', 'b', 'k'};

for i = 1:length(L_vals)
    L = L_vals(i);
    
    % Solve
    [~, D0, ~, Z, T] = solve_perivascular_flow(...
        param_fig5, param_fig5.Pa, param_fig5.Pb, param_fig5.Pe, ...
        param_fig5.H_amp, param_fig5.Ee_val, param_fig5.Me_val, L, 50);
    
    % Time average
    n_avg = floor(0.5 * length(T));
    D0_avg = mean(D0(:, end-n_avg+1:end), 2);
    
    plot(Z, D0_avg, line_styles{i}, 'Color', colors_L{i}, ...
        'LineWidth', 2, 'DisplayName', sprintf('L = %d', L));
    hold on;
end

xlabel('Non-dimensional axial coordinate, Z');
ylabel('Average deformation, ⟨D_0⟩');
title('Effect of PVS Length (R_1=10, M_e=0.5, E_e=0.1, H=0.2)');
legend('Location', 'best');
grid on;
set(gca, 'FontSize', 11, 'FontWeight', 'bold');

% Subplot 2: Effect of curvature
subplot(2,1,2);
R1_vals = [10, 100, 1000];
line_styles_R = {'-', '--', '-.'};
colors_R = {'b', 'r', 'g'};

for i = 1:length(R1_vals)
    param_temp = param_fig5;
    param_temp.R1_val = R1_vals(i);
    
    % Solve
    [~, D0, ~, Z, T] = solve_perivascular_flow(...
        param_temp, param_temp.Pa, param_temp.Pb, param_temp.Pe, ...
        param_temp.H_amp, param_temp.Ee_val, param_temp.Me_val, 20, 50);
    
    % Time average
    n_avg = floor(0.5 * length(T));
    D0_avg = mean(D0(:, end-n_avg+1:end), 2);
    
    plot(Z, D0_avg, line_styles_R{i}, 'Color', colors_R{i}, ...
        'LineWidth', 2, 'DisplayName', sprintf('R_1 = %d', R1_vals(i)));
    hold on;
end

xlabel('Non-dimensional axial coordinate, Z');
ylabel('Average deformation, ⟨D_0⟩');
title('Effect of Curvature (L=20, M_e=0.5, E_e=0.1, H=0.2)');
legend('Location', 'best');
grid on;
set(gca, 'FontSize', 11, 'FontWeight', 'bold');
