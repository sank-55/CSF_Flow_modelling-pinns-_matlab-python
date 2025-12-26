%% ===========================================================
%          radial component of the velocity ( axial component mag is reducued relatively to see effect of porous BC
%% ---------------------------------------=================


clear; close all; clc;

%% 1. Parameters (Romano et al. 2020)
L = 5; Tfin = 2; Pa = 5; Pb = 0; Pe = 2; Me = 1; Ee = 0.1;
Hbar = 0.1; R1 = 10;
A0 = 1.0; A1 = 0.1; A2 = Ee * Me; 

%% 2. Neural Network Setup & Training
layers = [featureInputLayer(2); fullyConnectedLayer(40); tanhLayer; ...
          fullyConnectedLayer(40); tanhLayer; fullyConnectedLayer(1)];
net = dlnetwork(layers);
ZT_coll = dlarray([L*rand(1, 1200); Tfin*rand(1, 1200)], 'CB');

learning_rate = 0.003; num_epochs = 200;
average = []; averageSq = []; iteration = 0;

fprintf('Training PINN...\n');
for epoch = 1:num_epochs
    iteration = iteration + 1;
    [loss, gradients] = dlfeval(@pinnLossFunction, net, ZT_coll, A0, A1, A2, L, Tfin, Pa, Pb, Pe, Hbar, Ee);
    [net.Learnables, average, averageSq] = adamupdate(net.Learnables, gradients, average, averageSq, iteration, learning_rate);
end

%% ---------------- FIGURE 1: PRESSURE FIELD ANALYSIS ----------------
num_vis = 60;
z_vis = linspace(0, L, num_vis); t_vis = linspace(0, Tfin, num_vis);
[ZZ, TT] = meshgrid(z_vis, t_vis);
P_pred = extractdata(forward(net, dlarray([ZZ(:)'; TT(:)'], 'CB')));
PP = reshape(P_pred, size(ZZ));

fig1 = figure('Name', 'Pressure Analysis', 'Color', 'w', 'Position', [100 100 900 400]);
% Panel A: Spatio-Temporal Map
subplot(1,2,1);
imagesc(z_vis, t_vis, PP); colorbar; colormap(jet);
xlabel('Axial Position Z'); ylabel('Time T'); 
title('Spatio-Temporal Pressure Field P_0', 'Interpreter', 'none');

% Panel B: Midpoint Time History
subplot(1,2,2);
plot(t_vis, PP(:, round(num_vis/2)), 'b', 'LineWidth', 2);
grid on; xlabel('Time T'); ylabel('Pressure P_0', 'Interpreter', 'none');
title('Pressure Response at Z = L/2', 'Interpreter', 'none');

%% ---------------- FIGURE 2: VELOCITY FIELD & STREAMLINES ----------------
T_snap = 0.5 * Tfin;
num_R = 25; num_Z = 60;
r_norm = linspace(0, 1, num_R)'; % Normalized radial grid

% --- STEP 1: Use Realistic Parameters for Thin-Film Physics ---
% Based on Table 1: Pa should be small (e.g., 1e-3) so D is < 1
Pa_real = 1e-3; Pe_real = 0; Ee_real = 0.01; Hbar_real = 0.1;

% Get Pressure and Gradient from trained PINN
z_vis = linspace(0, L, num_Z);
ZT_snap = dlarray([z_vis; repmat(T_snap, 1, num_Z)], 'CB');
[P_snap_dl, grads_dl] = dlfeval(@getGradients, net, ZT_snap);
P_val = extractdata(P_snap_dl) * (Pa_real/Pa); % Scale to realistic range
PZ_val = extractdata(grads_dl(1,:)) * (Pa_real/Pa);

% --- STEP 2: Map to Physical Coordinates ---
Z_phys = zeros(num_R, num_Z);
R_phys = zeros(num_R, num_Z);
W_phys = zeros(num_R, num_Z);
U_phys = zeros(num_R, num_Z);

H = Hbar_real * sin(2*pi*(z_vis - T_snap));
D = (P_val - Pe_real)/Ee_real;

for i = 1:num_Z
    % Real Physical Gap: from H to (1 + D)
    R_in = H(i); 
    R_out = 1 + D(i); 

    % Map the R grid points to the actual gap space
    current_R = R_in + r_norm * (R_out - R_in);
    R_phys(:, i) = current_R;
    Z_phys(:, i) = z_vis(i);

    % Axial Velocity W0 (Parabolic profile in the gap)
    axial_profile = (current_R - R_in) .* (R_out - current_R);
    W_phys(:, i) = -0.5 * PZ_val(i) * axial_profile;

    % Radial Velocity U0 (Linear transition for leak)
    U_leak = Me * (P_val(i) - Pe_real);
    U_phys(:, i) = U_leak * (current_R - R_in) / (R_out - R_in);
end


V_magnitude = sqrt(W_phys.^2 + U_phys.^2) + 1e-12;
W_norm = W_phys ./ V_magnitude;
U_norm = U_phys ./ V_magnitude;

% 2. Use 'axis equal' to ensure 1 unit on X looks the same as 1 unit on Y
figure('Name', 'True Aspect Ratio Flow');
hold on;
quiver(Z_phys, R_phys, W_norm, U_norm, 0.5, 'Color', [0.5 0.5 0.5]);
axis equal; % CRITICAL: Makes horizontal and vertical scales identical


% --- STEP 3: Final Corrected Plot ---
figure('Name', 'Physically Correct PVS Flow', 'Color', 'w');
hold on;

% 1. Fill the Perivascular Space (Blue background)
fill([z_vis, fliplr(z_vis)], [H, fliplr(1+D)], [0.9 0.9 1], 'EdgeColor', 'none', 'DisplayName', 'PVS Gap');

% 2. Plot the Artery Wall and Glial Boundary
plot(z_vis, H, 'k', 'LineWidth', 2.5, 'DisplayName', 'Artery Wall (H)');
plot(z_vis, 1+D, 'r', 'LineWidth', 2.5, 'DisplayName', 'Glial Boundary (D)');

% 3. Plot Streamlines in the Mapped Space
streamline(Z_phys, R_phys, W_phys, U_phys, zeros(1,6), linspace(0.1, 0.9, 6));
quiver(Z_phys, R_phys, W_phys, U_phys, 1.2, 'Color', [0.4 0.4 0.4], 'AutoScale', 'on');



title('Corrected Perivascular Flow: Streamlines mapped to Gap', 'Interpreter', 'none');
xlabel('Axial Position Z'); ylabel('Radial Position R (Nondimensional)');
ylim([min(H)-0.5, max(1+D)+0.5]); grid on;
legend('Location', 'northeastoutside', 'Interpreter', 'none');



%% --- Helper Functions ---
function [loss, gradients] = pinnLossFunction(net, ZT, A0, A1, A2, L, Tfin, Pa, Pb, Pe, Hbar, Ee)
    P = forward(net, ZT);
    grads = dlgradient(sum(P), ZT, 'EnableHigherDerivatives', true);
    P_Z = grads(1,:); P_T = grads(2,:);
    P_ZZ = dlgradient(sum(P_Z), ZT, 'EnableHigherDerivatives', true);
    % Physics Residual (Equation 10)
    Forcing = Ee*(Hbar*-2*pi*cos(2*pi*(ZT(1,:)-ZT(2,:))));
    res = P_ZZ(1,:) + A0*P_T + A1*P_Z + A2*(P - Pe) - Forcing;

    % BC/IC Enforcement
    T_pts = Tfin * rand(1, 40);
    Z_0 = dlarray([zeros(1,40); T_pts], 'CB'); Z_L = dlarray([L*ones(1,40); T_pts], 'CB');
    Z_init = dlarray([L*rand(1,40); zeros(1,40)], 'CB');
    loss_bc = mean((forward(net,Z_0)-Pa).^2) + mean((forward(net,Z_L)-Pb).^2);
    loss_ic = mean((forward(net,Z_init)-(Pa + (Pb-Pa)*Z_init(1,:)/L)).^2);

    loss = mean(res.^2) + 30*loss_bc + 10*loss_ic;
    gradients = dlgradient(loss, net.Learnables);
end

function [P, grads] = getGradients(net, ZT)
    P = forward(net, ZT);
    grads = dlgradient(sum(P), ZT);
end
