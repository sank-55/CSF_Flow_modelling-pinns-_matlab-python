# Matlab code 
## --------------------------------------------------------------------------------------------------
%% Lid-Driven Cavity Flow Simulation
% Solves incompressible Navier-Stokes equations using Chorin's Projection Method
% Finite Difference Method with explicit time stepping

clear; clc; close all;

%% Simulation Parameters
% Domain parameters
L = 1.0;                    % Domain length (m)
Nx = 41;                    % Number of grid points in x-direction
Ny = 41;                    % Number of grid points in y-direction

% Physical parameters
rho = 1.0;                  % Density (kg/m³)
nu = 0.1;                   % Kinematic viscosity (m²/s)
u_lid = 1.0;                % Lid velocity (m/s)

% Time stepping parameters
dt = 0.001;                 % Time step (s)
T_total = 0.5;              % Total simulation time (s)
n_steps = round(T_total/dt); % Number of time steps

% Pressure Poisson iteration parameters
max_pressure_iter = 50;     % Maximum pressure iterations
pressure_tolerance = 1e-6;  % Pressure convergence tolerance
omega = 1.2;                % SOR relaxation parameter

% Visualization parameters
plot_interval = 20;         % Plot every N steps
record_video = false;       % Record animation (set to true to record)

%% Grid Generation
dx = L/(Nx-1);              % Grid spacing in x-direction
dy = L/(Ny-1);              % Grid spacing in y-direction

% Create grid coordinates
x = linspace(0, L, Nx);
y = linspace(0, L, Ny);
[X, Y] = meshgrid(x, y);

% Initialize velocity and pressure fields
u = zeros(Ny, Nx);          % x-velocity (horizontal)
v = zeros(Ny, Nx);          % y-velocity (vertical)
p = zeros(Ny, Nx);          % Pressure field

% Apply boundary conditions
u(end, :) = u_lid;          % Moving lid (top boundary)

%% Finite Difference Operators
% Create derivative operators (second order central differences)

% Function for central difference in x-direction
central_diff_x = @(f) ([f(:,2:end) f(:,end)] - [f(:,1) f(:,1:end-1)])/(2*dx);
central_diff_y = @(f) ([f(2:end,:); f(end,:)] - [f(1,:); f(1:end-1,:)])/(2*dy);

% Function for Laplace operator (5-point stencil)
laplace = @(f) (...
    ([f(:,2:end) f(:,end)] - 2*f + [f(:,1) f(:,1:end-1)])/(dx^2) + ...
    ([f(2:end,:); f(end,:)] - 2*f + [f(1,:); f(1:end-1,:)])/(dy^2) ...
);

% Function for divergence
divergence = @(u, v) central_diff_x(u) + central_diff_y(v);

% Function for gradient
gradient_x = @(f) ([f(:,2:end) f(:,end)] - [f(:,1) f(:,1:end-1)])/(2*dx);
gradient_y = @(f) ([f(2:end,:); f(end,:)] - [f(1,:); f(1:end-1,:)])/(2*dy);

%% Stability Check
% CFL condition for explicit scheme
u_max = max(abs(u(:)));
v_max = max(abs(v(:)));
CFL = dt * max(u_max, v_max) * (1/dx + 1/dy);
viscous_limit = dt * nu * (1/dx^2 + 1/dy^2);

fprintf('=== Lid-Driven Cavity Simulation ===\n');
fprintf('Grid: %dx%d, dx = dy = %.4f\n', Nx, Ny, dx);
fprintf('Time step: dt = %.6f s\n', dt);
fprintf('Total time: T = %.2f s\n', T_total);
fprintf('CFL number: %.4f\n', CFL);
fprintf('Viscous limit: %.4f\n', viscous_limit);

% FIXED: Use & instead of && for array comparisons, and use scalar conditions
if CFL > 0.5
    warning('CFL condition may be violated: CFL = %.4f > 0.5', CFL);
end
if viscous_limit > 0.25
    warning('Viscous stability condition may be violated: limit = %.4f > 0.25', viscous_limit);
end

%% Pre-allocate for animation
if record_video
    video_file = VideoWriter('lid_driven_cavity.avi');
    video_file.FrameRate = 20;
    open(video_file);
end

%% Main Time Stepping Loop
fprintf('\nStarting simulation...\n');

% Create figure for real-time visualization
fig = figure('Position', [100, 100, 1400, 900]);
time = 0;

% Progress bar
fprintf('Progress: 0%%');
progress_step = floor(n_steps/10);

for n = 1:n_steps
    % Update time
    time = n * dt;
    
    % Store old velocity for time advancement
    u_old = u;
    v_old = v;
    
    %% Step 1: Predictor Step - Solve Momentum Equation (without pressure)
    % Compute convection terms (using upwind for stability)
    
    % Upwind scheme for u-convection
    u_e = 0.5*(u + abs(u));
    u_w = 0.5*(u - abs(u));
    v_n = 0.5*(v + abs(v));
    v_s = 0.5*(v - abs(v));
    
    % Convection terms
    conv_u = (u_e.*(u - [u(:,1) u(:,1:end-1)])/dx + ...
              u_w.*([u(:,2:end) u(:,end)] - u)/dx + ...
              v_n.*(u - [u(1,:); u(1:end-1,:)])/dy + ...
              v_s.*([u(2:end,:); u(end,:)] - u)/dy);
    
    conv_v = (u_e.*(v - [v(:,1) v(:,1:end-1)])/dx + ...
              u_w.*([v(:,2:end) v(:,end)] - v)/dx + ...
              v_n.*(v - [v(1,:); v(1:end-1,:)])/dy + ...
              v_s.*([v(2:end,:); v(end,:)] - v)/dy);
    
    % Compute diffusion terms
    diff_u = nu * laplace(u);
    diff_v = nu * laplace(v);
    
    % Compute tentative velocity (Euler forward)
    u_tilde = u_old - dt * conv_u + dt * diff_u;
    v_tilde = v_old - dt * conv_v + dt * diff_v;
    
    % Apply velocity boundary conditions for tentative velocity
    % Top wall (moving lid)
    u_tilde(end, :) = u_lid;
    v_tilde(end, :) = 0;
    
    % Bottom wall (no-slip)
    u_tilde(1, :) = 0;
    v_tilde(1, :) = 0;
    
    % Left wall (no-slip)
    u_tilde(:, 1) = 0;
    v_tilde(:, 1) = 0;
    
    % Right wall (no-slip)
    u_tilde(:, end) = 0;
    v_tilde(:, end) = 0;
    
    %% Step 2: Pressure Correction - Solve Poisson Equation
    % Compute divergence of tentative velocity field
    div = divergence(u_tilde, v_tilde);
    
    % Right-hand side for pressure Poisson equation
    rhs = (rho/dt) * div;
    
    % Solve pressure Poisson equation using SOR method
    p_old = p;
    converged = false;
    
    for iter = 1:max_pressure_iter
        p_new = p_old;
        
        % Interior points (SOR iteration)
        for i = 2:Ny-1
            for j = 2:Nx-1
                p_new(i,j) = (1-omega)*p_old(i,j) + omega/4 * (...
                    p_new(i-1,j) + p_old(i+1,j) + ...
                    p_new(i,j-1) + p_old(i,j+1) - ...
                    dx*dy * rhs(i,j));
            end
        end
        
        % Boundary conditions for pressure
        % Top wall: Dirichlet (p = 0 for incompressibility)
        p_new(end, :) = 0;
        
        % Bottom, left, right walls: Neumann (dp/dn = 0)
        p_new(1, :) = p_new(2, :);          % Bottom
        p_new(:, 1) = p_new(:, 2);          % Left
        p_new(:, end) = p_new(:, end-1);    % Right
        
        % Check convergence
        residual = max(abs(p_new(:) - p_old(:)));
        if residual < pressure_tolerance
            converged = true;
            break;
        end
        
        p_old = p_new;
    end
    
    if ~converged && mod(n, 100) == 0
        fprintf('Pressure solver did not converge at step %d, residual = %.2e\n', n, residual);
    end
    
    p = p_new;
    
    %% Step 3: Corrector Step - Update Velocity with Pressure Gradient
    % Compute pressure gradients
    dp_dx = gradient_x(p);
    dp_dy = gradient_y(p);
    
    % Correct velocities
    u = u_tilde - (dt/rho) * dp_dx;
    v = v_tilde - (dt/rho) * dp_dy;
    
    % Apply final velocity boundary conditions
    % Top wall (moving lid)
    u(end, :) = u_lid;
    v(end, :) = 0;
    
    % Bottom wall (no-slip)
    u(1, :) = 0;
    v(1, :) = 0;
    
    % Left wall (no-slip)
    u(:, 1) = 0;
    v(:, 1) = 0;
    
    % Right wall (no-slip)
    u(:, end) = 0;
    v(:, end) = 0;
    
    %% Compute Diagnostics
    % Compute vorticity
    vorticity = gradient_x(v) - gradient_y(u);
    
    % Compute divergence of final velocity field (should be near zero)
    final_div = divergence(u, v);
    max_div = max(abs(final_div(:)));
    
    % Compute kinetic energy
    kinetic_energy = 0.5 * rho * sum(sum(u.^2 + v.^2)) * dx * dy;
    
    %% Visualization
    if mod(n, plot_interval) == 0 || n == n_steps
        % Update progress
        if mod(n, progress_step) == 0
            progress = floor(100 * n / n_steps);
            fprintf('\b\b\b\b%3d%%', progress);
        end
        
        % Clear figure
        clf;
        
        % Subplot 1: Velocity magnitude with streamlines
        subplot(2, 3, [1, 2, 4, 5]);
        hold on;
        
        % Compute velocity magnitude
        vel_mag = sqrt(u.^2 + v.^2);
        
        % Plot velocity magnitude contour
        contourf(X, Y, vel_mag, 20, 'LineStyle', 'none');
        colorbar;
        colormap(jet);
        caxis([0, u_lid]);
        
        % Plot streamlines
        startx = 0.1:0.1:0.9;
        starty = 0.1:0.1:0.9;
        [startX, startY] = meshgrid(startx, starty);
        streamline(X, Y, u, v, startX(:), startY(:));
        
        % Plot velocity vectors (every 2nd point for clarity)
        quiver_skip = 2;
        quiver(X(1:quiver_skip:end, 1:quiver_skip:end), ...
               Y(1:quiver_skip:end, 1:quiver_skip:end), ...
               u(1:quiver_skip:end, 1:quiver_skip:end), ...
               v(1:quiver_skip:end, 1:quiver_skip:end), ...
               'k', 'LineWidth', 0.5);
        
        xlabel('x (m)');
        ylabel('y (m)');
        title(sprintf('Lid-Driven Cavity Flow\nTime = %.3f s, Re = %.1f', ...
                      time, u_lid*L/nu));
        axis equal tight;
        xlim([0 L]);
        ylim([0 L]);
        grid on;
        
        % Subplot 2: Pressure field
        subplot(2, 3, 3);
        contourf(X, Y, p, 20, 'LineStyle', 'none');
        colorbar;
        colormap(parula);
        xlabel('x (m)');
        ylabel('y (m)');
        title('Pressure Field');
        axis equal tight;
        
        % Subplot 3: Vorticity field
        subplot(2, 3, 6);
        contourf(X, Y, vorticity, 20, 'LineStyle', 'none');
        colorbar;
        colormap(cool);
        xlabel('x (m)');
        ylabel('y (m)');
        title('Vorticity Field');
        axis equal tight;
        
        drawnow;
        
        % Record frame for video
        if record_video
            frame = getframe(fig);
            writeVideo(video_file, frame);
        end
    end
end

fprintf('\b\b\b\b100%%\n');

% Close video if recording
if record_video
    close(video_file);
    fprintf('Video saved as "lid_driven_cavity.avi"\n');
end

%% Post-processing and Analysis
fprintf('\n=== Simulation Complete ===\n');
fprintf('Final time: %.3f s\n', time);
fprintf('Maximum divergence: %.2e\n', max_div);
fprintf('Total kinetic energy: %.4f J\n', kinetic_energy);

%% Create Comprehensive Analysis Figure
figure('Position', [50, 50, 1400, 800]);

% Plot 1: Velocity profiles at different x-locations
subplot(2, 3, 1);
hold on;
x_locations = [0.25, 0.5, 0.75];
colors = {'r', 'g', 'b'};
for idx = 1:length(x_locations)
    [~, x_idx] = min(abs(x - x_locations(idx)));
    plot(u(:, x_idx), y, 'Color', colors{idx}, 'LineWidth', 2);
end
xlabel('u-velocity (m/s)');
ylabel('y (m)');
title('Horizontal Velocity Profiles');
legend(cellstr(num2str(x_locations', 'x = %.2f')), 'Location', 'best');
grid on;

% Plot 2: Velocity profiles at different y-locations
subplot(2, 3, 4);
hold on;
y_locations = [0.25, 0.5, 0.75];
for idx = 1:length(y_locations)
    [~, y_idx] = min(abs(y - y_locations(idx)));
    plot(x, v(y_idx, :), 'Color', colors{idx}, 'LineWidth', 2);
end
xlabel('x (m)');
ylabel('v-velocity (m/s)');
title('Vertical Velocity Profiles');
legend(cellstr(num2str(y_locations', 'y = %.2f')), 'Location', 'best');
grid on;

% Plot 3: Streamfunction contours
subplot(2, 3, 2);
% Compute streamfunction by solving Poisson equation
psi = compute_streamfunction(u, v, dx, dy, Nx, Ny);
contourf(X, Y, psi, 30, 'LineStyle', 'none');
colorbar;
colormap(jet);
xlabel('x (m)');
ylabel('y (m)');
title('Streamfunction');
axis equal tight;

% Plot 4: Vorticity magnitude
subplot(2, 3, 5);
contourf(X, Y, abs(vorticity), 30, 'LineStyle', 'none');
colorbar;
colormap(hot);
xlabel('x (m)');
ylabel('y (m)');
title('Vorticity Magnitude');
axis equal tight;

% Plot 5: Pathlines (particle tracking)
subplot(2, 3, 3);
hold on;
% Initialize particles
n_particles = 20;
particle_x = rand(1, n_particles) * L;
particle_y = rand(1, n_particles) * L;

% Track particles for a short time
particle_history = cell(n_particles, 1);
for p_idx = 1:n_particles
    particle_history{p_idx} = [particle_x(p_idx), particle_y(p_idx)];
end

for step = 1:100
    % Interpolate velocities at particle positions
    u_particles = interp2(X, Y, u, particle_x, particle_y, 'linear', 0);
    v_particles = interp2(X, Y, v, particle_x, particle_y, 'linear', 0);
    
    % Update particle positions (Euler integration)
    particle_x = particle_x + 0.01 * u_particles;
    particle_y = particle_y + 0.01 * v_particles;
    
    % Keep particles within domain
    particle_x = max(0, min(L, particle_x));
    particle_y = max(0, min(L, particle_y));
    
    % Store history
    for p_idx = 1:n_particles
        particle_history{p_idx} = [particle_history{p_idx}; ...
                                   particle_x(p_idx), particle_y(p_idx)];
    end
end

% Plot particle trajectories
for p_idx = 1:n_particles
    plot(particle_history{p_idx}(:,1), particle_history{p_idx}(:,2), ...
         'b-', 'LineWidth', 0.5);
end
scatter(particle_x, particle_y, 50, 'r', 'filled');
xlabel('x (m)');
ylabel('y (m)');
title('Particle Trajectories');
axis equal tight;
xlim([0 L]);
ylim([0 L]);
grid on;

% Plot 6: Centerline velocity comparison
subplot(2, 3, 6);
hold on;
% Vertical centerline velocity
center_x = round(Nx/2);
plot(u(:, center_x), y, 'b-', 'LineWidth', 2);

% Horizontal centerline velocity
center_y = round(Ny/2);
plot(x, v(center_y, :), 'r-', 'LineWidth', 2);

xlabel('Velocity (m/s)');
ylabel('Position (m)');
title('Centerline Velocities');
legend('u at x = L/2', 'v at y = L/2', 'Location', 'best');
grid on;

sgtitle(sprintf('Lid-Driven Cavity Flow Analysis (Re = %.0f)', u_lid*L/nu), ...
        'FontSize', 14, 'FontWeight', 'bold');

%% Display Flow Statistics
fprintf('\n=== Flow Statistics ===\n');
fprintf('Reynolds number: Re = %.2f\n', u_lid*L/nu);
fprintf('Maximum horizontal velocity: %.4f m/s\n', max(u(:)));
fprintf('Maximum vertical velocity: %.4f m/s\n', max(v(:)));
fprintf('Maximum pressure: %.4f Pa\n', max(p(:)));
fprintf('Minimum pressure: %.4f Pa\n', min(p(:)));
fprintf('Maximum vorticity: %.4f 1/s\n', max(abs(vorticity(:))));

% Compute circulation
circulation = sum(sum(vorticity)) * dx * dy;
fprintf('Total circulation: %.4f m²/s\n', circulation);

%% Helper Function for Streamfunction Calculation
function psi = compute_streamfunction(u, v, dx, dy, Nx, Ny)
    % Solve Poisson equation for streamfunction: ∇²ψ = -ω
    % where ω = ∂v/∂x - ∂u/∂y is vorticity
    
    % Initialize streamfunction
    psi = zeros(Ny, Nx);
    
    % Compute vorticity
    vort = zeros(Ny, Nx);
    for i = 2:Ny-1
        for j = 2:Nx-1
            vort(i,j) = (v(i,j+1) - v(i,j-1))/(2*dx) - ...
                        (u(i+1,j) - u(i-1,j))/(2*dy);
        end
    end
    
    % Solve Poisson equation using SOR
    max_iter = 1000;
    tolerance = 1e-6;
    omega = 1.8;
    
    for iter = 1:max_iter
        psi_old = psi;
        
        for i = 2:Ny-1
            for j = 2:Nx-1
                psi(i,j) = (1-omega)*psi_old(i,j) + omega/4 * (...
                    psi(i-1,j) + psi_old(i+1,j) + ...
                    psi(i,j-1) + psi_old(i,j+1) + ...
                    dx*dy * vort(i,j));
            end
        end
        
        % Boundary conditions (ψ = 0 on boundaries)
        psi(1,:) = 0;
        psi(end,:) = 0;
        psi(:,1) = 0;
        psi(:,end) = 0;
        
        % Check convergence
        residual = max(abs(psi(:) - psi_old(:)));
        if residual < tolerance
            break;
        end
    end
end

%% Create Additional Visualization: Velocity Field Animation Over Time
fprintf('\nCreating final visualization...\n');

figure('Position', [100, 100, 1200, 500]);

% Subplot 1: Final velocity field with streamlines
subplot(1, 2, 1);
vel_mag = sqrt(u.^2 + v.^2);
contourf(X, Y, vel_mag, 30, 'LineStyle', 'none');
hold on;
colorbar;
colormap(jet);
caxis([0 u_lid]);

% Plot streamlines
[startX, startY] = meshgrid(0.1:0.1:0.9, 0.1:0.1:0.9);
streamline(X, Y, u, v, startX(:), startY(:));

% Plot velocity vectors
quiver_skip = 3;
quiver(X(1:quiver_skip:end, 1:quiver_skip:end), ...
       Y(1:quiver_skip:end, 1:quiver_skip:end), ...
       u(1:quiver_skip:end, 1:quiver_skip:end), ...
       v(1:quiver_skip:end, 1:quiver_skip:end), ...
       'k', 'LineWidth', 1);

xlabel('x (m)');
ylabel('y (m)');
title(sprintf('Final Velocity Field (t = %.3f s)', time));
axis equal tight;
grid on;

% Subplot 2: Pressure and vorticity comparison
subplot(1, 2, 2);
hold on;

% Compute vorticity
vorticity = gradient_x(v) - gradient_y(u);

% Plot pressure contours
contour(X, Y, p, 15, 'r-', 'LineWidth', 1.5);
% Plot vorticity contours
contour(X, Y, vorticity, 15, 'b--', 'LineWidth', 1.5);

xlabel('x (m)');
ylabel('y (m)');
title('Pressure (red) and Vorticity (blue) Contours');
legend('Pressure', 'Vorticity', 'Location', 'best');
axis equal tight;
grid on;

sgtitle(sprintf('Lid-Driven Cavity Flow: Re = %.0f, Grid: %dx%d', ...
        u_lid*L/nu, Nx, Ny), 'FontSize', 14, 'FontWeight', 'bold');

fprintf('\nSimulation and visualization complete!\n');
## ----------------------------------------- end of script --------------------------------------

# plots (results) 



<img width="1084" height="449" alt="Screenshot 2025-12-29 004118" src="https://github.com/user-attachments/assets/306f190a-278f-4751-967c-7d834116575f" />
<img width="1184" height="654" alt="Screenshot 2025-12-29 004109" src="https://github.com/user-attachments/assets/eddc1ef8-539b-41b2-8d08-434072563c30" />
<img width="1175" height="641" alt="Screenshot 2025-12-29 004101" src="https://github.com/user-attachments/assets/a3c71325-a3ea-4218-90e3-7a94fbca3446" />


