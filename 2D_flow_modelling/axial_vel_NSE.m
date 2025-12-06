% pvs_hydraulic_resistance_fd.m
% Using Forward Difference Scheme
% Compute hydraulic resistance of an eccentric annulus (inner circle, outer ellipse)
% by solving Laplace/Poisson for axial velocity: Lap u = (1/mu)*dpdz, u=0 on walls.

clear; close all; clc;

%% Parameters (physical)
mu = 1e-3;              % dynamic viscosity (Pa.s) - typical for water-like CSF
dpdz = -1.0;            % chosen axial pressure gradient (Pa/m); pick -1 for scaling

%% Geometry parameters (all in meters)
R_inner = 20e-6;        % inner artery radius
a_outer = 60e-6;        % outer ellipse semi-axis (x-direction)
b_outer = 40e-6;        % outer ellipse semi-axis (y-direction)

% Domain box to contain ellipse (choose margin)
x_min = -a_outer*1.1; x_max = a_outer*1.1;
y_min = -b_outer*1.2; y_max = b_outer*1.2;

%% Grid / discretization
Nx = 200; Ny = 140;     % resolution (increase for accuracy)
x = linspace(x_min,x_max,Nx);
y = linspace(y_min,y_max,Ny);
dx = x(2)-x(1); dy = y(2)-y(1);
[X,Y] = meshgrid(x,y);

%% Function to check if point is inside outer ellipse AND outside inner circle (annulus)
isInside = @(xc) ( (X./a_outer).^2 + (Y./b_outer).^2 <= 1 ) & ...
                 ( (X - xc).^2 + Y.^2 >= R_inner^2 );

%% Sweep eccentricity (shift of inner circle center along +x)
ecc_shifts = linspace(0, a_outer - R_inner - 1e-9, 10); % up to touching (avoid exact touching)
Rvals = zeros(size(ecc_shifts));
Qvals = zeros(size(ecc_shifts));

for k=1:length(ecc_shifts)
    xc = ecc_shifts(k);            % inner circle center x location (eccentricity)
    mask = isInside(xc);           % logical mask of annular region (true = fluid node)
    
    % identify nodes to solve for (interior nodes)
    N = Nx*Ny;
    idx = reshape(1:N,Ny,Nx);     % linear index mapping
    fluid_idx = idx(mask);
    nF = numel(fluid_idx);
    
    % Build 5-point Laplacian (finite difference) for interior fluid nodes
    % For node (i,j): Lap u ≈ (u_{i+1,j}-2u_{i,j}+u_{i-1,j})/dx^2 + (u_{i,j+1}-2u_{i,j}+u_{i,j-1})/dy^2
    % We'll assemble sparse matrix A * u_vec = b, where b = (1/mu)*dpdz for interior nodes.
    
    I = zeros(nF*5,1); J = zeros(nF*5,1); V = zeros(nF*5,1);
    b = (1/mu)*dpdz * ones(nF,1);   % RHS (constant)
    cnt = 0;
    
    % Precompute neighbor offsets in linear index
    for n=1:nF
        lin = fluid_idx(n);                    % flattened index
        [j,i] = ind2sub([Ny,Nx], lin);        % j row (y), i col (x)
        % center
        cnt = cnt+1; I(cnt)=n; J(cnt)=n; V(cnt)= -2/(dx^2) -2/(dy^2);
        
        % left neighbor (i-1,j)
        if i>1
            linL = idx(j,i-1);
            if mask(j,i-1)   % neighbor is fluid
                nL = find(fluid_idx==linL); % small cost but fine for moderate grid
                cnt = cnt+1; I(cnt)=n; J(cnt)=nL; V(cnt)=1/(dx^2);
            else
                % boundary (wall) => Dirichlet u=0 contributes nothing to unknowns,
                % but shifts b: u_wall=0 so no addition needed since u_wall*coef = 0
            end
        end
        
        % right neighbor (i+1,j)
        if i<Nx
            linR = idx(j,i+1);
            if mask(j,i+1)
                nR = find(fluid_idx==linR);
                cnt = cnt+1; I(cnt)=n; J(cnt)=nR; V(cnt)=1/(dx^2);
            end
        end
        
        % down neighbor (i,j-1)
        if j>1
            linD = idx(j-1,i);
            if mask(j-1,i)
                nD = find(fluid_idx==linD);
                cnt = cnt+1; I(cnt)=n; J(cnt)=nD; V(cnt)=1/(dy^2);
            end
        end
        
        % up neighbor (i,j+1)
        if j<Ny
            linU = idx(j+1,i);
            if mask(j+1,i)
                nU = find(fluid_idx==linU);
                cnt = cnt+1; I(cnt)=n; J(cnt)=nU; V(cnt)=1/(dy^2);
            end
        end
    end
    
    % Trim zeros
    I = I(1:cnt); J = J(1:cnt); V = V(1:cnt);
    A = sparse(I,J,V,nF,nF);
    
    % Solve linear system
    % Use backslash; matrix is symmetric negative definite (we used center negative),
    % so multiply both sides by -1 to get standard positive-definite Laplacian:
    A_pos = -A;
    b_pos = -b;
    % Solve
    u_vec = A_pos \ b_pos;
    
    % Compose full u field
    U = zeros(size(X));
    U(mask) = u_vec;
    
    % compute Q = integral u dA
    Q = sum(U(mask),'all') * dx * dy;
    Qvals(k) = Q;
    Rvals(k) = -dpdz / Q;  % per unit length (since dpdz was per meter)
    
    fprintf('ecc shift = %.2e m, Q = %.3e [m^2/s], R = %.3e [Pa s / m^3]\n', xc, Q, Rvals(k));
    
    % optional: store last U for plotting
    if k==1 || k==round(length(ecc_shifts)/2) || k==length(ecc_shifts)
        figure; pcolor(x*1e6,y*1e6,U); shading interp; colorbar;
        axis equal; title(sprintf('u (um/s) at ecc shift = %.1f um', xc*1e6));
        xlabel('x (um)'); ylabel('y (um)');
    end
end

% Plot R vs eccentric shift (and normalized by R0 where shift=0)
figure;
R0 = Rvals(1);
plot(ecc_shifts*1e6, Rvals/R0,'-o','LineWidth',1.5);
xlabel('eccentric shift ( \mum )'); ylabel('Normalized R (R/R_{ecc=0})');
title('Hydraulic resistance vs eccentricity (normalized)');
grid on;

