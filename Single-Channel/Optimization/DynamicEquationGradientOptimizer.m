% Define parameters
a = 1;  % Parameter for x equation
b = 0.25;  % Coupling parameter for x -> y
c = 1;  % Parameter for y equation
d = 2;  % Coupling parameter for y -> x

% Define the system of ODEs
ode_system = @(t, z) [
    -a * z(1) + b * z(2);  % dx/dt
    -c * z(2) + d * z(1)   % dy/dt
];

% Initial conditions for x(0) and y(0)
initial_conditions = [1; 1];  % For example, x(0) = 1, y(0) = 1

% Time span for the solution
tspan = [0 10];  % From t=0 to t=10

% Solve the ODEs
[t, solution] = ode45(ode_system, tspan, initial_conditions);

% Extract the solutions for x and y
x = solution(:, 1);
y = solution(:, 2);

% Plot the results
figure;
subplot(2, 1, 1);
plot(t, x, 'r', 'LineWidth', 2);
xlabel('Time t');
ylabel('x(t)');
title('Solution for x(t)');

subplot(2, 1, 2);
plot(t, y, 'b', 'LineWidth', 2);
xlabel('Time t');
ylabel('y(t)');
title('Solution for y(t)');
