clear; clc; close all; 

% Load the trained Q-Table
try
    load('Q_Table_Addestrata.mat', 'Q_table');
    disp('Q-Table loaded successfully.');
catch
    error('Q_Table_Addestrata.mat file not found. Run main.m first');
end

% Parameters 
max_steps = 3000; 
initial_state = [0, 0, 0, 0]; 

% Array to save the history of position and angles
history_x = zeros(max_steps, 1);
history_theta = zeros(max_steps, 1);
steps_survived = 0;


current_state_continuous = initial_state;

for step = 1 : max_steps
    
    chosen_force = chooseBestAction(current_state_continuous, Q_table); %chooses best action and returns corresponding force
    
    s = current_state_continuous;
    next_state_cont = SimulatePendel(chosen_force, s(1), s(2), s(3), s(4)); % next state
    
    [~, failed] = getRewardAndFailure(next_state_cont); %check if it fails
    
    % save data for the plot
    history_x(step) = next_state_cont(1); % save position 
    history_theta(step) = rad2deg(next_state_cont(3)); % Save angle in degrees
    steps_survived = step;

    %update state
    current_state_continuous = next_state_cont;
    
    if failed %if you fail, exit
        disp(['Failure at step: ', num2str(step)]);
        break; 
    end
end

if ~failed %if we haven't failed
    disp(['Success! Survived for all ', num2str(max_steps), ' steps.']);
end

% Clean up the arrays for the plot, remove trailing zeros if it failed
history_x = history_x(1:steps_survived);
history_theta = history_theta(1:steps_survived);


% Plot 1: Cart Position (x)
figure;
plot(history_x);
title('Test: Cart Position (x) over time');
xlabel('Step (0.02s per step)');
ylabel('Position x (meters)');
grid on;
ylim([-2.5, 2.5]); % Failure limits [cite: 165]
hold on;
plot([-max_steps, max_steps], [2.4, 2.4], 'r--', 'LineWidth', 1.5);
plot([-max_steps, max_steps], [-2.4, -2.4], 'r--', 'LineWidth', 1.5);
hold off;

% Plot 2: Pendulum Angle (theta)
figure;
plot(history_theta);
title('Test: Pendulum Angle (\theta) over time');
xlabel('Step (0.02s per step)');
ylabel('Angle \theta (degrees)');
grid on;
ylim([-13, 13]); % Failure limits [cite: 165]
hold on;
plot([-max_steps, max_steps], [12, 12], 'r--', 'LineWidth', 1.5);
plot([-max_steps, max_steps], [-12, -12], 'r--', 'LineWidth', 1.5);
hold off;
