clear; clc; 

gamma = 0.99;
actions = [-10, 10];
num_actions = 2;

num_episodes = 10000; 
max_steps = 3000;    

kappa = 10;            % Initial k
max_kappa = 20 ;          % max k  
kappa_increase = 1.0005; % Growth factor 

% First time I created the table from scratch, then I loaded the one from the previous training

%num_total_states = 5000; % (This comes from 10*5*20*5, as defined in getState.m)
%Q_table = rand(num_total_states, num_actions) * 0.01;

disp('Loading trained Q-Table...');
load('Q_Table_Addestrata.mat'); % Load the saved Q-table
num_total_states = 5000; 
disp('Loading complete.');

step_history = zeros(num_episodes, 1); %saves the number of steps reached for each episode


%% TRAINING CYCLE

for episode = 1 : num_episodes
    
    current_state_continuous = [0, 0, 0, 0]; % Initial state
    current_state_idx = getState(current_state_continuous); % Translate to index / discretize
    
    for step = 1 : max_steps
        
        
        % Get Q-values for this state
        q_values = Q_table(current_state_idx, :);
        
        % this is a method I found to reduce the size of Q-values to facilitate calculations
        q_relative = q_values - max(q_values);  %Subtract the max
       
        probabilities = kappa .^ q_relative;  % Calculate probabilities
        normalized_probs = probabilities / sum(probabilities);

        % Choose an action (1 or 2)
        r = rand(); 
        if r < normalized_probs(1)
            action_idx = 1; % Choose action 1
        else
            action_idx = 2; % Choose action 2 
        end

        force = actions(action_idx); %calculate force
        
        % Execute the Action
        s = current_state_continuous;
        next_state_cont = SimulatePendel(force, s(1), s(2), s(3), s(4)); %calculate next state

        [reward, failed] = getRewardAndFailure(next_state_cont); %get the reward / failure

        next_state_idx = getState(next_state_cont); %discretize next state
        
        current_q = Q_table(current_state_idx, action_idx);
        
        if failed
            q_target = reward; % if failed, reward is just the penalty
        else
            % if not failed Q = r + gamma * max Q
            max_future_q = max(Q_table(next_state_idx, :));
            q_target = reward + gamma * max_future_q;
        end
        
        % update Q-value
        Q_table(current_state_idx, action_idx) = q_target;
        
        % update current state
        current_state_continuous = next_state_cont;
        current_state_idx = next_state_idx;
        
        if failed
            break; % if failed, move to the next episode
        end
    end
    
    step_history(episode) = step; % Save the score (steps) obtained in the episode
    
    % increment KAPPA 
    kappa = min(max_kappa, kappa * kappa_increase);
    
    % print every 100 episodes
    if mod(episode, 100) == 0
        disp(['Episode: ', num2str(episode), ', Steps: ', num2str(step), ', Kappa: ', num2str(kappa)]);
    end
    
end 

% Save the trained Q-table
save('Q_Table_Addestrata.mat', 'Q_table');

% Show a plot 
figure;
plot(step_history);
title('Training Progress: Steps survived per episode');
xlabel('Episode');
ylabel('Steps Survived');
grid on;
