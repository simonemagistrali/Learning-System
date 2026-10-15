
function force = chooseBestAction(continuous_state, Q_table)
    % chooses only the best action

    % possible actions
    actions = [-10, 10]; 

    % discretize
    state_idx = getState(continuous_state);

    % 3. Extract the Q-values for that state
    q_values_for_state = Q_table(state_idx, :);

    %find the index of the largest value
    [~, action_idx] = max(q_values_for_state);

    % return the force corresponding to that index
    force = actions(action_idx);
end
