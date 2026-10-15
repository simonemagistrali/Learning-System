function [reward, failed] = getRewardAndFailure(next_state)

    % Extract states
    new_x = next_state(1);
    new_theta_rad = next_state(3);
    new_vel = next_state(2);
    new_theta_vel = next_state(4);

    % Failure limits
    x_limit = 2.4;
    theta_limit_degrees = 12;

    % Convert angle to DEGREES
    new_theta_degrees = rad2deg(new_theta_rad);

    % Check if we failed
    failed = new_x < -x_limit || ...
             new_x > x_limit || ...
             new_theta_degrees < -theta_limit_degrees || ...
             new_theta_degrees > theta_limit_degrees;

    % Assign the reward 
    if failed
        reward = -100; % penalty for failure
    else
        reward = 1;    % if we survive
    end

end
