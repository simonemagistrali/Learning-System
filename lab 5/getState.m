function state_index = getState(continuous_state)
   
    % position discretized into 10 parts
    x_bins = linspace(-2.4, 2.4, 9); 
    

    % velocity discretized into 5 parts
    x_dot_bins = linspace(-2, 2, 4);

    % angle discretized into 20 parts
    theta_bins = linspace(-12, 12, 19);

    % angular velocity discretized into 5 parts
    theta_dot_bins = linspace(-2, 2, 4);


    % convert angle to DEGREES before discretizing
    theta_degrees = rad2deg(continuous_state(3));

    % discretize
    % "inf" is for out-of-range
    s1 = discretize(continuous_state(1), [-inf, x_bins, inf]);
    s2 = discretize(continuous_state(2), [-inf, x_dot_bins, inf]);
    s3 = discretize(theta_degrees,       [-inf, theta_bins, inf]);
    s4 = discretize(continuous_state(4), [-inf, theta_dot_bins, inf]);
    

    
    % The dimensions of our table
    state_dimensions = [10, 5, 20, 5]; 
    
    % map 4 indices (s1, s2, s3, s4) to a single number, like a hash table
    state_index = sub2ind(state_dimensions, s1, s2, s3, s4);
end
