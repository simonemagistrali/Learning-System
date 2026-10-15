import math

DATA_FILE = "city 1.txt"
GOAL_CITY = "F"


def load_city_data(filename):
    adjacency_list = {}

    try:
        with open(filename, 'r') as f:
            # Skip the first header lines
            for _ in range(3):
                next(f)

            # Read each line of data
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    city1, city2, distance_str = parts
                    distance = int(distance_str)

                    # Ensure both cities exist in the dictionary
                    if city1 not in adjacency_list:
                        adjacency_list[city1] = []
                    if city2 not in adjacency_list:
                        adjacency_list[city2] = []

                    # Add the connection in both directions
                    adjacency_list[city1].append((city2, distance))
                    adjacency_list[city2].append((city1, distance))

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except Exception as e:
        print(f"Error while reading the file: {e}")
        return None

    return adjacency_list

def initialize_values(all_cities, goal_city):
    """Initializes the value dictionary V."""
    V = {}
    for city in all_cities:
        if city == goal_city:
            V[city] = 0.0
        else:
            V[city] = -math.inf  # Negative infinity
    return V


def find_shortest_path(start_city, goal_city, V_star, adj_list):
    """Finds the optimal path from a start city using the optimal V* values."""

    if V_star[start_city] == -math.inf:
        return [f"No path from {start_city} to {goal_city}"]

    path = [start_city]
    current_city = start_city

    # As long as we haven't reached the destination
    while current_city != goal_city:

        best_neighbor = None  # We are looking for the best neighbor

        best_value = -math.inf

        # Check all neighbors
        for (neighbor, distance) in adj_list[current_city]:
            reward = -distance
            V_star_neighbor = V_star[neighbor]

            # Calculate r + V*(s')
            value = reward + V_star_neighbor

            if value > best_value:
                best_value = value
                best_neighbor = neighbor

        # Add the best move to the path and continue from there
        path.append(best_neighbor)
        current_city = best_neighbor

    return path


if __name__ == "__main__":
    print("Loading map data...")
    map_data = load_city_data(DATA_FILE)

    # Check if the map was created
    if map_data:
        print(f"Data loaded successfully for {len(map_data)} cities.")

        print("Initializing V0 values...")

        # Get the list of all cities from the map data
        all_cities_list = list(map_data.keys())

        # Create the base V0 dictionary
        V_k = initialize_values(all_cities_list, GOAL_CITY)

        print("Running Value Iteration...")

        # Use V_k to read old values and V_kplus1 to write new ones
        V_kplus1 = V_k.copy()

        iteration = 0

        while True:  # The loop continues until convergence
            iteration += 1

            # Copy V_kplus1 values (from the last iteration) into V_k (for this iteration)
            V_k = V_kplus1.copy()

            converged = True

            # Calculate V_k+1 for each city
            for city in all_cities_list:

                if city == GOAL_CITY:
                    V_kplus1[city] = 0.0
                    continue  # Move to the next city

                # Bellman equation
                # We need to find the max(reward + neighbor_value)

                action_values = []  # List for the values of all possible "reward + neighbor_value"

                # Look at all neighbors of 'city'
                for (neighbor, distance) in map_data.get(city, []):

                    reward = -distance     # Use "minus" because we want to maximize the total reward

                    # Neighbor's value from iteration k (V_k)
                    V_k_neighbor = V_k[neighbor]      # Using V_k gets the value from the previous iteration

                    # Calculate r + gamma * V_k(s')
                    # gamma = 1.0
                    value = reward + 1.0 * V_k_neighbor

                    action_values.append(value)

                # V*(s) = max_a (...)
                # Find the maximum value among all possible actions (neighbors)
                if action_values:
                    V_kplus1[city] = max(action_values)
                else:
                    # If a city has no neighbors (and is not the goal), it has no path
                    V_kplus1[city] = -math.inf


                # Check for convergence
                if V_kplus1[city] != V_k[city]:
                    converged = False

            if converged:
                break


        V_optimal = V_kplus1

        print(f"Convergence reached after {iteration} iterations.")

        # --- Print the Results (Task 1) ---
        print("\nCalculated Optimal Values (V*)")
        for city in sorted(V_optimal.keys()):
            print(f"  V*({city}): {V_optimal[city]:.0f}")

        print("\nActual Minimum Distances (Cost)")
        for city in sorted(V_optimal.keys()):
            real_distance = -V_optimal[city]
            print(f"  Minimum distance from {city} to {GOAL_CITY}: {real_distance:.0f}")

        # Calculate and print the path from each city
        print("\nOptimal Paths to Goal:")
        for city in sorted(all_cities_list):
            path = find_shortest_path(city, GOAL_CITY, V_optimal, map_data)
            print(f"  {city}: {' -> '.join(path)}")
