import random
import heapq
import numpy
import numpy as np
import matplotlib
matplotlib.use("TkAgg")   # oppure "Qt5Agg" se hai PyQt5 installato
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker



def generate_initial_population(population_size, num_nodes):
    population = []
    for i in range(0, population_size):
        city_indices = list(range(1, int(num_nodes)))
        random.shuffle(city_indices)
        population.append(city_indices)

    return population

def print_population(population, population_size):
    for i in range(0, population_size):
        print(i, ':', population[i])

def calculate_paths_length(population, population_size, num_cities, nodes_coordinates):

    paths_lengths = []
    for i in range(0, population_size):
        path_length = numpy.sqrt(pow(nodes_coordinates[0][0] - nodes_coordinates[population[i][0]][0], 2) + pow(nodes_coordinates[0][1] - nodes_coordinates[population[i][0]][1], 2))
        for j in range(0, int(num_cities) - 2):
            path_length += numpy.sqrt(pow(nodes_coordinates[population[i][j]][0] - nodes_coordinates[population[i][j + 1]][0], 2) + pow(nodes_coordinates[population[i][j]][1] - nodes_coordinates[population[i][j + 1]][1], 2))

        path_length += numpy.sqrt(pow(nodes_coordinates[population[i][int(num_cities) - 2]][0] - nodes_coordinates[0][0], 2) + pow(nodes_coordinates[population[i][int(num_cities) - 2]][1] - nodes_coordinates[0][1], 2))

        paths_lengths.append(path_length)

    return paths_lengths


def generate_new_population(sorted_population, population_size, num_cities, tournament_size, mutation_probability, crossover_probability, num_elite, lunghezza_percorsi, nodelist):
    new_population = []
    for j in range(0, num_elite):
        new_population.append(sorted_population[j])

    for i in range(0, population_size - num_elite):
        tournament_index = random.sample(range(len(sorted_population)), tournament_size)
        parent1 = sorted_population[min(tournament_index)]

        tournament_index = random.sample(range(len(sorted_population)), tournament_size)
        parent2 = sorted_population[min(tournament_index)]

        crossover_decision =  random.randint(1,100)
        if crossover_decision <= crossover_probability:

            start_index = random.randint(0, int(num_cities) - 2)
            end_index = random.randint(start_index + 1, int(num_cities) - 1)

            static_part = []
            dynamic_part = []

            for j in range(0, int(num_cities) - 1):
                if j >= start_index and j <= end_index:
                    static_part.append(parent1[j])
                else:
                    dynamic_part.append(parent1[j])

            ordered_dynamic_part = []

            for j in range(0, int(num_cities) - 1):
                if parent2[j] in dynamic_part:
                    ordered_dynamic_part.append(parent2[j])

            offspring = []
            s = 0
            d = 0
            for j in range(0, int(num_cities) - 1):
                if j >= start_index and j <= end_index:
                    offspring.append(static_part[s])
                    s += 1
                else:
                    offspring.append(ordered_dynamic_part[d])
                    d += 1


            mutation_decision = random.randint(1, 100)

            if mutation_decision <= mutation_probability:
                mutation_type = random.randint(1,2)
                if mutation_type == 1:
                    gene1_index = random.randint(0, int(num_cities) - 2)
                    gene2_index = random.randint(0, int(num_cities) - 2)
                    temp = offspring[gene1_index]
                    offspring[gene1_index] = offspring[gene2_index]
                    offspring[gene2_index] = temp
                if mutation_type == 2:
                    start_index = random.randint(0, int(num_cities) - 2)
                    end_index = random.randint(0, int(num_cities) - 1)
                    offspring[start_index:end_index] = offspring[start_index:end_index][::-1]



            new_population.append(offspring)

        else:
            parent_choice = random.randint(1,2)
            if parent_choice == 1:
                new_population.append(parent1)
            else:
                new_population.append(parent2)

    return new_population



def calculate_population_diversity(population):
    n = len(population)
    if n <= 1:
        return 0.0

    length = len(population[0])
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            differences = sum(1 for a, b in zip(population[i], population[j]) if a != b)
            distances.append(differences / length)

    return sum(distances) / len(distances)


def plot_best_path(best_path, nodes_coordinates, title="Best TSP Path"):
    """
    Mostra un grafico con le coordinate delle città e le linee del percorso
    della soluzione migliore trovata.

    Args:
        best_path (list): La lista degli indici delle città (escluso il nodo 0 iniziale/finale)
                          nell'ordine del percorso.
        nodes_coordinates (list of lists/numpy array): Lista delle coordinate [x, y] di tutte le città,
                                                       dove nodes_coordinates[0] è il punto di partenza/arrivo.
        title (str): Titolo del grafico.
    """
    # Il percorso completo include il nodo di partenza (indice 0) all'inizio e alla fine.
    # Il nodo di partenza (0) non è incluso in best_path.
    full_path_indices = [0] + best_path + [0]

    # Estrai le coordinate X e Y del percorso completo
    x_coords = [nodes_coordinates[i][0] for i in full_path_indices]
    y_coords = [nodes_coordinates[i][1] for i in full_path_indices]

    # Crea un nuovo grafico
    plt.figure(figsize=(10, 8))

    # Disegna il percorso con delle linee
    plt.plot(x_coords, y_coords, color='blue', linestyle='-', linewidth=1.5, marker='o', markerfacecolor='red',
             markersize=5, label='Path')

    # Segna il punto di partenza/arrivo (nodo 0) con un colore diverso
    plt.plot(x_coords[0], y_coords[0], 's', color='green', markersize=8, label='Start/End Node (0)')

    # Segna tutti gli altri nodi
    plt.scatter(x_coords[1:-1], y_coords[1:-1], color='red', s=30, zorder=5, label='Cities')

    # Aggiungi etichette per i numeri dei nodi (opzionale)
    for i, (x, y) in enumerate(nodes_coordinates):
        if i != 0:
            plt.text(x, y + 0.5, str(i), fontsize=9, ha='center')
        else:
            plt.text(x, y + 0.5, '0 (Start/End)', fontsize=9, ha='center', color='green')

    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.legend()
    plt.axis('equal')  # Per assicurare che le proporzioni siano corrette
    plt.show()



if __name__ == "__main__":
    infile = open('data/berlin52.tsp', 'r')

    Name = infile.readline().strip().split()[1]  # NAME
    FileType = infile.readline().strip().split()[1]  # TYPE
    Comment = infile.readline().strip().split()[1]  # COMMENT
    num_cities = infile.readline().strip().split()[1]  # DIMENSION
    EdgeWeightType = infile.readline().strip().split()[1]  # EDGE_WEIGHT_TYPE
    infile.readline()

    nodelist = []
    N = int(num_cities)
    for i in range(0, int(num_cities)):
        x, y = infile.readline().strip().split()[1:]
        nodelist.append([float(x), float(y)])

    infile.close()

    population_size = 500
    max_generations = 2000
    initial_mutation_probability = 50
    tournament_size = 3
    crossover_probability = 90
    num_elite = 2

    population = generate_initial_population(population_size, num_cities)


    print_population(population, population_size)

    mutation_probability = initial_mutation_probability

    min_lengths_history = []
    population_diversity = []


    #plt.ion()
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    fig1.canvas.manager.set_window_title('Convergence Statistics')

    linea, = ax1.plot([], [], label="Minimum length")
    ax1.set_ylabel("Minimum length")
    ax1.grid()

    linea2, = ax2.plot([], [], color="orange", label="Population diversity")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Diversity (0-1)")
    ax2.set_ylim(0, 1)
    ax2.grid()


    generation=1
    convergence_counter=0
    while generation<max_generations:
        print("generation:", generation)

        paths_lengths = calculate_paths_length(population, population_size, num_cities, nodelist)

        population_with_fitness = list(zip(population, paths_lengths))

        sorted_population_with_fitness = sorted(population_with_fitness, key=lambda item: item[1])

        sorted_population = [individuo for individuo, fitness in sorted_population_with_fitness]
        sorted_lengths = [fitness for individuo, fitness in sorted_population_with_fitness]

        min_length = sorted_lengths[0]
        print("min_length:", sorted_lengths[0])

        min_lengths_history.append(min_length)


        diversity = calculate_population_diversity(population)
        population_diversity.append(diversity)


        """
        linea.set_xdata(range(len(min_lengths_history)))
        linea.set_ydata(min_lengths_history)
        ax1.relim()
        ax1.autoscale_view()

        ax1.set_ylim(5000, 20000)
        ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x / 1000)}k"))

        linea2.set_xdata(range(len(population_diversity)))
        linea2.set_ydata(population_diversity)
        ax2.relim()
        ax2.autoscale_view()

        #plt.draw()
        #plt.pause(0.1)
        """

        if min_length<8000:
            print("trovata rotta con lunghezza adatta:", min_length)
            print(sorted_population[0])
            break

        population = generate_new_population(sorted_population, population_size, num_cities, tournament_size, mutation_probability, crossover_probability, num_elite, paths_lengths, nodelist)
        generation += 1
        convergence_counter +=1




    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    fig1.canvas.manager.set_window_title('Convergence Statistics')

    ax1.plot(range(len(min_lengths_history)), min_lengths_history, label="Minimum length")
    ax1.set_ylabel("Minimum length")
    ax1.grid()
    ax1.set_ylim(5000, 20000)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x / 1000)}k"))

    ax2.plot(range(len(population_diversity)), population_diversity, color="orange", label="Population diversity")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Diversity (0-1)")
    ax2.set_ylim(0, 1)
    ax2.grid()

    plot_best_path(population[0], nodelist)

    plt.show()



    #plt.ioff()
    #plt.show()


