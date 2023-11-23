# Import the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import the classes from the other files
from metamodel import Metamodel
from architecture import Architecture
from optimization import Optimization
from environments import Environments

# Define the main function
def main():
    """
    This function runs the AGI agent in the environments and tasks, using the metamodel, the architecture, and the optimization approach.
    The function creates an object of the Metamodel class, an object of the Architecture class, an object of the Optimization class, and an object of the Environments class.
    The function calls the optimize method of the Optimization class to get the optimal solutions for the AGI agent.
    The function then runs the AGI agent in the environments and tasks, using the optimal solutions, and collects the performance, complexity, diversity, and regret metrics.
    The function also renders the current state of the environments and tasks, and displays the metrics of the AGI agent.
    """
    # Create an object of the Metamodel class
    metamodel = Metamodel()

    # Create an object of the Architecture class
    architecture = Architecture(metamodel)

    # Create an object of the Optimization class
    optimization = Optimization(metamodel, architecture, n_obj=4, n_var=12, n_pop=100, n_gen=50, lb=[-1.0] * 12, ub=[1.0] * 12)

    # Create an object of the Environments class
    environments = Environments(metamodel, architecture)

    # Call the optimize method of the Optimization class to get the optimal solutions for the AGI agent
    optimal_solutions = optimization.optimize()

    # Initialize the lists for storing the metrics of the AGI agent
    performance_list = []
    complexity_list = []
    diversity_list = []
    regret_list = []

    # Run the AGI agent in the environments and tasks, using the optimal solutions
    for solution in optimal_solutions:
        # Set the parameters, policies, and architectures of the AGI agent according to the solution
        architecture.set_parameters(solution[:optimization.n_var // 3])
        architecture.set_policies(solution[optimization.n_var // 3: 2 * optimization.n_var // 3])
        architecture.set_architectures(solution[2 * optimization.n_var // 3:])

        # Reset the environments and tasks to get the initial observations, rewards, dones, and infos
        obs, reward, done, info = environments.reset()

        # Initialize the variables for storing the metrics of the AGI agent
        performance = 0
        complexity = 0
        diversity = 0
        regret = 0

        # Loop until any of the environments and tasks is done
        while not any(done):
            # Get the actions of the AGI agent for each environment and task, using the architecture object
            actions = architecture.get_actions(obs)

            # Perform one step of interaction between the AGI agent and each environment and task, using the environments object
            obs, reward, done, info = environments.step(actions)

            # Render the current state of each environment and task, using the environments object
            environments.render()

            # Update the metrics of the AGI agent, using the reward and info values
            performance += sum(reward)
            complexity += architecture.get_complexity()
            diversity += architecture.get_diversity()
            regret += architecture.get_regret()

        # Append the metrics of the AGI agent to the lists
        performance_list.append(performance)
        complexity_list.append(complexity)
        diversity_list.append(diversity)
        regret_list.append(regret)

    # Close the environments and tasks, using the environments object
    environments.close()

    # Display the metrics of the AGI agent, using the matplotlib and seaborn libraries
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x=performance_list, y=complexity_list, hue=diversity_list, size=regret_list, palette="rainbow", legend="full")
    plt.xlabel("Performance")
    plt.ylabel("Complexity")
    plt.title("Metrics of the AGI agent")
    plt.show()

# Call the main function
if __name__ == "__main__":
    main()
