# Import the necessary libraries
import gym
import gym.spaces
import gym.wrappers
import gym.envs
import gym.envs.atari

# Define the Atari games environment and task class
class AtariGamesEnv(gym.Env):
    """
    This class defines the Atari games environment and task for the AGI agent, using the OpenAI Gym interface.
    The environment and task consist of playing Atari 2600 games, such as Breakout, Pong, Space Invaders, etc.
    The environment and task provide the necessary interfaces and functions for the AGI agent, such as the observation space, the action space, the reward function, and the termination condition.
    """

    def __init__(self, game):
        """
        This method initializes the Atari games environment and task with the following attributes:
        - metadata: a dictionary that stores the metadata of the environment and task, such as the version and the render modes.
        - game: a string that represents the name of the Atari game to be played, such as "Breakout-v0", "Pong-v0", "SpaceInvaders-v0", etc.
        - env: an object of the gym.make class that creates the Atari game environment and task, using the game parameter.
        - obs: a numpy array that represents the observation of the environment and task, which is the RGB image of the game screen.
        - action: an integer that represents the action of the environment and task, which is the joystick movement of the game controller.
        - reward: a float that represents the reward of the environment and task, which is the score difference between the AGI agent and the opponent.
        - done: a boolean that represents the termination condition of the environment and task, which is True if the game is over, and False otherwise.
        - info: a dictionary that represents the additional information of the environment and task, such as the number of lives and the level of the game.
        """
        self.metadata = {"render.modes": ["human", "rgb_array"], "version": "0.1.0"}
        self.game = game
        self.env = gym.make(game)
        self.obs = None
        self.action = None
        self.reward = None
        self.done = None
        self.info = None

    def reset(self):
        """
        This method resets the Atari games environment and task to an initial state, and returns the initial observation, reward, done, and info.
        The method resets the env attribute and assigns the initial observation, reward, done, and info to the obs, reward, done, and info attributes.
        The method returns the obs, reward, done, and info attributes as a tuple.
        """
        # Reset the env attribute
        self.env.reset()

        # Assign the initial observation, reward, done, and info to the obs, reward, done, and info attributes
        self.obs = self.env.observation_space.sample()
        self.action = self.env.action_space.sample()
        self.reward = 0
        self.done = False
        self.info = {}

        # Return the obs, reward, done, and info attributes as a tuple
        return (self.obs, self.reward, self.done, self.info)

    def step(self, actions):
        """
        This method performs one step of interaction between the AGI agent and the Atari games environment and task.
        The actions parameter is a list of objects that represents the actions of the AGI agent in the Atari games environment and task, which are the joystick movements of the game controller.
        The method updates the observations, rewards, dones, and infos of the AGI agent in the Atari games environment and task, and returns them as a tuple of lists.
        """
        # Perform one step of interaction between the AGI agent and the Atari games environment and task
        for i in range(len(actions)):
            # Get the action of the AGI agent for the current game
            action = actions[i]

            # Apply the action to the env attribute and get the observation, reward, done, and info
            obs, reward, done, info = self.env.step(action)

            # Assign the observation, reward, done, and info to the obs, reward, done, and info attributes
            self.obs = obs
            self.reward = reward
            self.done = done
            self.info = info

            # Update the observations, rewards, dones, and infos of the AGI agent in the Atari games environment and task
            self.obs[i] = self.obs
            self.rewards[i] = self.reward
            self.dones[i] = self.done
            self.infos[i] = self.info

        # Return the observations, rewards, dones, and infos of the AGI agent in the Atari games environment and task
        return (self.obs, self.reward, self.done, self.info)

    def render(self, mode="human"):
        """
        This method renders the current state of the Atari games environment and task.
        The mode parameter is a string that represents the mode of rendering, which can be "human" or "rgb_array".
        The method displays or returns the RGB image of the game screen, using the env attribute.
        """
        # Check the mode of rendering
        if mode == "human":
            # Display the RGB image of the game screen, using the env attribute
            self.env.render(mode=mode)
        elif mode == "rgb_array":
            # Return the RGB image of the game screen, using the env attribute
            return self.env.render(mode=mode)
        else:
            # Raise an exception if the mode is invalid
            raise ValueError("Invalid mode: {}".format(mode))

    def close(self):
        """
        This method closes the Atari games environment and task.
        The method releases any resources used by the environment and task, such as the env attribute.
        """
        # Release the env attribute
        self.env.close()
