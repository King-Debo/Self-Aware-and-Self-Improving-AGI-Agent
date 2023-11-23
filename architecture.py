# Import the necessary libraries
import opencog
from opencog.atomspace import AtomSpace, TruthValue, types
from opencog.type_constructors import *
from opencog.scheme_wrapper import scheme_eval, scheme_eval_h
from opencog.bindlink import execute_atom
from opencog.pln import *
from opencog.ure import *
from opencog.pymoses import *
from opencog.openpsi import *
from opencog.spacetime import *
from opencog.nlp import *

# Define the architecture class
class Architecture:
    """
    This class defines the modular and hierarchical architecture for the AGI agent, following the guidelines of the OpenCog framework.
    The architecture consists of several modules, such as the perception module, the memory module, the reasoning module, the planning module, the action module, and the communication module.
    Each module implements the corresponding cognitive function and provides the necessary interfaces and functions for the AGI agent.
    The modules are connected by the AtomSpace, which is a hypergraph-based knowledge representation that allows for efficient and flexible manipulation of heterogeneous data and information.
    """

    def __init__(self, metamodel):
        """
        This method initializes the architecture with the following attributes:
        - metamodel: an object of the Metamodel class that provides the metamodel and the framework for the AGI agent.
        - atomspace: an object of the AtomSpace class that provides the hypergraph-based knowledge representation for the AGI agent.
        - perception_module: an object of the PerceptionModule class that implements the perception function and provides the perception functions for the AGI agent.
        - memory_module: an object of the MemoryModule class that implements the memory function and provides the memory functions for the AGI agent.
        - reasoning_module: an object of the ReasoningModule class that implements the reasoning function and provides the reasoning functions for the AGI agent.
        - planning_module: an object of the PlanningModule class that implements the planning function and provides the planning functions for the AGI agent.
        - action_module: an object of the ActionModule class that implements the action function and provides the action functions for the AGI agent.
        - communication_module: an object of the CommunicationModule class that implements the communication function and provides the communication functions for the AGI agent.
        """
        self.metamodel = metamodel
        self.atomspace = AtomSpace()
        self.perception_module = PerceptionModule(self.metamodel, self.atomspace)
        self.memory_module = MemoryModule(self.metamodel, self.atomspace)
        self.reasoning_module = ReasoningModule(self.metamodel, self.atomspace)
        self.planning_module = PlanningModule(self.metamodel, self.atomspace)
        self.action_module = ActionModule(self.metamodel, self.atomspace)
        self.communication_module = CommunicationModule(self.metamodel, self.atomspace)

    def get_metamodel(self):
        """
        This method returns the metamodel of the architecture.
        """
        return self.metamodel

    def get_atomspace(self):
        """
        This method returns the atomspace of the architecture.
        """
        return self.atomspace

    def get_perception_module(self):
        """
        This method returns the perception module of the architecture.
        """
        return self.perception_module

    def get_memory_module(self):
        """
        This method returns the memory module of the architecture.
        """
        return self.memory_module

    def get_reasoning_module(self):
        """
        This method returns the reasoning module of the architecture.
        """
        return self.reasoning_module

    def get_planning_module(self):
        """
        This method returns the planning module of the architecture.
        """
        return self.planning_module

    def get_action_module(self):
        """
        This method returns the action module of the architecture.
        """
        return self.action_module

    def get_communication_module(self):
        """
        This method returns the communication module of the architecture.
        """
        return self.communication_module
