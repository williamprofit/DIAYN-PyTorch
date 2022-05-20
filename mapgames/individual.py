import torch

# Class defining the lower-level instance: the individuals
class Individual:
    def __init__(
        self,
        desc,
        fitness,
        actor_fn=None,
        state_dict=None,
        hand_bd=None,
        centroid=None,
        sensory_info=None,
        parent_fitness=None,
        parent_bd=None,
        type=None,
        parent_1_id=None,
        parent_2_id=None,
        novel=False,
        delta_f=0,
        delta_bd=0,
        parent_delta_f=0,
        variation_type=None,
    ):
        """
          Initialise the individual
          Input:
            - x {Actor} - controller
            - desc {list}
            - fitness {float}
            - hand_bd {list} - optional hand coded BD for projection using AURORA
            - centroid {int} - id of corresponding centroid
            - sensory_info {list} - raw description as given by the environment. Used by AURORA as desc in the encoded BD.
          Output: /
        """
        self.id = next(self._ids)  # get a unique id for each individual
        Individual.current_id = self.id
        self.actor_fn = actor_fn
        self.state_dict = state_dict
        self.desc = desc
        self.sensory_info = sensory_info
        self.fitness = fitness
        self.fitness_score = self.fitness
        self.hand_bd = hand_bd
        self.centroid = centroid

        self.type = type
        self.parent_1_id = parent_1_id
        self.parent_2_id = parent_2_id

        self.novel = novel  # considered not novel until proven

        # Compare to previous elite in the niche
        self.delta_f = delta_f  # set to zero until the niche is filled

        # Compare to parent
        self._parent_fitness = None
        self._parent_bd = None
        self.delta_bd = delta_bd  # zero until measured
        self.parent_delta_f = parent_delta_f

        self.variation_type = variation_type

        # Compute parent-offspring distance
        self.parent_delta_f = 0  # for simplicity
        self.parent_delta_bd = 0  # for simplicity

    @property
    def parent_fitness(self):
        return self._parent_fitness

    @parent_fitness.setter
    def parent_fitness(self, v):
        self._parent_fitness = v
        if self._parent_fitness is not None:
            self.parent_delta_f = self.fitness - self._parent_fitness

    @property
    def parent_bd(self):
        return self._parent_bd

    @parent_bd.setter
    def parent_bd(self, v):
        self._parent_bd = v
        if self._parent_bd is not None:
            self.parent_delta_bd = sum([abs(x) for x in (self.desc - self._parent_bd)])

    @classmethod
    def get_counter(self):
        """Get the individual unique id"""
        return self._ids

    def save(self, filename):
        torch.save(self.state_dict, filename)

    def load(self, filename):
        self.state_dict = torch.load(filename, map_location=torch.device("cpu"))
