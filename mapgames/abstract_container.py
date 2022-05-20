import abc
from typing import List

from mapgames import Individual

def write_array(a, f):
    for i in a:
        f.write(str(i) + " ")

class AbstractContainer(object):
    @abc.abstractmethod
    def _attempt_add_individual(
        self,
        individual: Individual,
    ) -> bool:
        ...

    def attempt_add_population(
        self,
        individuals_list: List[Individual],
    ) -> List[bool]:
        # individuals_not_dead_list = [
        #     individual for individual in individuals_list if not individual.is_dead
        # ]

        # was_added_list = [
        #     self._attempt_add_individual(individual)
        #     for individual in individuals_not_dead_list
        # ]

        was_added_list = [
            self._attempt_add_individual(individual) for individual in individuals_list
        ]

        return was_added_list

    @abc.abstractmethod
    def _direct_add_individual(self, individual: Individual):
        ...

    def direct_add_population(
        self,
        individuals_list: List[Individual],
    ):
        # individuals_not_dead_list = [
        #     individual for individual in individuals_list if not individual.is_dead
        # ]
        # for individual in individuals_not_dead_list:
        #     self._direct_add_individual(individual)

        for individual in individuals_list:
            self._direct_add_individual(individual)

    @abc.abstractmethod
    def get_individual_at_index(self, flattened_index):
        ...

    def get_individuals_at_indexes(self, list_flattened_indexes):

        return [
            self.get_individual_at_index(flattened_index)
            for flattened_index in list_flattened_indexes
        ]

    def get_all_individuals(self):
        return self.get_individuals_at_indexes(range(len(self)))

    @abc.abstractmethod
    def __len__(self):
        ...

    def iteration_update(self, iteration_index: int):
        ...

    @abc.abstractmethod
    def clear(self):
        ...

    @abc.abstractmethod
    def refresh(self):
        ...

    @abc.abstractmethod
    def empty_copy(self):
        """
        Returns a copy of the container without the individuals
        """
        ...

    def save(
        self,
        n_evals,
        archive_name,
        save_path,
        save_models=False,
        prefixe="archive",
        suffixe="",
    ):
        """
        Save the archive status
        Input:
        - n_evals {int} - number of evaluations for file name
        - archive_name {str} - main file name
        - save_path {str}
        - save_models {bool} - also save the archive for resume as a model
        - prefixe {str} - prefixe for file name
        - suffixe {str} - prefixe for file name
        Output: /
        """

        filename = (
            f"{save_path}/{prefixe}_{archive_name}_" + str(n_evals) + f"{suffixe}.dat"
        )

        if len(self) <= 0:
            return

        with open(filename, "w") as f:
            for k in self.get_all_individuals():
                f.write(str(k.fitness) + " ")

                if k.centroid != None:
                    write_array(k.centroid, f)

                write_array(k.desc, f)
                f.write(str(k.id) + " ")
                f.write("\n")
                if save_models:
                    k.save(f"{save_path}/models/{archive_name}_actor_" + str(k.id))

    @abc.abstractmethod
    def plot(self):
        ...
