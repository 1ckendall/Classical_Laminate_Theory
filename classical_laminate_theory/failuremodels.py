from abc import ABC, abstractmethod


class FailureModel:
    @abstractmethod
    def failure_check(self, local_stress, local_strain):
        raise NotImplementedError


class TsaiHill(FailureModel):
    def __init__(self, X11, X22, S12):
        self.X11 = X11
        self.X22 = X22
        self.S12 = S12

    def failure(self, local_stress, local_strain):
        criterion = (
            (local_stress[0] / self.X11) ** 2
            - (local_stress[0] * local_stress[1]) / (self.X11**2)
            + (local_stress[1] / self.X22) ** 2
            + (local_stress[2] / self.S12) ** 2
        )
        if criterion < 1:
            return False
        return True


# PUCK, HASHIN, CHRISTENSEN, LARC05
