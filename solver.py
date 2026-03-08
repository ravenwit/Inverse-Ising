from abc import ABC, abstractmethod
import numpy as np
import copy
from .samples import Samples
from .machines import Machine, BoltzmanMachine, NEMachine

class InverseIsing(ABC):
    def __init__(self, data: Samples, non_equilibrium=False, partition_number=1, symmetric=True) -> None:
        self.data = copy.deepcopy(data)
        self.ne = non_equilibrium
        self.spins = self.data.spins
        self.N = self.data.N
        self.Z = partition_number
        self.true_magnetization = self.data.magnetization()
        self.true_correlation = self.data.correlation()
        self.symmetric = symmetric

    def calc_loss(self, machine: Machine):
        energies = []
        samples = self.data.data
        if self.ne:
            # Pseudo-likelihood for NE
            for index, sample in enumerate(samples[:-1]):
                ef = machine.effective_field(sample)
                term1 = np.dot(samples[index + 1], ef)
                term2 = np.sum(np.log(2 * np.cosh(ef)))
                energies.append(- term1 + term2)
        else:
            # Likelihood for Equilibrium
            # Note: This is computationally expensive as it requires Z, usually approximated
            for sample in samples:
                energies.append(machine.energy(sample))

        return np.mean(np.array(energies)) # + np.log(self.Z) if known

    def _infer_step(self, machine: Machine, lr):
        couplings = copy.deepcopy(machine.couplings)
        fields = copy.deepcopy(machine.fields)

        if self.ne: 
            infer_magnetization = machine.compute_magnetization(self.data)
            infer_correlation = machine.compute_correlation(self.data)
        else:
            infer_magnetization = machine.compute_magnetization()
            infer_correlation = machine.compute_correlation()

        # Gradient Ascent: Delta J = <s_i s_j>_data - <s_i s_j>_model
        fields += lr * (self.true_magnetization - infer_magnetization)
        couplings += lr * (self.true_correlation - infer_correlation)

        # Enforce symmetry ONLY for Equilibrium models
        if not self.ne and self.symmetric:
            couplings = (couplings + couplings.T) / 2
            
        # For NE models, we strictly preserve asymmetry to capture causality

        return couplings, fields
    
    def infer_parameters(self, atol, lr, stop=3, iter=0):
        infer_couplings = np.random.rand(self.N, self.N) * 1e-8
        infer_fields = np.random.rand(self.N) * 1e-8

        if self.ne:
            infer_machine = NEMachine(self.spins, infer_couplings, infer_fields)
        else:
            infer_machine = BoltzmanMachine(self.spins, infer_couplings, infer_fields)

        count = 1
        loss = []
        
        while True:
            infer_couplings, infer_fields = self._infer_step(infer_machine, lr)
            infer_machine.update_parameters(infer_couplings, infer_fields)
            
            _loss = self.calc_loss(infer_machine)
            loss.append(_loss)
            
            if count % 10 == 0:
                print(f"Iteration {count}, Loss: {_loss:.4f}")

            count += 1
            if iter > 0 and count >= iter:
                break
            # Add convergence checks here if needed
        
        return infer_machine, loss
