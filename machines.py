from abc import ABC, abstractmethod
import numpy as np
import copy
from .samples import BoltzmanSamples, NESamples

class Machine(ABC):
    def __init__(self, 
                 spins, couplings, fields, 
                 eq_step, calc_step, random_indexing) -> None:
        self.couplings = couplings
        self.fields = fields
        self.N = couplings.shape[0]
        self.random_indexing = random_indexing
        self.spins = spins  # [-1, 1]

        self.opposite_spin = {
            1.0: -1.0,
            -1.0: 1.0,
            1: -1,
            -1: 1
        }

        self.eq_step = eq_step
        self.calc_step = calc_step

    def update_parameters(self, couplings, fields):
        self.couplings = copy.deepcopy(couplings)
        self.fields = copy.deepcopy(fields)
        self.N = len(couplings)

    def generate_random_sample(self):
        return np.random.choice(self.spins, size=self.N)
    
    def generate_sample(self):
        sample = self.generate_random_sample()
        for _ in range(self.eq_step):
            sample = self.monte_carlo(sample)
        return sample
    
    def generate_ensamble(self, n_sample, sample_type="Boltzman", triangular=False):
        samples = []
        samples.append(self.generate_sample())
        for i in range(n_sample):
            samples.append(self.monte_carlo(samples[-1]))
        samples = np.array(samples)
        if sample_type == "Boltzman":
            return BoltzmanSamples(samples, triangular)
        elif sample_type == "Non":
            return NESamples(samples, triangular)
        
    def _possible_confs(self):
        import itertools
        return np.array(list(itertools.product([-1, 1], repeat=self.N)))

    def calc_Z(self):
        # Exact partition function calculation
        Z = 0.0
        samples = self._possible_confs()
        for sample in samples:
            Z += np.exp(-self.energy(sample))
        return Z

    @abstractmethod
    def energy(self, sample):
        pass

    @abstractmethod
    def effective_field(self, sample, index=None):
        pass

    @abstractmethod
    def monte_carlo(self, sample):
        pass

    @abstractmethod
    def compute_magnetization(self):
        pass

    @abstractmethod
    def compute_correlation(self):
        pass

class BoltzmanMachine(Machine):
    def __init__(self, 
                 spins, couplings, fields, eq_step=int(1e3),
                 calc_step=int(2000), random_indexing=True) -> None:
        
        super().__init__(spins, couplings, fields,
                          eq_step, calc_step, random_indexing)
        
    def effective_field(self, sample, index=None):
        if index is None:
            return np.dot(self.couplings, sample) + self.fields
        return np.dot(self.couplings[index], sample) + self.fields[index]
    
    def energy(self, sample):
        # Using upper triangular for standard Ising energy definition to avoid double counting
        triangular_couplings = np.triu(self.couplings, k=1)
        spins_cross = np.outer(sample, sample)
        return -np.sum(triangular_couplings * spins_cross) - np.dot(self.fields, sample)

    def monte_carlo(self, sample):
        sample = copy.deepcopy(sample)
        indices = np.arange(self.N)
        if self.random_indexing:
            np.random.shuffle(indices)
        
        # Sequential updates for Detailed Balance
        for i in indices:
            h_eff = self.effective_field(sample, i)
            # Energy change if spin i flips: Delta E = 2 * s_i * h_i
            del_E = 2 * sample[i] * h_eff
            
            # Correct Glauber probability P(flip) = 1 / (1 + exp(Delta E))
            prob_flip = 1.0 / (1.0 + np.exp(del_E))
            
            if np.random.rand() < prob_flip:
                sample[i] = self.opposite_spin[sample[i]]

        return sample
    
    def compute_magnetization(self, data=None):
        # Model expectation
        samples = []
        samples.append(self.generate_sample())
        for _ in range(self.calc_step):
            samples.append(self.monte_carlo(samples[-1]))
        return np.mean(samples, axis=0)
    
    def compute_correlation(self, data=None):
        # Model expectation
        samples = []
        samples.append(self.generate_sample())
        for _ in range(self.calc_step):
            samples.append(self.monte_carlo(samples[-1]))
        samples_array = np.array(samples)
        
        outer_products = samples_array[:, :, np.newaxis] * samples_array[:, np.newaxis, :]
        return np.mean(outer_products, axis=0)
    

class NEMachine(Machine):
    def __init__(self, 
                 spins, couplings, fields, 
                 eq_step=100, calc_step=50, random_indexing=True) -> None:
        super().__init__(spins, couplings, fields, 
                         eq_step, calc_step, random_indexing)
 
    def energy(self, sample):
        # Standard Ising energy for compatibility
        triangular_couplings = np.triu(self.couplings, k=1)
        spins_cross = np.outer(sample, sample)
        return -np.sum(triangular_couplings * spins_cross) - np.dot(self.fields, sample)

    def effective_field(self, sample, index=None):
        # Removes self-interaction diagonal
        if index is not None:
            self_interaction = self.couplings[index, index] * sample[index]
            return np.dot(self.couplings[index], sample) + self.fields[index] - self_interaction
        else:
            return np.dot(self.couplings, sample) + self.fields - np.diag(self.couplings) * sample

    def monte_carlo(self, sample):
        sample = copy.deepcopy(sample)
        indices = np.arange(self.N)
        if self.random_indexing:
            np.random.shuffle(indices)

        # Sequential updates
        for i in indices:
            h_eff = self.effective_field(sample, i)
            del_E = 2 * sample[i] * h_eff
            
            # Correct Glauber flip probability
            prob_flip = 1.0 / (1.0 + np.exp(del_E))
            
            if np.random.rand() < prob_flip:
                sample[i] = self.opposite_spin[sample[i]]
        return sample
                
    def compute_magnetization(self, data):
        # Exact Kinetic Expectation: < tanh(h_i(t)) >
        T = data.n_samples 
        data_arr = data.data
        sum_mag = np.zeros(self.N)
        
        for t in range(T - 1):
            sum_mag += np.tanh(self.effective_field(data_arr[t]))
            
        return sum_mag / (T - 1)
    
    def compute_correlation(self, data):
        # Exact Kinetic Gradient Expectation: < tanh(h_i(t)) s_j(t) >
        T = data.n_samples
        data_arr = data.data
        corr = np.zeros((self.N, self.N))
        
        for t in range(T - 1):
            # Effective field on target spins at t+1 given state at t
            eff_fields = self.effective_field(data_arr[t])
            tanh_fields = np.tanh(eff_fields)
            
            # Outer product: influence of source j (cols) on target i (rows)
            corr += np.outer(tanh_fields, data_arr[t])
            
        return corr / (T - 1)
