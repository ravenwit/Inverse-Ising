from abc import ABC, abstractmethod
import numpy as np

class Samples(ABC):
    def __init__(self, data, generate=False, big=False) -> None:
        data = np.array(data)
        self.n_samples, self.N = data.shape
        self.data = data
        self.spins = np.unique(self.data)
        self.generate = generate
        self.big = big

    @abstractmethod
    def correlation(self):
        pass
    
    @abstractmethod
    def magnetization(self):
        return np.mean(self.data, axis=0)


class NESamples(Samples):
    def __init__(self, data, generate=False, big=False) -> None:
        super().__init__(data, generate)

    def magnetization(self):
        # Target magnetization for t+1
        return np.mean(self.data[1:, :], axis=0)
       
    def correlation(self):
        # Time-delayed correlation: <s_i(t+1) s_j(t)>
        corr = np.zeros((self.N, self.N))
        # Vectorized implementation for speed
        s_t = self.data[:-1, :]
        s_t1 = self.data[1:, :]
        
        # Outer product sum over time
        for t in range(self.n_samples - 1):
            corr += np.outer(s_t1[t], s_t[t])
            
        corr /= (self.n_samples - 1)
        return corr


class BoltzmanSamples(Samples):
    def __init__(self, data, generate=False, big=False) -> None:
        super().__init__(data, generate, big)

    def magnetization(self):
        return super().magnetization()

    def correlation(self):
        if self.big:
            mean_correlation = np.zeros((self.N, self.N))
            for sample in self.data:
                mean_correlation += np.outer(sample, sample)
            return mean_correlation / self.n_samples
        else:
            outer_products = self.data[:, :, np.newaxis] * self.data[:, np.newaxis, :]
            mean_correlation = np.mean(outer_products, axis=0)
            return mean_correlation
