import numpy as np
import copy
from .machines import BoltzmanMachine

class CriticalityAnalyzer:
    """
    Executes a thermodynamic temperature sweep on an inferred BoltzmanMachine 
    to map response functions and locate phase transitions.
    """
    def __init__(self, inferred_machine):
        # Store the exact inferred parameters as the unscaled reference baseline
        self.N = inferred_machine.N
        self.J_ref = np.copy(inferred_machine.couplings)
        self.h_ref = np.copy(inferred_machine.fields)
        self.machine = inferred_machine

    def _unscaled_energy(self, sample):
        """Calculates the baseline H*(s) without temperature scaling."""
        triangular_J = np.triu(self.J_ref, k=1)
        spins_cross = np.outer(sample, sample)
        return -np.sum(triangular_J * spins_cross) - np.dot(self.h_ref, sample)

    def sample_observables(self, T_tilde, n_samples=5000, n_burn=2000, n_skip=10):
        """
        Runs MCMC at synthetic temperature T_tilde and returns the intensive 
        specific heat (Cv) and magnetic susceptibility (chi).
        """
        # 1. Scale the machine's parameters for the MCMC transition dynamics
        # T_tilde is dimensionless temperature scalar
        # Avoid division by zero
        if T_tilde == 0:
            T_tilde = 1e-10
            
        self.machine.couplings = self.J_ref / T_tilde
        self.machine.fields = self.h_ref / T_tilde
        
        # Initialize a random state
        state = self.machine.generate_random_sample()
        
        # 2. Thermal Equilibration (Burn-in)
        for _ in range(n_burn):
            state = self.machine.monte_carlo(state)
            
        # Arrays to store the unscaled physical observables
        energies = np.zeros(n_samples)
        magnetizations = np.zeros(n_samples)
        
        # 3. Production Sampling
        for i in range(n_samples):
            for _ in range(n_skip): # Decorrelate samples
                state = self.machine.monte_carlo(state)
                
            energies[i] = self._unscaled_energy(state)
            magnetizations[i] = np.sum(state)
            
        # 4. Compute Fluctuation-Dissipation Observables
        var_E = np.var(energies)
        var_M = np.var(magnetizations)
        
        Cv = var_E / (self.N * (T_tilde ** 2))
        chi = var_M / (self.N * T_tilde)
        
        return Cv, chi

    def temperature_sweep(self, T_range=None):
        """
        Sweeps across a spectrum of T_tilde to map the thermodynamic landscape.
        """
        if T_range is None:
            T_range = np.linspace(0.5, 2.0, 30)
            
        cv_curve = []
        chi_curve = []
        
        print(f"Starting temperature sweep across {len(T_range)} points...")
        for T in T_range:
            cv, chi = self.sample_observables(T)
            cv_curve.append(cv)
            chi_curve.append(chi)
            
        # Restore original parameters after the sweep is complete
        self.machine.couplings = self.J_ref
        self.machine.fields = self.h_ref
        
        return T_range, np.array(cv_curve), np.array(chi_curve)
