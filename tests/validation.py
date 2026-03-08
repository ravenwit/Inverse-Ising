import sys
import os
import unittest
import numpy as np

# Add parent directory to path to import InverseIsing package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from InverseIsing.machines import BoltzmanMachine, NEMachine
from InverseIsing.criticality import CriticalityAnalyzer

class TestThermodynamics(unittest.TestCase):

    def test_exact_partition_function(self):
        """
        Verifies that calc_Z returns the exact analytical partition function 
        for a 2-spin system.
        Analytical Z = 4 * cosh(J) for N=2, h=0
        """
        J_val = 1.5
        # 2 spins, J=1.5, h=0
        couplings = np.array([[0, J_val], [J_val, 0]])
        fields = np.zeros(2)
        machine = BoltzmanMachine(spins=[-1, 1], 
                                  couplings=couplings, 
                                  fields=fields)
        
        analytical_Z = 4 * np.cosh(J_val)
        computed_Z = machine.calc_Z()
        
        print(f"Analytical Z: {analytical_Z}, Computed Z: {computed_Z}")
        self.assertAlmostEqual(computed_Z, analytical_Z, places=5)

    def test_detailed_balance_logic(self):
        """
        Verifies the flip probability calculation logic.
        """
        # Since we can't inspect internal variables easily, we can verify behavior.
        # If we set a very high field, the spin should align with it.
        # If h_i = 10, spin should become +1 with high probability.
        
        machine = BoltzmanMachine(spins=[-1, 1], 
                                  couplings=np.zeros((1,1)), 
                                  fields=np.array([10.0]))
        
        # Start with opposing spin
        sample = np.array([-1.0])
        
        # Run MC step
        sample = machine.monte_carlo(sample)
        
        # Should be flipped to +1 because delta E = 2 * (-1) * 10 = -20 (energy decrease)
        # Actually delta E = 2 * s * h = 2 * (-1) * 10 = -20.
        # Wait, if s=-1 and h=10, energy is -h*s = -(-10) = 10.
        # If s=1 and h=10, energy is -h*s = -10.
        # So s=1 is lower energy.
        # Delta E (change if flip) = E_new - E_old = -10 - 10 = -20.
        # Prob flip = 1 / (1 + exp(-20)) approx 1.
        
        self.assertEqual(sample[0], 1.0)
        
    def test_negative_temperature_fix(self):
        """
        Verifies that the system doesn't maximize energy (negative temp behavior).
        """
        # If we have a single spin with field -10, it should align to -1 (lower energy).
        # Energy = -h*s. If s=1, E=-(-10)=10. If s=-1, E=-10.
        # So it should prefer -1.
        
        machine = BoltzmanMachine(spins=[-1, 1], 
                                  couplings=np.zeros((1,1)), 
                                  fields=np.array([-10.0]))
        
        sample = np.array([1.0])
        # Run many steps to equilibrate
        for _ in range(20):
            sample = machine.monte_carlo(sample)
            
        self.assertEqual(sample[0], -1.0)

if __name__ == '__main__':
    unittest.main()
