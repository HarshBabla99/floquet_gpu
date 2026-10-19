import numpy as np

# All energies in GHz (without 2pi)
# All times in ns

tpi = 2. * np.pi

params = {
    # transmon params
    'EJ': tpi * 20, 
    'EC': tpi * 0.2, 
    'ng': 0., 
    'ncut': 41,
    'transmon_dim': 15,
    #
    # kerr cat params
    'K': tpi * 0.5e-3, # Enters the Hamiltonian as H = -K adag^2 a^2, so K = -(E02 - 2*E01 + E00)/2
    'delta_const': 0.0,
    'resonator_dim' : 40, # truncation: max(photon_nums) +  5*sqrt(max(photon_nums)) ~ 35 for max(photon_nums) = 15
    'omega_r_values' : tpi * np.linspace(6.5, 7.5, 501), # omega_r values are spaced by 2 MHz.
    'photon_nums': np.linspace(0, 15, 201),
    #
    # coupling
    # chi > 2*K*max(photon_nums), to ensure eps_2 < chi/2 i.e. e doesn't bifurcate
    'chi' : tpi * 20e-3,
}
