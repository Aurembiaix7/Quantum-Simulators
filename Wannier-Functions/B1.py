import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# Constants
k = 1
N_bands = 5
N_x = 100
L = 4*np.pi/k
x = np.linspace(-L/2, L/2, N_x)
dx = x[1] - x[0]

def hamiltonian(q, V0, N_bands):
    n_vec = np.arange(-N_bands//2, N_bands//2 + 1)
    T = 0.5 * (q + 2*n_vec*k)**2
    diagonal = T + V0/2
    H = np.diag(diagonal)
    for i in range(1, N_bands):
        H[i-1, i] = -V0/4
        H[i, i-1] = -V0/4
    return H

# Bloch wavefunctions at q=0
V0_list = [1, 5, 10]
fig, axes = plt.subplots(3, 2, figsize=(15, 12))

for i_V0, V0 in enumerate(V0_list):
    q = 0
    H = hamiltonian(q, V0, N_bands)
    energies, coeffs = eigh(H)
    
    for i_band, band in enumerate([0, 1]):
        # Reconstruct wave function(x)
        psi = np.zeros(N_x, dtype=complex)
        n_vec = np.arange(-N_bands//2, N_bands//2 + 1)
        for n_idx, n in enumerate(n_vec):
            psi += coeffs[n_idx, band] * np.exp(1j * 2*n*k*x)
        
        # Normalization
        norm = np.sqrt(np.sum(np.abs(psi)**2) * dx)
        if norm > 1e-12:
            psi = psi / norm
        
        # Plot
        #axes[i_V0, i_band].plot(x*k/np.pi, np.real(psi), 'b-', linewidth=2, label='Re[ψ]')
        axes[i_V0, i_band].plot(x*k/np.pi, np.abs(psi)**2, 'r--', linewidth=2, label='|ψ|²')
        
        # Lattice sites
        for site in [-1, 0, 1]:
            axes[i_V0, i_band].axvline(site, color='k', linestyle=':', alpha=0.7)
        
        axes[i_V0, i_band].set_title(f'V₀={V0}, Band {band+1} (E={energies[band]:.2f})')
        axes[i_V0, i_band].set_xlim(-2, 2)
        #axes[i_V0, i_band].legend()
        axes[i_V0, i_band].grid(True, alpha=0.3)

#plt.suptitle('Wavefunctions at q=0', fontsize=16)
plt.tight_layout()
plt.show()
