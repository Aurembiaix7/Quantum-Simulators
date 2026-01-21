import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# Constants
k = 1
N_bands = 5
N_x = 100
L = 4*np.pi/k
x_grid = np.linspace(-L/2, L/2, N_x)
dx = x_grid[1] - x_grid[0]
N_q = 50
q_min = -k
q_max = k
q_array = np.linspace(q_min, q_max, N_q)

def hamiltonian(q, V0, N_bands):
    n_vec = np.arange(-N_bands//2, N_bands//2 + 1)
    T = 0.5 * (q + 2*n_vec*k)**2
    diagonal = T + V0/2
    H = np.diag(diagonal)
    for i in range(1, N_bands):
        H[i-1, i] = -V0/4
        H[i, i-1] = -V0/4
    return H

def bloch_wave_lowest (q, V0, N_bands, x_grid):
    H= hamiltonian(q, V0, N_bands)
    energies, coeffs = eigh (H)
    band = 0
    n_vec = np.arange(-N_bands//2, N_bands//2 + 1)
    
    psi_q = np.zeros_like(x_grid, dtype=complex)
    for n_idx, n in enumerate(n_vec):
            psi_q += coeffs[n_idx, band] * np.exp(1j * (q + 2*n*k) * x_grid)
        
    norm = np.sqrt(np.sum(np.abs(psi_q)**2) * dx)
    psi_q /= norm
    return psi_q, energies[band]

def wannier_low (V0, x_grid, q_array, xj =0.0):
    w = np.zeros_like(x_grid, dtype=complex)
    for q in q_array:
        psi_q, _ = bloch_wave_lowest(q, V0, N_bands, x_grid)
        w += np.exp(-1j*q*xj)*psi_q
    w /= np.sqrt(len(q_array))

    norm = np.sqrt(np.sum(np.abs(w)**2 * dx))

    w/= norm
    return w

# plot
V0_list = [1, 5, 10]
fig, (ax1, ax2) = plt.subplots(1, 2 ,figsize=(15, 12))

for V0 in V0_list:
     w = wannier_low(V0, x_grid, q_array, xj=0 )
     ax1.plot (x_grid * k/np.pi, np.real (w), label = f'V0={V0}ER')
     ax2.plot (x_grid * k / np.pi, np.abs(w)**2, label = f'V0={V0}ER')

for site in range (-2,3):
     ax1.axvline(site, color = 'k', linestyle=':', alpha=0.3)
     ax2.axvline(site, color = 'k', linestyle=':', alpha=0.3)
ax1.set_xlim(-2, 2)
ax1.set_xlabel('x * k / π')
ax1.set_ylabel('Wannier function')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.set_xlim(-2, 2)
ax2.set_xlabel('x * k / π')
ax2.set_ylabel('Wannier function')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

