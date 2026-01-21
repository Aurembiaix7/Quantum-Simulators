import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# Constants 
hbar = 1.0
m = 1.0
k = 1.0
Er = hbar**2 * k**2 / (2 * m)
d = np.pi / k  

print(f"Recoil energy: Er = {Er}")
print(f"Lattice spacing: d = {d}")

V0 = 10.0
V0_er = V0 * Er
N_bands = 11

def hamiltonian(q, V0_total, N_bands):
    "Full lattice Hamiltonian"
    n = np.arange(-(N_bands//2), N_bands//2 + 1)
    T = hbar**2 / (2 * m) * (q + 2 * n * k)**2
    diagonal = T + V0_total / 2
    
    H = np.diag(diagonal)
    for i in range(1, N_bands):
        H[i-1, i] = H[i, i-1] = -V0_total / 4
    return H

def harmonic_energies(V0_over_Er):
    "Harmonic oscillator levels for V0 in units of Er"
    omega = 2 * np.sqrt(V0_over_Er)  # in units of Er
    E_ho = omega * (np.arange(5) + 0.5)
    return E_ho, omega


q_edge = k
H_edge = hamiltonian(q_edge, V0_er, N_bands)
energies_edge, _ = eigh(H_edge)

gap_numeric = (energies_edge[1] - energies_edge[0]) / Er

# Harmonic Oscillation
E_ho, omega = harmonic_energies(V0)
gap_ho = omega  # E1 - E0 = hbar omega

print(f"\nNumerical (full lattice) at q = k:")
print(f"E0 = {energies_edge[0]/Er:.2f} ER")
print(f"E1 = {energies_edge[1]/Er:.2f} ER")
print(f"Gap = {gap_numeric:.2f} ER")

print(f"\nHarmonic approximation:")
print(f"ω = {omega:.4f} ER")
print(f"Predicted gap = {gap_ho:.2f} ER")

print(f"numeric band gap and HO{gap_numeric:.3f} vs {gap_ho:.3f} ER")
print(f"relative error {100*(gap_numeric-gap_ho)/gap_ho:.2f}%")


# Band structure 
q_near_edge = np.linspace(0.8*k, k, 50)
E0_vals, E1_vals = [], []

for q in q_near_edge:
    H = hamiltonian(q, V0_er, N_bands)
    energies, _ = eigh(H)
    E0_vals.append(energies[0])
    E1_vals.append(energies[1])


print(f"{'V0/ER':<6} {'Num Gap':<8} {'HO Gap':<8} {'Error%':<8}")

V0_test = [1, 5, 10]
for v0 in V0_test:
    H_test = hamiltonian(k, v0 * Er, N_bands)
    e_test, _ = eigh(H_test)
    gap_num = (e_test[1] - e_test[0]) / Er
    
    eho_test, omega_test = harmonic_energies(v0)
    gap_ho_test = omega_test
    
    error = 100 * (gap_num - gap_ho_test) / gap_ho_test
    print(f"{v0:<6.0f} {gap_num:<8.3f} {gap_ho_test:<8.3f} {error:<8.1f}")
