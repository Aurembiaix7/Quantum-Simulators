import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

#Constants
hbar = 1
m = 1
k = 1 #2*np.pi/lb
Er = hbar**2*k**2/2*m
print(Er)

#Parameter
V0_list = [1,5,10]
N_bands = 5 #plane wave (-2,-1,0,1,2)
N_q = 100 
q_min = -k
q_max = k

#q-grid 
q_array = np.linspace(q_min, q_max, N_q)

def hamiltonian(q, V0, N_bands):
    n = np.arange(-N_bands//2, N_bands//2 +1)
    T = hbar**2/2*m * (q+2*n*k)**2
    diagonal = T + V0/2
    
    H = np.diag(diagonal)
    for i in range(1, N_bands):
        H[i-1, i] = H[i, i-1] = -V0/4
    #np.fill_diagonal(H[1:], -V0/4)
    #np.fill_diagonal (H[:-1], -V0/4)
    return H

#Band Structure
plt.figure(figsize = (10, 8))
colours = ['blue', 'red','green']

for i, V0 in enumerate(V0_list):
    V0_er = V0 * Er
    all_E = []

    for  q in q_array:
        H = hamiltonian(q,V0_er, N_bands)
        energies, _ = eigh(H)
        all_E.append(energies)

    all_E = np.array(all_E)
    for band in range (3):
        plt.plot(q_array/k, all_E[:,band]/Er, #
             color = colours[i], linewidth = 2, label = f'Vo{V0} ER band{band+1}' 
             if band==0 else"")
     


plt.xlabel ('q/k') #Reduced Brillouin Zone
plt.ylabel ('E') 
plt.xlim(-1,1)
plt.legend()
plt.ylim (-5,20)
plt.tight_layout()
plt.show ()

print ('\nBand gaps at zone edge (q=k):')
for i, V0 in enumerate (V0_list):
    V0_er = V0 * Er
    H_edge = hamiltonian(k, V0_er, N_bands)
    energies_edge, _ = eigh(H_edge)
    energies = np.zeros(N_bands)  
    gap1 = energies_edge[1] - energies_edge[0]
    print (f'V0 = {V0} ER: gap ={gap1/Er:.3f} ER')

q_array = np.linspace(-k, k, N_q)  
print(q_array[:5])
print(k)
print(f"q range: {q_array.min():.3f} to {q_array.max():.3f}")


for i, V0 in enumerate(V0_list):
    plt.figure(figsize=(10, 6))  
    
    all_E = []
    for q in q_array:
        H = hamiltonian(q, V0, N_bands)
        energies, _ = eigh(H)
        all_E.append(energies)
    all_E = np.array(all_E)
    
    for band in range(4):
        plt.plot(q_array/k, all_E[:,band], linewidth=2)
    
    plt.title(f'Band Structure V0 = {V0} E_R')
    plt.xlabel('q/k')
    plt.ylabel('E / E_R')
    plt.xlim(-1, 1)
    plt.ylim(-1, 13)
    plt.grid(True, alpha=0.3)
    plt.show()  