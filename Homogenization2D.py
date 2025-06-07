import numpy as np
import matplotlib.pyplot as plt
import time
from numba import njit
# ================================
# 1. Material Parameters
# ================================
E_matrix, nu_matrix = 1e3, 0.30
E_incl, nu_incl = 200e9, 0.3

def plane_strain_stiffness(E, nu):
    lam = E * nu / ((1 + nu)*(1 - 2*nu))
    mu = E / (2 * (1 + nu))
    C11 = lam + 2*mu
    C12 = lam
    C66 = 2*mu
    return np.array([[C11, C12, 0],
                     [C12, C11, 0],
                     [0,   0,   C66]])


Emax = max(E_matrix, E_incl)
Emin = min(E_matrix, E_incl)
nuBar = (nu_matrix+nu_incl)/2
C0 = plane_strain_stiffness(Emax, nuBar)
C_incl = plane_strain_stiffness(Emin, nuBar)

# ================================
# 2. Domain and Inclusion
# ================================
nx = ny = 512
x = np.arange(nx)
y = np.arange(ny)
X, Y = np.meshgrid(x, y)
radius = nx // 8
mask = (X - nx/2)**2 + (Y - ny/2)**2 <= radius**2

# Plot the mask
# plt.imshow(mask, cmap='gray', origin='lower')
# plt.colorbar(label='Mask')
# plt.title("Inclusion Mask")
# plt.axis('off')
# plt.tight_layout()
# plt.show()

DeltaC = np.zeros((ny, nx, 3, 3))
DeltaC[mask] = C_incl - C0


# ================================
# 3. Green Operator Construction
# ================================
time_start = time.time()
freq_x = np.fft.fftfreq(nx)
freq_y = np.fft.fftfreq(ny)
kx, ky = np.meshgrid(freq_x, freq_y, indexing='ij')
Gamma_fft = np.zeros((ny, nx, 3, 3), dtype=np.complex128)



@njit(cache=True)
def constructGreensFunction(nx,ny,kx,ky,C0):
    S0 = np.linalg.inv(C0)
    Gamma_fft = np.zeros((ny, nx, 3, 3), dtype=np.complex128)
    for i in range(ny):
        for j in range(nx):
            k1 = 2*np.pi * kx[i, j]
            k2 = 2*np.pi * ky[i, j]
            if (np.abs(k1) < 1e-12) or (np.abs(k2) < 1e-12):
                continue
        
            K = np.array([[k1*C0[0,0], k1*C0[0,1], k2*C0[0,2]],
                            [k2*C0[1,0], k2*C0[1,1], k1*C0[1,2]]])
            P = np.array([[k1, 0,  k2],
                            [0,  k2, k1]])
            A = K @ S0 @ K.T   
            A_inv = np.linalg.inv(A) # 2 x 2 matrix inversion
            Gamma_fft[i, j] = - S0 @ K.T @ A_inv @ P
            
    return Gamma_fft

Gamma_fft = constructGreensFunction(nx,ny,kx,ky,C0)

time_end = time.time()
print(f"Green operator constructed in {time_end - time_start:.2f} seconds.")    
# ================================
# 4. Iterative Solver
# ================================

alpha = 0.5
max_iters = 500
tol = 1e-8
time_start = time.time()
for mode in range(3):
    eps0 = np.zeros(3)
    eps0[mode] = 0.01  # Initial strain in one direction
    eps_field = np.full((ny, nx, 3), eps0)
    for n in range(max_iters):
        tau = np.einsum('ijab,ijb->ija', DeltaC, eps_field)
        tau_k = np.zeros_like(tau, dtype=np.complex128)
        for c in range(3):
            tau_k[..., c] = np.fft.fft2(tau[..., c], norm="ortho")
        eps_k = np.einsum('ijab,ijb->ija', Gamma_fft, tau_k)
        eps_conv = np.zeros_like(eps_field, dtype=float)
        for c in range(3):
            eps_conv[..., c] = np.fft.ifft2(eps_k[..., c], norm="ortho").real
        eps_new = eps_field + alpha * (eps0 - eps_field - eps_conv)
        if np.linalg.norm(eps_new - eps_field) < tol * np.linalg.norm(eps0) * nx * ny:
            eps_field = eps_new
            print(f"Converged in {n} iterations.")
            break
        eps_field = eps_new
    else:
        print("Did not converge.")

time_end = time.time()

# Recover the homogenized elasticity tensor.
# For a linear elastic composite, the effective (homogenized) stiffness is defined
# by the averaged stress response to a given average strain.
#
# In our iterative procedure the macroscopic (applied) strain eps0 was imposed so that
# the cell average of the computed local strain equals eps0. Thus, the local stress field is
#
#   σ(x) = C(x) ε(x) = [C0 + ΔC(x)] ε(x)
#
# and the macroscopic stress is
#
#   σ̄ = <σ(x)> = (1/vol) ∫ σ(x) dx.
#
# Since for the current loading mode the macroscopic strain is eps0,
# the corresponding column of the homogenized elasticity tensor is given by
#
#   C_hom(:,i) = σ̄ / eps0_i.
#
# Note:
# For a full homogenized tensor, you should solve for three independent applied strains.
# Here we compute the effective stiffness for the current (last) loading mode.
# (In our simulation, eps0 was set to 0.01 in one component.)
#
# Compute the local stress field:
sigma_field = np.einsum('ij,klj->kli', C0, eps_field) + np.einsum('klij,klj->kli', DeltaC, eps_field)
# Compute the spatial average (macroscopic stress):
sigma_avg = np.mean(sigma_field, axis=(0, 1))
# Recover the effective stiffness column.
# In this load, the nonzero component of the applied strain was 0.01.
C_hom_column = sigma_avg / 0.01

print("Homogenized elasticity tensor column for the applied loading:")
print(C_hom_column)

# To get the full tensor, repeat the procedure for three independent macroscopic strain states.
print(f"Iterative solver completed in {time_end - time_start:.2f} seconds.")  
print("Final strain field shape:", eps_field.shape)  
# ================================
# 5. Plotting
# ================================
plt.imshow(eps_field[..., 0], cmap='viridis')
plt.colorbar(label='Strain ε_xx')
plt.title("Eyre-Milton: Strain Field ε_xx")
plt.axis('off')
plt.tight_layout()
plt.show()
