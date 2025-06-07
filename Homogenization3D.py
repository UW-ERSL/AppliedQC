import numpy as np
import matplotlib.pyplot as plt
import time
from numpy.fft import fftn, ifftn
from numba import njit
# ================================
# 1. Material Parameters (3D)
# ================================
def isotropic_stiffness_3d(E, nu):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    C = np.array([[lam + 2*mu, lam,         lam,         0,    0,    0],
                  [lam,         lam + 2*mu, lam,         0,    0,    0],
                  [lam,         lam,         lam + 2*mu, 0,    0,    0],
                  [0,           0,           0,          mu,   0,    0],
                  [0,           0,           0,          0,    mu,   0],
                  [0,           0,           0,          0,    0,    mu]])
    return C

E_matrix, nu_matrix = 1e3, 0.30
E_incl, nu_incl   = 200e9, 0.3

Emax = max(E_matrix, E_incl)
Emin = min(E_matrix, E_incl)
nuBar = (nu_matrix + nu_incl) / 2.
C0 = isotropic_stiffness_3d(Emax, nuBar)
C_incl = isotropic_stiffness_3d(Emin, nuBar)

# ================================
# 2. Domain and Inclusion (3D)
# ================================
nx = ny = nz = 128
x = np.arange(nx)
y = np.arange(ny)
z = np.arange(nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

radius = nx // 8
mask = (X - nx/2)**2 + (Y - ny/2)**2 + (Z - nz/2)**2 <= radius**2

# Prepare DeltaC: difference in stiffness on each voxel (in Voigt 6x6 format)
DeltaC = np.zeros((nx, ny, nz, 6, 6))
DeltaC[mask] = C_incl - C0

# ================================
# 3. Green Operator Construction (3D)
# ================================
# Fourier frequencies (using the same ordering as real-space: (nx, ny, nz))
@njit(cache=True)
def constructGreensFunction(nx,ny,nz,kx,ky,kz,C0):
    S0 = np.linalg.inv(C0)
    Gamma_fft = np.zeros((nx, ny, nz, 6, 6), dtype=np.complex128)

    # The mapping L(k) from displacement u (3) to strain ε (6) (in Voigt form)
    def L_matrix(k1, k2, k3):
        return np.array([
            [k1,       0,       0      ],
            [0,       k2,       0      ],
            [0,        0,      k3      ],
            [0,     k3/2,    k2/2     ],
            [k3/2,     0,     k1/2    ],
            [k2/2,   k1/2,      0      ]
        ])

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                k1 = kx[i,j,k]
                k2 = ky[i,j,k]
                k3 = kz[i,j,k]
                # Avoid singularity for zero wavevector:
                if np.abs(k1) < 1e-12 and np.abs(k2) < 1e-12 and np.abs(k3) < 1e-12:
                    continue
                Lk = L_matrix(k1, k2, k3)  # shape (6,3)
                A = Lk.T @ C0 @ Lk         # shape (3,3)
                A_inv = np.linalg.inv(A)
                # Green operator in Fourier space:
                # According to the derivation: Gamma(k) = - L(k) @ A_inv @ L(k).T
                Gamma_fft[i, j, k] = - Lk @ A_inv @ Lk.T
    return Gamma_fft

time_start = time.time()
freq_x = np.fft.fftfreq(nx)
freq_y = np.fft.fftfreq(ny)
freq_z = np.fft.fftfreq(nz)
kx, ky, kz = np.meshgrid(freq_x, freq_y, freq_z, indexing='ij')
# Scale to angular frequencies:
kx *= 2 * np.pi
ky *= 2 * np.pi
kz *= 2 * np.pi
Gamma_fft = constructGreensFunction(nx,ny,nz,kx,ky,kz, C0)    
time_end = time.time()
print(f"Green operator constructed in {time_end - time_start:.2f} seconds.")

# ================================
# 4. Iterative Solver (3D)
# ================================
alpha = 0.5
max_iters = 500
tol = 1e-8

# eps_field shape: (nx, ny, nz, 6)
# Loop over 6 independent prescribed macroscopic strains
time_start = time.time()
for mode in range(6):
    eps0 = np.zeros(6)
    eps0[mode] = 0.01  # Apply a small strain in the chosen component
    eps_field = np.full((nx, ny, nz, 6), eps0, dtype=float)
    
    for n in range(max_iters):
        # Polarization stress: tau = DeltaC : eps_field (Einstein contraction over the last index)
        tau = np.einsum('ijkab,ijkb->ijka', DeltaC, eps_field)
        # FFT of tau over spatial dimensions
        tau_k = fftn(tau, axes=(0,1,2), norm="ortho")
        # Convolution with the Green operator in Fourier space:
        eps_k = np.einsum('ijkab,ijkb->ijka', Gamma_fft, tau_k)
        eps_conv = ifftn(eps_k, axes=(0,1,2), norm="ortho").real
        
        eps_new = eps_field + alpha * (eps0 - eps_field - eps_conv)
        diff = np.linalg.norm(eps_new - eps_field)
        if diff < tol * np.linalg.norm(eps0) * nx * ny * nz:
            eps_field = eps_new
            print(f"Mode {mode+1} converged in {n} iterations.")
            break
        eps_field = eps_new
    else:
        print(f"Mode {mode+1} did not converge.")

time_end = time.time()
print(f"Iterative solver completed in {time_end - time_start:.2f} seconds.")
print("Final strain field shape:", eps_field.shape)

# ================================
# 5. Plotting a Slice (3D)
# ================================
# Plot the central slice of the ε_xx component (index 0)
mid_slice = nz // 2
plt.imshow(eps_field[:,:,mid_slice,0], cmap='viridis', origin='lower')
plt.colorbar(label='Strain ε_xx')
plt.title("3D Eyre-Milton: Central Slice of Strain Field ε_xx")
plt.axis('off')
plt.tight_layout()
plt.show()
