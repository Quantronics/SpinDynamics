"""Helper functions used to generate the hyperfine constants."""
import numpy as np
import matplotlib.pyplot as plt
from src.utility import generate_S
import pandas as pd

# Physical constants
mu_B = 9.27401007831e-24   # Bohr magneton in J/T
h    = 6.6260693e-34       # Plank constant
mu_0 = 12.566370614e-7     # Vacuum permeability
mu_N = 5.0507836991e-27    # Nuclear magneton in J/T


def get_hyperfine_tensor(xyz, S, I, g_a, g_b, g_c, g_N):
    """
    Calculates the hyperfine tensor and hamiltonian due to the dipole-dipole interaction between two spins
    The expression for the hyperfine tensor used is detailed in Le Dantec's Thesis (2022), p36
 
    --------------------------------------------------------------------------------------------------------------
    
        - xyz (np.array): Cartesian coordinates of the nuclear spin
        - B (np.array): External magnetic field applied to the system
        - return_xyz_tensor (bool): return the tensor and hamiltonian on the original basis (True) 
                                    or in the rotated basis (z' along B) (False)
        
    --------------------------------------------------------------------------------------------------------------
    
        - Tdd (np.array): hyperfine coupling tensor
        - Hdd (np.array): hamiltonian of interaction
        
    """
    
    x, y, z = xyz
    r = np.linalg.norm(xyz)
    
    (Sx,Sy,Sz) = generate_S(S)
    nS = int(2*S+1)
    
    (Ix,Iy,Iz) = generate_S(I)
    nI = int(2*I+1)
    
    Tdd = np.zeros((3, 3)) # dipole-dipole tensor
    prefactor = g_N * r**-5 * mu_B * mu_N * mu_0/(4*np.pi)

    # diagonal
    Tdd[0,0] = g_a * (r**2 - 3*x**2)
    Tdd[1,1] = g_b * (r**2 - 3*y**2)
    Tdd[2,2] = g_c * (r**2 - 3*z**2)
    
    # xy
    Tdd[0,1] = g_a * (-3)*x*y
    Tdd[1,0] = g_b * (-3)*y*x
    
    # xz
    Tdd[0,2] = g_a * (-3)*x*z
    Tdd[2,0] = g_c * (-3)*z*x
    
    # yz
    Tdd[1,2] = g_b * (-3)*y*z
    Tdd[2,1] = g_c * (-3)*z*y
    
    Tdd *= prefactor
    
    Hdd = (
        Tdd[0,0] * np.kron(Sx, Ix) + Tdd[1,1] * np.kron(Sy, Iy) + Tdd[2,2] * np.kron(Sz, Iz) + 
        Tdd[0,1] * np.kron(Sx, Iy) + Tdd[1,0] * np.kron(Sy, Ix) +
        Tdd[0,2] * np.kron(Sx, Iz) + Tdd[2,0] * np.kron(Sz, Ix) +
        Tdd[1,2] * np.kron(Sy, Iz) + Tdd[2,1] * np.kron(Sz, Iy)
    )
    return Tdd, Hdd


def get_hyperfine_constants(atom_pos, S, I, B_field, g_a, g_b, g_c, g_N):
    """
    Given a tugnsten ion and a magnetic field calculate the constants A and B from the secular approximation.
    This needs testing when the magnetic field is applied in the y direction
    
    --------------------------------------------------------------------------------------------------------------
    
        - B_field (np.array): External magnetic field applied to the system
        
    --------------------------------------------------------------------------------------------------------------
    
        - A (float): coupling constant along Sz'Iz
        - B (float): coupling constant along Sz'Ix
        
    """
    
    (Sx,Sy,Sz) = generate_S(S)
    nS = int(2*S+1)
    
    (Ix,Iy,Iz) = generate_S(I)
    nI = int(2*I+1)
    
    # Calculate the Zeeman hamiltonians and find their respective unitary rotations for diagonalization
    h_Zeeman_spin = mu_B * (B_field[0]*g_a*Sx + B_field[1]*g_b*Sy + B_field[2]*g_c*Sz)
    h_Zeeman_nucl = mu_N * g_N * (B_field[0]*Ix + B_field[1]*Iy + B_field[2]*Iz)

    _, rotmat_spin = np.linalg.eigh(h_Zeeman_spin)
    _, rotmat_nucl = np.linalg.eigh(h_Zeeman_nucl)
    
    # Calculate the general rotation matrix
    rotmat = np.kron(np.identity(nS), rotmat_nucl) @ np.kron(rotmat_spin, np.identity(nI))
        
    # Obtain the dipole-dipole hamiltonian for the given ion and rotate it to the new frame
    _, h_dipole = get_hyperfine_tensor(atom_pos, S, I, g_a, g_b, g_c, g_N)
    rot_h_dipole = rotmat.T.conj() @ h_dipole @ rotmat
    
    Sz_pp = rotmat.T.conj() @ np.kron(Sz, np.identity(nI)) @ rotmat
    Iz_p = rotmat.T.conj() @ np.kron(np.identity(nS), Iz) @ rotmat
    Ix_p = rotmat.T.conj() @ np.kron(np.identity(nS), Ix) @ rotmat

    # Through the secular approximation obtain the constants A and B
    # When we plot the different elementary matrices (SxIx, SxIy, ..., SzIz) that compose the hyperfine dipole-dipole Hamiltonian S.A.I, 
    # we see that the SzIz element is the only one to have a non-zero coefficient on [0,0], and that SzIx is the only one to have a 
    # non-zero coefficient on [0,1]. Therefore, we extract A and B by selecting the coefficients on [0,0] and [0,1] from the dipole-dipole Hamiltonian.
    A = abs(rot_h_dipole[0,0]) / (h*1e3 * S*I)
    B = abs(rot_h_dipole[0,1]) / (h*1e3 * S*I)
    return A, B