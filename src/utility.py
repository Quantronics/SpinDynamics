"""Utility functions."""
import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from scipy.sparse import dia_matrix
from scipy.optimize import curve_fit

def generate_S(S: float) -> (Qobj, Qobj, Qobj):
    mS = np.arange(S, -S-1, -1)
    nS = len(mS)
    S_plus = Qobj(dia_matrix((np.sqrt(S*(S+1)-mS*(mS+1)),1), shape = (nS, nS)).A)
    S_minus = Qobj(np.transpose(S_plus))
    Sx = Qobj(0.5*(S_plus+S_minus))
    Sy = Qobj(-0.5*1j*(S_plus-S_minus))
    Sz = Qobj(dia_matrix((mS,0), shape = (nS, nS)).A)

    return (Sx,Sy,Sz)

