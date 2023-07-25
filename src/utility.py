"""Utility functions."""
import numpy as np
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


def initialize_time_list(N_pulses, tau, pulse_duration, kappa):
    """Initialize a time list with increased resolution when pulses are applied.
    Note: If there are duplicate time instants, the solver will not run the simulation.

    Parameters
    ----------
    N_pulses: int
        Number of pulses to apply to the e- spin (N_pulses//2 unit
        control sequences in total).

    tau: float
        Free evolution time of the sequence (the interpulse is 2*tau).

    pulse_duration: float
        Duration of a single pulse.

    Returns
    -------
    t: np.ndarray
        Time list to be used by the solver.
    """
    t = []
    t1 = tau - 0.1 * pulse_duration
    t2 = tau + pulse_duration + 4 / kappa

    # Setting up a good resolution for the first pi/2 pulse
    t = np.linspace(0, pulse_duration + 4 / kappa, 200).tolist()
    t += np.linspace(pulse_duration + 4 / kappa + 1e-3, t1 - 1e-3, 50).tolist()
    t += np.linspace(t1, t2, 100).tolist()
    t += np.linspace(t2 + 1e-3, 2 * tau - 1e-3, 50).tolist()

    for k in range(1, N_pulses):
        step = k * 2 * tau
        t_before_pulse = np.linspace(0 + step, t1 - 1e-3 + step, 50).tolist()
        t_during_pulse = np.linspace(t1 + step, t2 + step, 100).tolist()
        t_after_pulse = np.linspace(t2 + 1e-3 + step, 2 * tau - 1e-3 + step, 50).tolist()
        t += (t_before_pulse + t_during_pulse + t_after_pulse)

    # Adding time instants for final pi/2 pulse
    t += np.linspace(N_pulses * 2 * tau, N_pulses * 2 * tau + t2, 200).tolist()

    return np.array(t)


def generate_identity_ops(*args):
    """Generates a tuple of an arbitrary number of qeye operators with
    the dimensions provided by the user.
    The resulting tuple can then be passed to any QuTip function,
    especially expect().
    
    Parameters
    ----------
    *args: arbitrarily-long list of operators dimensions
    
    Returns
    -------
    ops: tuple of qeye operators with dimensions given in *args.
    """
    ops = ()
    n_spins = len(args)
    for i in range(n_spins):
        ops += (qeye(args[i]),)
        
    return ops


def fit_cos(tt, yy):
    """Fits cos to the input time sequence and return the best fitting
    paremeters. The initial guess is automatically determined in an optimal
    way by using the FFT.
    
    Parameters
    ----------
    tt: np.ndarray of float values
        Evenly-spaced time sequence.
        
    yy: np.ndarray of float values
        Data to fit.
    
    Returns
    -------
    Fitting parameters of the best matching cos function
    """
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    def cos_func(t, A, w, p, c):  return A * np.cos(w*t + p) + c

    popt, pcov = curve_fit(cos_func, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.cos(w*t + p) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}