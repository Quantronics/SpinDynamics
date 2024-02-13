import numpy as np
from qutip import *
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import curve_fit
from cmath import exp

############################## Pulse Definitions ##############################
omega_LO = 0
def ErfRising(t, sigma, amplitude, t0):
    return amplitude * (erf((t - t0) / sigma) + 1) / 2
def square_pulse(t, args):
    pulse_duration = args['pulse_duration']
    amplitude = args['amplitude']
    t0 = args['t0']
    if t > (t0 - pulse_duration/2) and t < (t0 + pulse_duration/2):
        return amplitude/2 
    else:
        return 0
def square_pulse_p(t, args): return square_pulse(t, args) * np.exp(+1j*((args['detuning']+omega_LO)*t + args['phase']))
def square_pulse_m(t, args): return square_pulse(t, args) * np.exp(-1j*((args['detuning']+omega_LO)*t + args['phase']))        
def gaussian_pulse(t, args):
    sigma = args['sigma_gauss']
    amplitude = args['amplitude']
    t0 = args['t0']
    if t > (t0-sigma) and t < (t0+sigma):
        return amplitude/2 * np.exp(-((t - t0) ** 2) / (2 * (sigma/6) ** 2))
    else: 
        return 0
def gaussian_pulse_p(t, args): return gaussian_pulse(t, args) * np.exp(+1j*((args['detuning']+omega_LO)*t + args['phase']))
def gaussian_pulse_m(t, args): return gaussian_pulse(t, args) * np.exp(-1j*((args['detuning']+omega_LO)*t + args['phase']))
def flattop_pulse(t, args):
    pulse_duration = args['pulse_duration']
    sigma = args['sigma_raise']
    amplitude = args['amplitude']
    t0 = args['t0']
    if t > (t0 - pulse_duration/2 - 6*sigma) and t < (t0 - pulse_duration/2):
        return ErfRising(t, sigma, amplitude/2, t0 - pulse_duration/2 - 3*sigma) 
    elif t > (t0 - pulse_duration/2) and t < (t0 + pulse_duration/2):
        return amplitude/2 
    elif t > (t0 + pulse_duration/2) and t < (t0 + pulse_duration/2 + 6*sigma):
        return (amplitude/2-ErfRising(t, sigma, amplitude/2, t0 + pulse_duration/2 + 3*sigma))
    else:
        return 0
def flattop_pulse_p(t, args): return flattop_pulse(t, args) * np.exp(+1j*((args['detuning']+omega_LO)*t + args['phase']))
def flattop_pulse_m(t, args): return flattop_pulse(t, args) * np.exp(-1j*((args['detuning']+omega_LO)*t + args['phase']))
def pulse_sequence_p(t, args):
    """
    Generate an arbitrary pulse sequence based on the argument dictionary using the Qutip notation
    Input:
        - t (float): time in microseconds
        - args (dict): dictionary with two elements:
            -- pulses: (list): list of strings that can be "s", "g" or "ft" refering to either square, gauss or flattop pulse.
            -- args_pulses: (list): list of same length as pulses that contains the arguments for the corresponding pulse.
    Outputs:
        - H (array): hamiltonian in the style of Qutip
    """
    H = 0+0j
    pulse_map_p = {
        "s": square_pulse_p,
        "g": gaussian_pulse_p,
        "ft": flattop_pulse_p
    }
    for pulse, args_pulse in zip(args["pulses"], args["args_pulses"]):
        try: pulse_fnc = pulse_map_p[pulse]
        except KeyError: raise KeyError("The only accepted pulse types are 's', 'g' or 'ft'")
        H += pulse_fnc(t, args_pulse)
    return H
def pulse_sequence_m(t, args):
    H = 0+0j
    pulse_map_m = {
        "s": square_pulse_m,
        "g": gaussian_pulse_m,
        "ft": flattop_pulse_m
    }
    for pulse, args_pulse in zip(args["pulses"], args["args_pulses"]):
        try: pulse_fnc = pulse_map_m[pulse]
        except KeyError: raise KeyError("The only accepted pulse types are 's', 'g' or 'ft'")
        H += pulse_fnc(t, args_pulse)
    return H

############################## Multiprocessing Functions ##############################

def mp_raman_spec_ft(H, ket0, dur, sigma_raise, det, A, Delta, seq_args, Sz, nS):
    """
    Get Raman Rabi with flattop pulses
    Parameters
    ----------
    H: Qobj
        H0 + H(t)
        
    ket0: Qobj
        initial state
        
    dur: float
        pulse duration (us)

    sigma_raise: float
        sigma_raise * 6 is the ramp up time of a flattop pulse (us)
        
    seq_args: dictionary of float values
        parameters that define the pulse shape
            
    Sz: Qobj
        Pauli Z operator
        
    nS: int
        Dimension of the electron spin subspace
    """
    t0 = dur/2+sigma_raise*6
    t = np.linspace(0, t0*2, int(t0/2))
    seq_args["args_pulses"][0]["pulse_duration"] = dur
    seq_args["args_pulses"][1]["pulse_duration"] = dur
    seq_args["args_pulses"][0]["t0"] = t0
    seq_args["args_pulses"][1]["t0"] = t0
    seq_args["args_pulses"][0]["detuning"] = -(A)/2 - Delta + det
    seq_args["args_pulses"][1]["detuning"] = -(A)/2 - Delta
    result = mesolve(H, ket0, t, [], [], args=seq_args)

    return expect(tensor(qeye(nS), Sz), result.states)[-1]

def mp_raman_spec_s(H, ket0, dur, sigma_raise, det, A, Delta, seq_args, Sz, nS):
    """
    Get Raman Rabi with square pulses
    Parameters
    ----------
    H: Qobj
        H0 + H(t)
        
    ket0: Qobj
        initial state
        
    dur: float
        pulse duration (us)

    sigma_raise: float
        sigma_raise * 6 is the ramp up time of a flattop pulse (us)
        
    seq_args: dictionary of float values
        parameters that define the pulse shape
            
    Sz: Qobj
        Pauli Z operator
        
    nS: int
        Dimension of the electron spin subspace
    """
    t0 = dur/2+sigma_raise*3
    t = np.linspace(0, t0*2, int(t0))
    seq_args["args_pulses"][0]["pulse_duration"] = dur+sigma_raise*6
    seq_args["args_pulses"][1]["pulse_duration"] = dur+sigma_raise*6
    seq_args["args_pulses"][0]["t0"] = t0
    seq_args["args_pulses"][1]["t0"] = t0
    seq_args["args_pulses"][0]["detuning"] = -(A)/2 + Delta + det
    seq_args["args_pulses"][1]["detuning"] = -(A)/2 + Delta
    result = mesolve(H, ket0, t, [], [], args=seq_args)

    return expect(tensor(qeye(nS), Sz), result.states)[-1]

def mp_raman_rabi_ft(H, ket0, dur, sigma_raise, seq_args, Sz, nS):
    """
    Get Raman Rabi with flattop pulses
    Parameters
    ----------
    H: Qobj
        H0 + H(t)
        
    ket0: Qobj
        initial state
        
    dur: float
        pulse duration (us)

    sigma_raise: float
        sigma_raise * 6 is the ramp up time of a flattop pulse (us)
        
    seq_args: dictionary of float values
        parameters that define the pulse shape
            
    Sz: Qobj
        Pauli Z operator
        
    nS: int
        Dimension of the electron spin subspace
    """
    t0 = dur/2+sigma_raise*6
    t = np.linspace(0, t0*2, int(t0/2))
    seq_args["args_pulses"][0]["pulse_duration"] = dur
    seq_args["args_pulses"][1]["pulse_duration"] = dur
    seq_args["args_pulses"][0]["t0"] = t0
    seq_args["args_pulses"][1]["t0"] = t0
    result = mesolve(H, ket0, t, [], [], args=seq_args)

    return expect(tensor(qeye(nS), Sz), result.states)[-1]

def mp_raman_rabi_s(H, ket0, t, c_ops, dur, sigma_raise, seq_args, Sz, nS):
    """
    Get Raman Rabi with square pulses
    Parameters
    ----------
    H: Qobj
        H0 + H(t)
        
    ket0: Qobj
        initial state
        
    dur: float
        pulse duration (us)

    sigma_raise: float
        sigma_raise * 6 is the ramp up time of a flattop pulse (us)
        
    seq_args: dictionary of float values
        parameters that define the pulse shape
            
    Sz: Qobj
        Pauli Z operator
        
    nS: int
        Dimension of the electron spin subspace
    """

    seq_args["args_pulses"][0]["pulse_duration"] = dur+sigma_raise*6
    seq_args["args_pulses"][1]["pulse_duration"] = dur+sigma_raise*6
    seq_args["args_pulses"][0]["t0"] = t0
    seq_args["args_pulses"][1]["t0"] = t0
    result = mesolve(H, ket0, t, c_ops, [], args=seq_args)

    return expect(tensor(qeye(nS), Sz), result.states), expect(tensor(Sz, qeye(nS)), result.states)

def mp_get_allowed_rabi(H, ket0, allowed_chevron_det, amp, A, dur, Sz, nI):
    gamma_R = (0.8*1000)**-1 # (0.8 ms)**-1
    c_ops = np.sqrt(gamma_R) * tensor(sigmam(), qeye(nI))
    t = np.linspace(0, dur+100, int(dur))
    args = {
        'sigma_raise': 1,
        'sigma_gauss': dur,
        'pulse_duration': dur,
        'amplitude': amp,
        't0': dur/2,
        'detuning': -A/2 + allowed_chevron_det, # 2pi*f in MHz
        'phase': 0,
    }
    result = mesolve(H, ket0, t, [], [], args=args)

    return expect(tensor(Sz, qeye(nI)), result.states)[-1]
