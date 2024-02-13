import numpy as np
from qutip import *

def print_x(x, y):
    return x*y

def get_allowed_rabi(H, ket0, allowed_chevron_det, amp, A, dur, Sz, nI):

    t = np.linspace(0, 500, 100)

    args = {
        'sigma_raise': 1,
        'sigma_gauss': dur,
        'pulse_duration': dur,
        'amplitude': amp,
        't0': 30 + dur/2,
        'detuning': -A/2 + allowed_chevron_det, # 2pi*f in MHz
        'phase': 0,
    }
    result = mesolve(H, ket0, t, [], [], args=args)

    return expect(tensor(Sz, qeye(nI)), result.states)[-1]

def parallelize_raman_rabi_ft(H0, H_p, H_m, ket0, dur, sigma_raise, seq_args):
    """
    Get Raman Rabi
    :amp1: sideband drive amplitude (MHz * 2pi)
    :amp2: allowed drive amplitude (MHz * 2pi)
    :Delta: Raman detuning (MHz * 2pi)
    :pulse_duration: pulse duration (us) of the flat part for a flattop pulse (or equivalent for a square pulse with the same area)
    :resonant_freq: position of the peak obtain by fitting the spectroscopy ((MHz * 2pi))
    :sigma_raise: sigma_raise * 6 is the ramp up time of a flattop pulse (us)
    :pulse_shape: 'ft': flattop, 's': square pulse
    """
    ####### Drive at resonance for Raman Rabi #######

    t0 = dur/2+sigma_raise*6
    t = np.linspace(0, t0*2, int(t0/5))
    seq_args["args_pulses"][0]["pulse_duration"] = dur
    seq_args["args_pulses"][1]["pulse_duration"] = dur
    seq_args["args_pulses"][0]["t0"] = t0
    seq_args["args_pulses"][1]["t0"] = t0

    result = mesolve([H0, H_p, H_m], ket0, t, [], [], args=seq_args)


    return expect(tensor(qeye(nS), Sz), result.states)[-1]

# def parallelize_raman_rabi(amp1, amp2, Delta, dur, resonant_freq, chevron_det, sigma_raise, pulse_shape):
#     """
#     Get Raman Rabi
#     :amp1: sideband drive amplitude (MHz * 2pi)
#     :amp2: allowed drive amplitude (MHz * 2pi)
#     :Delta: Raman detuning (MHz * 2pi)
#     :pulse_duration: pulse duration (us) of the flat part for a flattop pulse (or equivalent for a square pulse with the same area)
#     :resonant_freq: position of the peak obtain by fitting the spectroscopy ((MHz * 2pi))
#     :sigma_raise: sigma_raise * 6 is the ramp up time of a flattop pulse (us)
#     :pulse_shape: 'ft': flattop, 's': square pulse
#     """
#     ####### Drive at resonance for Raman Rabi #######
#     pulses, args_pulses = [], []
#     pulses.append(pulse_shape)
#     pulses.append(pulse_shape)

#     args_pulses.append({
#         'sigma_raise': sigma_raise,
#         'amplitude': amp1,
#         'phase': 0
#     })
#     args_pulses.append({
#         'sigma_raise': sigma_raise,
#         'amplitude': amp2,
#         'phase': 0
#     })
#     seq_args = {"pulses": pulses, "args_pulses": args_pulses}
#     H_p = [tensor(sigmap(),qeye(nI)), pulse_sequence_p]
#     H_m = [tensor(sigmam(),qeye(nI)), pulse_sequence_m]
#     ket0 = H0.eigenstates()[1][3] # down-down state

#     if pulse_shape == "ft":
#         t0 = dur/2+sigma_raise*6
#         t = np.linspace(0, t0*2, int(t0/5))
#         seq_args["args_pulses"][0]["pulse_duration"] = dur
#         seq_args["args_pulses"][1]["pulse_duration"] = dur
#         seq_args["args_pulses"][0]["t0"] = t0
#         seq_args["args_pulses"][1]["t0"] = t0
#     if pulse_shape == "s":
#         t0 = dur/2+sigma_raise*3
#         t = np.linspace(0, t0*2, int(t0/5))
#         seq_args["args_pulses"][0]["pulse_duration"] = dur+sigma_raise*6
#         seq_args["args_pulses"][1]["pulse_duration"] = dur+sigma_raise*6
#         seq_args["args_pulses"][0]["t0"] = t0
#         seq_args["args_pulses"][1]["t0"] = t0

#     seq_args["args_pulses"][0]["detuning"] = -(A)/2 - Delta + resonant_freq
#     seq_args["args_pulses"][1]["detuning"] = -(A)/2 - Delta + chevron_det
#     result = mesolve([H0, H_p, H_m], ket0, t, [], [], args=seq_args)

#     # seq_args["args_pulses"][0]["detuning"] = 0
#     # seq_args["args_pulses"][1]["detuning"] = 0
#     # p = [pulse_sequence_p(_t, seq_args) for _t in t]
#     # plt.title("Raman Pulse Sequence (zero detuning to see pulse shape)")
#     # plt.plot(t/1000, p, 'k', alpha=0.5, label = 'sigma_raise = %.f ms'%(sigma_raise/1000))
#     # plt.xlabel("Time (ms)")
#     # plt.ylabel("Amplitude (rad/us)")

#     return expect(tensor(qeye(nS), Sz), result.states)[-1]