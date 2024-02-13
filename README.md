# Pulse-level simulation of spin dynamics

The purpose of these notebooks is to provide the user with ready-to-use tools to simulate the dynamics of any electron spin coupled to one or more nuclear spins down to the pulse-level.
While control pulses used in most Electron Paramagnetic Resonance (EPR) or Electron Spin Resonance (ESR) experiments can be considered as instantaneous compared to the other relevant time durations (such as the Larmor period of spins), the dynamics of the spins can become very complex when working with non-infinitely short pulses (as it can be when working with microwave fields in microcavities for example) especially since there are no known analytical formulas to predict the signal emitted by the spins when addressed by such pulses.
In this case, simulating forward in time the dynamics of an electron-nuclear spin system while taking into account the real shape of the pulse sequence applied to it can be convenient to the experimenter trying to reproduce real data.

This repository contains some examples of simulations used to grasp a better understanding of the ESR experiments on rare-earth ions done at CEA Saclay in the [Quantronics Group](https://iramis.cea.fr/spec/Pres/Quantro/static/index.html).
The user can freely edit the parameters of these simulations to adapt it to any electron-nuclear spin system.
