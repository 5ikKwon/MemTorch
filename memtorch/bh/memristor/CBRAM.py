import math

import numpy as np
import torch

import memtorch
from memtorch.utils import clip, convert_range

from .Memristor import Memristor as Memristor

class CBRAM(Memristor):
    """CBRAM memristor model (DOI: 10.1109/TED.2011.2116120).

    Parameters
    ----------
    time_series_resolution : float
        Time series resolution (s).

    L : float
        Device length (nm).
    v_h : float
        CF vertical growth (cm/s).
    v_r : float
        CF lateral growth (cm/s). 
    E_a : float
        Activation energy (eV).
    rho_on : float
        CF resistivity (ohm*cm)
    rho_off : float
        Oxide resistivity (ohm*cm)
    A : float
        Area of bottom CF(nm^2)
    R_th : float
        Thermal resistance(Kevin/Watts)
    beta : float
        Fitting parameter for lateral expansion.
    alpha : float
        Fitting parameter for vertical growth. 
    C : float
        Fitting parameter realted with R_set(V). (R_set = C/I_comp.)
    I_comp : float
        Compliance current.(A)
    kT = 0.025
    r, h, T,R are variable.
    flag is flag.
    """

    """
    Equations : 
    vertical growth : dh/dt = v_h*exp(-E_a/kT)*sinh(Z*q*E*a/(2*k*T)) -eq(3)
                    E = V/(L+(rho_on/rho_off - 1)*h)                -eq(4)
                    -> dh/dt = v_h*exp(-E_a/kT)*sinh(alpha*q*V/(2*k*T)) -eq(10)
                    R_off = (rho_on*h +rho_off*(L-h))/A             -eq(5)
                    h update
    lateral growh :  dr/dt = v_r*exp(-E_a/kT)*sinh(beta*q*V/(k*T))  -eq(6)
                    R_on = (rho_on*L/(pi*r*R))                      -eq(7)
                    R -> radius of the Bottom of area.              
                    r -> radius of the top of area.
                    T = T0 + V^2*R_th/R_on                          -eq(8) 
                    R_set = C/I_comp                                -eq(9)
                    r and R update compensating Temperature.
                    r is determined from R_set = C/I_comp. (R_set = R_on(initial r and R))
    
    """
    def __init__(
        self,
        time_series_resolution=1e-6,
        L = 50,
        v_h = 0.35,
        v_r = 700,
        E_a = 0.5,
        rho_on = 4,
        rho_off = 1.33e4,
        A = 1330,
        R_th = 10e5,
        beta = 0.3,
        alpha = 0.5,
        C = 0.08,
        I_comp = 0.001,
        **kwargs
    ):

        args = memtorch.bh.unpack_parameters(locals())
        super(CBRAM, self).__init__(
           args.time_series_resolution
        )
        self.L = args.L
        self.v_h = args.v_h
        self.v_r = args.v_r
        self.E_a = args.E_a
        self.rho_on = args.rho_on
        self.rho_off = args.rho_off
        self.A = args.A
        self.R_th = args.R_th
        self.beta = args.beta
        self.alpha = args.alpha
        self.C = args.C
        self.I_comp = args.I_comp
        self.g = args.g
        self.kT = 0.025
        self.q = 1.602e-19
        self.h = 0
        self.r = 0
        self.T = 300
        self.R = np.sqrt(args.A/np.pi)
        self.flag = True
        self.istouched = False

    def dhdt(self, voltage):
        """
        Method to determine the derivative of the vertical expansion.

        Parameters
        ----------
        voltage : float
            The current applied voltage (V).

        Returns
        -------
        float
            The derivative of the vertical expansion.
        """
        return self.v_h*np.exp(-self.E_a/self.kT)*np.sinh(self.alpha*voltage/(self.kT))

    def drdt(self, voltage, R_on):
        """
        Method to determine the derivative of the lateral expansion.

        Parameters
        ----------
        voltage : float
            The current applied voltage (V).

        Returns
        -------
        float
            The derivative of the lateral expansion.
        """
        self.T = self.kT + (voltage^2*(self.R_th/R_on))/(1.602e-19)
        return self.v_r*np.exp(-self.E_a/(self.T*self.kT/300))*np.sinh(self.beta*voltage/(self.T*self.kT/300))
    
    def current(self, voltage):
        if(self.istouched):
            R_off = (self.rho_on*self.h+self.rho_off*(self.L-self.h))/self.A
            return voltage/R_off
        else:
            R_on = (self.rho_on*self.L)/(np.pi*self.r*self.R)
        return voltage/R_on


####
    def simulate(self, voltage_signal, return_current=False):
        len_voltage_signal = 1
        try:
            len_voltage_signal = len(voltage_signal)
        except:
            voltage_signal = [voltage_signal]

        if return_current:
            current = np.zeros(len_voltage_signal)

        np.seterr(all="raise")
        for t in range(0, len_voltage_signal):
            if(np.max(voltage_signal) == voltage_signal[t] | np.min(voltage_signal) == voltage_signal[t]):
                self.flag = ~self.flag

            if (self.flag):
                if(~self.istouched):
                    self.h = self.h + (self.dhdt(voltage_signal[t]).self.time_series_resolution)
                    if (self.h>=self.L):
                        self.h = self.L
                        self.istouched = True
                        self.r = (self.rho_on*self.L)/(np.pi*self.R*(self.C/self.I_comp)) # self.r initialization
                else:
                    self.r = self.r + (self.drdt(voltage_signal[t])*self.time_series_resolution)
                    self.R = self.R + (self.drdt(voltage_signal[t])*self.time_series_resolution)

            else:
                if(self.istouched):
                    self.r = self.r - (self.drdt(voltage_signal[t])*self.time_series_resolution)
                    self.R = self.R - (self.drdt(voltage_signal[t])*self.time_series_resolution)
                    if(self.r<=0):
                        self.r = 0
                        self.istouched = False
                else:
                    self.h = self.h -(self.dhdt(voltage_signal[t])*self.time_series_resolution)
                    if(self.h<=0):
                        self.h = 0

            current_ = self.current(voltage_signal[t])
            if voltage_signal[t] != 0:
                self.g = current_ / voltage_signal[t] # do we need to save conductance?

            if return_current:
                current[t] = current_

        if return_current:
            return current    
####



    def plot_hysteresis_loop(
        self,
        duration=200e-9,
        voltage_signal_amplitude=1,
        voltage_signal_frequency=50e6,
        return_result=False,
    ):
        return super(CBRAM, self).plot_hysteresis_loop(
            self,
            duration=duration,
            voltage_signal_amplitude=voltage_signal_amplitude,
            voltage_signal_frequency=voltage_signal_frequency,
            return_result=return_result,
        )

    def plot_bipolar_switching_behaviour(
        self,
        voltage_signal_amplitude=1.5,
        voltage_signal_frequency=50e6,
        log_scale=True,
        return_result=False,
    ):
        return super().plot_bipolar_switching_behaviour(
            self,
            voltage_signal_amplitude=voltage_signal_amplitude,
            voltage_signal_frequency=voltage_signal_frequency,
            log_scale=log_scale,
            return_result=return_result,
        )