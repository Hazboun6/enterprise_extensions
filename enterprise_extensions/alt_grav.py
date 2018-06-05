from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
import scipy as sp
from enterprise import constants as const

class response_fcns(object):
    """
    Class for calculating PTA response functions in the computational frame.
    Reproduces the response functions for pulsars to all six polarizations of
    gravitational waves, 4 of which do not occur in GR.

    Based on Eqns (31)-(42) of Gair, Romano and Taylor, 2015.
    """
    def __init__(self, zeta, f=None, L1=None, L2=None):
        self.zeta = zeta
        self.L1 = L1
        self.L2 = L2
        self.f = f

    def common(self, theta, phi):
        return (1+np.sin(self.zeta)*np.sin(theta)*np.cos(phi)
                +np.cos(theta)*np.cos(self.zeta))

    def psr_term1(self, theta, phi):
        return np.exp(2*np.pi*1j*self.f*self.L1*(1+np.cos(theta))/const.c)

    def psr_term2(self, theta, phi):
        return np.exp(2*np.pi*1j*self.f*self.L2/const.c \
               * self.common(theta, phi))

    def R1_plus(self, theta, phi, psr_term=True):
        if psr_term:
            factor = 1 - self.psr_term1(theta, phi)
        else:
            factor = 1
        return 1/2 * (1 - np.cos(theta)) * factor

    def R2_plus(self, theta, phi, psr_term=True):
        if psr_term:
            factor = 1 - self.psr_term2(theta, phi)
        else:
            factor = 1
        return 1/2*((1 - np.sin(self.zeta)*np.sin(theta)*np.cos(phi)
                    - np.cos(theta)*np.cos(self.zeta))
                    - (2*np.sin(self.zeta)**2*np.sin(phi)**2)
                    / self.common(theta, phi) ) * factor

    def R1_cross(self, theta, phi, psr_term=True):
        return 0

    def R2_cross(self, theta, phi, psr_term=True):
        if psr_term:
            factor = 1 - self.psr_term2(theta, phi)
        else:
            factor = 1
        return -1/2*((np.sin(self.zeta)**2*np.cos(theta)*np.sin(2*phi)
                     -np.sin(2*self.zeta)*np.sin(theta)*np.sin(phi)) \
                     / self.common(theta, phi)) * factor

    def R1_breath(self, theta, phi, psr_term=True):
        if psr_term:
            factor = 1 - self.psr_term1(theta, phi)
        else:
            factor = 1
        return 1/2 * (1 - np.cos(theta)) * factor

    def R2_breath(self, theta, phi, psr_term=True):
        if psr_term:
            factor = 1 - self.psr_term2(theta, phi)
        else:
            factor = 1
        return 1/2 * (1 - np.sin(self.zeta)*np.sin(theta) * np.cos(phi)
                      - np.cos(theta)*np.cos(self.zeta) ) * factor

    def R1_SL(self, theta, phi, psr_term=True):
        if psr_term:
            factor = 1 - self.psr_term1(theta, phi)
        else:
            factor = 1
        return 1/np.sqrt(2) * np.cos(theta)**2/(1+np.cos(theta)) * factor

    def R2_SL(self, theta, phi, psr_term=True):
        if psr_term:
            factor = 1 - self.psr_term2(theta, phi)
        else:
            factor = 1
        return 1/np.sqrt(2) * (np.sin(self.zeta) * np.sin(theta) * np.cos(phi)
                               + np.cos(theta) * np.cos(self.zeta)) \
                               / self.common(theta, phi) * factor

    def R1_XL(self, theta, phi, psr_term=True):
        if psr_term:
            factor = 1 - self.psr_term1(theta, phi)
        else:
            factor = 1
        return -np.cos(theta) * np.sin(theta) /(1+np.cos(theta)) * factor

    def R2_XL(self, theta, phi, psr_term=True):
        if psr_term:
            factor = 1 - self.psr_term2(theta, phi)
        else:
            factor = 1
        return (np.sin(self.zeta) * np.sin(theta) * np.cos(phi)
                + np.cos(theta) * np.cos(self.zeta)) \
             * (np.sin(self.zeta) * np.cos(theta) * np.cos(phi)
                - np.sin(theta) * np.cos(self.zeta)) / self.common(theta, phi) * factor

    def R1_YL(self, theta, phi, psr_term=True):
        return 0

    def R2_YL(self, theta, phi, psr_term=True):
        if psr_term:
            factor = 1 - self.psr_term2(theta, phi)
        else:
            factor = 1
        return -np.sin(phi) * np.sin(self.zeta) \
               * (np.sin(self.zeta) * np.sin(theta) * np.cos(phi) \
                  + np.cos(theta) * np.cos(self.zeta)) \
                  / self.common(theta, phi) * factor

def hd_orf(zeta):
    return 1/3*(3/2*((1-np.cos(zeta))/2) * np.log(((1-np.cos(zeta))/2))
                - 1/4*((1-np.cos(zeta))/2) + 1/2)

def breath_orf(zeta):
    return 2/3*(3/8+1/8*np.cos(zeta))

def vec_long_orf(zeta):
    return 1/3*(3/2*np.log(2/(1-np.cos(zeta)))-2*np.cos(zeta)-3/2)

class orf():
    def __init__(self, zeta, f=None, L1=None, L2=None):
        self.zeta = zeta
        self.f = f
        self.L1 = L1
        self.L2 = L2
        #Make this a class factory with the response functions as input?
