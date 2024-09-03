"""
Parameters:
ee - expected exposure
pd - probability of default
r - risk-free rate
t - time to maturity

"""


import numpy as np

def discount_factor(t, r):
  """
  Calculates the discount factor at time t given risk-free rate r.
  """
  return np.exp(-r * t)

def expected_exposure(ee, pd):
  """
  Calculates expected exposure adjusted for probability of default (PD).

  """
  return ee * pd

def cva_delta_exposure(ee, pd, df):
  """
  Calculates CVA using the delta-exposure method.
  """
  return expected_exposure(ee, pd) * df


ee = 10000  
pd = 0.01  
r = 0.05  
t = 1 

df = discount_factor(t, r)
cva = cva_delta_exposure(ee, pd, df)

print("CVA (delta-exposure):", cva)
