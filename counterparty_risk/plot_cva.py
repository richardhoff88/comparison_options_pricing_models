"""
Parameters:
ee - expected exposure
pd - probability of default
r - risk-free rate
t - time to maturity

"""

import numpy as np
import matplotlib.pyplot as plt

def discount_factor(t, r):
  """
  Calculates the discount factor at time t given risk-free rate r.

  Args:
      t: Time to maturity
      r: Risk-free rate

  Returns:
      Discount factor at time t
  """
  return np.exp(-r * t)


def expected_exposure(t, ee_function):
  """
  Calculates the expected exposure at time t using an EE function.

  Args:
      t: Time
      ee_function: Function to calculate EE at a given time

  Returns:
      Expected Exposure at time t
  """
  return ee_function(t)

def probability_of_default(t, pd_function):
  """
  Calculates the probability of default at time t using a PD function.

  Args:
      t: Time
      pd_function: Function to calculate PD at a given time

  Returns:
      Probability of default at time t
  """
  return pd_function(t)

def cva_delta_exposure(t, ee_function, pd_function, r):
  """
  Calculates CVA using the delta-exposure method.

  Args:
      t: Time
      ee_function: Function to calculate EE at a given time
      pd_function: Function to calculate PD at a given time
      r: Risk-free rate

  returns CVA
  """
  ee = expected_exposure(t, ee_function)
  pd = probability_of_default(t, pd_function)
  df = discount_factor(t, r)
  return ee * pd * df


def plot_cva(time_range, ee_function, pd_function, r):
  """
  Plots CVA over time.
  """
  cva_values = [cva_delta_exposure(t, ee_function, pd_function, r) for t in time_range]
  plt.plot(time_range, cva_values)
  plt.xlabel("Time")
  plt.ylabel("CVA")
  plt.title("CVA Over Time")
  plt.show()

def expected_exposure(time_grid, exposure_profile):
  """
  linear interpolation to calculate expected exposure.
  more straightforward option than Monte Carlo simulation
  (which would be better)
  """
  ee = np.interp(time_grid, time_grid, exposure_profile)
  ee[ee < 0] = 0 
  return ee

def cva_sensitivity_analysis(ee_values, pd, r, t):
  """
  Calculates CVA for different expected exposure values.
  """
  cva_values = []
  for ee in ee_values:
      cva = cva_delta_exposure(ee, pd, r, t)
      cva_values.append(cva)
  return cva_values

def pd_example(t):
  # replace this
  return 0.01 + 0.001 * t

time_range = np.linspace(0, 5, 100)
r = 0.05

plot_cva(time_range, expected_exposure, pd_example, r)