import torch
import scipy.signal as signal


def discount(x, gamma):
  """
  Compute discounted sum of future values
  out[i] = in[i] + gamma * in[i+1] + gamma^2 * in[i+2] + ...
  """
  return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def inverse_distance(h, h_i, epsilon=1e-3):
  return 1 / (torch.dist(h, h_i) + epsilon)
