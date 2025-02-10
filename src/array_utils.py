"""Numpy extensions."""

import validation_utils

import numpy as np
from scipy import linalg

from typing import List


def cross_multi_dim(arr_a, arr_b):
  """Computes cross product of the outer axes of `a` and `b`.

  Args:
    a: Array of shape `batch_a + [3]`
    b: Array of shape `batch_b + [3]`

  Returns:
    Array of shape `batch_a + batch_b + [3]` containing cross products
    of `a` and `b`.
  """
  return np.apply_along_axis(np.cross, axis=-1, arr=arr_a, b=arr_b)

# TODO: Test
def outer_exponential(arr):
  """Computes matrix exponential on outer axes of `arr`."""
  shape = list(arr.shape)
  flat = np.reshape(arr, [-1, ] + shape[-2:])

  # Take exponent of individual matrices.
  exp = [linalg.expm(a) for a in flat]

  # Stack and reshape to original shape.
  return np.reshape(np.stack(exp, 0), shape)


def repeat_first(arr):
    return np.concatenate([arr[0, np.newaxis], arr])


def repeat_final(arr):
    return np.concatenate([arr, arr[-1, np.newaxis]])


def repeat_both_ends(arr):
    return np.concatenate([arr[0, np.newaxis], arr, arr[-1, np.newaxis]])


def interleave(a: np.ndarray, b: np.ndarray, axis=0):
  """Interleave `a` and `b` along first axis starting with elements of `a`."""
  c = np.empty((a.shape[0] + b.shape[0],) + a.shape[1:], dtype=a.dtype)
  c[0::2] = a
  c[1::2] = b
  return c


def interleave_arb_axis(a, b, axis):
  """Interleaves `a` and `b` along arbitrary axis."""
  a = np.swapaxes(a, 0, axis)
  b = np.swapaxes(b, 0, axis)
  return np.swapaxes(interleave(a, b), 0, axis)


def diagonal_coords(
    mat_shape: List,
    offset: List[int] = None,
    origin_indices: List[int] = None,
):
  """Calculate coordinates for placing N-dim `mat` as an 2N-dim diagonal.

  Ex. A 2D diagonal matrix could be created from 1D vector `v`.

    # coords = diagonal_component(mat, offset = (0,))
    # coords = coords.reshape([-1, 2])
    # mat = sparse.COO(coords, mat, fill_value=0)

  NOTE: This function does not verify that `mat` and `offset` will properly fit
  in a 2N-dim square matrix.

  Args:
    mat: `np.ndarray` of dimension N to used as filling in a 2N-dimensional
      diagonal.
    offset: Offset of mat within diagonal. Should have length = N.
    origin_indices: Offset of elment in the first slot of `mat` in the diagonal
      matrix.

  Returns:
    coordinates: Array of shape `mat_shape + [coordinates]`. I.e. one con
    find the indices of `mat[i_1, ..., i_n]` in the final array by looking
    at `coordinates[i_1, ..., i_n]`. Note that when
    flattened to `[-1, 2N]` the array has shape `[point_count, 2N]`.
  """
  dim = len(mat_shape)
  if offset is not None:
    if len(offset) != dim:
      raise ValueError("""`offset` should have same dimension as `mat`.
      Expected {} but got {}""".format(dim, len(offset)))
  else:
    offset = [0, ] * dim
  if origin_indices is not None:
    if len(origin_indices) != dim:
      raise ValueError("""`origin_indices` should have same dimension as `mat`.
      Expected {} but got {}""".format(dim, len(origin_indices)))
    if any(i + o < 0 for i, o in zip(origin_indices, offset)):
      raise ValueError("""`origin_indices` incompatible with `offset`. Should
      all be non-negative but got {}""".format(
        [i + o for i, o in zip(origin_indices, offset)]))
  else:
    origin_indices = [0, ] * dim
  coords = [np.arange(s) + i for s, i in zip(mat_shape, origin_indices)]

  # `inner_coord` describes the first `N` coordinates of the diagonal matrix.
  inner_coord = np.stack(np.meshgrid(*coords, indexing='ij'), -1)

  # `outer_coord` gives the last `N` coordinates of the diagonal matrix.
  outer_coord = inner_coord + np.broadcast_to(offset, (1,) * dim + (dim,))

  return np.concatenate((inner_coord, outer_coord), axis=-1)


def fill_diagonal(arr, offset=0, fill=0):
  """Sets diagonal elements of 2D `arr`."""

  validation_utils.validate_shape([None, None], arr)

  if offset > 0:
    np.fill_diagonal(arr[:, offset:], fill)
  if offset <= 0:
    np.fill_diagonal(arr[-1 * offset:], fill)
  return arr
