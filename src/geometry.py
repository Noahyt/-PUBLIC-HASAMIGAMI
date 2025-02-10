"""Basic functions for cartesian geometry.

These functions serve to compute general relationships including:

* orientation
* angles
"""

import array_utils

import numpy as np


_LEFT_HAND = "LEFT"
_RIGHT_HAND = "RIGHT"
_Z_HAT = np.array([0,0,1.])

__TWO_PI = 2 * np.pi


def switch_handedness(angle):
  """Switches between left- and right-handed measurements in radians.

  Args:
    `angle`: Array of angles. Must be positive and in the range
  [0, 2 * np.pi].

  Returns:
    Array containing other handed version of `angle`.
  """
  return __TWO_PI - angle


def point_plane_orientation(
  plane_vertices: np.ndarray,
  comparison_vertices: np.ndarray,
):
  """
  Consider an ordered set of three vertices `a`, `b`, `c`. These vertices define
  an oriented plane (using a right-handed convention). This function computes
  the orientation of `comparison_vertices` relative to this plane.

  Visually, this function computes whether a point `d` is "above" or "below"
  the plane defined by `a`, `b`, `c`.

  #
  #    a       c
  #     \\  d  /
  #      \\ | /
  #        b
  #

  The orientation is represented as a binary value with `+1` being above the
  plane and `-1` being below.

  Args:
    plane_vertices: Array of shape [batch, 3, 3] where the second dimension
      indexes (`a`, `b`, `c`) vertices in the description above and the last
      index corresponds to spatial coordinates.
    comparison_vertices: Array of shape [batch, comparison_count, 3] where the
      second index corresponds to different possible choices of `d` and the
      final index corresponds to spatial coordinate.

  Returns:
    Array of shape `[batch, comparison_count]` with binary entries representing
      the orientation."""
  if plane_vertices.shape[0] != comparison_vertices.shape[0]:
    raise ValueError("Incompatible batch dimension.")

  b_coordinate = plane_vertices[:, np.newaxis, 1]

  # recenter all cordinates relative to coordinate `b` for each batch.
  plane_vertices = plane_vertices - b_coordinate
  comparison_vertices = comparison_vertices - b_coordinate

  # Compute normals to ABC plane.
  plane_normal = normal(plane_vertices[:, 2], plane_vertices[:, 0])

  # If dot product of `d` and normal is positive -> moutain fold, else valley.
  return np.sign(_dot_along(plane_normal[:, np.newaxis], comparison_vertices))


def normal(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
    normalize: bool = False,
):
  """Computes normal to plane defiend by vectors `a` and `b`.

  The normal is taken to be:

  n = (axb/|axb|)

  The orientation of the normal vector follows from the right-hand rule. In the
  diagram below, the normal projects out of the plane.

     * b
    /
   /
  *-----* a


  Args:
    vector_a: Array of shape `[batch , 3]`.
    vector_b: Same as `vector_a`.
    normalize: If `True` then returns normalized vector.

  Returns:
    Array of shape `[batch, 3]`.
  """
  normal = np.cross(vector_a, vector_b)
  if normalize:
    normal = unit_vector(normal)
  return normal


def norm(vector):
    """Convenience wrapper for np.linalg.norm"""
    return np.linalg.norm(vector, axis=-1)


def unit_vector(vector, axis=-1):
  """Returns unit vector where the norm is computed along `axis`."""
  return vector / np.linalg.norm(vector, axis=-1)[..., np.newaxis]


def angle_between(v1, v2):
  """Returns the angle in radians between vectors 'v1' and 'v2'."""
  v1 = unit_vector(v1)
  v2 = unit_vector(v2)
  return np.arccos(np.clip(_dot_along(v1, v2), -1.0, 1.0))


def signed_angle_between(vector_a, vector_b, plane_normal):
  """Signed angle in radians between `vector_a` and `vector_b`.

  This function computes the signed angle [0, 2 * np.pi) based on a given
  normal to the plane containing `vector_a` and `vector_b`.

  In the diagram below if `plane_normal` projects from the page, then the
  angle between `a` and `b` is the acute angle; if `plane_normal` goes
  into the page, then the angle is the obtuse angle.

       * b
      /
     /
    *-----* a

  args:
    vector_a: Array of shape `batch + [3,]`
    vector_b: Same as `vector_a`.
    plane_normal: Array of shape compatible with `vector_a` describing normal
      to surface containing `vector_a` and `vector_b`

  returns:
    Array of shape `batch` containing angles measured between pairs of
      `vector_a` and `vector_b`.
  """
  vector_a = unit_vector(vector_a)
  vector_b = unit_vector(vector_b)
  normal = unit_vector(plane_normal)

  a_cross_b = np.cross(vector_a, vector_b)

  # TODO: Make this a standalone function to check orthogonal vectors.
  # Check normal is normal to `vector_a` and `vector_b`.
  if not np.allclose(np.cross(normal, a_cross_b), np.zeros_like(a_cross_b)):
    raise ValueError("`normal` not orthogonal to `vector_a` and `vector_b`.")

  cos_ = _dot_along(vector_a, vector_b)
  sin_ = _dot_along(normal, a_cross_b)

  # `shift_angle` ensures output is in range [0, 2 * pi).
  return shift_angle(np.arctan2(sin_, cos_))


def internal_angle(a, b, c):
  """Computes internal angle between ordered points."""
  a, c = a - b, c - b
  return angle_between(a, c)


def _dot_along(arr_a, arr_b, axis=-1):
  """Computes dot product of two arrays along axis."""
  return np.sum(arr_a * arr_b, axis=axis)


def perpendicular_component(vector, axis):
  """Returns component of `vector` perpendicular to `axis`"""
  axis = unit_vector(axis)
  return vector - _dot_along(vector, axis)[..., np.newaxis] * axis


def law_of_cos(a, b, theta):
  """Evaluates the law of cosines and returns intercepted length.

  Args:
    a, b: `np.ndarray` representing lengths.
    theta: `np.ndarray` of

  Returns:
    Array containing length intercepted by `a` and `b`.
  """
  return np.sqrt(a ** 2 + b ** 2 - 2 * a * b * np.cos(theta))


def shift_angle(
  angle,
  offset: float = 0,
):
  """Sets angle range betwee `offset` and 2 * pi + offset.

  Example: `shift_angle(angles)` will shift all `angle` values to be positive
  values between `(0, np.pi)`.

  Args:
    angle: Array of angles.
    offset: Float or array of floats compatible with shape of `angle`.

  Returns:
    Array of same shape as `angle` containing angles under offset.
  """
  # Get angle mod `2 * np.pi`.
  angle = np.mod(angle, __TWO_PI)

  return angle + offset


def rotate_about(
  points: np.array,
  axis: np.array,
  theta: np.array,
):
  """Rotates `points` about `axis` by `theta` in right handed manner.

  Args:
    points: Array of shape `batch + [3]`
    axis: See `rotation_from_axis_angle`.
    theta: See `rotation_from_axis_angle`.

  Returns:
    Array of same shape as `points` containing result of rotation.
  """
  # TODO: Validate shape.
  rotation_matrix = rotation_matrix_from_axis_angle(axis, theta)

  # Expand dimensions of points so we can perform dot product with
  # rotation matrix.
  points = np.expand_dims(points, -2)

  return _dot_along(rotation_matrix, points)


def rotation_matrix_from_axis_angle(
  axis,
  theta
):
  """Creates rotation mattrix using rodrigues formula.

  See definition using matrix exponential:
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula#Matrix_notation

  Args:
    axis: Array of shape `batch + [3]`.
    theta: Array of shape `batch`.

  Returns:
    Rotation matrix, of shape `batch + [3, 3]`
  """
  # TODO: Validate shape
  # Ensure `axis` are unit vectors.
  axis = unit_vector(axis)

  # The `K` matrix has shape `batch + [3, 3]`
  K_cross_matrix = -1 * array_utils.cross_multi_dim(axis, np.eye(3))

  # Expand dimensions of `theta` so it is compatible with `K_cross_matrix`.
  theta = theta[..., np.newaxis, np.newaxis]

  return array_utils.outer_exponential(theta * K_cross_matrix)


def euler_2(x, z, alpha, beta,):
    """Computes coordinates of `z` axis after movement by `alpha`, `beta`.

    Euler angles are described: https://en.wikipedia.org/wiki/Euler_angles

    Note that all rays are taken to have one end at origin.

    Args:
    x:
    z:
    alpha:
    beta:

    Returns:
    Array containing coordinates of new ray.
    """
    # Define `N` axis by rotating `x` by `alpha`.
    N = rotate_about(x, axis=z, theta=alpha)

    # Rotate `a` by `theta`.
    z_prime = rotate_about(z, axis=N, theta=beta)

    # Return unit ray.
    return unit_vector(z_prime)
