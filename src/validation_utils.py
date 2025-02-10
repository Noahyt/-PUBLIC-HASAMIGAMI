"""Utilities for working with numpy arrays."""


# TODO: Make an option to allow broadcasting.
def is_compatible(a, b):
  """Tests that `a` and `b` are compatible.

  Note that `is_compatible` is distinct from checking broadcast compatibility
  as `a` and `b` must have the same length.

  Examples:
    a = [1, 2, 4], b = [1, 2, 4]
    is_compatible(a, b) == True

    a = [None, 2, 4], b = [3, 2, 4]
    is_compatible(a, b) == True

    a = [None, 2, 4], b = [2, 4]
    is_compatible(a, b) == False

    a = [1, 2, 4], b = [None 3, 2]
    is_compatible(a, b) == False

  Args:
    a, b: List representing shape of arrays.

  Returns:
    True if `a` and `b` are compatible; False if not.
  """
  if len(a) != len(b):
    return False
  for a_, b_ in zip(a, b):
    if a_ == b_ or a_ is None or b_ is None:
      pass
    else:
      return False
  return True


def validate_shape(expected, arr, name="array"):
  """Validates that `a` is compatible with `expected`, if not raises error."""
  s = arr.shape
  if not is_compatible(expected, s):
    raise ValueError(
      "{} expected to have shape compatible with {} but got {}.".format(
        name, expected, s))
