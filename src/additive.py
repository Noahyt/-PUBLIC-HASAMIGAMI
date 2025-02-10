"""Construct shearing structures."""

import numpy as np
from numpy import linalg

import geometry


def norm(v):
    """new comment"""
    return linalg.norm(v)


def unit_vector(v):
    return v / norm(v)


def validate_construction_p(v1, v2, v_star, p):
    """Returns the denominator of `p` construction.
    If denominator is <0 then the construction is valid (the new pivot lies past growth front).
    If denominator is >0 then the construction is invalid (The new pivot lies behind front.).
    """
    # Construct vectors.
    r_1 = v2 - v1
    s2 = p - v1
    s3 = p - v2
    l_2_top = norm(s2)
    l_3_top = norm(s3)

    # Constant for solving new length.
    kappa_2 = l_2_top - l_3_top
    r_1_norm = norm(r_1)

    # s4 unit vector.
    s4_unit = unit_vector(v_star - v2)

    # length of s4 to satisfy flat-shearability.
    return kappa_2 + np.dot(r_1, s4_unit)


def validate_construction():
    """Returns value of denominators in construction procedure."""
    pass


def construct_p(v1, v2, v_star, p):
    """Constructs pivot in cell bounded by v1, v2, and v_star.
    This construction satisfies a flat-shearability condition.
    """
    # Construct vectors.
    r_1 = v2 - v1
    s2 = p - v1
    s3 = p - v2
    l_2_top = norm(s2)
    l_3_top = norm(s3)

    # Constant for solving new length.
    kappa_2 = l_2_top - l_3_top
    r_1_norm = norm(r_1)

    # s4 unit vector.
    s4_unit = unit_vector(v_star - v2)

    # length of s4 to satisfy flat-shearability.
    l4 = (kappa_2**2 - r_1_norm**2) / (2 * (kappa_2 + np.dot(r_1, s4_unit)))

    # Construct new pivot.
    return v2 + l4 * s4_unit


def construct_p_kappa(v1, v2, v_star, kappa_1, return_validation=False):
    """Constructs pivot in cell bounded by v1, v2, and v_star.
    This construction satisfies a flat-shearability condition.
    """
    # Construct vectors.
    r_1 = v2 - v1

    # Constant for solving new length.
    r_1_norm = norm(r_1)

    # s4 unit vector.
    s4_unit = unit_vector(v_star - v2)

    # length of s4 to satisfy flat-shearability.
    l4 = (kappa_1**2 - r_1_norm**2) / (2 * (kappa_1 + np.dot(r_1, s4_unit)))

    # Construct new pivot.
    p = v2 + l4 * s4_unit

    if return_validation:
        return p, np.sign(l4)

    return p


def construct_p_by_angle(v0, v1, normal, theta_1, theta_2, kappa, return_validation=False):
    t = v1 - v0
    t_hat = geometry.unit_vector(t)
    t_norm = norm(t)

    # unit vector in direction of l4.
    s4_unit = v_hat_direction_by_angle(
        t_hat, normal, np.array([theta_1]), np.array([theta_2])
    )

    # length of s4 to satisfy flat-shearability.
    l4 = (kappa**2 - t_norm**2) / (2 * (kappa + np.dot(t, s4_unit)))

    new_p = v1 + l4 * s4_unit

    # Construct new pivot.
    if return_validation:
        return new_p, np.sign(l4)

    return new_p


def v_hat_direction_by_angle(t, normal, theta_1, theta_2):
    """Constructs unit vector in direction of `v_hat` parameterized by angles."""

    return geometry.euler_2(normal, -1 * t, theta_1, theta_2)[0]


def v_by_angle(v0, v1, normal, theta_1, theta_2, length=1.0):
    """Returns new v based on angles. Defaults to unit vector length."""
    t = geometry.unit_vector(v1 - v0)
    return v1 + length * v_hat_direction_by_angle(t, normal, theta_1, theta_2)


def construct_v_star(v1, v2, v3, p_left, p_top):
    """Constructs new vertex to connect adjacent cells.
    This construction satisfies a flat-shearability condition.
    """
    s1_left = p_left - v1
    s4_left = p_left - v2

    s2_top = p_top - v2
    s3_top = p_top - v3

    R = v3 - p_left

    # Calculate known lengths.
    l1_left = norm(s1_left)
    l4_left = norm(s4_left)
    l_2_top = norm(s2_top)
    l_3_top = norm(s3_top)
    R_norm = norm(R)
    # print(f"R norm {R_norm}")

    # Construct constant.
    kappa_1 = l_2_top - l_3_top
    kappa_2 = kappa_1 + l4_left

    # Need unit vector in direction of `s1` to extend.
    s1_left_unit = unit_vector(s1_left)

    lambda_1_left = (R_norm**2 - kappa_2**2) / (
        2 * (kappa_2 + np.dot(R, s1_left_unit))
    )

    # Construct new point (bottom left vertex of new cell).
    return p_left + lambda_1_left * s1_left_unit


def construct_v_star_kappa(v1, v2, v3, p_left, kappa_1, return_validation=False):
    """Constructs new vertex to connect adjacent cells.
    This construction satisfies a flat-shearability condition.
    """
    s1_left = p_left - v1
    s4_left = p_left - v2

    R = v3 - p_left

    # Calculate known lengths.
    l1_left = norm(s1_left)
    l4_left = norm(s4_left)
    R_norm = norm(R)

    # Construct constant.
    kappa_2 = kappa_1 + l4_left

    # Need unit vector in direction of `s1` to extend.
    s1_left_unit = unit_vector(s1_left)

    lambda_1_left = (R_norm**2 - kappa_2**2) / (
        2 * (kappa_2 + np.dot(R, s1_left_unit))
    )

    # Construct new point (bottom left vertex of new cell).
    new_v = p_left + lambda_1_left * s1_left_unit

    if return_validation:
        v_valid = np.sign(lambda_1_left)
        return new_v, v_valid
    
    return new_v


def construct_cell(v1, v2, v3, p_left, p_top):
    """Constructs next cell in sequence based on cells to left and top.
    This construction satisfies a flat-shearability condition.
    """
    # Construct new point v_{i+1,j}
    v_star = construct_v_star(v1, v2, v3, p_left, p_top)

    # Construct p_{i,j}
    p = construct_p(v2, v3, v_star, p_top)

    return v_star, p


def construct_cell_kappa(v1, v2, v3, p_left, kappa_1, return_validation=False):
    """Constructs next cell in sequence based on cells to left and top.
    This construction satisfies a flat-shearability condition.
    """
    # Construct new point v_{i+1,j}
    out = construct_v_star_kappa(v1, v2, v3, p_left, kappa_1, return_validation=return_validation)
    if return_validation:
        v_star, v_valid = out
    else:
        v_star = out

    # Construct p_{i,j}
    out = construct_p_kappa(v2, v3, v_star, kappa_1, return_validation=return_validation)
    if return_validation:
        p, p_valid = out
    else:
        p = out

    if return_validation:
        return v_star, p, p_valid, v_valid

    return v_star, p


def construct_v_free_boundary(v, p, length):
    """Constructs a new v at a boundary by extending the internal edge by `length`."""
    s = p - v
    s_length = norm(s)
    s_unit = unit_vector(s)
    return v + s_unit * (length + s_length)


def construct_strip_initial_angles(
    v_top, kappa, normal, theta_1, theta_2, left_leg_length, right_leg_length,
    return_validation=False,
):
    """Builds strip from vertices, angles of left leg, and kappas."""

    # Construct left pivot.
    out = construct_p_by_angle(
        v_top[0], v_top[1], normal, theta_1, theta_2, kappa[0], return_validation=return_validation
    )
    if return_validation:
        p_left_init, p_valid_init = out
        p_valid = [p_valid_init]
        v_valid = []
    else:
        p_left_init = out



    # Construct left leg.
    v_left_init = construct_v_free_boundary(v_top[1], p_left_init, left_leg_length)

    new_v = [v_left_init]
    new_p = [p_left_init]

    # Build strip.
    for v0, v1, v2, k in zip(v_top[:-2], v_top[1:-1], v_top[2:], kappa[1:]):
        p_side = new_p[-1]

        out = construct_cell_kappa(v0, v1, v2, p_side, k, return_validation=return_validation)

        if return_validation:
            v_star, p, p_valid_, v_valid_ = out
            p_valid.append(p_valid_)
            v_valid.append(v_valid_)
        else:
            v_star, p = out

        new_v.append(v_star)
        new_p.append(p)

    # Construct right leg.
    final_v = construct_v_free_boundary(v_top[-2], new_p[-1], right_leg_length)
    new_v.append(final_v)

    new_v = np.stack(new_v)
    new_p = np.stack(new_p)

    if return_validation:
        p_valid = np.stack(p_valid)
        v_valid = np.stack(v_valid)
        return new_v, new_p, v_valid, p_valid

    return new_v, new_p


def normal_from_top_pivot_vertex(v_seed, p_seed):
    """Compute normal to seed strip.
    Normal is defined  by `l2` and `l3`"""

    # Construct l2, l3.
    l2 = v_seed[:-1] - p_seed
    l3 = v_seed[1:] - p_seed

    return geometry.normal(l2, l3, normalize=True)


def kappa_from_top_pivot_vertex(v_seed, p_seed):
    """Compute `kappa` (leg length difference) from seed strip."""
    l2 = v_seed[:-1] - p_seed
    l2_length = geometry.norm(l2)

    l3 = v_seed[1:] - p_seed
    l3_length = geometry.norm(l3)

    # Constant for solving new length.
    return l2_length - l3_length


def construct_strip_top_pivots_given(
    v_top, p_top, theta_1, theta_2, left_leg_length, right_leg_length, return_validation
):
    """Computes next set of pivots and vertices from row of pivots and vertices.
    Used for construction in bulk of structure (not boundary where normal is given).
    """
    # Construct normal.
    normal = normal_from_top_pivot_vertex(v_top[:2], p_top[:2])

    # Construct kappa.
    kappa = kappa_from_top_pivot_vertex(v_top, p_top)

    return construct_strip_initial_angles(
        v_top, kappa, normal, theta_1, theta_2, left_leg_length, right_leg_length, return_validation
    )


def construct_strip(v_left_init, v_top, p_top, final_length=None, return_validation=False):
    # Construct `p_left`.
    p_left_init = construct_p(v_top[0], v_top[1], v_left_init, p_top[0])

    new_v = [v_left_init]
    new_p = [p_left_init]

    # Build strip.
    for v0, v1, v2, p_back in zip(v_top[:-2], v_top[1:-1], v_top[2:], p_top[1:]):
        p_side = new_p[-1]

        # print(f"cell {len(new_v)}")

        v_star, p = construct_cell(v0, v1, v2, p_side, p_back)

        new_v.append(v_star)
        new_p.append(p)

    # If `final_length` not given, set final edge to same length as first.
    if final_length is None:
        final_length = norm(v_left_init - p_left_init)

    final_v = construct_v_free_boundary(v_top[-2], new_p[-1], final_length)
    new_v.append(final_v)

    new_v = np.stack(new_v)
    new_p = np.stack(new_p)

    if return_validation:
        pass

    return new_v, new_p


def construct_surface_top_left_angles(
    v_top, kappa_top, normal_top, theta_1, theta_2, left_leg_length, right_leg_length,
    return_validation=False
):
    """Constructs surface from boundaries on top and left."""

    # Build first row.
    out = construct_strip_initial_angles(
        v_top,
        kappa_top,
        normal_top,
        theta_1[0],
        theta_2[0],
        left_leg_length[0],
        right_leg_length[0],
        return_validation,
    )
    if return_validation:
        new_v, new_p, new_v_valid, new_p_valid = out
        v_valid = [new_v_valid]
        p_valid = [new_p_valid]
    else:
        new_v, new_p = out

    v = [v_top, new_v]
    p = [new_p]

    # Build subsequent rows.
    if len(theta_1) > 1:
        for t1, t2, ll, rl in zip(
            theta_1[1:], theta_2[1:], left_leg_length[1:], right_leg_length[1:]
        ):
            v_ = v[-1]
            p_ = p[-1]

            out = construct_strip_top_pivots_given(
                v_top=v_,
                p_top=p_,
                theta_1=t1,
                theta_2=t2,
                left_leg_length=ll,
                right_leg_length=rl,
                return_validation=return_validation
            )

            if return_validation:
                new_v, new_p, new_v_valid, new_p_valid = out
                v_valid.append(new_v_valid)
                p_valid.append(new_p_valid)
            else:
                new_v, new_p = out

            v.append(new_v)
            p.append(new_p)

    if return_validation:
        return v, p, v_valid, p_valid

    return v, p


def construct_axisymmetric_surface(v_top, v_left):
    """Constructs axisymmetric surface from vertices on top and left."""

    v = [v_top]
    # choose p to be midpoints since we know `kappa=0` everywhere for axisymmetric structures.
    p = [v_top[:-1] + (v_top[1:] - v_top[:-1]) / 2]

    for vl_ in v_left:
        v_ = v[-1]
        p_ = p[-1]

        # Pad since fist vertex is also last.
        v_ = np.pad(v_, [[0, 1], [0, 0]], mode="wrap")

        new_v, new_p = construct_strip(vl_, v_, p_)

        # Remove terminal `v`
        new_v = new_v[:-1]

        v.append(new_v)
        p.append(new_p)

    return v, p

def construct_periodic_loop(v_top, kappa_top, normal_top, theta_1, theta_2, return_validation=False):
    """Constructs periodic loop surface from vertices on top and angles for v2."""
    


    # Pad twice.
    # Constructing a structure with compatible pivots requires we "re-construct" the first cell to 
    # find the correct leg lengths (equivalently vertex `x_{i+1,0}`)

    v_top = np.array(v_top)

    v_top_padded = np.pad(v_top[:-1], [[1, 2], [0, 0]], mode="wrap")
    kappa_top_padded = np.pad(kappa_top, [[1, 1]], mode="wrap")
    print(kappa_top_padded.shape)
    print(v_top_padded.shape)
    print(v_top_padded)
    
    # Build first row.
    out = construct_strip_initial_angles(
        v_top_padded,
        kappa_top_padded,
        normal_top,
        theta_1[0],
        theta_2[0],
        1.,
        1.,
        return_validation,
    )
    print(out)
    if return_validation:
        new_v, new_p, new_v_valid, new_p_valid = out
        v_valid = [new_v_valid]
        p_valid = [new_p_valid]
    else:
        new_v, new_p = out

    
    print("here")
    print(new_v)

    
    # remove boundary vertices and pivots (leaving only the overlapping vertices v_{i+1,0} == v_{i+1, N+1}.
    new_v = new_v[1:-1]
    new_p = new_p[1:-1]

    v = [v_top, new_v]
    p = [new_p]

    # Build subsequent rows.
    if len(theta_1) > 1:
        for t1, t2, in zip(theta_1[1:], theta_2[1:]):
            v_ = v[-1]
            p_ = p[-1]

            # Pad.
            v_ = np.pad(v_[:-1], [[1, 2], [0, 0]], mode="wrap")
            p_ = np.pad(p_, [[1, 1], [0, 0]], mode="wrap")

            out = construct_strip_top_pivots_given(
                v_top=v_,
                p_top=p_,
                theta_1=t1,
                theta_2=t2,
                left_leg_length=1.,
                right_leg_length=1.,
                return_validation=return_validation
            )

            if return_validation:
                new_v, new_p, new_v_valid, new_p_valid = out
                v_valid.append(new_v_valid)
                p_valid.append(new_p_valid)
            else:
                new_v, new_p = out

            # Remove pad.
            new_v = new_v[1:-1]
            new_p = new_p[1:-1]

            print(new_v.shape)

            v.append(new_v)
            p.append(new_p)

    if return_validation:
        return v, p, v_valid, p_valid

    return v, p


def construct_surface_top_left(
    p_top, v_top, v_left, final_length=None, axisymmetric=False
):
    """Constructs surface from boundaries on top and left."""

    v = [v_top]
    p = [p_top]
    if final_length is None:
        final_length = [
            None,
        ] * len(v_left)

    for v_left_init, fl_ in zip(v_left, final_length):
        v_ = v[-1]
        p_ = p[-1]

        if axisymmetric:
            v_ = np.pad(v_, [[0, 1], [0, 0]], mode="wrap")

        new_v, new_p = construct_strip(v_left_init, v_, p_, fl_)

        if axisymmetric:
            new_v = new_v[:-1]

        v.append(new_v)
        p.append(new_p)

    return v, p


def construct_loop_kappa(v, kappa, construction_steps):
    """Same as `construct_loop` but giving kappa instead of preceding pivots."""

    # Construct first pivot.
    loop_count = len(kappa)

    p_left_init = construct_p_kappa(v[0], v[1], v[-1], kappa[-loop_count])

    p = np.array([p_left_init])

    # Construct.
    for step in range(construction_steps):
        new_v, new_p = construct_cell_kappa(
            v[-loop_count - 1],
            v[-loop_count],
            v[-loop_count + 1],
            p[-1],
            kappa[-loop_count + 1],
        )

        # New points.
        p = np.append(p, [new_p], axis=0)
        v = np.append(v, [new_v], axis=0)

        # Compute new kappa.
        new_kappa = kappa_from_top_pivot_vertex(v[-2:], p[-2, np.newaxis])
        kappa.append(new_kappa)

    return v, p


def construct_loop(v, p, construction_steps):
    """Builds structure assuming it is a helix."""
    # Number of cells to look back.

    v = [v_ for v_ in v]
    p = [p_ for p_ in p]

    # Construct first pivot.
    loop_length = len(v) - 1

    p_left_init = construct_p(v[0], v[1], v[-1], p[0])

    p.append(p_left_init)

    for _ in range(construction_steps):
        new_v, new_p = construct_cell(
            v[-loop_length - 1],
            v[-loop_length],
            v[-loop_length + 1],
            p[-1],
            p[-loop_length],
        )

        p.append(new_p)
        v.append(new_v)

    return v[loop_length:], p[loop_length:]


# CODE TO VALIDATE STRUCTURES.



def hyperbola(v1, v2, kappa):
    """Constructs hyperbola parameters."""

    t = v2 - v1
    t_norm = geometry.norm(t)

    a = -kappa / 2
    c = t_norm / 2
    b = np.sqrt(c**2 - a**2)

    return a, b, c, b / a


def slope_to_angle(slope):
    return np.arctan(slope)


def valid_theta_1(x1, x2, kappa):
    """Calculates valid theta given top vertices and kappa."""
    _, _, _, slope = hyperbola(x1, x2, kappa)
    theta = slope_to_angle(slope)
    return np.where(
        kappa < 0, theta, np.pi + theta
    )  # when kappa=0. (floating point) returns -pi/2


def actual_theta_1(x1, x2, p, normal):
    """Calculates theta from vertex and central pivots."""
    t = x2 - x1
    v2 = p - x2
    return geometry.signed_angle_between(-t, v2, normal)


def valid_theta_2(p_left, x2, kappa_2):
    """Computes valid theta 2, the angle between \vec{R} and v2."""
    return valid_theta(p_left, x2, kappa_2)


def actual_theta_2(p_left, x2, p):
    """Computes angle between \vec{R} and v2."""
    return actual_theta(p_left, x2, p)


def kappa_1(x1, x2, p_prev):
    """Calculates the leg length constant used for construction.

    Args:
        p_i_minus_1: pivot in upper cell.
        x1: lower left vertex.
        x2: lower right vertex.
            
    Returns:
        kappa: scalar.
    """
    s2_prev = p_prev - x1
    s3_prev = p_prev - x2
    l_2_prev = geometry.norm(s2_prev)
    l_3_prev = geometry.norm(s3_prev)
    return l_2_prev - l_3_prev


def kappa_2(x1, x2, p_prev, p_left):
    """Computes `kappa_2`"""
    s4_left = p_left - x1
    l_4_left = geometry.norm(s4_left)
    k1 = kappa_1(x1, x2, p_prev)
    return k1 + l_4_left


def lattice_kappa_1(v, p):
    """Compute value of kappa for lattice given by `v`, `p`."""
    x1 = v[1:, :-1]
    x2 = v[1:, 1:]
    p_prev = p
    return kappa_1(x1, x2, p_prev)


def lattice_kappa_2(v, p):
    """Compute value of kappa_2 for lattice given by `v`, `p`."""
    x1 = v[1:-1, 1:-1]
    x2 = v[1:-1, 2:]
    p_prev = p[:-1, 1:]
    p_left = p[1:, :-1]
    return kappa_2(x1, x2, p_prev, p_left)


def boundary_lattice_kappa_2(x_top, p_first_row, kappa_1_boundary):
    """Computes value of `kappa_2` for top row of lattice."""
    s4_left = p_first_row - x_top[1:]
    l_4_left = geometry.norm(s4_left)
    return kappa_1_boundary[1:] + l_4_left[:-1]


def compute_all_kappa(v, p, kappa_top):
    """Computes value of `kappa` in all cells of lattice."""
    k1_bulk = lattice_kappa_1(v, p)
    return np.concatenate([[kappa_top], k1_bulk])


def compute_all_kappa_2(v, p, kappa_top):
    """Computes all values of `kappa_2` for lattice."""
    k2_bulk = lattice_kappa_2(v, p)
    k2_edge = boundary_lattice_kappa_2(v[1], p[1], kappa_top)

    return np.concatenate([[k2_edge], k2_bulk])


def actual_valid_theta_1_difference(v, p, kappa_top):
    """Calculates difference \theta_{valid} - \theta.

    If the quantity is positive, then the construction is valid (grows forward).
    """
    # First, construct all `kappa`. Then discard bottom row.
    k1_ = compute_all_kappa(v, p, kappa_top)[:-1]

    valid_theta_1_ = valid_theta_1(v[:-1, :-1], v[:-1, 1:], k1_)
    actual_theta_1_ = actual_theta_1(v[:-1, :-1], v[:-1, 1:], p)

    return valid_theta_1_ - actual_theta_1_


def actual_valid_theta_2_difference(v, p, kappa_top):
    """Calculates difference \theta_{valid} - \theta.

    If the quantity is positive, then the construction is valid (grows forward).
    """
    # First, construct all `kappa`. Then discard bottom row.
    k2_ = compute_all_kappa_2(v, p, kappa_top)[:-1]

    valid_theta_2_ = valid_theta_2(v[:-1, :-1], v[:-1, 1:], k2_)
    actual_theta_2_ = actual_theta_2(v[:-1, :-1], v[:-1, 1:], p)

    return valid_theta_2_ - actual_theta_2_
