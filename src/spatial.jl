# Spatial calculations

export
    vector1D,
    vector,
    wrap_coords,
    wrap_coords_vec,
    maxwell_boltzmann,
    random_velocities,
    random_velocities!,
    bond_angle,
    torsion_angle,
    remove_CM_motion!

"""
    vector1D(c1, c2, side_length)

Displacement between two 1D coordinate values from c1 to c2, accounting for
the bounding box.
The minimum image convention is used, so the displacement is to the closest
version of the coordinate accounting for the periodic boundaries.
"""
function vector1D(c1, c2, side_length)
    if c1 < c2
        return (c2 - c1) < (c1 - c2 + side_length) ? (c2 - c1) : (c2 - c1 - side_length)
    else
        return (c1 - c2) < (c2 - c1 + side_length) ? (c2 - c1) : (c2 - c1 + side_length)
    end
end

"""
    vector(c1, c2, box_size)

Displacement between two coordinate values from c1 to c2, accounting for
the bounding box.
The minimum image convention is used, so the displacement is to the closest
version of the coordinates accounting for the periodic boundaries.
"""
vector(c1, c2, box_size) = vector1D.(c1, c2, box_size)

@generated function vector(c1::SVector{N}, c2::SVector{N}, box_size) where N
    quote
        Base.Cartesian.@ncall $N SVector{$N} i->vector1D(c1[i], c2[i], box_size[i])
    end
end

square_distance(i, j, coords, box_size) = sum(abs2, vector(coords[i], coords[j], box_size))

# Pad a vector to 3D to allow operations such as the cross product
function vector_pad3D(c1::SVector{2, T}, c2::SVector{2, T}, box_size::SVector{2, T}) where T
    SVector{3, T}(
        vector1D(c1[1], c2[1], box_size[1]),
        vector1D(c1[2], c2[2], box_size[2]),
        zero(T),
    )
end

vector_pad3D(c1::SVector{3}, c2::SVector{3}, box_size::SVector{3}) = vector(c1, c2, box_size)

# Trim a vector back to 2D if required
trim3D(v::SVector{3, T}, box_size::SVector{2}) where T = SVector{2, T}(v[1], v[2])
trim3D(v::SVector{3}, box_size::SVector{3}) = v

"""
    wrap_coords(c, side_length)

Ensure a 1D coordinate is within the simulation box and return the coordinate.
"""
wrap_coords(c, side_length) = c - floor(c / side_length) * side_length

"""
    wrap_coords_vec(c, box_size)

Ensure a coordinate is within the simulation box and return the coordinate.
"""
wrap_coords_vec(v, box_size) = wrap_coords.(v, box_size)

const mb_conversion_factor = uconvert(u"u * nm^2 * ps^-2 * K^-1", Unitful.k)

"""
    velocity(mass, temperature; dims=3)
    velocity(mass, temperature, k; dims=3)

Generate a random velocity from the Maxwell-Boltzmann distribution, with
optional custom Boltzmann constant.
"""
function AtomsBase.velocity(mass, temp, k=mb_conversion_factor; dims::Integer=3, rng=Random.GLOBAL_RNG)
    k_strip = (unit(mass) == NoUnits) ? ustrip(k) : k
    return SVector([maxwell_boltzmann(mass, temp, k_strip; rng=rng) for i in 1:dims]...)
end

function velocity_3D(mass, temp, k=mb_conversion_factor; rng=Random.GLOBAL_RNG)
    return SVector(
        maxwell_boltzmann(mass, temp, k; rng=rng),
        maxwell_boltzmann(mass, temp, k; rng=rng),
        maxwell_boltzmann(mass, temp, k; rng=rng),
    )
end

function velocity_2D(mass, temp, k=mb_conversion_factor; rng=Random.GLOBAL_RNG)
    return SVector(
        maxwell_boltzmann(mass, temp, k; rng=rng),
        maxwell_boltzmann(mass, temp, k; rng=rng),
    )
end

"""
    maxwell_boltzmann(mass, temperature; rng=Random.GLOBAL_RNG)
    maxwell_boltzmann(mass, temperature, k; rng=Random.GLOBAL_RNG)

Generate a random speed along one dimension from the Maxwell-Boltzmann
distribution, with optional custom Boltzmann constant.
"""
function maxwell_boltzmann(mass, temp, k; rng=Random.GLOBAL_RNG)
    T = typeof(convert(AbstractFloat, ustrip(temp)))
    σ = sqrt(k * temp / mass)
    return rand(rng, Normal(zero(T), T(ustrip(σ)))) * unit(σ)
end

function maxwell_boltzmann(mass, temp; rng=Random.GLOBAL_RNG)
    k = unit(temp) == NoUnits ? ustrip(mb_conversion_factor) : mb_conversion_factor
    return maxwell_boltzmann(mass, temp, k; rng=rng)
end
"""
    random_velocities(sys, temp)

Generate random velocities from the Maxwell-Boltzmann distribution
for a `System`.
"""
function random_velocities(sys::AbstractSystem{3}, temp; rng=Random.GLOBAL_RNG)
    if isa(sys.coords, CuArray)
        return cu(velocity_3D.(Array(mass.(sys.atoms)), temp, sys.k; rng=rng))
    else
        return velocity_3D.(mass.(sys.atoms), temp, sys.k; rng=rng)
    end
end

function random_velocities(sys::AbstractSystem{2}, temp; rng=Random.GLOBAL_RNG)
    if isa(sys.coords, CuArray)
        return cu(velocity_2D.(Array(mass.(sys.atoms)), temp,sys.k; rng=rng))
    else
        return velocity_2D.(mass.(sys.atoms), temp,sys.k; rng=rng)
    end
end

"""
    random_velocities!(sys, temp)

Set the velocities of a `System` to random velocities generated from the
Maxwell-Boltzmann distribution.
"""
function random_velocities!(sys, temp; rng=Random.GLOBAL_RNG)
    sys.velocities = random_velocities(sys, temp; rng=rng)
    return sys
end

# Sometimes domain error occurs for acos if the value is > 1.0 or < -1.0
acosbound(x::Real) = acos(clamp(x, -1, 1))

"""
    bond_angle(coord_i, coord_j, coord_k, box_size)
    bond_angle(vec_ji, vec_jk)

Calculate the bond or pseudo-bond angle in radians between three
coordinates or two vectors.
The angle between j→i and j→k is returned in the range 0 to π.
"""
function bond_angle(coords_i, coords_j, coords_k, box_size)
    vec_ji = vector(coords_j, coords_i, box_size)
    vec_jk = vector(coords_j, coords_k, box_size)
    return bond_angle(vec_ji, vec_jk)
end

function bond_angle(vec_ji, vec_jk)
    acosbound(dot(vec_ji, vec_jk) / (norm(vec_ji) * norm(vec_jk)))
end

"""
    torsion_angle(coord_i, coord_j, coord_k, coord_l, box_size)
    torsion_angle(vec_ij, vec_jk, vec_kl)

Calculate the torsion angle in radians defined by four coordinates or
three vectors.
The angle between the planes defined by atoms (i, j, k) and (j, k, l) is
returned in the range -π to π.
"""
function torsion_angle(coords_i, coords_j, coords_k, coords_l, box_size)
    vec_ij = vector(coords_i, coords_j, box_size)
    vec_jk = vector(coords_j, coords_k, box_size)
    vec_kl = vector(coords_k, coords_l, box_size)
    return torsion_angle(vec_ij, vec_jk, vec_kl)
end

function torsion_angle(vec_ij, vec_jk, vec_kl)
    cross_ij_jk = vec_ij × vec_jk
    cross_jk_kl = vec_jk × vec_kl
    θ = atan(
        ustrip(dot(cross_ij_jk × cross_jk_kl, normalize(vec_jk))),
        ustrip(dot(cross_ij_jk, cross_jk_kl)),
    )
    return θ
end

# Used to write an rrule that can override the Zygote sum adjoint
sum_svec(arr) = sum(arr)

"""
    remove_CM_motion!(system)

Remove the centre of mass motion from a system.
"""
function remove_CM_motion!(sys)
    masses = mass.(sys.atoms)
    cm_momentum = sum_svec(sys.velocities .* masses)
    cm_velocity = cm_momentum / sum(masses)
    sys.velocities = sys.velocities .- (cm_velocity,)
    return sys
end
