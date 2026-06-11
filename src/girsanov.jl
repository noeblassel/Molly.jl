
export NoReweighting, TrajectoryReweighting

abstract type AbstractReweighting end

"""
    NoReweighting()

Trivial trajectory reweighting that performs no work and produces no weights.

This is the default `trajectory_reweighting` keyword argument to [`simulate!`](@ref).
"""
struct NoReweighting <: AbstractReweighting end
function _reweighting_callback!(::NoReweighting, args...; kwargs...) end


mutable struct OverdampedLangevinReweighting{T,F,FB,PF1,PF2,IM} <: AbstractReweighting
    force_perturbations::F
    log_weights::Vector{T}

    force_buffer::FB
    Δη_squared_prefactor::PF1
    η_dot_Δη_prefactor::PF2
    inv_masses::IM
end

raw"""
    TrajectoryReweighting(sys, sim; force_perturbations)
    TrajectoryReweighting(sys, sim; feature_basis, linear_parameters)

Girsanov trajectory reweighting for the [`OverdampedLangevin`](@ref) and
[`LangevinSplitting`](@ref) simulators (`"ABOBA"` splitting). Accumulates path
log-weights for reweighting trajectories sampled under `sim` to a target dynamics
that differs only in its force term. Pass the result to [`simulate!`](@ref) via
the `trajectory_reweighting` keyword; running log-weights accumulate in
`log_weights`. Not compatible with `sim.remove_CM_motion != 0`.

Provide exactly one of:
- `force_perturbations`: an iterable of interactions giving the difference in
  force functions between target and reference dynamics.
  Each of these interactions should implement a method for
  `AtomsCalculators.forces!`, see [General interactions](@ref).
- `feature_basis` and `linear_parameters`: a callable `feature_basis(sys)`
  returning an `(N atoms) × (N features)` array of 3-vectors (the force feature
  matrix `D(x)`), and an
  `(N features) × (N ensembles)` matrix `Θ`, for linear perturbations
  ```math
  \delta F(x) = \Theta^\top D(x)
  ```
  allowing efficient reweighting to many target dynamics at once.
"""
struct TrajectoryReweighting{S<:AbstractReweighting,L} <: AbstractReweighting
    scheme::S
    log_weights::L
end

function _make_reweighting(sys, sim::OverdampedLangevin, force_perturbations)
    T = float_type(sys)

    force_buffer = zero_forces(sys)
    Δη_squared_prefactor = sim.dt / (2sim.friction * sys.k * sim.temperature)
    η_dot_Δη_prefactor = sqrt(sim.dt / 2sim.friction) / (sys.k * sim.temperature)

    inv_masses = inv.(sys.masses)

    return OverdampedLangevinReweighting(force_perturbations, T[], force_buffer, Δη_squared_prefactor, η_dot_Δη_prefactor, inv_masses)
end

function _reweighting_callback!(rw::OverdampedLangevinReweighting{T},
    noise_velocity,
    sys,
    step_n,
    n_threads,
    buffers,
    neighbors
) where {T}

    fill!(rw.force_buffer, zero(eltype(rw.force_buffer)))

    for inter in rw.force_perturbations
        AtomsCalculators.forces!(rw.force_buffer, sys, inter; neighbors=neighbors, step_n=step_n,
            n_threads=n_threads, buffers=buffers)
    end

    running_log_weight = isempty(rw.log_weights) ? zero(T) : last(rw.log_weights)

    Δη_squared = ustrip(NoUnits, rw.Δη_squared_prefactor * dot(rw.force_buffer, rw.force_buffer .* rw.inv_masses))

    η_dot_Δη = ustrip(NoUnits, rw.η_dot_Δη_prefactor * dot(rw.force_buffer, noise_velocity))

    running_log_weight += η_dot_Δη - Δη_squared / 2

    push!(rw.log_weights, running_log_weight)
end

mutable struct MultipleOverdampedLangevinReweighting{T,FB,LP,FM,SFM,VB,GB,PF1,PF2,IM} <: AbstractReweighting
    feature_basis::FB
    linear_parameters::LP

    log_weights::Vector{Vector{T}}

    feature_buffer::FM
    scaled_feature_buffer::SFM
    v_buffer::VB
    gram_buffer::GB
    Δη_squared_prefactor::PF1
    η_dot_Δη_prefactor::PF2
    sqrt_inv_masses::IM
end

function _make_reweighting(sys, sim::OverdampedLangevin, feature_basis, linear_parameters)
    T = float_type(sys)

    # compute force feature basis once for setup

    feature_buffer = zero(feature_basis(sys)) # Nat × P array of SVector{3}, where Nat is the number of atoms

    if (size(feature_buffer, 1) != length(sys)) || (size(feature_buffer, 2) != size(linear_parameters, 1)) || (ndims(feature_buffer) != 2)
        throw(DimensionMismatch("Force descriptor basis has shape $(size(feature_buffer)), should be ($(length(sys)), $(size(linear_parameters,1)))."))
    end

    if size(first(feature_buffer)) != (size(first(sys.coords)))
        throw(DimensionMismatch("Components of the force descriptor basis have shape $(size(first(feature_buffer))), should be $(size(first(sys.coords)))."))
    end

    if ndims(linear_parameters) != 2
        throw(DimensionMismatch("`linear_parameters` should be two-dimensional, got `ndims(linear_parameters)=$(ndims(linear_parameters))`"))
    end

    Δη_squared_prefactor = sim.dt / (2sim.friction * sys.k * sim.temperature)
    η_dot_Δη_prefactor = sqrt(sim.dt / 2sim.friction) / (sys.k * sim.temperature)

    sqrt_inv_masses = sqrt.(inv.(sys.masses))

    v_buffer = feature_buffer' * sys.velocities
    scaled_feature_buffer = zero(sqrt_inv_masses .* feature_buffer)
    gram_buffer = zero(scaled_feature_buffer' * scaled_feature_buffer)

    return MultipleOverdampedLangevinReweighting(feature_basis, linear_parameters, Vector{T}[], feature_buffer, scaled_feature_buffer, v_buffer, gram_buffer, Δη_squared_prefactor, η_dot_Δη_prefactor, sqrt_inv_masses)
end

function _reweighting_callback!(rw::MultipleOverdampedLangevinReweighting{T},
    noise_velocity,
    sys,
    step_n,
    n_threads,
    buffers,
    neighbors
) where {T}

    rw.feature_buffer .= rw.feature_basis(sys)  # D matrix n_atoms × n_features matrix of 3-vectors
    Θ = rw.linear_parameters                    # n_features × n_params

    mul!(rw.v_buffer, rw.feature_buffer', noise_velocity)    # n_features

    rw.scaled_feature_buffer .= rw.sqrt_inv_masses .* rw.feature_buffer
    mul!(rw.gram_buffer, rw.scaled_feature_buffer', rw.scaled_feature_buffer)

    Δη_squared = ustrip_vec(NoUnits, rw.Δη_squared_prefactor .* vec(sum(Θ .* (rw.gram_buffer * Θ); dims=1)))
    η_dot_Δη = ustrip_vec(NoUnits, rw.η_dot_Δη_prefactor .* (Θ' * rw.v_buffer))

    n_params = size(Θ, 2)

    running_log_weight = isempty(rw.log_weights) ? zeros(T, n_params) : copy(last(rw.log_weights))
    running_log_weight .+= η_dot_Δη .- Δη_squared ./ 2

    push!(rw.log_weights, running_log_weight)
end

const _girsanov_implemented_splittings = ["ABOBA"]
const _girsanov_intercept_count = Dict("ABOBA" => 1)

function _girsanov_prefactors_langevin(::Val{:ABOBA}, sys, sim)
    M_inv = inv.(masses(sys))
    α_eff = exp.(-sim.friction * sim.dt .* M_inv / count('O', sim.splitting))
    σ_eff = sqrt.((1 * unit(eltype(α_eff))) .- (α_eff .^ 2))

    scaling = sim.dt .* (α_eff .+ 1) ./ (2 .* σ_eff)
    Δη_squared_prefactor = (scaling .^ 2) .* M_inv ./ (sys.k * sim.temperature)
    η_dot_Δη_prefactor = scaling ./ (sys.k * sim.temperature)

    return ((Δη_squared_prefactor,), (η_dot_Δη_prefactor,))
end

mutable struct LangevinSplittingReweighting{K,T,F,NB,FB,PF1,PF2} <: AbstractReweighting
    splitting::String
    force_perturbations::F

    noise_velocity_buffer::NTuple{K,NB}
    force_buffer::NTuple{K,FB}

    Δη_squared_prefactor::NTuple{K,PF1}
    η_dot_Δη_prefactor::NTuple{K,PF2}

    log_weights::Vector{T}
end

function _make_reweighting(sys, sim::LangevinSplitting, force_perturbations)

    splitting = String(sim.splitting)

    if !(splitting in _girsanov_implemented_splittings)
        throw(ArgumentError("Splitting $(splitting) is not available for Girsanov reweighting. Available splittings: $(join(_girsanov_implemented_splittings,", "))."))
    end

    T = float_type(sys)
    K = _girsanov_intercept_count[splitting]

    zero_force = zero_forces(sys)
    zero_velocity = zero(sys.velocities)

    force_buffer = NTuple{K}(copy(zero_force) for i = 1:K)
    noise_velocity_buffer = NTuple{K}(copy(zero_velocity) for i = 1:K)
    log_weights = T[]

    Δη_squared_prefactor, η_dot_Δη_prefactor = _girsanov_prefactors_langevin(Val(Symbol(splitting)), sys, sim)

    return LangevinSplittingReweighting(
        String(splitting), force_perturbations, noise_velocity_buffer, force_buffer,
        Δη_squared_prefactor, η_dot_Δη_prefactor, log_weights,
    )
end

function _reweighting_callback!(rw::LangevinSplittingReweighting, args...)
    if rw.splitting == "ABOBA"
        _reweighting_callback_aboba!(rw, args...)
    end
end


function _reweighting_callback_aboba!(rw::LangevinSplittingReweighting{K,T},
    noise,
    sys,
    step_n,
    n_threads,
    buffers,
    neighbors,
    op_index) where {K,T}

    if op_index == 1 # after first A step
        fill!(rw.force_buffer[1], zero(eltype(rw.force_buffer[1])))

        for inter in rw.force_perturbations
            AtomsCalculators.forces!(rw.force_buffer[1], sys, inter; neighbors=neighbors, step_n=step_n,
                n_threads=n_threads, buffers=buffers)
        end

    elseif op_index == 3 # after O step

        rw.noise_velocity_buffer[1] .= noise

    elseif op_index == 5 # after last A step

        Δη_squared = ustrip(NoUnits, dot(rw.force_buffer[1], rw.Δη_squared_prefactor[1] .* rw.force_buffer[1]))

        η_dot_Δη = ustrip(NoUnits, dot(rw.force_buffer[1], rw.η_dot_Δη_prefactor[1] .* rw.noise_velocity_buffer[1]))

        running_log_weight = isempty(rw.log_weights) ? zero(T) : last(rw.log_weights)
        running_log_weight += η_dot_Δη - Δη_squared / 2

        push!(rw.log_weights, running_log_weight)

    end
end

mutable struct MultipleLangevinSplittingReweighting{K,T,FB,LP,NB,FM,SFM,VB,GB,PF1,PF2} <: AbstractReweighting
    splitting::String
    feature_basis::FB
    linear_parameters::LP

    log_weights::Vector{Vector{T}}

    noise_velocity_buffer::NTuple{K,NB}
    feature_buffer::NTuple{K,FM}
    scaled_feature_buffer::NTuple{K,SFM}
    v_buffer::VB
    gram_buffer::GB

    sqrt_Δη_squared_prefactor::NTuple{K,PF1}
    η_dot_Δη_prefactor::NTuple{K,PF2}
end

function _make_reweighting(sys, sim::LangevinSplitting, feature_basis, linear_parameters)

    splitting = String(sim.splitting)

    if !(splitting in _girsanov_implemented_splittings)
        throw(ArgumentError("Splitting $(splitting) is not available for Girsanov reweighting. Available splittings: $(join(_girsanov_implemented_splittings,", "))."))
    end

    T = float_type(sys)
    K = _girsanov_intercept_count[splitting]

    # compute force feature basis once for setup
    feature_buffer_ = zero(feature_basis(sys)) # n_atom × n_params array of 3-vectors

    if size(feature_buffer_, 1) != length(sys)
        throw(DimensionMismatch("The number of force descriptors does not match the number of atoms in the system. `force_feature_basis` should return an array of 3-vectors, of shape (N atoms) x (N features)"))
    end

    if size(first(feature_buffer_)) != (size(first(sys.coords)))
        throw(DimensionMismatch("`force_feature_basis` should return an array of vectors shaped like physical coordinates."))
    end

    if size(feature_buffer_, 2) != size(linear_parameters, 1)
        throw(DimensionMismatch("`size(linear_parameters,1)` should be equal to the number of force descriptors."))
    end

    Δη_squared_prefactor, η_dot_Δη_prefactor = _girsanov_prefactors_langevin(Val(Symbol(splitting)), sys, sim)

    sqrt_Δη_squared_prefactor = map(p -> sqrt.(p), Δη_squared_prefactor)

    feature_buffer = NTuple{K}(copy(feature_buffer_) for i = 1:K)
    noise_velocity_buffer = NTuple{K}(zero(η_dot_Δη_prefactor[i] .* sys.velocities) for i = 1:K)

    scaled_feature_buffer = NTuple{K}(zero(sqrt_Δη_squared_prefactor[i] .* feature_buffer_) for i = 1:K)

    v_buffer = zero(feature_buffer_' * (η_dot_Δη_prefactor[1] .* sys.velocities))
    gram_buffer = zero(scaled_feature_buffer[1]' * scaled_feature_buffer[1])

    log_weights = Vector{T}[]

    return MultipleLangevinSplittingReweighting(
        String(splitting), feature_basis, linear_parameters, log_weights,
        noise_velocity_buffer, feature_buffer, scaled_feature_buffer, v_buffer, gram_buffer,
        sqrt_Δη_squared_prefactor, η_dot_Δη_prefactor,
    )
end

function _reweighting_callback!(rw::MultipleLangevinSplittingReweighting, args...)
    if rw.splitting == "ABOBA"
        _reweighting_callback_aboba!(rw, args...)
    end
end


function _reweighting_callback_aboba!(rw::MultipleLangevinSplittingReweighting{K,T},
    noise,
    sys,
    step_n,
    n_threads,
    buffers,
    neighbors,
    op_index) where {K,T}

    if op_index == 1 # after first A step
        rw.feature_buffer[1] .= rw.feature_basis(sys)  # D matrix, n_atoms × n_features of 3-vectors

    elseif op_index == 3 # after O step
        rw.noise_velocity_buffer[1] .= rw.η_dot_Δη_prefactor[1] .* noise

    elseif op_index == 5 # after last A step
        Θ = rw.linear_parameters

        mul!(rw.v_buffer, rw.feature_buffer[1]', rw.noise_velocity_buffer[1])

        rw.scaled_feature_buffer[1] .= rw.sqrt_Δη_squared_prefactor[1] .* rw.feature_buffer[1]
        mul!(rw.gram_buffer, rw.scaled_feature_buffer[1]', rw.scaled_feature_buffer[1])

        Δη_squared = ustrip_vec(NoUnits, vec(sum(Θ .* (rw.gram_buffer * Θ); dims=1)))
        η_dot_Δη = ustrip_vec(NoUnits, Θ' * rw.v_buffer)

        n_params = size(Θ, 2)

        running_log_weight = isempty(rw.log_weights) ? zeros(T, n_params) : copy(last(rw.log_weights))
        running_log_weight .+= η_dot_Δη .- Δη_squared ./ 2

        push!(rw.log_weights, running_log_weight)

    end

end

function TrajectoryReweighting(sys, sim;
    force_perturbations=nothing, feature_basis=nothing, linear_parameters=nothing)

    if !iszero(sim.remove_CM_motion)
        throw(ArgumentError("TrajectoryReweighting is not compatible with sim.remove_CM_motion = 1 "))
    end

    scheme = if force_perturbations !== nothing
        (feature_basis === nothing && linear_parameters === nothing) ||
            throw(ArgumentError("Supply either `force_perturbations`, or " *
                                "`feature_basis` and `linear_parameters`, not both."))
        _make_reweighting(sys, sim, force_perturbations)
    elseif feature_basis !== nothing && linear_parameters !== nothing
        _make_reweighting(sys, sim, feature_basis, linear_parameters)
    else
        throw(ArgumentError("Supply either `force_perturbations`, or both " *
                            "`feature_basis` and `linear_parameters`."))
    end

    return TrajectoryReweighting{typeof(scheme),typeof(scheme.log_weights)}(scheme, scheme.log_weights)
end

_reweighting_callback!(rw::TrajectoryReweighting, args...) =
    _reweighting_callback!(rw.scheme, args...)