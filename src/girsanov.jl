
export NoReweighting, OverdampedLangevinReweighting, LangevinSplittingReweighting, MultipleOverdampedLangevinReweighting, MultipleLangevinSplittingReweighting

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

"""
    OverdampedLangevinReweighting(sys, sim, force_perturbations)

Girsanov trajectory reweighting for the [`OverdampedLangevin`](@ref) simulator.

Accumulates path log-weights for future reweighting of trajectories sampled
under `sim` to a target dynamics whose force function differs by the sum of
`force_perturbations`. The running log-weights are stored in `log_weights`
and updated once per step.
Not compatible with `sim.remove_CM_motion != 0`.

# Arguments
- `sys`: the [`System`](@ref) being simulated.
- `sim::OverdampedLangevin`: the reference simulator generating the trajectory.
- `force_perturbations`: an iterable of interactions giving the
    difference in force functions between the target and reference dynamics.

Interactions in `force_perturbations` should implement a method for `AtomsCalculators.forces!`, see [General interactions](@ref).
"""
function OverdampedLangevinReweighting(sys, sim, force_perturbations)
    T = float_type(sys)

    if !iszero(sim.remove_CM_motion)
        throw(ArgumentError("OverdampedLangevinReweighting is not compatible with sim.remove_CM_motion = 1 "))
    end

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


raw"""
    MultipleOverdampedLangevinReweighting(args...)

Simultaneous reweighting to multiple path ensembles, for linear perturbations of the form

    ```math
        \delta F(x) = \theta^\top D(x)
    ```

    where \(D(x)\) is a basis of descriptors, and \(\theta\) is a vector of linear weights.

"""
mutable struct MultipleOverdampedLangevinReweighting{T,FB,LP,FM,VB,GB,PF1,PF2,IM} <: AbstractReweighting
    feature_basis::FB
    linear_parameters::LP

    log_weights::Vector{Vector{T}}

    feature_buffer::FM
    v_buffer::VB
    gram_buffer::GB
    Δη_squared_prefactor::PF1
    η_dot_Δη_prefactor::PF2
    sqrt_inv_masses::IM
end

# """
#     MultipleOverdampedLangevinReweighting(sys, sim, feature_basis, linear_parameters)

# Girsanov trajectory reweighting for the [`OverdampedLangevin`](@ref) simulator.

# Accumulates path log-weights for future reweighting of trajectories sampled
# under `sim` to a target dynamics whose force function differs by the sum of
# `force_perturbations`. The running log-weights are stored in `log_weights`
# and updated once per step.
# Not compatible with `sim.remove_CM_motion != 0`.

# # Arguments
# - `sys`: the [`System`](@ref) being simulated.
# - `sim::OverdampedLangevin`: the reference simulator generating the trajectory.
# - `force_perturbations`: an iterable of interactions giving the
#     difference in force functions between the target and reference dynamics.

# """
function MultipleOverdampedLangevinReweighting(sys, sim, feature_basis, linear_parameters)
    T = float_type(sys)

    if !iszero(sim.remove_CM_motion)
        throw(ArgumentError("MultipleOverdampedLangevinReweighting is not compatible with sim.remove_CM_motion = 1 "))
    end

    # compute force feature basis once for setup

    feature_buffer = zero(feature_basis(sys)) # Nat × P array of SVector{3}, where Nat is the number of atoms

    if size(feature_buffer, 1) != length(sys)
        throw(DimensionMismatch("The number of force descriptors does not match the number of atoms in the system. `force_feature_basis` should return an array of 3-vectors, of shape (N atoms) x (N features)"))
    end

    if size(first(feature_buffer)) != (size(first(sys.coords)))
        throw(DimensionMismatch("`force_feature_basis` should return an array of vectors shaped like physical coordinates."))
    end

    if size(feature_buffer, 2) != size(linear_parameters, 1)
        throw(DimensionMismatch("`size(linear_parameters,1)` should be equal to the number of force descriptors."))
    end

    Δη_squared_prefactor = sim.dt / (2sim.friction * sys.k * sim.temperature)
    η_dot_Δη_prefactor = sqrt(sim.dt / 2sim.friction) / (sys.k * sim.temperature)

    sqrt_inv_masses = sqrt.(inv.(sys.masses))

    v_buffer = feature_buffer' * sys.velocities
    gram_buffer = zero((sqrt_inv_masses .* feature_buffer)' * (sqrt_inv_masses .* feature_buffer))

    return MultipleOverdampedLangevinReweighting(feature_basis, linear_parameters, Vector{T}[], feature_buffer, v_buffer, gram_buffer, Δη_squared_prefactor, η_dot_Δη_prefactor, sqrt_inv_masses)
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

    rw.feature_buffer .*= rw.sqrt_inv_masses
    mul!(rw.gram_buffer, rw.feature_buffer', rw.feature_buffer)

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

"""
    LangevinSplittingReweighting(splitting, sys, sim, force_perturbations)

Girsanov trajectory reweighting for the [`LangevinSplitting`](@ref) simulator.

Accumulates path log-weights for future reweighting of trajectories sampled
under `sim` to a target dynamics whose force function differs by the sum of
`force_perturbations`. The running log-weights are stored in `log_weights`
and updated once per step.
Currently only the `"ABOBA"` splitting is supported; `splitting` must match
`sim.splitting`. Not compatible with `sim.remove_CM_motion != 0`.

# Arguments
- `splitting::AbstractString`: the splitting scheme; must match `sim.splitting`
    and appear in `Molly._girsanov_implemented_splittings`.
- `sys`: the [`System`](@ref) being simulated.
- `sim::LangevinSplitting`: the reference simulator generating the trajectory.
- `force_perturbations`: an iterable of interactions giving the
    difference in force functions between the target and reference dynamics.

Interactions in `force_perturbations` should implement a method for `AtomsCalculators.forces!`, see [General interactions](@ref).
"""
function LangevinSplittingReweighting(splitting, sys, sim, force_perturbations)

    if !(splitting in _girsanov_implemented_splittings)
        throw(ArgumentError("Splitting $(splitting) is not available for Girsanov reweighting. Available splittings: $(join(_girsanov_implemented_splittings,", "))."))
    end

    if splitting != sim.splitting
        throw(ArgumentError("Reweighting splitting ($(splitting)) does not match simulator splitting ($(sim.splitting))."))
    end

    if !iszero(sim.remove_CM_motion)
        throw(ArgumentError("LangevinSplittingReweighting is not compatible with sim.remove_CM_motion = 1 "))
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

mutable struct MultipleLangevinSplittingReweighting{K,T,FB,LP,NB,FM,VB,GB,PF1,PF2} <: AbstractReweighting
    splitting::String
    feature_basis::FB
    linear_parameters::LP

    log_weights::Vector{Vector{T}}

    noise_velocity_buffer::NTuple{K,NB}
    feature_buffer::NTuple{K,FM}
    v_buffer::VB
    gram_buffer::GB

    sqrt_Δη_squared_prefactor::NTuple{K,PF1}
    η_dot_Δη_prefactor::NTuple{K,PF2}
end

raw"""
    MultipleLangevinSplittingReweighting(splitting, sys, sim, feature_basis, linear_parameters)

Simultaneous reweighting to multiple path ensembles for the [`LangevinSplitting`](@ref)
simulator, for linear perturbations of the form

    ```math
        \delta F(x) = \theta^\top D(x)
    ```

    where \(D(x)\) is a basis of descriptors, and \(\theta\) is a vector of linear weights.

One running log-weight is accumulated per column of `linear_parameters` and stored
in `log_weights`. Currently only the `"ABOBA"` splitting is supported; `splitting`
must match `sim.splitting`. Not compatible with `sim.remove_CM_motion != 0`.

# Arguments
- `splitting::AbstractString`: the splitting scheme; must match `sim.splitting`
    and appear in `Molly._girsanov_implemented_splittings`.
- `sys`: the [`System`](@ref) being simulated.
- `sim::LangevinSplitting`: the reference simulator generating the trajectory.
- `feature_basis`: a callable mapping `sys` to an `(N atoms) × (N features)` array of
    3-vectors `D(x)`, the basis of force descriptors.
- `linear_parameters`: an `(N features) × (N ensembles)` matrix `Θ` whose columns
    are the linear weights `θ` defining each target dynamics.
"""
function MultipleLangevinSplittingReweighting(splitting, sys, sim, feature_basis, linear_parameters)

    if !(splitting in _girsanov_implemented_splittings)
        throw(ArgumentError("Splitting $(splitting) is not available for Girsanov reweighting. Available splittings: $(join(_girsanov_implemented_splittings,", "))."))
    end

    if splitting != sim.splitting
        throw(ArgumentError("Reweighting splitting ($(splitting)) does not match simulator splitting ($(sim.splitting))."))
    end

    if !iszero(sim.remove_CM_motion)
        throw(ArgumentError("MultipleLangevinSplittingReweighting is not compatible with sim.remove_CM_motion = 1 "))
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

    v_buffer = zero(feature_buffer_' * (η_dot_Δη_prefactor[1] .* sys.velocities))
    gram_buffer = zero((sqrt_Δη_squared_prefactor[1] .* feature_buffer_)' * (sqrt_Δη_squared_prefactor[1] .* feature_buffer_))

    log_weights = Vector{T}[]

    return MultipleLangevinSplittingReweighting(
        String(splitting), feature_basis, linear_parameters, log_weights,
        noise_velocity_buffer, feature_buffer, v_buffer, gram_buffer,
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

        rw.feature_buffer[1] .*= rw.sqrt_Δη_squared_prefactor[1]
        mul!(rw.gram_buffer, rw.feature_buffer[1]', rw.feature_buffer[1])

        Δη_squared = ustrip_vec(NoUnits, vec(sum(Θ .* (rw.gram_buffer * Θ); dims=1)))
        η_dot_Δη = ustrip_vec(NoUnits, Θ' * rw.v_buffer)

        n_params = size(Θ, 2)

        running_log_weight = isempty(rw.log_weights) ? zeros(T, n_params) : copy(last(rw.log_weights))
        running_log_weight .+= η_dot_Δη .- Δη_squared ./ 2

        push!(rw.log_weights, running_log_weight)

    end

end