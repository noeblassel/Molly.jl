
export NoReweighting, OverdampedLangevinReweighting, LangevinSplittingReweighting


abstract type AbstractReweighting end

struct NoReweighting <: AbstractReweighting end
function _reweighting_callback!(::NoReweighting, args...; kwargs...) end

mutable struct OverdampedLangevinReweighting{T,F} <: AbstractReweighting
    force_perturbations::F
    log_weights::Vector{T}
    function OverdampedLangevinReweighting{T,F}(force_perturbations, log_weights) where {T,F}
        return new{T,F}(force_perturbations, log_weights)
    end
end

function OverdampedLangevinReweighting(sys, force_perturbations)
    T = float_type(sys)
    return OverdampedLangevinReweighting{T, typeof(force_perturbations)}(force_perturbations, T[])
end

function _reweighting_callback!(rw::OverdampedLangevinReweighting{T},
    noise_velocity,
    sys,
    sim,
    step_n,
    n_threads,
    buffers,
    neighbors
) where {T}

    if !iszero(sim.remove_CM_motion)
        throw(ArgumentError("OverdampedLangevinReweighting is not compatible with sim.remove_CM_motion = 1 "))
    end

    force_perturbation = zero_forces(sys)

    for inter in rw.force_perturbations
        AtomsCalculators.forces!(force_perturbation, sys, inter; neighbors=neighbors, step_n=step_n,
            n_threads=n_threads, buffers=buffers)
    end

    running_log_weight = isempty(rw.log_weights) ? zero(T) : last(rw.log_weights)

    Δη_squared_prefactor = sim.dt / (2sim.friction * sys.k * sim.temperature)
    Δη_squared = ustrip(NoUnits, Δη_squared_prefactor * dot(force_perturbation, force_perturbation ./ masses(sys)))

    η_dot_Δη_prefactor = sqrt(sim.dt / 2sim.friction) / (sys.k * sim.temperature)
    η_dot_Δη = ustrip(NoUnits, η_dot_Δη_prefactor * dot(force_perturbation, noise_velocity))

    running_log_weight += η_dot_Δη - Δη_squared / 2

    push!(rw.log_weights, running_log_weight)
end

const _girsanov_implemented_splittings = ["ABOBA"]
const _girsanov_intercept_count = Dict("ABOBA" => 1)

mutable struct LangevinSplittingReweighting{K,T,F,NB,FB} <: AbstractReweighting
    splitting::String
    force_perturbations::F

    noise_velocity_buffer::NTuple{K,NB}
    force_buffer::NTuple{K,FB}

    log_weights::Vector{T}

    function LangevinSplittingReweighting{K,T,F,NB,FB}(
            splitting, force_perturbations, noise_velocity_buffer, force_buffer, log_weights,
        ) where {K,T,F,NB,FB}
        return new{K,T,F,NB,FB}(splitting, force_perturbations,
                                noise_velocity_buffer, force_buffer, log_weights)
    end
end

function LangevinSplittingReweighting(splitting, sys, force_perturbations)
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

    F = typeof(force_perturbations)
    NB = typeof(zero_velocity)
    FB = typeof(zero_force)

    return LangevinSplittingReweighting{K,T,F,NB,FB}(
        String(splitting), force_perturbations, noise_velocity_buffer, force_buffer, log_weights,
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
    sim,
    step_n,
    n_threads,
    buffers,
    neighbors,
    op_index,
    α_eff,
    σ_eff
) where {K,T}

    if op_index == 1 # after first A step
        force_perturbation = zero_forces(sys)

        for inter in rw.force_perturbations
            AtomsCalculators.forces!(force_perturbation, sys, inter; neighbors=neighbors, step_n=step_n,
                n_threads=n_threads, buffers=buffers)
        end

        rw.force_buffer[1] .= force_perturbation

    elseif op_index == 3 # after O step

        rw.noise_velocity_buffer[1] .= noise

    elseif op_index == 5 # after last A step

        scaled_perturbation = rw.force_buffer[1] .* sim.dt .* (α_eff .+ 1) ./ (2 .* σ_eff)

        Δη_squared = ustrip(NoUnits,
            dot(scaled_perturbation, scaled_perturbation ./ masses(sys)) / (sys.k * sim.temperature))

        η_dot_Δη = ustrip(NoUnits,
            dot(scaled_perturbation, rw.noise_velocity_buffer[1]) / (sys.k * sim.temperature))

        running_log_weight = isempty(rw.log_weights) ? zero(T) : last(rw.log_weights)
        running_log_weight += η_dot_Δη - Δη_squared / 2

        push!(rw.log_weights, running_log_weight)

    end


end