
export NoReweighting, OverdampedLangevinReweighting


abstract type AbstractReweighting end

struct NoReweighting <: AbstractReweighting end
function _reweighting_callback!(::NoReweighting, args...; kwargs...) end

mutable struct OverdampedLangevinReweighting{F,LW} <: AbstractReweighting
    force_perturbations::F
    log_weights::LW
end

OverdampedLangevinReweighting(force_perturbations) = OverdampedLangevinReweighting(force_perturbations, Float64[])

function _reweighting_callback!(rw::OverdampedLangevinReweighting,
    noise_velocity,
    sys,
    sim,
    step_n,
    n_threads,
    buffers,
    neighbors
)

    if !iszero(sim.remove_CM_motion)
        throw(ArgumentError("OverdampedLangevinReweighting is not compatible with sim.remove_CM_motion = 1 "))
    end

    force_perturbation = zero_forces(sys)

    for inter in rw.force_perturbations
        AtomsCalculators.forces!(force_perturbation, sys, inter; neighbors=neighbors, step_n=step_n,
            n_threads=n_threads, buffers=buffers)
    end

    running_log_weight = isempty(rw.log_weights) ? 0.0 : last(rw.log_weights)

    Δη_squared_prefactor = sim.dt / (2sim.friction * sys.k * sim.temperature)
    Δη_squared = ustrip(NoUnits, Δη_squared_prefactor * dot(force_perturbation, force_perturbation ./ masses(sys)))

    η_dot_Δη_prefactor = sqrt(sim.dt / 2sim.friction) / (sys.k * sim.temperature)
    η_dot_Δη = ustrip(NoUnits, η_dot_Δη_prefactor * dot(force_perturbation, noise_velocity))

    running_log_weight +=  η_dot_Δη - 0.5 * Δη_squared

    push!(rw.log_weights, running_log_weight)
end


# TODO implement captured O-step and B-step for Langevin-Girsanov
# TODO implement default callback for simple perturbations of the potential (Q specific vs general inters)
# TODO implement VACF unit test