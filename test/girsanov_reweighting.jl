"""
A constant drift applied as a biasing force on the first atom of a system.
"""
struct ConstantDrift{T}
    f::SVector{3,T}
end

function AtomsCalculators.forces!(fs, sys, inter::ConstantDrift; kwargs...)
    fs[1] = fs[1] + inter.f
    return fs
end

"""
A harmonic potential with stiffness h
V(x) = h|x|²/2
"""
struct HarmonicPotential{T}
    h::T
end

function AtomsCalculators.forces!(fs, sys, inter::HarmonicPotential; neighbors=nothing, kwargs...)
    fs .-= inter.h * sys.coords
end

"""
Constructs a unitless system consisting of a single 1D particle in a harmonic energy well,
initialized in canonical equilibrium.
The state is represented as a monoatomic 3D system, with two spurious dimensions.
The first coordinate and velocity are logged so that (auto)correlation functions can be estimated.
"""
function harmonic_oscillator(h; temp=1.0, rng=Random.default_rng())
    atoms = [Atom(mass=1.0)]
    coords = [sqrt(temp / h) * SVector{3}(randn(rng), 0.0, 0.0)]
    velocities = [sqrt(temp) * SVector{3}(randn(rng), 0.0, 0.0)]

    one_d_coord(sys, args...; kwargs...) = first(sys.coords)[1]
    one_d_velocity(sys, args...; kwargs...) = first(sys.velocities)[1]
    loggers = (coords=GeneralObservableLogger(one_d_coord, Float64, 1),
        velocities=GeneralObservableLogger(one_d_velocity, Float64, 1))

    return System(
        atoms=atoms,
        coords=coords,
        boundary=CubicBoundary(Inf),
        velocities=velocities,
        general_inters=(HarmonicPotential(h),),
        loggers=loggers,
        force_units=NoUnits,
        energy_units=NoUnits,
        k=1.0,
    )
end

"""
Returns a `make_system` method producing identical Lennard-Jones fluids on each call.
"""
function lennard_jones_system_maker(; n_atoms, temp)
    boundary = CubicBoundary(10.0u"nm")
    coords = place_atoms(n_atoms, boundary; min_dist=0.3u"nm")
    velocities = [random_velocity(10.0u"g/mol", temp) .* 0.01 for i in 1:n_atoms]
    atoms = [Atom(mass=10.0u"g/mol", charge=0.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1")
             for i in 1:n_atoms]

    make_system() = System(
        atoms=atoms,
        coords=copy(coords),
        boundary=boundary,
        velocities=copy(velocities),
        pairwise_inters=(LennardJones(use_neighbors=true),),
        neighbor_finder=DistanceNeighborFinder(
            eligible=trues(n_atoms, n_atoms),
            n_steps=10,
            dist_cutoff=2.0u"nm",
        ),
    )
    return make_system
end

@testset "Girsanov functionality / unit compatibility" begin

    n_steps = 200
    temp = 300.0u"K"
    make_system = lennard_jones_system_maker(; n_atoms=100, temp=temp)

    biasing_forces = [ConstantDrift(SVector(1.0, 0.0, 0.0) .* u"kJ * mol^-1 * nm^-1")]

    @testset "OverdampedLangevinReweighting with units" begin
        sys = make_system()
        simulator = OverdampedLangevin(; dt=0.002u"ps", temperature=temp,
            friction=1.0u"ps^-1", remove_CM_motion=false)
        rw = TrajectoryReweighting(sys, simulator; force_perturbations=biasing_forces)

        simulate!(sys, simulator, n_steps; trajectory_reweighting=rw,
            rng=Xoshiro(2026))

        @test length(rw.log_weights) == n_steps
        @test eltype(rw.log_weights) <: Real
        @test all(isfinite, rw.log_weights)
    end

    @testset "ABOBA LangevinSplittingReweighting with units" begin
        sys = make_system()
        simulator = LangevinSplitting(dt=0.002u"ps", temperature=temp,
            friction=10.0u"g * mol^-1 * ps^-1",
            splitting="ABOBA", remove_CM_motion=false)
        rw = TrajectoryReweighting(sys, simulator; force_perturbations=biasing_forces)

        simulate!(sys, simulator, n_steps; trajectory_reweighting=rw,
            rng=MersenneTwister(2026))

        @test length(rw.log_weights) == n_steps
        @test eltype(rw.log_weights) <: Real
        @test all(isfinite, rw.log_weights)
    end

end


@testset "Girsanov validation 1D" begin

    @testset "OverdampedLangevinReweighting validation" begin
        n_samps = 10000
        n_steps = 300
        tol = 0.1

        rng = Xoshiro(2026)

        h0, h1 = 0.7, 1.2 # original vs target stiffness parameters
        acf(h, t, temp) = exp(-h * t) * (temp / h) # analytical autocovariance (from closed-from of the Ornstein-Uhlenbeck process)
        std_acf(h, t, temp) = sqrt((1 + exp(-2 * h * t)) * (temp / h)^2) # standard deviation of E[X_0 X_t] where $X_0$ is stationary (from closed-form of the OU process)

        acfs = Vector{Float64}[]
        rw_acfs = Vector{Float64}[]

        dt = 0.011
        temp = 1.32

        biasing_forces = [HarmonicPotential(h1 - h0)]

        for i = 1:n_samps

            sys = harmonic_oscillator(h0; temp=temp, rng=rng)
            simulator = OverdampedLangevin(; dt=dt, temperature=temp, friction=1.0, remove_CM_motion=false)
            rw_girsanov = TrajectoryReweighting(sys, simulator; force_perturbations=biasing_forces)

            simulate!(sys, simulator, n_steps; trajectory_reweighting=rw_girsanov)

            weights = exp.(rw_girsanov.log_weights)
            xtrace = values(sys.loggers.coords)
            ic_likelihood_ratio = sqrt(h1 / h0) * exp(-(h1 - h0) * first(xtrace)^2 / (2temp)) # Boltzmann factor for initial condition (explicit normalization here)

            push!(acfs, first(xtrace) * xtrace[2:n_steps+1])
            push!(rw_acfs, ic_likelihood_ratio * first(xtrace) * xtrace[2:n_steps+1] .* weights)

        end

        acf_hat_h1 = mean(rw_acfs)

        times = dt * (1:n_steps)
        acf_ana_h1 = acf.(h1, times, temp)

        @test all(i -> abs(acf_hat_h1[i] - acf_ana_h1[i]) < tol, 1:n_steps)
        println(maximum(abs, acf_hat_h1 - acf_ana_h1))

    end

    @testset "Validation of Girsanov reweighting (LangevinSplittingReweighting)" begin

        for splitting in ["ABOBA"]

            n_samps = 10000
            n_steps = 300
            tol = 0.1

            rng = Xoshiro(2026)

            h0, h1 = 0.7, 1.2 # original vs target stiffness parameters
            γ = 0.63

            """
                Closed-form velocity autocorrelation function for harmonic oscillator.
                (this can be derived by solving the underdamped Langevin equation, which is linear in this case).
            """
            function harmonic_vacf(γ, β, h)
                δ = γ^2 / 4 - h
                z = sqrt(complex(δ))
                λ = -γ / 2

                if 4h == γ^2 # degenerate case (critically damped)
                    return t -> exp(λ * t) * (1 + λ * t) / β
                else # non-degenerate case (overdamped if δ>0, underdamped if δ<0)
                    λ1, λ2 = λ - z, λ + z
                    return t -> real((exp(λ1 * t) * (γ + λ2) - exp(λ2 * t) * (γ + λ1)) / (2z * β))
                end
            end

            vacfs = Vector{Float64}[]
            rw_vacfs = Vector{Float64}[]

            dt = 0.011
            temp = 1.32

            biasing_forces = [HarmonicPotential(h1 - h0)]

            for i = 1:n_samps

                sys = harmonic_oscillator(h0; temp=temp, rng=rng)
                simulator = LangevinSplitting(dt=dt, temperature=temp, friction=γ, splitting="ABOBA", remove_CM_motion=false)
                rw_girsanov = TrajectoryReweighting(sys, simulator; force_perturbations=biasing_forces)
                simulate!(sys, simulator, n_steps; trajectory_reweighting=rw_girsanov)

                weights = exp.(rw_girsanov.log_weights)
                xtrace = values(sys.loggers.coords)
                vtrace = values(sys.loggers.velocities)
                ic_likelihood_ratio = sqrt(h1 / h0) * exp(-(h1 - h0) * first(xtrace)^2 / (2temp)) # Boltzmann factor for initial condition (explicit normalization here)

                push!(vacfs, first(vtrace) * vtrace[2:n_steps+1])
                push!(rw_vacfs, ic_likelihood_ratio * first(vtrace) * vtrace[2:n_steps+1] .* weights)

            end

            vacf_hat_h1 = mean(rw_vacfs)

            times = dt * (1:n_steps)
            f1 = harmonic_vacf(γ, 1 / temp, h1)
            vacf_ana_h1 = f1.(times)

            @test all(i -> abs(vacf_hat_h1[i] - vacf_ana_h1[i]) < tol, 1:n_steps)
            println(maximum(abs, vacf_hat_h1 - vacf_ana_h1))


        end

    end

end

@testset "TrajectoryReweighting unitless equivalence" begin

    harmonic_feature(sys) = reshape(-sys.coords, 1, 1)

    h0, h1 = 0.5, 0.9
    dt, temp, n_steps = 0.01, 1.1, 150
    biasing_forces = [HarmonicPotential(h1 - h0)]
    Θ = reshape([h1 - h0], 1, 1)

    sim_od = OverdampedLangevin(; dt=dt, temperature=temp, friction=1.0, remove_CM_motion=false)
    sim_ul = LangevinSplitting(; splitting="ABOBA", dt=dt, temperature=temp, friction=1.0, remove_CM_motion=false)

    rng = Xoshiro(2026)
    sys = harmonic_oscillator(h0; temp=temp, rng=rng)
    rw_single_od = TrajectoryReweighting(sys, sim_od; force_perturbations=biasing_forces)
    simulate!(sys, sim_od, n_steps; trajectory_reweighting=rw_single_od, rng=rng)
    rw_single_ul = TrajectoryReweighting(sys, sim_ul; force_perturbations=biasing_forces)
    simulate!(sys, sim_ul, n_steps; trajectory_reweighting=rw_single_ul, rng=rng)

    rng = Xoshiro(2026)
    sys = harmonic_oscillator(h0; temp=temp, rng=rng)
    rw_multiple_od = TrajectoryReweighting(sys, sim_od; feature_basis=harmonic_feature, linear_parameters=Θ)
    simulate!(sys, sim_od, n_steps; trajectory_reweighting=rw_multiple_od, rng=rng)
    rw_multiple_ul = TrajectoryReweighting(sys, sim_ul; feature_basis=harmonic_feature, linear_parameters=Θ)
    simulate!(sys, sim_ul, n_steps; trajectory_reweighting=rw_multiple_ul, rng=rng)

    @test maximum(abs, rw_single_od.log_weights - reduce(vcat, rw_multiple_od.log_weights)) < 1e-10
    @test maximum(abs, rw_single_ul.log_weights - reduce(vcat, rw_multiple_ul.log_weights)) < 1e-10
    @test rw_single_od.log_weights === rw_single_od.scheme.log_weights

    @testset "invalid keyword combinations throw" begin
        sys = harmonic_oscillator(h0; temp=temp, rng=Xoshiro(1))
        @test_throws ArgumentError TrajectoryReweighting(sys, sim_od)
        @test_throws ArgumentError TrajectoryReweighting(sys, sim_od; feature_basis=harmonic_feature)
        @test_throws ArgumentError TrajectoryReweighting(sys, sim_od; linear_parameters=Θ)
        @test_throws ArgumentError TrajectoryReweighting(sys, sim_od;
            force_perturbations=biasing_forces, feature_basis=harmonic_feature)
    end
end

@testset "TrajectoryReweighting unit compatibility" begin

    n_steps = 100
    temp = 300.0u"K"
    make_system = lennard_jones_system_maker(; n_atoms=50, temp=temp)

    drift = 1.0u"kJ * mol^-1 * nm^-1"
    biasing_forces = [ConstantDrift(SVector(1.0, 0.0, 0.0) .* drift)]

    # equivalent linear force feature
    function drift_feature(sys)
        D = [SVector(0.0, 0.0, 0.0) for _ in 1:length(sys)]
        D[1] = SVector(1.0, 0.0, 0.0)
        return reshape(D, length(sys), 1)
    end
    Θ = reshape([drift], 1, 1)

    sims = (OverdampedLangevin(; dt=0.002u"ps", temperature=temp, friction=1.0u"ps^-1",
            remove_CM_motion=false),
        LangevinSplitting(; dt=0.002u"ps", temperature=temp,
            friction=10.0u"g * mol^-1 * ps^-1", splitting="ABOBA", remove_CM_motion=false))

    perturbations = (("force_perturbations", (; force_perturbations=biasing_forces)),
        ("feature_basis", (; feature_basis=drift_feature, linear_parameters=Θ)))

    @testset "$mode / $(nameof(typeof(sim)))" for (mode, kw) in perturbations, sim in sims
        sys = make_system()
        rw = TrajectoryReweighting(sys, sim; kw...)
        simulate!(sys, sim, n_steps; trajectory_reweighting=rw, rng=Xoshiro(2026))
        w = reduce(vcat, rw.log_weights)
        @test length(w) == n_steps
        @test eltype(w) <: Real
        @test all(isfinite, w)
    end
end
