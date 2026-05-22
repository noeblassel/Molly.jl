@testset "Girsanov functionality / unit compatibility" begin

    struct ConstantDrift{T}
        f::SVector{3,T}
    end

    function AtomsCalculators.forces!(fs,
        sys,
        inter::ConstantDrift
        ;
        kwargs...)
        fs[1] = fs[1] + inter.f
        return fs
    end

    n_atoms = 100
    n_steps = 200
    temp = 300.0u"K"
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
        ),)

    biasing_forces = [ConstantDrift(
        SVector(1.0, 0.0, 0.0) .* u"kJ * mol^-1 * nm^-1")]

    @testset "OverdampedLangevinReweighting with units" begin
        sys = make_system()
        simulator = OverdampedLangevin(; dt=0.002u"ps", temperature=temp,
            friction=1.0u"ps^-1", remove_CM_motion=false)
        rw = OverdampedLangevinReweighting(sys, simulator, biasing_forces)

        simulate!(sys, simulator, n_steps; trajectory_reweighting=rw,
            rng=MersenneTwister(2026))

        @test length(rw.log_weights) == n_steps
        @test eltype(rw.log_weights) <: Real
        @test all(isfinite, rw.log_weights)
    end

    @testset "ABOBA LangevinSplittingReweighting with units" begin
        sys = make_system()
        simulator = LangevinSplitting(dt=0.002u"ps", temperature=temp,
            friction=10.0u"g * mol^-1 * ps^-1",
            splitting="ABOBA", remove_CM_motion=false)
        rw = LangevinSplittingReweighting("ABOBA", sys, simulator, biasing_forces)

        simulate!(sys, simulator, n_steps; trajectory_reweighting=rw,
            rng=MersenneTwister(2026))

        @test length(rw.log_weights) == n_steps
        @test eltype(rw.log_weights) <: Real
        @test all(isfinite, rw.log_weights)
    end

end


@testset "Girsanov validation 1D" begin

    """
    A harmonic potential with stifness h
    V(x) = h|x|²/2
    """
    struct HarmonicPotential{T}
        h::T
    end

    function AtomsCalculators.forces!(fs,
        sys,
        inter::HarmonicPotential
        ;
        neighbors=nothing,
        kwargs...)
        fs .-= inter.h * sys.coords
    end

    """
    Constructs a unitless system consisting of a single 1D particle in a harmonic energy well, initialized in canonical equilibrium.
    The state is represented as a monoatomic 3D system, with two spurious dimensions.
    """
    function harmonic_oscillator(h; temp=1.0, rng=Random.default_rng())
        atoms = [Atom(mass=1.0)]
        coords = [sqrt(temp / h) * SVector{3}(randn(rng), 0.0, 0.0)]
        velocities = [sqrt(temp) * SVector{3}(randn(rng), 0.0, 0.0)]
        boundary = CubicBoundary(Inf)

        harm_pot = HarmonicPotential(h)

        one_d_coord_wrapper(s, args...; kwargs...) = first(sys.coords)[1]
        one_d_velocity_wrapper(s, args...; kwargs...) = first(sys.velocities)[1]

        loggers = (coords=GeneralObservableLogger(one_d_coord_wrapper, Float64, 1),
            velocities=GeneralObservableLogger(one_d_velocity_wrapper, Float64, 1))

        sys = System(
            atoms=atoms,
            coords=coords,
            boundary=boundary,
            velocities=velocities,
            general_inters=(harm_pot,),
            loggers=loggers,
            force_units=NoUnits,
            energy_units=NoUnits,
            k=1.0
        )

        return sys
    end

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
            rw_girsanov = OverdampedLangevinReweighting(sys, simulator, biasing_forces)

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
        println(maximum(abs,acf_hat_h1 - acf_ana_h1))

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
                rw_girsanov = LangevinSplittingReweighting("ABOBA", sys, simulator, biasing_forces)
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
                    println(maximum(abs,vacf_hat_h1 - vacf_ana_h1))


        end

    end

end

