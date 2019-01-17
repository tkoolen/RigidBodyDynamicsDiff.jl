using RigidBodyDynamicsDiff
using RigidBodyDynamics
using BenchmarkTools
using ForwardDiff
using DiffResults
using Random
using Profile

struct MyTag end

include(joinpath(@__DIR__, "..", "test", "forwarddiff_compatible.jl"))

function create_benchmark_suite()
    # T = Float32
    T = Float64
    suite = BenchmarkGroup()
    urdf = joinpath(dirname(pathof(RigidBodyDynamics)), "..", "test", "urdf", "atlas.urdf")
    mechanism = parse_urdf(urdf, scalar_type=T)
    state = MechanismState{T}(mechanism)
    rand!(state)

    q = configuration(state)
    v = velocity(state)
    v̇ = similar(velocity(state))
    nv = num_velocities(mechanism)

    statecache = StateCache(mechanism)
    resultcache = DynamicsResultCache(mechanism)
    diffcache = DifferentialCache{MyTag}(state)

    # Mass matrix
    let
        Mvec = zeros(nv * nv)
        mass_matrix_fd! = forwarddiff_compatible(mass_matrix!, statecache, resultcache)
        suite["mass_matrix_fd!"] = @benchmarkable begin
            $mass_matrix_fd!($Mvec, $q)
        end evals=10

        fdjacresult = DiffResults.JacobianResult(Mvec, q)
        config = ForwardDiff.JacobianConfig(mass_matrix_fd!, Mvec, q, ForwardDiff.Chunk(3)) # Chunk(4) and higher results in inference issues
        suite["mass_matrix! ForwardDiff"] = @benchmarkable begin
            ForwardDiff.jacobian!($fdjacresult, $mass_matrix_fd!, $Mvec, $q, $config)
        end evals=10

        dM = zeros(T, nv * nv, nv)
        suite["mass_matrix!"] = @benchmarkable begin
            setdirty!($state)
            mass_matrix_differential!($dM, $diffcache)
        end evals = 10
    end

    # Dynamics bias
    let
        dynamics_bias_fd! = forwarddiff_compatible(dynamics_bias!, statecache, resultcache, v)
        c = zeros(nv)
        suite["dynamics_bias_fd!"] = @benchmarkable begin
            $dynamics_bias_fd!($c, $q)
        end evals=10

        fdjacresult = DiffResults.JacobianResult(c, q)
        config = ForwardDiff.JacobianConfig(dynamics_bias_fd!, c, q, ForwardDiff.Chunk(3)) # Chunk(4) and higher results in inference issues
        suite["dynamics_bias! ForwardDiff"] = @benchmarkable begin
            ForwardDiff.jacobian!($fdjacresult, $dynamics_bias_fd!, $c, $q, $config)
        end evals=10

        dc = zeros(T, nv, nv)
        suite["dynamics_bias!"] = @benchmarkable begin
            setdirty!($state)
            dynamics_bias_differential!($dc, $diffcache)
        end evals = 10
    end

    # Inverse dynamics
    let
        inverse_dynamics_vec! = forwarddiff_compatible(inverse_dynamics!, statecache, resultcache, v, v̇)
        τ = similar(velocity(state))
        suite["inverse_dynamics_vec!"] = @benchmarkable begin
            $inverse_dynamics_vec!($τ, $q)
        end evals=10

        fdjacresult = DiffResults.JacobianResult(τ, q)
        config = ForwardDiff.JacobianConfig(inverse_dynamics_vec!, τ, q, ForwardDiff.Chunk(3)) # Chunk(4) and higher results in inference issues
        suite["inverse_dynamics! ForwardDiff"] = @benchmarkable begin
            ForwardDiff.jacobian!($fdjacresult, $inverse_dynamics_vec!, $τ, $q, $config)
        end evals=10

        dτ = zeros(T, nv, nv)
        suite["inverse_dynamics!"] = @benchmarkable begin
            setdirty!($state)
            inverse_dynamics_differential!($dτ, $v̇, $diffcache)
        end
    end

    return suite
end

function runbenchmarks()
    @show Threads.nthreads()
    suite = create_benchmark_suite()
    Profile.clear_malloc_data()
    overhead = BenchmarkTools.estimate_overhead()
    Random.seed!(1)
    results = run(suite, verbose=true, overhead=overhead, gctrial=false)
    for result in results
        println("$(first(result)):")
        display(last(result))
        println()
    end
end

runbenchmarks()
