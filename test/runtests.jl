using RigidBodyDynamicsDiff
using RigidBodyDynamics
using ForwardDiff
using DiffResults
using Random
using Test

using RigidBodyDynamics: velocity_to_configuration_derivative_jacobian
using RigidBodyDynamics: configuration_derivative_to_velocity_jacobian


@show Threads.nthreads()

struct MyTag end

@testset "mass_matrix_differential!" begin
    Random.seed!(1)
    urdf = joinpath(dirname(pathof(RigidBodyDynamics)), "..", "test", "urdf", "atlas.urdf")
    mechanism = parse_urdf(urdf, root_joint_type=QuaternionFloating{Float64}())
    state = MechanismState(mechanism)
    result = DynamicsResult(mechanism)
    rand!(state)

    nq = num_positions(state)
    nv = num_velocities(state)
    q = configuration(state)
    normalize = Ref(false)

    mass_matrix_vec! = function (Mvec, q::AbstractVector{T}) where T
        set_configuration!(fdstate, q)
        if normalize[]
            normalize_configuration!(fdstate)
        end
        mass_matrix!(fdresult, fdstate)
        copyto!(Mvec, fdresult.massmatrix)
    end

    Mvec = zeros(nv * nv)
    fdjacresult = DiffResults.JacobianResult(Mvec, q)
    config = ForwardDiff.JacobianConfig(mass_matrix_vec!, Mvec, q, ForwardDiff.Chunk(1))
    fdstate = MechanismState{eltype(config)}(mechanism)
    fdresult = DynamicsResult{eltype(config)}(mechanism)

    cache = DifferentialCache{MyTag}(state)
    dc = Matrix{Float64}(undef, nv^2, nv)
    mass_matrix_differential!(dc, cache)

    # Without normalization:
    normalize[] = false
    ForwardDiff.jacobian!(fdjacresult, mass_matrix_vec!, Mvec, q, config)
    fdMjac = DiffResults.jacobian(fdjacresult)
    @test dc ≈ fdMjac * velocity_to_configuration_derivative_jacobian(state) atol = 1e-10

    # Without normalization, the normal component in dM is zero (since we're only working
    # in the tangent plane), but the ForwardDiff version *does* have a normal component,
    # so the following is not true:
    #   @test dM * configuration_derivative_to_velocity_jacobian(state) ≈ fdMjac atol = 1e-10
    # But with normalization:
    normalize[] = true
    ForwardDiff.jacobian!(fdjacresult, mass_matrix_vec!, Mvec, q, config)
    fdMjac = DiffResults.jacobian(fdjacresult)
    @test dc * configuration_derivative_to_velocity_jacobian(state) ≈ fdMjac atol = 1e-10
    @test dc ≈ fdMjac * velocity_to_configuration_derivative_jacobian(state) atol = 1e-10
end

@testset "dynamics_bias_differential!" begin
    Random.seed!(1)
    urdf = joinpath(dirname(pathof(RigidBodyDynamics)), "..", "test", "urdf", "atlas.urdf")
    mechanism = parse_urdf(urdf)#, root_joint_type=QuaternionFloating{Float64}())
    state = MechanismState(mechanism)
    result = DynamicsResult(mechanism)
    rand!(state)
    zero_velocity!(state)
    velocity(state)[end] = 1.0

    nq = num_positions(state)
    nv = num_velocities(state)
    q = configuration(state)
    v = velocity(state)
    normalize = Ref(false)

    dynamics_bias_vec! = function (cvec, q::AbstractVector{T}) where T
        set_velocity!(fdstate, v)
        set_configuration!(fdstate, q)
        if normalize[]
            normalize_configuration!(fdstate)
        end
        dynamics_bias!(fdresult, fdstate)
        copyto!(cvec, fdresult.dynamicsbias)
    end

    cvec = zeros(nv)
    fdjacresult = DiffResults.JacobianResult(cvec, q)
    config = ForwardDiff.JacobianConfig(dynamics_bias_vec!, cvec, q, ForwardDiff.Chunk(1))
    fdstate = MechanismState{eltype(config)}(mechanism)
    fdresult = DynamicsResult{eltype(config)}(mechanism)

    cache = DifferentialCache{MyTag}(state)
    dc = Matrix{Float64}(undef, nv, nv)
    dc .= NaN
    dynamics_bias_differential!(dc, cache)

    # Without normalization:
    normalize[] = false
    ForwardDiff.jacobian!(fdjacresult, dynamics_bias_vec!, cvec, q, config)
    fdcjac = DiffResults.jacobian(fdjacresult)
    @test dc ≈ fdcjac * velocity_to_configuration_derivative_jacobian(state) atol = 1e-10

    # Without normalization, the normal component in dc is zero (since we're only working
    # in the tangent plane), but the ForwardDiff version *does* have a normal component,
    # so the following is not true:
    #   @test dc * configuration_derivative_to_velocity_jacobian(state) ≈ fdcjac atol = 1e-10
    # But with normalization:
    normalize[] = true
    ForwardDiff.jacobian!(fdjacresult, dynamics_bias_vec!, cvec, q, config)
    fdcjac = DiffResults.jacobian(fdjacresult)
    @test dc * configuration_derivative_to_velocity_jacobian(state) ≈ fdcjac atol = 1e-10
    @test dc ≈ fdcjac * velocity_to_configuration_derivative_jacobian(state) atol = 1e-10
end

@testset "benchmarks" begin
    include(joinpath(@__DIR__, "..", "perf", "runbenchmarks.jl"))
end
