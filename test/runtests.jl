using RBDJac
using RigidBodyDynamics
using ForwardDiff
using DiffResults
using Random
using Test

@testset "mass_matrix_jacobian!" begin
    Random.seed!(1)
    urdf = joinpath(dirname(pathof(RigidBodyDynamics)), "..", "test", "urdf", "atlas.urdf")
    mechanism = parse_urdf(urdf)
    state = MechanismState(mechanism)
    result = DynamicsResult(mechanism)
    rand!(state)

    nq = num_positions(state)
    nv = num_velocities(state)
    q = configuration(state)

    mass_matrix_vec! = function (Mvec, q::AbstractVector{T}) where T
        set_configuration!(fdstate, q)
        mass_matrix!(fdresult, fdstate)
        copyto!(Mvec, fdresult.massmatrix)
    end

    Mvec = zeros(nv * nv)
    fdjacresult = DiffResults.JacobianResult(Mvec, q)
    config = ForwardDiff.JacobianConfig(mass_matrix_vec!, Mvec, q, ForwardDiff.Chunk(1))
    fdstate = MechanismState{eltype(config)}(mechanism)
    fdresult = DynamicsResult{eltype(config)}(mechanism)
    ForwardDiff.jacobian!(fdjacresult, mass_matrix_vec!, Mvec, q, config)
    fdMjac = DiffResults.jacobian(fdjacresult);

    D = ForwardDiff.Dual{Nothing, Float64, 1}
    jacstates = [MechanismState{D}(mechanism) for _ = 1 : Threads.nthreads()] # TODO: false sharing
    jacresults = [DynamicsResult{D}(mechanism) for _ = 1 : Threads.nthreads()] # TODO: false sharing
    Mjac = similar(fdMjac)
    mass_matrix_jacobian!(Mjac, state, jacstates, jacresults)
    @test Mjac ≈ fdMjac atol = 1e-10
end
