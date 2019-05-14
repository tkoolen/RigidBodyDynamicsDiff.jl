module RigidBodyDynamicsDiff

export
    DifferentialCache,
    mass_matrix_differential!,
    dynamics_bias_differential!,
    inverse_dynamics_differential!,
    dynamics_differential!

using RigidBodyDynamics
using ForwardDiff
using LinearAlgebra
using StaticArrays

using RigidBodyDynamics: predsucc, successorid, isdirty, Spatial.colwise, Spatial.hat, Spatial.se3_commutator
using RigidBodyDynamics: update_transforms!, update_bias_accelerations_wrt_world!, update_twists_wrt_world!
using RigidBodyDynamics: update_motion_subspaces!, update_spatial_inertias!, update_crb_inertias!
using RigidBodyDynamics: configuration_index_to_joint_id, velocity_index_to_joint_id, supports
using RigidBodyDynamics: velocity_to_configuration_derivative_jacobian, velocity_to_configuration_derivative_jacobian!
using RigidBodyDynamics.CustomCollections: SegmentedBlockDiagonalMatrix
using ForwardDiff: Dual

# TODO:
# * more frame checks for timederiv methods

function timederiv(H::Transform3D, twist::Twist, ::Type{Tag}) where Tag
    # @framecheck H.from twist.body
    # @framecheck H.to twist.base
    # @framecheck twist.frame twist.base
    R = rotation(H)
    p = translation(H)
    ω = angular(twist)
    v = linear(twist)
    dR = colwise(×, ω, R)
    dp = ω × p + v
    T = promote_type(eltype(dR), eltype(dp))
    dH = hcat(
        vcat(dR, zeros(similar_type(dR, Size(1, 3)))),
        vcat(dp, zeros(similar_type(dR, Size(1, 1))))
    )
    Transform3D(H.from, H.to,
        map(Dual{Tag}, H.mat, dH))
end

function timederiv(J::GeometricJacobian, twist::Twist, ::Type{Tag}) where Tag
    # TODO: more frame checks needed?
    @framecheck J.frame twist.frame
    ω = angular(twist)
    v = linear(twist)
    Jω = angular(J)
    Jv = linear(J)
    dJω = colwise(×, ω, Jω)
    dJv = colwise(×, v, Jω) + colwise(×, ω, Jv)
    GeometricJacobian(J.body, J.base, J.frame,
        map(Dual{Tag}, Jω, dJω),
        map(Dual{Tag}, Jv, dJv))
end

function timederiv(twist1::Twist, twist2::Twist, ::Type{Tag}) where Tag
    @framecheck twist1.frame twist2.frame
    dangular, dlinear = se3_commutator(angular(twist2), linear(twist2), angular(twist1), linear(twist1))
    Twist(twist1.body, twist1.base, twist1.frame,
        map(Dual{Tag}, angular(twist1), dangular),
        map(Dual{Tag}, linear(twist1), dlinear))
end

function timederiv(accel::SpatialAcceleration, twist::Twist, ::Type{Tag}) where Tag
    @framecheck accel.frame twist.frame
    dangular, dlinear = se3_commutator(angular(accel), linear(accel), angular(twist), linear(twist))
    SpatialAcceleration(accel.body, accel.base, accel.frame,
        map(Dual{Tag}, angular(accel), dangular),
        map(Dual{Tag}, linear(accel), dlinear))
end

function timederiv(inertia::SpatialInertia, twist::Twist, ::Type{Tag}) where Tag
    # TODO: more frame checks needed?
    @framecheck inertia.frame twist.frame
    ω = angular(twist)
    v = linear(twist)
    Iω = inertia.moment
    mc = inertia.cross_part
    m = inertia.mass
    mv = m * v
    dmc = mv + ω × mc
    ωhat = hat(ω)
    vhat = hat(v)
    mchat = hat(mc)
    dIω = ωhat * Iω - Iω * ωhat - vhat * mchat - mchat * vhat
    SpatialInertia(inertia.frame,
        map(Dual{Tag}, Iω, dIω),
        map(Dual{Tag}, mc, dmc),
        Dual{Tag}(m, zero(m)))
end

struct DifferentialCache{Tag, T, DS<:MechanismState{Dual{Tag, T, 1}}, DR<:DynamicsResult{Dual{Tag, T, 1}}}
    dualstates::Vector{DS}
    dualresults::Vector{DR}
end

function DifferentialCache{Tag}(mechanism::Mechanism{T}) where {Tag, T}
    if !mapreduce(has_fixed_subspaces, &, tree_joints(mechanism), init=true)
        throw(ArgumentError("Can only handle Mechanisms with fixed motion subspaces."))
    end
    D = ForwardDiff.Dual{Tag, T, 1}
    n = Threads.nthreads()
    dualstates = Vector{typeof(MechanismState{D}(mechanism))}(undef, n)
    dualresults = Vector{typeof(DynamicsResult{D}(mechanism))}(undef, n)
    Threads.@threads for i = 1 : n
        # Create MechanismStates and DynamicsResults in separate threads to avoid the possibility of false sharing.
        id = Threads.threadid()
        dualstate = dualstates[id] = MechanismState{D}(mechanism)
        dualresults[id] = DynamicsResult{D}(mechanism)
        dualstate.q .= NaN
        dualstate.v .= NaN
        dualstate.s .= NaN
    end
    DifferentialCache(dualstates, dualresults)
end

function copy_differential_column!(dest::Matrix, src::AbstractVector{<:Dual}, destcol::Integer)
    deststart = (destcol - 1) * size(dest, 1)
    @inbounds @simd for row in eachindex(src)
        dest[deststart + row] = ForwardDiff.partials(src[row], 1)
    end
end

function copy_differential_column!(dest::Matrix, src::Symmetric{<:Dual}, destcol::Integer)
    deststart = (destcol - 1) * size(dest, 1)
    n = size(src, 1)
    upper = src.uplo === 'u'
    @inbounds for col in Base.OneTo(n)
        @simd for row in col : n
            u_index = (row - 1) * n + col
            l_index = (col - 1) * n + row
            data_index = ifelse(upper, u_index, l_index)
            val = ForwardDiff.partials(src.data[data_index], 1)
            dest[deststart + u_index] = val
            dest[deststart + l_index] = val
        end
    end
end

function dual_state_init!(dualstate::MechanismState{<:Dual{Tag}}, state::MechanismState{T}, v_index::Integer,
        transforms::Bool, motion_subspaces::Bool, inertias::Bool, crb_inertias::Bool,
        twists::Bool, bias_accelerations::Bool) where {Tag, T}
    dualstate.q .= NaN # TODO: actually compute this?
    copyto!(dualstate.v, state.v)
    copyto!(dualstate.s, state.s)
    setdirty!(dualstate)
    k = velocity_index_to_joint_id(state, v_index)
    twist = Twist(state.motion_subspaces.data[v_index], SVector(one(T)))

    # TODO:
    if bias_accelerations
        transforms = true
    end

    if transforms
        dtransforms_to_root = dualstate.transforms_to_root
        @inbounds for i in state.treejointids
            i_successor = successorid(i, state)
            H = transform_to_root(state, i_successor, false)
            if supports(k, i_successor, state)
                dtransforms_to_root[i_successor] = timederiv(H, twist, Tag)
            else
                dtransforms_to_root[i_successor] = H
            end
        end
        dualstate.transforms_to_root.dirty = false
    end

    if motion_subspaces
        dmotion_subspaces = dualstate.motion_subspaces.data
        @inbounds for i in state.treejointids
            i_successor = successorid(i, state)
            for v_index_i in velocity_range(state, i)
                S = state.motion_subspaces.data[v_index_i]
                if supports(k, i_successor, state)
                    dmotion_subspaces[v_index_i] = timederiv(S, twist, Tag)
                else
                    dmotion_subspaces[v_index_i] = S
                end
            end
        end
        dualstate.motion_subspaces.dirty = false
    end

    if inertias
        dinertias = dualstate.inertias
        @inbounds for i in state.treejointids
            i_successor = successorid(i, state)
            I_i = state.inertias[i_successor]
            if supports(k, i_successor, state)
                dinertias[i_successor] = timederiv(I_i, twist, Tag)
            else
                dinertias[i_successor] = I_i
            end
        end
        dualstate.inertias.dirty = false
    end

    if crb_inertias
        dcrb_inertias = dualstate.crb_inertias
        k_successor = successorid(k, state)
        @inbounds for i in state.treejointids
            i_successor = successorid(i, state)
            Ic_i = state.crb_inertias[i_successor]
            if supports(k, i_successor, state) || supports(i, k_successor, state)
                p = max(i, k)
                Ic_p = state.crb_inertias[successorid(p, state)]
                Ic_p_dual = timederiv(Ic_p, twist, Tag)
                Iω = map(Dual{Tag}, Ic_i.moment, map(ForwardDiff.partials, Ic_p_dual.moment))
                mc = map(Dual{Tag}, Ic_i.cross_part, map(ForwardDiff.partials, Ic_p_dual.cross_part))
                m = Dual{Tag}(Ic_i.mass, ForwardDiff.partials(Ic_p_dual.mass))
                dcrb_inertias[i_successor] = SpatialInertia(Ic_i.frame, Iω, mc, m)
            else
                dcrb_inertias[i_successor] = Ic_i
            end
        end
        dualstate.crb_inertias.dirty = false
    end

    if twists
        dtwists = dualstate.twists_wrt_world
        @inbounds for i in state.treejointids
            i_successor = successorid(i, state)
            twist_i = twist_wrt_world(state, i_successor, false)
            if supports(k, i_successor, state)
                relative_twist_dual = timederiv(relative_twist(state, i_successor, first(predsucc(k, state))), twist, Tag)
                ω = map(Dual{Tag}, twist_i.angular, map(ForwardDiff.partials, relative_twist_dual.angular))
                v = map(Dual{Tag}, twist_i.linear, map(ForwardDiff.partials, relative_twist_dual.linear))
                dtwists[i_successor] = Twist(twist_i.body, twist_i.base, twist_i.frame, ω, v)
            else
                dtwists[i_successor] = twist_i
            end
        end
        dualstate.twists_wrt_world.dirty = false
    end

    # if bias_accelerations
    #     dbias_accelerations = dualstate.bias_accelerations_wrt_world
    #     @inbounds for i in state.treejointids
    #         κ_i = state.ancestor_joint_masks[i]
    #         i_successor = successorid(i, state)
    #         accel_i = bias_acceleration(state, i_successor, false)
    #         if κ_i[k]
    #             dbias_accelerations[i_successor] = timederiv(accel_i, twist, Tag)
    #         else
    #             dbias_accelerations[i_successor] = accel_i
    #         end
    #     end
    #     dualstate.bias_accelerations_wrt_world.dirty = false
    # end

    nothing
end

function threaded_differential!(f::F, dest::Matrix, state::MechanismState, cache::DifferentialCache;
        transforms::Bool=false, motion_subspaces::Bool=false, inertias::Bool=false, crb_inertias::Bool=false, twists::Bool=false, bias_accelerations::Bool=false) where F
    let state = state
        transforms && update_transforms!(state)
        motion_subspaces && update_motion_subspaces!(state)
        inertias && update_spatial_inertias!(state)
        crb_inertias && update_crb_inertias!(state)
        twists && update_twists_wrt_world!(state)
        bias_accelerations && update_bias_accelerations_wrt_world!(state)
    end
    let dest = dest, cache = cache, state = state
        nv = num_velocities(state)
        Threads.@threads for v_index in Base.OneTo(nv)
            id = Threads.threadid()
            dualstate = cache.dualstates[id]
            dual_state_init!(dualstate, state, v_index, transforms, motion_subspaces,
                inertias, crb_inertias, twists, bias_accelerations)
            dualresult = cache.dualresults[id]
            copy_differential_column!(dest, f(dualresult, dualstate), v_index)
            nothing
        end
    end
    return dest
end

function mass_matrix_differential!(dest::Matrix, state::MechanismState, cache::DifferentialCache)
    nv = num_velocities(state)
    size(dest) == (nv^2, nv) || throw(DimensionMismatch())
    threaded_differential!(mass_matrix!, dest, state, cache;
        motion_subspaces=true, crb_inertias=true)
end

function dynamics_bias_differential!(dest::Matrix, state::MechanismState, cache::DifferentialCache)
    nv = num_velocities(state)
    size(dest) == (nv, nv) || throw(DimensionMismatch())
    threaded_differential!(dynamics_bias!, dest, state, cache;
        motion_subspaces=true, inertias=true, bias_accelerations=true, twists=true)
end

function inverse_dynamics_differential!(dest::Matrix, state::MechanismState, v̇::SegmentedVector{JointID}, cache::DifferentialCache)
    nv = num_velocities(state)
    size(dest) == (nv, nv) || throw(DimensionMismatch())
    f = let v̇ = v̇
        function (result, state)
            # TODO: abusing the dynamicsbias and totalwrenches fields of DynamicsResult.
            inverse_dynamics!(result.dynamicsbias, result.jointwrenches, result.accelerations, state, v̇, result.totalwrenches)
            result.dynamicsbias
        end
    end
    threaded_differential!(f, dest, state, cache;
        transforms=true, motion_subspaces=true, inertias=true, bias_accelerations=true, twists=true)
end

function dynamics_differential!(dest::Matrix, result::DynamicsResult, state::MechanismState, τ::SegmentedVector{JointID}, cache::DifferentialCache)
    nv = num_velocities(state)
    size(dest) == (nv, nv) || throw(DimensionMismatch())
    dynamics!(result, state, τ)
    inverse_dynamics_differential!(dest, state, result.v̇, cache)
    uplo = result.massmatrix.uplo
    # LAPACK.potrs!(result.massmatrix.uplo, result.L, dest)
    # @inbounds map!(-, dest, dest)
    LAPACK.potri!(uplo, result.L)
    @inbounds dest .= BLAS.symm('L', uplo, -1.0, result.L, dest)
    # BLAS.symm!('L', uplo, 'N', 'N', -1, result.L, dest)
    return dest
end

end # module
