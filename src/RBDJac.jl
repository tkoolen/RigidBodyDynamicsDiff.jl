module RBDJac

export
    mass_matrix_jacobian!

using RigidBodyDynamics
using ForwardDiff
using LinearAlgebra
using StaticArrays

using ForwardDiff: Dual
using RigidBodyDynamics: successorid, isdirty, Spatial.colwise, Spatial.hat

# TODO:
# * tagged Duals
# * need mapping from q index / v index to JointID (stored as `Vector{JointID}` in `MechanismState`)
# * frame checks for timederiv methods

function timederiv(H::Transform3D, twist::Twist)
    @framecheck H.from twist.body
    @framecheck H.to twist.base
    @framecheck twist.frame twist.base
    R = rotation(H)
    p = translation(H)
    ω = angular(twist)
    v = linear(twist)
    top = [colwise(×, ω, R) ω × p + v]
    bottom = zeros(similar_type(top, Size(1, 4)))
    dH = [top; bottom]
    Transform3D(H.from, H.to, map(Dual, H.mat, dH))
end

function timederiv(J::GeometricJacobian, twist::Twist)
    # TODO: more frame checks needed?
    @framecheck J.frame twist.frame
    ω = angular(twist)
    v = linear(twist)
    Jω = angular(J)
    Jv = linear(J)
    dJω = colwise(×, ω, Jω)
    dJv = colwise(×, v, Jω) + colwise(×, ω, Jv)
    GeometricJacobian(J.body, J.base, J.frame, map(Dual, Jω, dJω), map(Dual, Jv, dJv))
end

function timederiv(inertia::SpatialInertia, twist::Twist)
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
    SpatialInertia(inertia.frame, map(Dual, Iω, dIω), map(Dual, mc, dmc), Dual(m, zero(m)))
end

function configuration_jacobian_init!(jacstate::MechanismState{D}, state::MechanismState{T}, l::Int,
        compute_transforms::Bool, compute_motion_subspaces::Bool, compute_inertias::Bool, compute_crb_inertias::Bool) where {D, T}
    @boundscheck begin
        jacstate.mechanism === state.mechanism || error()
        num_positions(state) === num_velocities(state) || error() # FIXME
        length(joints(state.mechanism)) === num_velocities(state) || error() # FIXME
        length(state.treejoints.data) === 1 || error() # FIXME
        mapreduce(has_fixed_subspaces, &, state.treejoints, init=true) || error()
    end

    setdirty!(jacstate)
    copyto!(jacstate.q, state.q)
    copyto!(jacstate.v, state.v)
    copyto!(jacstate.s, state.s)
    jacstate.q[l] = ForwardDiff.Dual(state.q[l], one(T))

    k = JointID(l) # FIXME
    j_l = Twist(state.motion_subspaces.data[l], SVector(one(T))) # j_l?
    if compute_transforms
        # RigidBodyDynamics.update_transforms!(state)
        dtransforms_to_root = jacstate.transforms_to_root
        @inbounds for i in state.treejointids
            κ_i = state.ancestor_joint_masks[i]
            bodyid = successorid(i, state)
            H = transform_to_root(state, bodyid)
            if κ_i[k]
                dtransforms_to_root[bodyid] = timederiv(H, j_l)
            else
                dtransforms_to_root[bodyid] = H
            end
        end
        jacstate.transforms_to_root.dirty = false
    end

    if compute_motion_subspaces
        # RigidBodyDynamics.update_motion_subspaces!(state)
        @inbounds for i in state.treejointids
            κ_i = state.ancestor_joint_masks[i]
            bodyid = successorid(i, state)
            J_i = state.motion_subspaces.data[i.value] # FIXME: should be a v index
            if κ_i[k]
                jacstate.motion_subspaces.data[i] = timederiv(J_i, j_l)
            else
                jacstate.motion_subspaces.data[i] = J_i
            end
        end
        jacstate.motion_subspaces.dirty = false
    end

    if compute_inertias
        # RigidBodyDynamics.update_spatial_inertias!(state)
        dinertias = jacstate.inertias
        @inbounds for i in state.treejointids
            κ_i = state.ancestor_joint_masks[i]
            bodyid = successorid(i, state)
            I_i = state.inertias[bodyid]
            if κ_i[k]
                dinertias[bodyid] = timederiv(I_i, j_l)
            else
                dinertias[bodyid] = I_i
            end
        end
        jacstate.inertias.dirty = false
    end

    if compute_crb_inertias
        # RigidBodyDynamics.update_crb_inertias!(state)
        dcrbinertias = jacstate.crb_inertias
        κ_k = state.ancestor_joint_masks[k]
        @inbounds for i in state.treejointids
            κ_i = state.ancestor_joint_masks[i]
            bodyid = successorid(i, state)
            Ic_i = state.crb_inertias[bodyid]
            if κ_i[k] || κ_k[i]
                p = max(i, k)
                Ic_p = state.crb_inertias[successorid(p, state)]
                Ic_p_dual = timederiv(Ic_p, j_l)
                Dual = ForwardDiff.Dual
                Iω = map(Dual, Ic_i.moment, map(ForwardDiff.partials, Ic_p_dual.moment))
                mc = map(Dual, Ic_i.cross_part, map(ForwardDiff.partials, Ic_p_dual.cross_part))
                m = Dual(Ic_i.mass, ForwardDiff.partials(Ic_p_dual.mass))
                dcrbinertias[bodyid] = SpatialInertia(Ic_i.frame, Iω, mc, m)
            else
                dcrbinertias[bodyid] = Ic_i
            end
        end
        jacstate.crb_inertias.dirty = false
    end

    jacstate
end

function mass_matrix_jacobian!(Mjac::Matrix, state::MechanismState, jacstates::Vector{<:MechanismState}, jacresults::Vector{<:DynamicsResult})
    nq = num_positions(state)
    nv = num_velocities(state)
    RigidBodyDynamics.update_motion_subspaces!(state)
    RigidBodyDynamics.update_crb_inertias!(state)
    if !(Threads.nthreads() === length(jacstates) === length(jacresults))
        error("The lengths of the jacstates and jacresults vectors must be equal to the number of threads.")
    end
    @boundscheck size(Mjac) === (nv^2, nq) || throw(DimensionMismatch())
    let Mjac = Mjac, state = state, jacstates = jacstates, jacresults = jacresults, nq = nq
        Threads.@threads for qindex = 1 : nq
            id = Threads.threadid()
            @inbounds jacstate = jacstates[id]
            @inbounds jacresult = jacresults[id]
            configuration_jacobian_init!(jacstate, state, qindex, false, true, false, true)
            mass_matrix!(jacresult, jacstate)
            M_dual = jacresult.massmatrix
            jacindex = (qindex - 1) * size(Mjac, 1)
            @inbounds for i in eachindex(M_dual)
                jacindex += 1
                Mjac[jacindex] = ForwardDiff.partials(M_dual[i])[1]
            end
        end
    end
end

end # module
