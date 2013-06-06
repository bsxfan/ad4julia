#default factorize
factorize(X::Matrix) = lufact(X)

function At_ldiv_B{T<:BlasFloat}(A::LU{T}, B::StridedVecOrMat{T})
    if A.info > 0; throw(SingularException(A.info)); end
    LAPACK.getrs!('T', A.factors, A.ipiv, copy(B))
end

function At_ldiv_Bt{T<:BlasFloat}(A::LU{T}, B::StridedVecOrMat{T})
    if A.info > 0; throw(SingularException(A.info)); end
    LAPACK.getrs!('T', A.factors, A.ipiv, transpose(B))
end

function A_ldiv_Bt{T<:BlasFloat}(A::LU{T}, B::StridedVecOrMat{T})
    if A.info > 0; throw(SingularException(A.info)); end
    LAPACK.getrs!('N', A.factors, A.ipiv, transpose(B))
end

function logdet2{T<:FloatingPoint}(A::LU{T})  # return log(abs(det)) and sign(det)
    m, n = size(A)
    dg = diag(A.factors)
    s = (bool(sum(A.ipiv .!= 1:n) % 2) ? -one(T) : one(T)) * prod(sign(dg))
    return sum(log(abs(dg))) , s 
end

function logdet{T<:FloatingPoint}(A::LU{T})
    d,s = logdet2(A)
    if s<0 error("DomainError: determinant is negative") end
    return d
end

function logdet{T<:FloatingPoint}(A::LU{Complex{T}})
    m, n = size(A); if m!=n error("matrix must be square") end
    if A.info > 0; return zero(typeof(A.factors[1])); end
    s = sum(log(diag(A.factors))) 
    s += (s + (bool(sum(A.ipiv .!= 1:n) % 2) ? complex(0,pi) : 0) )% 2pi
    if s>pi s -= 2pi elseif s<= -pi s += 2pi end
    return s    
end
