export factorize, logdet2



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

    (/){T}(B::Matrix{T},A::LU{T}) = At_ldiv_Bt(A,B).'

    function logdet2{T<:Real}(A::LU{T})  # return log(abs(det)) and sign(det)
        m, n = size(A); if m!=n error("matrix must be square") end
        dg = diag(A.factors)
        s = (bool(sum(A.ipiv .!= 1:n) % 2) ? -one(T) : one(T)) * prod(sign(dg))
        return sum(log(abs(dg))) , s 
    end

    function logdet{T<:Real}(A::LU{T})
        d,s = logdet2(A)
        if s<0 error("determinant is negative") end
        return d
    end

    function logdet{T<:Complex}(A::LU{T})
        m, n = size(A); if m!=n error("matrix must be square") end
        s = sum(log(diag(A.factors))) + (bool(sum(A.ipiv .!= 1:n) % 2) ? complex(0,pi) : 0) 
        r,a = reim(s); a = a % 2pi; if a>pi a -=2pi elseif a<=-pi a+=2pi end
        return complex(r,a)    
    end

    logdet{T<:BlasFloat}(A::Matrix{T}) = logdet(factorize(A))
    logdet2{T<:BlasFloat}(A::Matrix{T}) = logdet2(factorize(A))