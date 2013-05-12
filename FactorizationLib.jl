############################# Cholesky  #####################################


# first expand the applicability of Cholesky to more types and operators
#(remember FloatScalar == Linalg.BlasFloat)
(\){T<:FloatReal,S<:RealScalar}(C::Cholesky{T},B::Vector{S}) = C\convert(Vector{T},B)
(\){T<:FloatReal,S<:RealScalar}(C::Cholesky{T},B::Matrix{S}) = C\convert(Matrix{T},B)

function (/){T<:FloatScalar}(B::StridedVecOrMat{T},C::Cholesky{T}) 
   if size(B,1)==1 
    return (C\B.').' #'
  else
    return B*inv(C)
  end
end
(/){T<:FloatScalar,S<:RealScalar}(B::Matrix{S},C::Cholesky{T}) = convert(Matrix{T},B)/C
(/){T<:FloatScalar,S<:RealScalar}(B::Vector{S},C::Cholesky{T}) = convert(Vector{T},B)/C

immutable DualCholesky{T<:FloatScalar}
  st::Cholesky{T} 
  di::Matrix{T}
end

function cholfact{T<:FloatScalar}(X::DualNum{Matrix{T}})
    return DualCholesky(cholfact(X.st),X.di)
end
function cholfact{T<:FloatScalar}(X::DualNum{T})
    return DualCholesky(cholfact([X.st]),[X.di])
end


/(a::DualNum,b::DualCholesky) = (y=a.st/b.st;dualnum(y, (a.di - y*b.di)/b.st))
/(a::Numeric,b::DualCholesky) = (y=a/b.st;dualnum(y, - y*b.di/b.st))
/(a::DualNum,b::Cholesky) = (y=a.st/b;dualnum(y, a.di/b))

\(a::DualCholesky,b::DualNum) = (y=a.st\b.st;dualnum(y, a.st\(b.di - a.di*y)))
\(a::DualCholesky,b::Numeric) = (y=a.st\b;dualnum(y, -(a.st\a.di*y)))
\(a::Cholesky,b::DualNum) = (y=a\b.st;dualnum(y, a\b.di))


inv(x::DualCholesky) = (y=inv(x.st);dualnum(y,-y*x.di*y))

det(x::DualCholesky) = (y=det(x.st);dualnum(y,y*trace(x.st\x.di)))
logdet(x::DualCholesky) = (dualnum(logdet(x.st),trace(x.st\x.di)))


############################# LU  ##############################


#temporary fix for problem with LU \
# general signature is: function (\)(A::LU, B::StridedVecOrMat)
function (\){T<:FloatScalar}(A::LU{T}, B::StridedVecOrMat{T}) 
    A.info>0 && throw(SingularException(A.info))
    LinAlg.LAPACK.getrs!('N', A.factors, A.ipiv, copy(B))
end

# let's define / also
function (/){T<:FloatScalar}(A::LU{T}, B::StridedVecOrMat{T}) 
    A.info>0 && throw(SingularException(A.info))
    size(B,1)>1 && return B*inv(C)
    (LinAlg.LAPACK.getrs!('T', A.factors, A.ipiv, B.')).'  #'
end


#now expand to more types 
(\){T<:FloatReal,S<:RealScalar}(C::LU{T},B::Vector{S}) = C\convert(Vector{T},B)
(\){T<:FloatReal,S<:RealScalar}(C::LU{T},B::Matrix{S}) = C\convert(Matrix{T},B)
(\){T<:FloatComplex,S<:ComplexScalar}(C::LU{T},B::Vector{S}) = C\convert(Vector{T},B)
(\){T<:FloatComplex,S<:ComplexScalar}(C::LU{T},B::Matrix{S}) = C\convert(Matrix{T},B)
(\){T<:FloatComplex,S<:RealScalar}(C::LU{T},B::Vector{S}) = C\convert(Vector{T},B)
(\){T<:FloatComplex,S<:RealScalar}(C::LU{T},B::Matrix{S}) = C\convert(Matrix{T},B)

(/){S<:RealScalar,T<:FloatReal}(B::Vector{S},C::LU{T}) = convert(Vector{T},B)/C
(/){S<:RealScalar,T<:FloatReal}(B::Matrix{S},C::LU{T},) = convert(Matrix{T},B)/C
(/){S<:ComplexScalar,T<:FloatComplex}(B::Vector{S},C::LU{T}) = convert(Vector{T},B)/C
(/){S<:ComplexScalar,T<:FloatComplex}(B::Matrix{S},C::LU{T}) = convert(Matrix{T},B)/C
(/){S<:RealScalar,T<:FloatComplex}(B::Vector{S},C::LU{T}) = convert(Vector{T},B)/C
(/){S<:RealScalar,T<:FloatComplex}(B::Matrix{S},C::LU{T},) = convert(Matrix{T},B)/C

immutable DualLU{T<:FloatScalar}
  st::LU{T} 
  di::Matrix{T}
end

function lufact{T<:FloatScalar}(X::DualNum{Matrix{T}})
    return DualLU(lufact(X.st),X.di)
end
function lufact{T<:FloatScalar}(X::DualNum{T})
    return DualLU(lufact([X.st]),[X.di])
end


/(a::DualNum,b::DualLU) = (y=a.st/b.st;dualnum(y, (a.di - y*b.di)/b.st))
/(a::Numeric,b::DualLU) = (y=a/b.st;dualnum(y, - y*b.di/b.st))
/(a::DualNum,b::LU) = (y=a.st/b;dualnum(y, a.di/b))

\(a::DualLU,b::DualNum) = (y=a.st\b.st;dualnum(y, a.st\(b.di - a.di*y)))
\(a::DualLU,b::Numeric) = (y=a.st\b;dualnum(y, -(a.st\a.di*y)))
\(a::LU,b::DualNum) = (y=a\b.st;dualnum(y, a\b.di))


inv(x::DualLU) = (y=inv(x.st);dualnum(y,-y*x.di*y))

det(x::DualLU) = (y=det(x.st);dualnum(y,y*trace(x.st\x.di)))



