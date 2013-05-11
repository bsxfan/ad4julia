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
(/){T<:FloatScalar}(B::Matrix,C::Cholesky{T}) = convert(Matrix{T},B)/C
(/){T<:FloatScalar}(B::Vector,C::Cholesky{T}) = convert(Vector{T},B)/C

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

