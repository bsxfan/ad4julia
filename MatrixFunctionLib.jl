######## Matrix Function Library #######################

sum(x::DualNum,ii...) = dualnum(sum(x.st,ii...),sum(x.di,ii...))

trace(x::DualNum,ii...) = dualnum(trace(x.st,ii...),trace(x.di,ii...))

diag{T<:FloatArray}(x::DualNum{T},k...) = dualnum(diag(x.st,k...),diag(x.di,k...))

diagm{T<:FloatArray}(x::DualNum{T},k...) = dualnum(diagm(x.st,k...),diagm(x.di,k...))


#scale(Matrix,Scalar)
scale{X<:FloatMatrix,Y<:FloatScalar}(x::DualNum{X},y::DualNum{Y}) = 
    dualnum(scale(x.st,y.st),scale(x.di,y.st)+scale(x.st,y.di))
scale{X<:FloatMatrix,Y<:Number}(x::DualNum{X},y::Y) = 
    dualnum(scale(x.st,y),scale(x.di,y))
scale{X<:Array,Y<:FloatScalar}(x::X,y::DualNum{Y}) = 
    dualnum(scale(x,y.st),scale(x,y.di))

	
#scale(Matrix,Vector)
scale{X<:FloatMatrix,Y<:FloatVector}(x::DualNum{X},y::DualNum{Y}) = 
    dualnum(scale(x.st,y.st),scale(x.di,y.st)+scale(x.st,y.di))
scale{X<:FloatMatrix,Y<:Vector}(x::DualNum{X},y::Y) = 
    dualnum(scale(x.st,y),scale(x.di,y))
scale{X<:Matrix,Y<:FloatVector}(x::X,y::DualNum{Y}) = 
    dualnum(scale(x,y.st),scale(x,y.di))

#scale(Vector,Matrix)	
scale{X<:FloatVector,Y<:FloatMatrix}(x::DualNum{X},y::DualNum{Y}) = 
    dualnum(scale(x.st,y.st),scale(x.di,y.st)+scale(x.st,y.di))
scale{X<:Vector,Y<:FloatMatrix}(x::X,y::DualNum{Y}) = 
    dualnum(scale(x,y.st),scale(x,y.di))
scale{X<:FloatVector,Y<:Matrix}(x::DualNum{X},y::Y) = 
    dualnum(scale(x.st,y),scale(x.di,y))

	
inv{T<:FloatMatrix}(x::DualNum{T}) = (y=inv(x.st);dualnum(y,-y*x.di*y))
inv{T<:FloatScalar}(x::DualNum{T}) = (y=inv(x.st);dualnum(y,-y^2*x.di))

det{T<:FloatMatrix}(x::DualNum{T}) = (LU=lufact(x.st);y=det(LU);dualnum(y,y*dot(vec(inv(LU)),vec(x.di.'))))



#chol (remember FloatScalar == Linalg.BlasFloat)
# first expand the applicability of Cholesky to more types and operators
(\){T<:FloatScalar}(C::Cholesky{T},B::Vector) = C\convert(Vector{T},B)
(\){T<:FloatScalar}(C::Cholesky{T},B::Matrix) = C\convert(Matrix{T},B)

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


