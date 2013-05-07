######## Matrix Function Library #######################
sum(x::DualNum,ii...) = dualnum(sum(x.st,ii...),sum(x.di,ii...))
trace(x::DualNum,ii...) = dualnum(trace(x.st,ii...),trace(x.di,ii...))
diag{T<:FloatArray}(x::DualNum{T},k...) = dualnum(diag(x.st,k...),diag(x.di,k...))
diagm{T<:FloatArray}(x::DualNum{T},k...) = dualnum(diagm(x.st,k...),diagm(x.di,k...))



scale{X<:FloatMatrix,Y<:Union(FloatVector,FloatScalar)}(x::DualNum{X},y::DualNum{Y}) = 
    dualnum(scale(x.st,y.st),scale(x.di,y.st)+scale(x.st,y.di))
scale{X<:FloatMatrix,Y<:Union(Vector,Number)}(x::DualNum{X},y::Y) = 
    dualnum(scale(x.st,y),scale(x.di,y))
scale{X<:Array,Y<:FloatScalar}(x::X,y::DualNum{Y}) = 
    dualnum(scale(x,y.st),scale(x,y.di))
scale{X<:Matrix,Y<:FloatVector}(x::X,y::DualNum{Y}) = 
    dualnum(scale(x,y.st),scale(x,y.di))

scale{X<:Union(FloatVector,FloatScalar),Y<:FloatMatrix}(x::DualNum{X},y::DualNum{Y}) = 
    dualnum(scale(x.st,y.st),scale(x.di,y.st)+scale(x.st,y.di))
scale{X<:Union(Vector,Number),Y<:FloatMatrix}(x::DualNum{X},y::Y) = 
    dualnum(scale(x.st,y),scale(x.di,y))
scale{X<:FloatScalar,Y<:Array}(x::DualNum{X},y::Y) = 
    dualnum(scale(x.st,y),scale(x.di,y))
scale{X<:FloatVector,Y<:Matrix}(x::DualNum{X},y::Y) = 
    dualnum(scale(x.st,y),scale(x.di,y))

	
inv{T<:FloatMatrix}(x::DualNum{T}) = (y=inv(x.st);dualnum(y,-y*x.di*y))
det{T<:FloatMatrix}(x::DualNum{T}) = (LU=lufact(x.st);y=det(LU);dualnum(y,y*dot(vec(inv(LU)),vec(x.di.'))))




#chol
# function logdet{T}(C::Cholesky{T})
    # dd = zero(T)
    # for i in 1:size(C.UL,1) dd += log(C.UL[i,i]) end
    # 2*dd
# end

#lu
