######## Matrix Function Library #######################
sum(x::DualNum,ii...) = dualnum(sum(x.st,ii...),sum(x.di,ii...))
trace(x::DualNum,ii...) = dualnum(trace(x.st,ii...),trace(x.di,ii...))
diag{T<:FloatArray}(x::DualNum{T},k...) = dualnum(diag(x.st,k...),diag(x.di,k...))
diagm{T<:FloatArray}(x::DualNum{T},k...) = dualnum(diagm(x.st,k...),diagm(x.di,k...))

diagmm{X<:FloatMatrix,Y<:FloatVector}(x::DualNum{X},y::DualNum{Y}) = 
    dualnum(diagmm(x.st,y.st),diagmm(x.di,y.st)+diagmm(x.st,y.di))
diagmm{X<:FloatMatrix,Y<:FloatVector}(x::DualNum{X},y::Y) = 
    dualnum(diagmm(x.st,y),diagmm(x.di,y))
diagmm{X<:FloatMatrix,Y<:FloatVector}(x::X,y::DualNum{Y}) = 
    dualnum(diagmm(x,y.st),diagmm(x,y.di))

diagmm{X<:FloatVector,Y<:FloatMatrix}(x::DualNum{X},y::DualNum{Y}) = 
    dualnum(diagmm(x.st,y.st),diagmm(x.di,y.st)+diagmm(x.st,y.di))
diagmm{X<:FloatVector,Y<:FloatMatrix}(x::DualNum{X},y::Y) = 
    dualnum(diagmm(x.st,y),diagmm(x.di,y))
diagmm{X<:FloatVector,Y<:FloatMatrix}(x::X,y::DualNum{Y}) = 
    dualnum(diagmm(x,y.st),diagmm(x,y.di))

inv{T<:FloatMatrix}(x::DualNum{T}) = (y=inv(x.st);dualnum(y,-y*x.di*y))
det{T<:FloatMatrix}(x::DualNum{T}) = (LU=lufact(x.st);y=det(LU);dualnum(y,y*dot(vec(inv(LU)),vec(x.di.'))))

#chol
# function logdet{T}(C::Cholesky{T})
    # dd = zero(T)
    # for i in 1:size(C.UL,1) dd += log(C.UL[i,i]) end
    # 2*dd
# end

#lu
