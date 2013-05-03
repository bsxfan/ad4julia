zero_differential_part!{T<:FloatMatrix}(x::DualNum{T}) = (fill!(x.di,0);x)
one_differential_part!{T<:FloatMatrix}(x::DualNum{T},ii...) = (fill!(x.di,0);x.di[ii...]=1;x)



function compareDualAndComplex(f,flags,args...)

  n = length(args)
  @assert n==length(flags)
  A = [args...]
  for i=1:n
    
  end  


end

compareDualAndComplex(f,args...) = compareDualAndComplex(f,trues(length(args)),args...)
