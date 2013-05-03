#some utilities
zero_differential_part!{T<:FloatMatrix}(x::DualNum{T}) = (fill!(x.di,0);x)
one_differential_part!{T<:FloatMatrix}(x::DualNum{T},ii...) = (fill!(x.di,0);x.di[ii...]=1;x)



function compareDualAndComplex(f,flags,args...)

  n = length(args)
  @assert n==length(flags)
  @assert all(arg->isa(arg,Numeric),args[flags])
  dc_args = deepcopy(args[flags])
  Y0 = f(args...)
  @assert dc_args == args[flags] # no side-effect allowed in differentiable arguments
  @assert isa(Y0,Numeric)  # tuple of Numerics would also do-able
  
  C = A = {args...} # convert tuple to Array{Any}
  j = 0
  for i=1:n           # i indexes all args
    if !flags[i]; 
	  continue; 
	else
	  j += 1          # j indexes differentiable args
	end
	if length(A[i])==1
	  A[i] = dualnum(A[i],1)
	  C[i] = complex(A[i],1.0e-20)
	  Y = f(A...)
	else
	  C[i] = complex(A[i])  # copies A[i] to new complex matrix
      A[i] = dualnum(A[i])  # doesn't copy 
      for k=1:length(A[i])
        one_differential_part!(A[i],k)  
		Y = f(A...)
	  end	  
	end
    A[i] = dc_args[j] # restore to original	  
  end  


end

compareDualAndComplex(f,args...) = compareDualAndComplex(f,trues(length(args)),args...)
