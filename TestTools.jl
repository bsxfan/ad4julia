#some utilities
#zero_differential_part!{T<:FloatMatrix}(x::DualNum{T}) = (fill!(x.di,0);x)
#one_differential_part!{T<:FloatMatrix}(x::DualNum{T},ii...) = (fill!(x.di,0);x.di[ii...]=1;x)

export compareDualAndComplex,compRndDualAndComplex


printredln(line) = println("\e[91;1m$(line)\e[0m")
printerrln(comment,value) = value>1e-6?printredln("$(comment): $(value)"):println("$(comment): $(value)")

compareDualAndComplex(f::Function,args) = compareDualAndComplex(f,args,trues(length(args)))
function compareDualAndComplex(f::Function,args,flags)

  n = length(args)
  println("Testing differentials: $(sum(flags)) of $n arguments.")
  @assert n==length(flags)
  @assert all(arg->isa(arg,RealNum),args[flags])
  dc_args = deepcopy(args[flags])
  Y0 = f(args...)
  @assert dc_args == args[flags] # no side-effect allowed in differentiable arguments
  @assert isa(Y0,RealNum)  # tuple of Numerics would also be do-able
  println("  function value size: $(size(Y0))")
  err = 0
  
  A = {args...} # convert tuple to Array{Any}
  C = copy(A)
  j = 0
  for i=1:n           # i indexes all args
    if !flags[i]; 
	  continue; 
	else
	  j += 1          # j indexes differentiable args
	end
	println("  argument $i, $(size(A[i])):")
	if length(A[i])==1
	  C[i] = complex(A[i],cstepSz)
	  A[i] = dualnum(A[i],1)
	  Yd = f(A...)
	  @show C
	  @show f(C...)
	  Yc = complex2dual(f(C...))
	  @show Yc
	  verr1 = max(abs(Yd.st-Y0))
	  verr2 = max(abs(Yc.st-Y0))
	  derr = max(abs(Yd.di-Yc.di))
	  err = max(err,verr1,verr2,derr)
	else
	  verr1 = 0
	  verr2 = 0
	  derr = 0
	  C[i] = flt_complex(A[i])  # copies A[i] to new complex matrix
      A[i] = dualnum(A[i])  # doesn't copy 
      for k=1:length(A[i])
		A[i].di[k] = 1
        C[i][k] += cstepSz*im 		
		
		
		Yd = f(A...)
		Yc = complex2dual(f(C...))
		
		A[i].di[k] = 0 #restore this element
		C[i][k] = complex(real(C[i][k]),0.0) #restore this element

		verr1 = max(verr1,max(abs(Yd.st-Y0)))
		verr2 = max(verr2,max(abs(Yc.st-Y0)))
		derr = max(derr,max(abs(Yd.di-Yc.di)))
		err = max(err,verr1,verr2,derr)
	  end	  
	end
	printerrln("    max abs function value error in dual step",verr1)
	printerrln("    max abs function value error in complex step",verr2)
	printerrln("    max abs error between dual and complex differentials",derr)
    A[i] = dc_args[j] # restore to original -- no need to copy, no further modification	  
    C[i] = dc_args[j] # restore to original	-- no need to copy, no further modification	    
  end
  
  rerr = 0
  for i=1:10
    rerr = max(rerr,compRndDualAndComplex(f,args,flags,true))
  end
  printerrln("  all input differentials random, dual vs complex",rerr)
  return max(err,rerr)
  #compRndDualAndComplex(f,args,flags,true)
  #return err
end


compRndDualAndComplex(f::Function,args) = compRndDualAndComplex(f,args,trues(length(args)),false)
compRndDualAndComplex(f::Function,args,flags) = compRndDualAndComplex(f,args,flags,false)
function compRndDualAndComplex(f,args,flags,quiet)

  n = length(args)
  if !quiet; println("Testing differentials: $(sum(flags)) of $n arguments."); end
  @assert n==length(flags)
  @assert all(arg->isa(arg,RealNum),args[flags])
  dc_args = deepcopy(args[flags])
  Y0 = f(args...)
  @assert dc_args == args[flags] # no side-effect allowed in differentiable arguments
  @assert isa(Y0,RealNum)  # tuple of Numerics would also be do-able

  if !quiet; println("  function value size: $(size(Y0))"); end

  
  A = {args...} # convert tuple to Array{Any}
  C = copy(A);
  j = 0
  for i=1:n           # i indexes all args
    if !flags[i]; 
	  continue; 
	else
	  j += 1          # j indexes differentiable args
	end
	if length(A[i])==1
	  A[i] = dualnum(A[i],randn())
	else
      A[i] = dualnum(A[i],randn(size(A[i])))  
	end
	C[i] = dual2complex(A[i])
  end  
  Yd = f(A...)
  Yc = complex2dual(f(C...))
  verr1 = max(abs(Yd.st-Y0))
  verr2 = max(abs(Yc.st-Y0))
  derr = max(abs(Yd.di-Yc.di))
  if !quiet
    printerrln("  max abs function value error in dual step",verr1)
    printerrln("  max abs function value error in complex step",verr2)
    printerrln("  max abs error between dual and complex differentials",derr)
  end
  err = max(verr1,verr2,derr)
  return err

end
