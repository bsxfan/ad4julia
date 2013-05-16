module RadMatrix

importall Base
export RadMat,backprop,radeval

type RadMat{T,N} 
    st:: Array{T,N} 
    gr:: Array{T,N}
    rcount:: Int
    wcount:: Int
    bp:: Function
    RadMat(X::Array{T,N},bp::Function) = new(X,zero(X),0,0,bp) 
end
RadMat{T,N}(X::Array{T,N},bp::Function) = RadMat{T,N}(X,bp)
RadMat(X) = RadMat(X,(G)->1)

rd(R::RadMat) = (R.rcount += 1; R.st)

# Accumulates gradient, then backpropagates to all inputs.
# Returns number of inputs for which backprop is complete.
function backprop{T,N}(R::RadMat{T,N},G::Array{T,N})
	R.gr += G
	R.wcount +=1 
	if R.wcount < R.rcount
		return 0
	elseif R.wcount > R.rcount
	    error("more writes than reads")
	else
		return R.bp(R.gr)
    end
end

function radeval(f::Function,args,g,flags=trues(length(args)))
	@assert length(args) == length(flags)
	flags = bool(flags)
	n = length(args)
	args = ntuple(n,i->flags[i]?RadMat(args[i]):args[i])
	Z = f(args...)
	z = rd(Z)
    @assert n == backprop(Z,g)
    dargs = args[flags]
    m = length(dargs)
    return tuple(z,ntuple(m,i->dargs[i].gr)...) 
end


#################### operator library ##############################
function (+)(X::RadMat,Y::RadMat)
    Z = rd(X) + rd(Y)
    bp = (G)-> backprop(X,G) + backprop(Y,G) 
    return RadMat(Z,bp) 
end

function (+)(X::RadMat,Y::Array)
    Z = rd(X) + Y
    bp = (G)-> backprop(X,G) 
    return RadMat(Z,bp) 
end

function (+)(X::Array,Y::RadMat)
    Z = X + rd(Y)
    bp = (G)-> backprop(Y,G) 
    return RadMat(Z,bp) 
end

end # module

