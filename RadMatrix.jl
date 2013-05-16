module RadMatrix

importall Base
export RadMat,backprop,radeval


# this could be specialized or generalized 
typealias BaseScalar Number
typealias BaseVector{T<:BaseScalar} Array{T,1}
typealias BaseMatrix{T<:BaseScalar} Array{T,2}
typealias BaseArray{T<:BaseScalar} Union(BaseVector{T},BaseMatrix{T})
typealias BaseNum{T<:BaseScalar} Union(T,BaseArray)

abstract RadNum{T:<BaseScalar}

type RadScalar{T} <: RadNum{T}
    st:: T 
    gr:: T
    rcount:: Int
    wcount:: Int
    bp:: Function
    RadScalar(X::T,bp::Function) = new(X,zero(X),0,0,bp) 
end
RadScalar{T,N}(X::BaseMatrix{T},bp::Function) = RadScalar{T,N}(X,bp)
RadScalar(X) = RadScalar(X,(G)->1)


type RadVec{T} <: RadNum{T}
    st:: BaseVector{T} 
    gr:: BaseVector{T}
    rcount:: Int
    wcount:: Int
    bp:: Function
    RadMat(X::BaseVector{T},bp::Function) = new(X,zero(X),0,0,bp) 
end
RadVec{T,N}(X::BaseMatrix{T},bp::Function) = RadVec{T,N}(X,bp)
RadVec(X) = RadVec(X,(G)->1)


type RadMat{T,N} <: RadNum{T}
    st:: BaseMatrix{T}
    gr:: BaseMatrix{T}
    rcount:: Int
    wcount:: Int
    bp:: Function
    RadMat(X::BaseMatrix{T},bp::Function) = new(X,zero(X),0,0,bp) 
end
RadMat{T,N}(X::BaseMatrix{T},bp::Function) = RadMat{T,N}(X,bp)
RadMat(X) = RadMat(X,(G)->1)

typealias RadOrNot{T} Union(RadNum{T},BaseNum{T})
israd(X::RadOrNot) = isa(X,RadNum)

rd(R::RadNum) = (R.rcount += 1; R.st)
rd(X::BaseNum) = X

# Accumulates gradient, then backpropagates to all inputs.
# Returns number of inputs for which backprop is complete.
function backprop(R::BaseNum,G) = 0
function backprop(R::RadNum,G)
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


#################### unary operator library ##############################
(.')(X::RadMat) = RadMat(rd(X).',G -> backprop(X,G.')) 
(-)(X::RadMat) = RadMat(-rd(X),G -> backprop(X,-G)) 

#################### binary operator library ##############################
(+)(X::RadOrNot, Y::RadOrNot) = RadMat(rd(X) + rd(Y), G -> backprop(X,G) + backprop(Y,G) ) 
(-)(X::RadOrNot, Y::RadOrNot) = RadMat(rd(X) - rd(Y), G -> backprop(X,G) + backprop(Y,-G) ) 


end # module

