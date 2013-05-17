module RadMatrix

using Procrustes

importall Base
export RadNum,RadScalar,RadVec,RadMat,  #types
       radnum,backprop,radeval                 #functions 


# this could be specialized or generalized 
typealias BaseScalar Number
#typealias BaseVec{T<:BaseScalar} Array{T,1}
typealias BaseMat{T<:BaseScalar} Array{T}
#typealias BaseArray{T<:BaseScalar} Union(BaseVec{T},BaseMat{T})
#typealias BaseNum{T<:BaseScalar} Union(T,BaseArray)
typealias BaseNum{T<:BaseScalar} Union(T,BaseMat)

abstract RadNum

# declare RadScalar, RadVec and RadMat
bpInputNode(G)=1 # input nodes do not backpropate further
for (BaseType,RadType) in ( (:BaseScalar,:RadScalar), (:BaseVec, :RadVec), (:BaseMat, :RadMat) )
	@eval begin
		type $RadType <: RadNum
		    st:: $BaseType
		    gr:: $BaseType
		    rcount:: Int
		    wcount:: Int
		    bp:: Function #backpropagates to all parents, returns number of inputs for which backprop completed
		    $RadType(X::$BaseType,bp::Function) = new(X,zero(X),0,0,bp) #constructor
		end
        # conversions from each BaseNum flavour to corresponding RadNum flavour
        radnum(X::$BaseType,bp::Function=bpInputNode) = $RadType(X,bp)  
	end
end

typealias RadOrNot Union(RadNum,BaseNum)
israd(X::RadOrNot) = isa(X,RadNum)


ndims(R::RadNum) = ndims(R.st) 
isscalar(R::RadOrNot) = ndims(R)==0
size(R::RadNum,ii...) = size(R.st,ii...) 
endof(R::RadNum) = endof(R.st)
length(R::RadNum) = length(R.st)




# reads value, counts references
rd(R::RadNum) = (R.rcount += 1; R.st)
rd(X::BaseNum) = X #for convenience, simplifies code below

# Accumulates gradient, then backpropagates to all inputs.
# Returns number of inputs for which backprop is complete.
backprop(R::BaseNum,G) = 0 #for convenience, simplifies code below
function backprop(R::RadNum,G)
	#R.gr += G
	R.gr = procrustean_add!(R.gr,G)
	R.wcount +=1 
	if R.wcount < R.rcount
		return 0
	elseif R.wcount > R.rcount
	    error("more writes than reads")
	else
		return R.bp(R.gr)
    end
end

#Evaluate y=f(args...) and differentiates w.r.t. each flagged argument
# returns y and a function to backpropagate gradients
function radeval(f::Function,
	             args,
	             flags=trues(length(args))
	            )
	@assert length(args) == length(flags)
	flags = bool(flags)
	n = length(args)
	args = ntuple(n,i->flags[i]?radnum(args[i]):args[i])
	Z = f(args...)
	y = rd(Z)
	function do_backprop(g::BaseNum) # g---the gradient to backpropagate---must be of same size as y
		@assert Z.rcount==1
    	if Z.wcount==0
    		@assert n == backprop(Z,g) 
    	else
    	    error("this function cannot be called more than once")
    	end
    	@assert Z.wcount==Z.rcount==1
    	dargs = args[flags]
    	m = length(dargs)
    	return ntuple(m,i->dargs[i].gr)
    end
    return y, do_backprop 
end




#################### unary operator library ##############################
(.')(X::RadNum) = radnum(rd(X).',G -> backprop(X,G.')) 
(-)(X::RadNum) = radnum(-rd(X),G -> backprop(X,-G)) 


#################### binary operator library ##############################
for (L,R) in ( (:RadNum,:BaseNum), (:BaseNum,:RadNum), (:RadNum,:RadNum) )
	@eval begin
        (+)(X::$L, Y::$R) = radnum(rd(X) + rd(Y), G -> backprop(X,G) + backprop(Y,G) ) 
        (-)(X::$L, Y::$R) = radnum(rd(X) - rd(Y), G -> backprop(X,G) + backprop(Y,-G) ) 
    end
end


end # module

