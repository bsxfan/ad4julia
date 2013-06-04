module RadMatrix

using CustomMatrix

importall Base
export RadNum,  RadNode, bpLeaf, #types
       radnum,backprop,radeval          #functions 



# Node in backpropagation DAG. Edges go from output to inputs. Inputs are leaves.
type RadNode{B} # B annotates type of original variable---not used currently
    gr # accumulates gradient from all fanouts
    rcount:: Int  # fanout in fwd pass
    wcount:: Int  # fanin in backward pass
    bp:: Function #backpropagates to all parents
end

# Wrapper for numeric type B, which will cause backpropagation DAG to be 
# constructed during forward calculations.
immutable RadNum{B}
		    st:: B            #standard numeric part
		    nd:: RadNode{B}   #backprop node
            RadNum(X::B,bp::Function) = new(X,RadNode{B}(zero(X),0,0,bp)) 
end
bpLeaf(G)=1 # input nodes do not backpropate further
radnum{B}(X::B,bp::Function=bpLeaf) = RadNum{B}(X,bp)  

israd(X) = isa(X,RadNum)

# defer several functions to standard part
# derivatives play no role here
for fun in {:size,:ndims,:endof,:length,:eltype,:start}
    @eval begin
        ($fun)(R::RadNum) = ($fun)(R.st)
    end
end
for fun in {:size,:next,:done}
    @eval begin
        ($fun)(R::RadNum,args...) = ($fun)(R.st,args...)
    end
end


# reads value, counts references
rd(R::RadNum) = (R.nd.rcount += 1; R.st)
rd(X) = X #default for convenience, simplifies code below

# Accumulates gradient, then backpropagates to all inputs.
# Returns number of inputs for which backprop is complete.
backprop(R,G) = 0 #for convenience, simplifies code below
function backprop(R::RadNode,G)
	@assert R.wcount <= R.rcount
	if R.wcount == R.rcount ## reset to backprop another gradient
      R.gr = zero!(R.gr)
      R.wcount = 0
	end
	# R.gr += G, in-place if possible, sums or broadcasts G as needed to fit R.gr
	R.gr = procrustean_update!(R.gr,G) 
	R.wcount +=1 
	if R.wcount < R.rcount # wait for more 
		return 0
	else
		return R.bp(R.gr) # go deeper
    end
end
zero!(n::Number) = zero(n)
zero!{E}(X::Array{E}) = fill!(X,zero(E))

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
 	dargs = args[flags]
   	m = length(dargs)
	Z = f(args...)
	y = rd(Z)
	nd = Z.nd
	function do_backprop(g) # g---the gradient to backpropagate---must be of same size as y
		@assert nd.rcount==1
  		@assert m == backprop(nd,g) 
    	@assert nd.wcount==nd.rcount==1
    	return m==1?dargs[1].nd.gr:ntuple(m,i->dargs[i].nd.gr)
    end
    return y, do_backprop 
end


include("radmatrix/testrad.jl")



#################### matrix wiring #######################################
reshape(R::RadNum,ii...) = (sz = size(R); radnum(
	reshape(rd(R),ii...),
	G->backprop(R.nd,reshape(G,sz))                )
) 
vec(R::RadNum) = reshape(R,length(R))



#################### unary operator library ##############################
transpose(X::RadNum) = radnum(rd(X).',G -> backprop(X.nd, G.')) 
(-)(X::RadNum) = radnum(-rd(X),G -> backprop(X.nd, -G)) 
(+)(X::RadNum) = radnum(+rd(X),G -> backprop(X.nd, +G)) 


#################### binary operator library ##############################
for (L,R) in { (:RadNum,:RadNum), (:RadNum,:Any), (:Any,:RadNum) }
	@eval begin
        (+)(X::$L, Y::$R) = radnum(rd(X) + rd(Y), G -> backprop(X.nd,G) + backprop(Y.nd,G) ) 
        (-)(X::$L, Y::$R) = radnum(rd(X) - rd(Y), G -> backprop(X.nd,G) + backprop(Y.nd,-G) ) 
        
        function (.*)(X::$L, Y::$R) 
        	Xst = rd(X)
        	Yst = rd(Y)
        	radnum(Xst .* Yst, G -> backprop(X.nd,G.*Yst) + backprop(Y.nd,G.*Xst) ) 
        end
        
        function (*)(X::$L, Y::$R) 
        	Xst = rd(X)
        	Yst = rd(Y)
        	radnum(Xst * Yst, G -> backprop(X.nd,G*Yst.') + backprop(Y.nd,Xst.'*G) ) 
        end


        #At_mul_B
        #A_mul_Bt
        #At_mul_Bt
    end
end



#################### matrix function library ##########################################
trace(R::RadNum) = radnum(trace(rd(R)),G->backprop(R.nd,diagm(G*ones(size(R,1))))) 




end # module

