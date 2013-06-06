module RadMatrix

using CustomMatrix

importall Base
export RadNum,  RadNode, bpLeaf, #types
       radnum,backprop,radeval,compare_jacobians          #functions 

import Base.LinAlg: BLAS, LAPACK, BlasFloat, LU

#default factorize
factorize(X::Matrix) = lufact(X)
function At_ldiv_B{T<:BlasFloat}(A::LU{T}, B::StridedVecOrMat{T})
    if A.info > 0; throw(SingularException(A.info)); end
    LAPACK.getrs!('T', A.factors, A.ipiv, copy(B))
end
function At_ldiv_Bt{T<:BlasFloat}(A::LU{T}, B::StridedVecOrMat{T})
    if A.info > 0; throw(SingularException(A.info)); end
    LAPACK.getrs!('T', A.factors, A.ipiv, transpose(B))
end
function A_ldiv_Bt{T<:BlasFloat}(A::LU{T}, B::StridedVecOrMat{T})
    if A.info > 0; throw(SingularException(A.info)); end
    LAPACK.getrs!('N', A.factors, A.ipiv, transpose(B))
end


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
for fun in {:size,:ndims,:endof,:length,:eltype,:start,:isscalar}
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
rd(R::RadNum) = (R.nd.rcount += 1; (R.st, R.nd,true) )
rd(X) = X,nothing,false #default for convenience, simplifies code below


# Accumulates gradient, then backpropagates to all inputs.
# Returns number of inputs for which backprop is complete.
backprop(::Nothing,G) = 0 #for convenience, simplifies code below
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
	y,nd = rd(f(args...))
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
unpackX = :((Xs,Xn) = rd(X))
@eval begin
    transpose(X::RadNum) = ( $unpackX;
        radnum(Xs.',G -> backprop(Xn, G.')) )

    (-)(X::RadNum) = ( $unpackX;
        radnum(-Xs,G -> backprop(Xn, -G)) ) 

    (+)(X::RadNum) = ( $unpackX;
        radnum(+Xs,G -> backprop(Xn, +G)) )
end

#################### binary operator library ##############################

unpackXY = :( (Xs,Xn,radX) = rd(X); (Ys,Yn,radY) = rd(Y); both = radX && radY )
for (L,R) in { (:RadNum,:RadNum), (:RadNum,:Any), (:Any,:RadNum) }
	@eval begin

        (+)(X::$L, Y::$R) = ( $unpackXY;
            radnum(Xs + Ys, G -> backprop(Xn,G) + backprop(Yn,G) ) 
           ) 
    
        (-)(X::$L, Y::$R) = ( $unpackXY;
            radnum(Xs - Ys, G -> backprop(Xn,G) + backprop(Yn,-G) ) 
           )
        
        (.*)(X::$L, Y::$R) = ( $unpackXY;
            if     both back = G -> backprop(Xn,G.*Ys) + backprop(Yn,G.*Xs)
            elseif radX back = G -> backprop(Xn,G.*Ys)
            elseif radY back = G ->                      backprop(Yn,G.*Xs) end;
            radnum(Xs .* Ys, back) 
            )
        

        (*)(X::$L, Y::$R) = if ndims(X)==0 || ndims(Y)==0 return X .* Y else
            $unpackXY
            if     both back = G -> backprop(Xn,G*Ys.') + backprop(Yn,Xs.'*G)
            elseif radX back = G -> backprop(Xn,G*Ys.') 
            elseif radY back = G ->                       backprop(Yn,Xs.'*G) end
        	radnum(Xs * Ys, back ) 
        end

        (\)(X::$L, Y::$R) = if ndims(X)==0 || ndims(Y)==0 return Y ./ X else
            $unpackXY
            FX = factorize(Xs)
            Z = FX \ Ys
            thin = size(Ys,1) > size(Ys,2) # For square X: Z,Y,G all have the same size
            radnum(Z, G -> backprop(Xn, thin?(FX.' \ -G) * Z.' : FX.' \ (-G * Z.')) #could use rankone for vector RHS
                         + backprop(Yn, FX.' \ G) ) # FX.'\ G is common
        end

        (/)(X::$L, Y::$R) = (Y.'\X.').' #' can be made more efficient


        #At_mul_B
        #A_mul_Bt
        #At_mul_Bt
    end
end




#################### matrix function library ##########################################
@eval begin
    trace(X::RadNum) = ( unpackX;
         radnum(trace(Xs),G->backprop(Xn,diagm(G*ones(size(Xs,1))))) 
        )
end



end # module

