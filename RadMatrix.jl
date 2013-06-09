module RadMatrix

using CustomMatrix, GenUtils

importall Base
export RadNum,  RadNode, bpLeaf, #types
       radnum,backprop,radeval,
       compare_jacobians,rad_jacobians,complexstep_jacobians          
       
import Base.LinAlg: BLAS, LAPACK, BlasFloat, LU

include("radmatrix/lux.jl") #extend capabilities of LU factorization

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
vec(X::RadNum) = reshape(X,length(X))

unpackX = :((Xs,Xn) = rd(X))
@eval begin
    reshape(X::RadNum,ii...) = ($unpackX; sz = size(Xs); radnum(
    	reshape(Xs,ii...),
    	G->backprop(Xn,reshape(G,sz))                          )
    ) 


    #default, may be wasteful for large X
    getindex(X::RadNum,ii...) = ($unpackX; sz = size(Xs); radnum(
        getindex(Xs,ii...),
        G->(B=zero(Xs);setindex!(B,G,ii...);backprop(Xn,B))     )
    )

    # getindex(X::RadNum,i::Int) = ($unpackX; k = length(Xs); radnum(
    #     Xs[i],
    #     G->backprop(Xn,onevec(k,i,G))                           )
    # )

    fill(X::RadNum,ii...) = ($unpackX;
        radnum(fill(Xs,ii...), G-> backprop(Xn,sum(G)))
        )


end


function setindex!(D::RadNum,S::RadNum,ii...) # no new node is made, we modify the existing one
    Ss, Sn = rd(S)
    setindex!(D.st,Ss,ii...)
    oldbp = D.nd.bp
    D.nd.bp = G -> backprop(Sn,getindex(G,ii...)) + oldbp(setindex!(G,0,ii...))
    return D
end

function setindex!(D::RadNum,S,ii...) # no new node is made, we modify the existing one
    setindex!(D.st,S,ii...)
    oldbp = D.nd.bp
    D.nd.bp = G -> oldbp(setindex!(G,0,ii...))
    return D
end    


#setindex!(D,S::RadNum,ii...) = error("cannot write $(typeof(S)) into $(typeof(D))")




#################### unary operator library ##############################
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
    
        (.+)(X::$L, Y::$R) = ( $unpackXY;
            radnum(Xs .+ Ys, G -> backprop(Xn,G) + backprop(Yn,G) ) 
           ) 

        (-)(X::$L, Y::$R) = ( $unpackXY;
            radnum(Xs - Ys, G -> backprop(Xn,G) + backprop(Yn,-G) ) 
           )

        (.-)(X::$L, Y::$R) = ( $unpackXY;
            radnum(Xs .- Ys, G -> backprop(Xn,G) + backprop(Yn,-G) ) 
           )
        
        (.*)(X::$L, Y::$R) = ( $unpackXY;
            if     both back = G -> backprop(Xn,G.*Ys) + backprop(Yn,G.*Xs)
            elseif radX back = G -> backprop(Xn,G.*Ys)
            elseif radY back = G ->                      backprop(Yn,G.*Xs) end;
            radnum(Xs .* Ys, back) 
            )
        
        (./)(X::$L, Y::$R) = ( $unpackXY;
            Z = Xs ./ Ys;
            if     both back = G -> backprop(Xn,G./Ys) + backprop(Yn,-G.*Z./Ys)
            elseif radX back = G -> backprop(Xn,G./Ys)
            elseif radY back = G ->                      backprop(Yn,-G.*Z./Ys) end;
            radnum(Z, back) 
            )

        (.\)(X::$L, Y::$R) = Y ./ X

        (*)(X::$L, Y::$R) = if ndims(X)==0 || ndims(Y)==0 return X .* Y else
            $unpackXY
            if     both back = G -> backprop(Xn,G*Ys.') + backprop(Yn,Xs.'*G)
            elseif radX back = G -> backprop(Xn,G*Ys.') 
            elseif radY back = G ->                       backprop(Yn,Xs.'*G) end
        	radnum(Xs * Ys, back ) 
        end

        A_mul_Bt(X::$L, Y::$R) = if ndims(X)==0 || ndims(Y)==0 return X .* Y else
            $unpackXY
            if     both back = G -> backprop(Xn,G*Ys) + backprop(Yn,G.'*Xs)
            elseif radX back = G -> backprop(Xn,G*Ys) 
            elseif radY back = G ->                       backprop(Yn,G.'*Xs) end
            radnum(Xs * Ys.', back ) 
        end

        At_mul_B(X::$L, Y::$R) = if ndims(X)==0 || ndims(Y)==0 return X .* Y else
            $unpackXY
            if     both back = G -> backprop(Xn,Ys*G.') + backprop(Yn,Xs*G)
            elseif radX back = G -> backprop(Xn,Ys*G.') 
            elseif radY back = G ->                       backprop(Yn,Xs*G) end
            radnum(Xs.' * Ys, back ) 
        end

        At_mul_Bt(X::$L, Y::$R) = if ndims(X)==0 || ndims(Y)==0 return X .* Y else
            $unpackXY
            if     both back = G -> backprop(Xn,Ys.'*G.') + backprop(Yn,G.'*Xs.')
            elseif radX back = G -> backprop(Xn,Ys.'*G.') 
            elseif radY back = G ->                       backprop(Yn,Ys.'*G.') end
            radnum(Xs.' * Ys.', back ) 
        end


        (\)(X::$L, Y::$R) = if ndims(X)==0 || ndims(Y)==0 return Y ./ X else
            $unpackXY
            FX = factorize(Xs)
            Z = FX \ Ys
            thin = size(Ys,1) > size(Ys,2) # For square X: Z,Y,G all have the same size
            radnum(Z, G -> backprop(Xn, thin?(FX.' \ -G) * Z.' : FX.' \ (-G * Z.')) #could use rankone for vector RHS
                         + backprop(Yn, FX.' \ G) ) # FX.'\ G is common
        end

        (/)(X::$L, Y::$R) = (Y.'\X.').' #'  can be made more efficient


        dot(X::$L, Y::$R) = ( $unpackXY;
            if     both back = G -> backprop(Xn,G*Ys) + backprop(Yn,Xs*G)
            elseif radX back = G -> backprop(Xn,G*Ys) 
            elseif radY back = G ->                       backprop(Yn,Xs*G) end;
            radnum(dot(Xs,Ys), back ) 
           ) 



        #At_mul_B
        #A_mul_Bt
        #At_mul_Bt


        hcat(X::$L, Y::$R) = ( $unpackXY;
            Z = [Xs Ys];
            n = size(Xs,2); k = size(Ys,2);
            radnum(Z, G -> backprop(Xn,G[:,1:n]) + backprop(Yn,G[:,n+1:n+k]) ) 
            ) 

        vcat(X::$L, Y::$R) = ( $unpackXY;
            Z = [Xs;Ys];
            m = size(Xs,1); k = size(Ys,1);
            radnum(Z, G -> backprop(Xn,G[1:m,:]) + backprop(Yn,G[m+1:m+k,:]) ) 
            ) 

    end
end




#################### matrix function library ##########################################
@eval begin
    trace(X::RadNum) = ( $unpackX;
         #radnum(trace(Xs),G->backprop(Xn,diagm(G*ones(size(Xs,1))))) 
         (m,n) = size(Xs); @assert m==n;
         radnum( trace(Xs),G->backprop(Xn,diagmat(m,G)) ) 
        )
    
    logdet(X::RadNum) = ( $unpackX;
        FX = factorize(Xs);
         radnum(logdet(FX),G->backprop(Xn,G*inv(FX).') ) 
        )

    det(X::RadNum) = ( $unpackX;
        FX = factorize(Xs); d = det(FX);
         radnum(d,G->backprop(Xn,d*G*inv(FX).') ) 
        )
    
    inv(X::RadNum) = ( $unpackX;
        FX = factorize(Xs);
        Z = inv(FX);
         radnum(Z, G->backprop(Xn,-(FX.' \ (G * Z.')) ) ) 
        )

    sum(X::RadNum) = ($unpackX;
        s = sum(Xs); (m,n) = size(Xs);
         radnum(s, G->backprop(Xn, repel(m,n,G)) ) 
        )

    sum(X::RadNum,i::Int) = if i<1 || i>2 error("sum(::RadNum,$i) not implemented") else
        $unpackX;
        s = sum(Xs,i); (m,n) = size(Xs)
        if     i==1 back = G->backprop(Xn, reprow(m,G))
        elseif i==2 back = G->backprop(Xn, repcol(G,n)) end
        radnum(s, back ) 
    end


    diag(X::RadNum) = ($unpackX;
         radnum(diag(Xs), G->backprop(Xn, diagmat(G)) ) 
        )

    diagm(X::RadNum) = ($unpackX;
         radnum(diagm(Xs), G->backprop(Xn, diag(G)) ) 
        )



end

#################### scalar function library ##########################################
for (F,dFdX) in {
    ( :exp,        :Y                    ),
    ( :log,        :(1./X)               ),
    ( :log1p,      :(1./(X+1))           ),
    ( :sin,        :(cos(X))             ),
    ( :cos,        :(-sin(X))            ),
    ( :tanh,       :(1-Y.*Y)              ),
    }
    @eval begin
        $F(R::RadNum) = ( (X,Xn) = rd(R);
            Y = $F(X);
            radnum( Y,G->backprop(Xn,G.*$dFdX) ) 
            )
    end
end


include("radmatrix/customlib.jl")


end # module

