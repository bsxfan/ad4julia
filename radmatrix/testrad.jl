function compare_jacobians(f::Function,args,flags=trues(length(args)) )

    Jr = rad_jacobians(f,args,flags)
    Jc = complexstep_jacobians(f,args,flags)
    if !isa(Jr,NTuple)
      Jr = (Jr,) 
      Jc = (Jc,)
    end 
    K = length(Jr)
    return mapreduce(k->max(abs(Jr[k]-Jc[k])),max,1:K)

end


rad_jacobians(f::Function,args,flags=trues(length(args)) ) = 
    reversemode_jacobians( radeval(f,args,flags)... )


function complexstep_jacobians(f::Function,args,flags=trues(length(args)) )
    @assert length(args) == length(flags)
    flags = bool(flags)
    cargs = {args...}
    step = 1e-20
    function fwd(DX...)
        k = 1
        for i=1:length(args)
            if flags[i]
                cargs[i] = args[i]+(im*step)*DX[k]
                k +=1
            end
        end
        return imag(f(cargs...))/step
    end 
    return forwardmode_jacobians(fwd,args[flags]...) 
end




function reversemode_jacobians(Y,g::Function)
	m = length(Y)
    DY = zero(Y) ; 
    DY[1] = 1
    DX = g(DY); 
    DY[1] = 0
    @assert sum(abs(DY)) == 0
    if !isa(DX,NTuple) DX = (DX,) end
  	K = length(DX); 
    Jacobians = map(DX) do X # create tuple of Jacobian matrices
        #J = Array(eltype(X),m,length(X)); J[1,:] = vec(X); J    
        J = Array(eltype(X),m,length(X)); J[1,:] = X; J    
    end
    for i = 2:m
		DY[i] = 1
        DX = g(DY)
        DY[i] = 0
        @assert sum(abs(DY)) == 0
        if !isa(DX,NTuple) DX = (DX,) end
	    for k = 1:K
		    J = Jacobians[k]
            #J[i,:] = vec(DX[k])
            J[i,:] = DX[k]
		end
    end
    return K==1?Jacobians[1]:Jacobians
end


setcomp!(A::Array,k::Int,i::Int,v) = if ndims(A[k])==0 A[k] = v else A[k][i] = v end

function forwardmode_jacobians(fwd::Function, X...)
    K = length(X)
    Jacobians = cell(K)
    DX = {zero(X[k]) for k=1:K}
    for k = 1:K
        setcomp!(DX,k,1,1)
        DY = fwd(DX...)
        setcomp!(DX,k,1,0)
        @assert sum(abs(DX[k])) == 0
        J = Array(eltype(DY),length(DY),length(DX[k]))
        J[:,1] = vec(DY)
        #J[:,1] = vec(DY)
        for j = 2:length(DX[k])
            setcomp!(DX,k,j,1)
            DY = fwd(DX...)
            setcomp!(DX,k,j,0)
            @assert sum(abs(DX[k])) == 0
            #J[:,j] = vec(DY)
            J[:,j] = DY
        end
        Jacobians[k] = J
    end
    return K==1?Jacobians[1]:Jacobians
end




