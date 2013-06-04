

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
        J = Array(eltype(X),m,length(X)); J[1,:] = vec(X); J    
    end
    for i = 2:m
		DY[i] = 1
        DX = g(DY)
        DY[i] = 0
        @assert sum(abs(DY)) == 0
        if !isa(DX,NTuple) DX = (DX,) end
	    for k = 1:K
		    J = Jacobians[k]
		    J[i,:] = vec(DX[k])
		end
    end
    return K==1?Jacobians[1]:Jacobians
end

rad_jacobians(f::Function,args,flags=trues(length(args)) ) = 
 	reversemode_jacobians( radeval(f,args,flags)... )


function forwardmode_jacobians(fwd::Function, X...)
    K = length(X)
    Jacobians = cell(K)
    DX = map(zero,X)
    for k = 1:K
        D = DX[k]
        D[1] = 1
        DY = fwd(DX...)
        D[1] = 0
        @assert sum(abs(D)) == 0
        J = Array(eltype(DY),length(DY),length(D))
        J[:,1] = vec(DY)
        for j = 2:length(D)
            D[j] = 1
            DY = fwd(DX...)
            D[j] = 0
            @assert sum(abs(D)) == 0
            J[:,j] = vec(DY)
        end
        Jacobians[k] = J
    end
    return K==1?Jacobians[1]:Jacobians
end


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


