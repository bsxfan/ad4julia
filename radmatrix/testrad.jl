

function compute_radjacobians(Y,g::Function)
	m = length(Y)
    DY = zeros(eltype(Y),size(Y)) ; 
    DY[1] = 1
    DX = g(DY); 
    if !isa(DX,NTuple) DX = (DX,) end
  	K = length(DX); 
  	lenX = map(length,DX); 
    DY[1] = 0
    @assert sum(abs(DY)) == 0
    Jacobians = map(DX) do X # create tuple of Jacobian matrices
        J = similar(X,m,length(X)); J[1,:] = vec(X); J    
    end
    for i = 2:m
		DY[i] = 1
        DX = g(DY)
        if !isa(DX,NTuple) DX = (DX,) end
        DY[i] = 0
        @assert sum(abs(DY)) == 0
	    for k = 1:K
		    J = Jacobians[k]
		    J[i,:] = vec(DX[k])
		end
    end
    return Jacobians
end

radjacobians(f::Function,args,flags=trues(length(args)) ) = 
 	compute_radjacobians( radeval(f,args,flags)... )
