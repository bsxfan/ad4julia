module Procrustes

export procrustean_add!

# Implements D += S with:
# - broadcasting S for k, such that size(S,k) == 1 < size(D,k)
# - summing S for k ,such that size(S,k) > size(D,k) ==1

procrustean_add!(D::Number,S::Number) = D+S  
procrustean_add!(D::Number,S::AbstractArray) = D+sum(S)  
procrustean_add!(D::AbstractArray,S::Number) = (for i=1:length(D);D[i] += S;end;D)  

function procrustean_add!(D::AbstractArray,S::AbstractArray)
    # sum if necessary
    for k=1:ndims(S)
    	if size(S,k) > size(D,k)
    		@assert size(D,k) == 1
    		S = sum(S,k) # this changes size(S,k), but not ndims(S)
    	end
    end

    # now add with broadcasting
    if length(S) ==1 
    	return procrustean_add!(D,S[1])     # collapse S to scalar, broadcast over D
    elseif length(D) == length(S)  #should maybe check shapes properly, not just length
        for i=1:length(D)     
            D[i] += S[i]
        end
    else
        error("broadcasting t.b.d.")
    end
     
    return D
end

# In Greek mythology, Procrustes was a rogue smith and bandit who attacked people by stretching them,
# or cutting off their legs, so as to force them to fit the size of an iron bed.

end #module