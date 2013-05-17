module Procrustes

export procrustean_add!

# Implements D += S with:
# - broadcasting S for k, such that size(S,k) == 1 < size(D,k)
# - summing S for k ,such that size(S,k) > size(D,k) ==1
# Notes: 
# 1. Can promote shape and type of D; 
# 2. This tries to update D in-place, but this is not always possible.
#    Aways use the return value, the identity of D may change


procrustean_add!(D::Number,S::Number) = D + S  
procrustean_add!(D::Number,S::AbstractArray) = D + sum(S)  

function procrustean_add!(D::AbstractArray,S::Number) 
    if eltype(D)==typeof(S)
        for i=1:length(D)
            D[i] += S # work in-place 
        end
        return D
    else
        return D+S  # creates a new matrix
    end
end

function procrustean_add!(D::AbstractArray,S::AbstractArray)
    # sum if necessary
    for k=1:ndims(S)
    	if size(S,k) > size(D,k)
    		@assert size(D,k) == 1
    		S = sum(S,k) # this changes size(S,k), but not ndims(S)
    	end
    end

    if length(S)==1; return procrustean_add!(D,S[1]); end  # collapse S to scalar, broadcast over D
    
    # add in-place of possible, broadcasting not implemented yet---will crash when adding vec to mat
    if eltype(D)==eltype(S) # work in-place
        D = reshape(D,promote_shape(size(D),size(S))) #new identity, same data, can pro
        for i=1:length(D)
            D[i] += S[i]  
        end
        return D
    else # create new
        return D + S 
    end
     
end

# In Greek mythology, Procrustes was a rogue smith and bandit who attacked people by stretching them,
# or cutting off their legs, so as to force them to fit the size of an iron bed.

end #module