module Procrustes

export procrustean_add!

# Implements D += S with:
# - broadcasting S for k, such that size(S,k) == 1 < size(D,k)
# - summing S for k ,such that size(S,k) > size(D,k) ==1
# Notes: 
# 1. Can promote shape and type of D.
# 2. Tries to update D in-place, but this is not always possible.
#    Aways use the return value, the identity of D may change.


procrustean_add!(d::Number,s::Number) = d + s  
procrustean_add!(d::Number,S) = d + sum(S)  

function procrustean_add!(D::Array,s::Number) 
    if accepts(D,s)
        for i=1:length(D)
            D[i] += s # work in-place 
        end
        return D
    else
        return D+s  # creates a new matrix
    end
end

function procrustean_add!(D::Array,S::Array)
    # sum if necessary
    for k=1:ndims(S)
        szD = size(D,k) ; szS = size(S,k)
    	if szS > szDk
    		if szDk != 1 error("cannot reduce size(S,$k)==$szS to $szDk") end
    		S = sum(S,k) # this changes size(S,k), but not ndims(S)
    	end
    end

    if length(S)==1; return procrustean_add!(D,S[1]); end  # collapse S to scalar, broadcast over D
    
    # add in-place if possible, broadcasting not implemented yet---will crash when adding vec to mat
    if eltype(D)==eltype(S) # work in-place
        D = reshape(D,promote_shape(size(D),size(S))) #new identity, same data
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


function update!(d::Number)


end #module