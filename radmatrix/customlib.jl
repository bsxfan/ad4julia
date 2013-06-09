export multiclass_xentropy

multiclass_xentropy(L::Matrix, labels::Vector{Int}) = multiclass_xentropy(L,labels,ones(size(L,1)))
function multiclass_xentropy(L::Matrix, labels::Vector{Int}, weights::Vector)
    m,n = size(L) # m classes, n trials
    @assert length(labels) == n
    lse = logsumexp(L)
    s = zeros(m)
    for j=1:n
        i = labels[j];
        s[i] += lse[j] - L[i,j] 
    end
    return dot(s,weights)
end