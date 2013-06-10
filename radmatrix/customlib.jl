export multiclass_xentropy

multiclass_xentropy(L, labels::Vector{Int}) = multiclass_xentropy(L,labels,ones(size(L,1)))
function multiclass_xentropy(L, labels::Vector{Int}, weights::Vector)
    m,n = size(L) # m classes, n trials
    @assert length(labels) == n
    lse = logsumexp(L)
    s = zeros(eltype(lse),m)
    for j=1:n
        i = labels[j];
        if 1<=i<=m s[i] += lse[j] - L[i,j] end
    end
    return dott(s,weights)
end

multiclass_xentropy(R::RadNum, labels::Vector{Int}) = multiclass_xentropy(R,labels,ones(size(R,1)))
function multiclass_xentropy(R::RadNum, labels::Vector{Int}, weights::Vector)
	L,Ln = rd(R)
    m,n = size(L) # m classes, n trials
    @assert length(labels) == n
    lse = logsumexp(L)
    s = zeros(eltype(lse),m)
    for j=1:n
        i = labels[j];
        if 1<=i<=m s[i] += lse[j] - L[i,j] end
    end
    Z = dott(s,weights)
    back(G) = (
    	DL = Array(eltype(lse),m,n);
    	for j=1:n lse_j = lse[j]
            for i=1:m
                lsm = L[i,j] - lse_j
    	        d = i==labels[j]?expm1(lsm):exp(lsm)
                DL[i,j] = G*weights[i]*d;
            end
    	end;
        return backprop(Ln,DL)
    )
    return radnum(Z,back)
end


###################################################################



function logsumexp(R::RadNum)
    X,Xn = rd(R)
    m,n = size(X)
    y = Array(eltype(X),n)
    for j=1:n
        mx = real(X[1,j])
        for i=2:m e = real(X[i,j]); if e>mx mx = e end end
        s = 0.0 
        for i=1:m s += exp(X[i,j]-mx) end
        y[j] = mx + log(s);
    end
    Y = reshape(y,1,n)
    back(G) = backprop(Xn,G.*exp(X.-(Y)))
    return radnum(Y,back)
end

###################################################################

