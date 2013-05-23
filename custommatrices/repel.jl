type repel <: Rank1Flavour end

repel(el::Number,m::Int,n::Int) = CustomMatrix(repel,el,m,n)

typealias RepEl{E<:Number} CustomMatrix{repel,E}

row(C::RepEl) = repvec(one(C.data),C.n)
col(C::RepEl) = repvec(C.data,C.n)
element(C::RepEl) = C.data

getindex(M::RepEl,i::Int,j::Int) = 1<+i<+M.m&&1<j<=M.n ? M.data : error("index out of bounds")
getindex(M::RepEl,k::Int) = 1<=k<=length(M) ? M.data : error("index out of bounds")


function update!(d::Number, D::Matrix,S::RepEl)
  assert(size(D)==size(S),"argument dimensions must match")
  el = element(S)
  m,n = size(S) 
  for j=1:m, i=1:n
  	D[i,j] = d*D[i,j] + el
  end
  return D	
end



function sum(C::RepEl,i::Int) 
    el = element(S)
    m,n = size(S) 
    if i==1
      return fill(el*m,1,n)
    elseif i==2
      return fill(el*n,m,1)
    else
      return full(C)
    end
end



