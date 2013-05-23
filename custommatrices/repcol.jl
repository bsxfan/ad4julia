type repcol <: Rank1Flavour end

repcol(col::VecOrMat,n::Int) = CustomMatrix(repcol,asvec(col),length(col),n)

typealias RepCol{E<:Number} CustomMatrix{repcol,E}

col(C::RepCol) = C.data
row(C::RepCol) = repvec(one(C.data[1]),C.n)


getindex(M::RepCol,i::Int,j::Int) = 1<=j<=M.n ? M.data[i] : error("index out of bounds")
getindex(M::RepCol,k::Int) = 1<=k<=length(M) ? M.data[1+(k-1)%M.m] : error("index out of bounds")

function update!(d::Number, D::Matrix,S::RepCol)
  assert(size(D)==size(S),"argument dimensions must match")
  col = S.data
  m,n = size(S) 
  for j=1:n, i=1:m 
      #'switching' with d=0 is about as fast as omitting the term d*D[i,j]
      # surprisingly, reading D[i,j] here is almost for free in terms of time
      D[i,j] = d*D[i,j] + col[i] 
  end 
  return D	
end


function sum(C::RepCol,i::Int) 
    if i==1
      return fill(sum(C.data),1,C.n) 
    elseif i==2
      return reshape(C.n*C.data,C.m,1)
    else
      return full(C)
    end
end

