type repcol <: RankOne end

repcol(column::Vector,n::Int) = CustomMatrix(repcol,column,length(column),n)
repcol(col::Matrix, n::Int) = repcol(asvec(col),n)

getindex(M::CustomMatrix{repcol},i::Int,j::Int) = M.data[i]
getindex(M::CustomMatrix{repcol},k::Int) = M.data[1+(k-1)%M.m] # k = i+m*(j-1)

function update!(d::Number, D::Matrix,S::CustomMatrix{repcol})
  assert(size(D)==size(S),"argument dimensions must match")
  col = S.data
  m,n = size(S) 
  for j=1:n, i=1:m 
      D[i,j] = d*D[i,j] + col[i] #'switch' d is about as fast as a test outside the loop
  end 
  return D	
end

transpose(C::CustomMatrix{repcol}) = CustomMatrix(reprow, C.data, C.n, C.m)

function sum(C::CustomMatrix{repcol},i::Int) 
    if i==1
      return fill(sum(C.data),1,C.n) 
    elseif i==2
      return reshape(C.n*C.data,C.m,1)
    else
      return full(C)
    end
end

*(M::Matrix, C::CustomMatrix{repcol}) = repcol(M*C.data,C.n)
*(C::CustomMatrix{repcol}, M::Matrix) = rankone(C.data,sum(M,1))
