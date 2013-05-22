type reprow <: RankOne end

reprow(m::Int,row::Vector) = CustomMatrix(reprow,row,m,length(row))
reprow(m::Int,row::Matrix) = reprow(m,asvec(row))


getindex(M::CustomMatrix{reprow},i::Int,j::Int) = M.data[j]
getindex(M::CustomMatrix{reprow},k::Int) = M.data[1+div(k-1,M.m)]


function update!(d::Number, D::Matrix,S::CustomMatrix{reprow})
  assert(size(D)==size(S),"argument dimensions must match")
  row = S.data
  m,n = size(S) 
  for (j,rj) in enumerate(row), i=1:m   #enumerate here is fast
  	D[i,j] = d*D[i,j] + rj
  end
  return D	
end


transpose(C::CustomMatrix{reprow}) = CustomMatrix(repcol, C.data, C.n, C.m)

function sum(C::CustomMatrix{reprow},i::Int) 
    if i==1
      return reshape(C.m*C.data,1,C.n) 
    elseif i==2
      return fill(sum(C.data),C.m,1)
    else
      return full(C)
    end
end

*(M::Matrix, C::CustomMatrix{reprow}) = rankone(sum(M,2),C.data)
*(C::CustomMatrix{reprow}, M::Matrix) = reprow(C.m,M.'*C.data)
