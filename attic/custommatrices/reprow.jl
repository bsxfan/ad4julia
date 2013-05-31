type reprow <: Rank1Flavour end

reprow(m::Int,row::VecOrMat) = CustomMatrix(reprow,asvec(row),m,length(row))

typealias RepRow{E<:Number} CustomMatrix{reprow,E}

row(C::RepRow) = C.data
col(C::RepRow) = repvec(one(C.data[1]),C.m)


getindex(M::RepRow,i::Int,j::Int) = 1<=i<=M.m ? M.data[j] : error("index out of bounds")
getindex(M::RepRow,k::Int) = 1<=k<=length(M) ? M.data[1+div(k-1,M.m)] : error("index out of bounds")


function update!(d::Number, D::Matrix,S::RepRow)
  assert(size(D)==size(S),"argument dimensions must match")
  row = S.data
  m,n = size(S) 
  for (j,rj) in enumerate(row), i=1:m   #enumerate here is fast
  	D[i,j] = d*D[i,j] + rj
  end
  return D	
end



function sum(C::RepRow,i::Int) 
    if i==1
      return reshape(C.m*C.data,1,C.n) 
    elseif i==2
      return fill(sum(C.data),C.m,1)
    else
      return full(C)
    end
end



