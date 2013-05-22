immutable blocksparse{T<:Number} <: SparseFlavour 
  block::Matrix{T}
  at::(Int,Int)
end
blocksparse{T}(block::Matrix{T},at::(Int,Int)) = {T}blocksparse(block,at)
eltype{T}(B::blocksparse{T}) = T

(+)(A::blocksparse,B::blocksparse) = 
  A.at==B.at && size(A)==size(B)? blocksparse(A.block+B.block,A.at) : error("block mismatch")

(*)(s::Number,data::blocksparse) = blocksparse(s*data.block,data.at)
(*)(data::blocksparse,s::Number) = *(s,data)

transpose(data::blocksparse) = blocksparse(data.block.',(data.at[2],data.at[1]) )
conj(data::blocksparse) = blocksparse(conj(data.block),data.at )
ctranspose(data::blocksparse) = blocksparse(data.block',(data.at[2],data.at[1]) )

function sum(B::blocksparse,i::Int)
    if i==1
      blocksparse(sum(B.block,i),(1,B.at[2]))  
    elseif i==2
      blocksparse(sum(B.block,i),(B.at[1],1))  
    else
      error("bad i")
    end
end

function blocksparse(block::Matrix,at::(Int,Int),sz::(Int,Int)) 
  for i=1:2 assert(1 <= at[i] <= sz[i] - size(block,i) + 1,
    "$(size(block)) block does not fit at $at in $sz matrix") 
    end
  return CustomMatrix(blocksparse,blocksparse(block,at),sz...)
end

function update!(d::Number, D::Matrix,S::CustomMatrix{blocksparse})
  assert(size(D)==size(S),"argument dimensions must match")
  i0,j0 = S.data.at
  block = S.data.block
  m,n = size(block) 
  atj = j0
  for j=1:n
    ati = i0
    for i=1:m
      D[ati,atj] = d*D[ati,atj] + block[i,j]
      ati += 1
    end
    atj += 1
  end
  return D  
end


function (+)(A::CustomMatrix{blocksparse},B::CustomMatrix{blocksparse}) 
  assert(size(A)==size(B),"size mismatch")
  return CustomMatrix(blocksparse,A.data+B.data,size(A)...) 
end

transpose(C::CustomMatrix{blocksparse}) = CustomMatrix(blocksparse,C.data.',C.n,C.m)
ctranspose{F,E<:Complex}(C::CustomMatrix{F,E}) = CustomMatrix(blocksparse,C.data',C.n,C.m)

function sum(C::CustomMatrix{blocksparse},i::Int) 
    if i==1
      println("here 1")
      sd = sum(C.data,i)
      println("here 2")
      S = CustomMatrix(blocksparse,sd,1,C.n)
      println("here 3")
      @which full(S)
      println(eltype(S))
      #return S
      @which zeros(eltype(S),size(S))
      return full(S)
    elseif i==2
      S = CustomMatrix(blocksparse,sum(C.data,i),C.m,1)
      return full(S)
    else
      return full(C)
    end
end
