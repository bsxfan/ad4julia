module Buggy

importall Base

export sparseblock

abstract Flavour
  abstract DenseFlavour <: Flavour
  abstract SparseFlavour <: Flavour


immutable CustomMatrix{F<:Flavour,E,D} 
    data::D
    m::Int
    n::Int
end
CustomMatrix{D}(F::DataType,data::D,m::Int,n::Int) = CustomMatrix{F,eltype(data),D}(data,m,n)
size(M::CustomMatrix) = (M.m,M.n)
eltype{F,E}(C::CustomMatrix{F,E}) = E
copy{F}(M::CustomMatrix{F}) = CustomMatrix(F,M.data,M.m,M.n) 


full(M::CustomMatrix) = full(M,eltype(M))
full{F<:SparseFlavour}(M::CustomMatrix{F},elty::DataType) = zeros(elty,size(M))
#full{F<:SparseFlavour,E}(M::CustomMatrix{F,E}) = zeros(E,size(M))



immutable blocksparse{T<:Number} <: SparseFlavour 
  block::Matrix{T}
  at::(Int,Int)
end
blocksparse{T}(block::Matrix{T},at::(Int,Int)) = {T}blocksparse(block,at)
eltype{T}(B::blocksparse{T}) = T




function sum(B::blocksparse,i::Int)
    if i==1
      blocksparse(sum(B.block,i),(1,B.at[2]))  
    elseif i==2
      blocksparse(sum(B.block,i),(B.at[1],1))  
    else
      error("bad i")
    end
end

function sparseblock(block::Matrix,at::(Int,Int),sz::(Int,Int)) 
  for i=1:2 assert(1 <= at[i] <= sz[i] - size(block,i) + 1,
    "$(size(block)) block does not fit at $at in $sz matrix") 
    end
  return CustomMatrix(blocksparse,blocksparse(block,at),sz...)
end




function sum(C::CustomMatrix{blocksparse},i::Int) 
    if i==1
      S = CustomMatrix(blocksparse,sum(C.data,i),1,C.n)
      return S
    elseif i==2
      S = CustomMatrix(blocksparse,sum(C.data,i),C.m,1)
      return full(S)
    else
      return full(C)
    end
end








end # module