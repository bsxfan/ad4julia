module Buggy

bug = true

importall Base

export blocksparse

abstract Flavour
  abstract DenseFlavour <: Flavour
  abstract SparseFlavour <: Flavour


if bug 
  @eval begin
    immutable CustomMatrix{F<:Flavour,E,D} #<: AbstractArray{E}
        data::D
        m::Int
        n::Int
    end
    eltype{F,E}(C::CustomMatrix{F,E}) = E
  end 
else 
  @eval begin
    immutable CustomMatrix{F<:Flavour,E,D} <: AbstractArray{E}
        data::D
        m::Int
        n::Int
    end
  end
end


CustomMatrix{D}(F::DataType,data::D,m::Int,n::Int) = CustomMatrix{F,eltype(data),D}(data,m,n)
size(M::CustomMatrix) = (M.m,M.n)

full(M::CustomMatrix) = full(M,eltype(M))
full{F<:SparseFlavour}(M::CustomMatrix{F},elty::DataType) = zeros(elty,size(M))
#full{F<:SparseFlavour,E}(M::CustomMatrix{F,E}) = zeros(E,size(M))


type blocksparse <: SparseFlavour end
immutable blockdata{T<:Number} 
  block::Matrix{T}
  at::(Int,Int)
end
blockdata{T}(block::Matrix{T},at::(Int,Int)) = {T}blockdata(block,at)
eltype{T}(B::blockdata{T}) = T




function sum(B::blockdata,i::Int)
    if i==1
      blockdata(sum(B.block,i),(1,B.at[2]))  
    elseif i==2
      blockdata(sum(B.block,i),(B.at[1],1))  
    else
      error("bad i")
    end
end

function blocksparse(block::Matrix,at::(Int,Int),sz::(Int,Int)) 
  for i=1:2 assert(1 <= at[i] <= sz[i] - size(block,i) + 1,
    "$(size(block)) block does not fit at $at in $sz matrix") 
    end
  return CustomMatrix(blocksparse,blockdata(block,at),sz...)
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