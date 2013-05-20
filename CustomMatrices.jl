module CustomMatrices

importall Base

export DenseFlavour,repcol,reprow,
       SparseFlavour,repdiag,fulldiag,blocksparse

abstract Flavour
abstract DenseFlavour <: Flavour
#reprow
#repcol
abstract SparseFlavour <: Flavour
#repdiag
#fulldiag
#blocksparse



immutable CustomMatrix{F<:Flavour,E,D} <: AbstractMatrix{E}
    data::D
    m::Int
    n::Int
end
CustomMatrix{D}(F::DataType,data::D,m::Int,n::Int) = CustomMatrix{F,eltype(data),D}(data,m,n)


full(M::CustomMatrix) = full(M,eltype(M))
full{F<:SparseFlavour}(M::CustomMatrix{F},elty::DataType) = add!(zeros(elty,size(M)),M)
full{F<:DenseFlavour}(M::CustomMatrix{F},elty::DataType) = update!(0,Array(elty,size(M)),M) #faster

show(io::IO,M::CustomMatrix) = (print(io,typeof(M));println("->");show(io,full(M)))
size(M::CustomMatrix) = (M.m,M.n)
copy{F}(M::CustomMatrix{F}) = CustomMatrix(F,M.data,M.m,M.n) 

add!(D::Matrix,S::CustomMatrix) = update!(1,D,S)
copy!{F<:DenseFlavour}(D::Matrix,S::CustomMatrix{F}) = update!(0,D,S)

#custom to full
convert{D<:Number,S<:Number,F<:Flavour}(::Type{Matrix{D}},M::CustomMatrix{F,S}) = full(M,D)
#custom to custom (same flavour)
convert{D<:Number,S<:Number,F<:Flavour}(::Type{CustomMatrix{F,D}},
                                        M::CustomMatrix{F,S}) = D==S?M:CustomMatrix(F,convert(D,M.data),M.m,M.n) 


function (+){F}(A::CustomMatrix{F},B::CustomMatrix{F}) 
  assert(size(A)==size(B),"size mismatch")
  return CustomMatrix(F,A.data+B.data,size(A)...) 
end

  

###################################################################
type repcol <: DenseFlavour end
repcol(column::Vector,n::Int) = CustomMatrix(repcol,column,length(column),n)
getindex(M::CustomMatrix{repcol},i,j) = M.data[i]
getindex(M::CustomMatrix{repcol},k) = M.data[1+(k-1)%M.m] # k = i+m*(j-1)

function update!(d:<Number, D::Matrix,S::CustomMatrix{repcol})
  assert(size(D)==size(S),"argument dimensions must match")
  col = S.data
  m,n = size(S) 
  for j=1:n, i=1:m 
      D[i,j] = d*D[i,j] + col[i] #'switch' d is about as fast as a test outside the loop
  end 
  return D	
end


###################################################################
type reprow <: DenseFlavour end
reprow(row::Vector,m::Int) = CustomMatrix(reprow,row,m,length(row))
getindex(M::CustomMatrix{repcol},i,j) = M.data[j]
getindex(M::CustomMatrix{repcol},k) = M.data[1+div(k-1,M.m)]


function update!(d::Number, D::Matrix,S::CustomMatrix{reprow})
  assert(size(D)==size(S),"argument dimensions must match")
  row = S.data
  m,n = size(S) 
  for (j,rj) in enumerate(row), i=1:m   #enumerate here is fast
  	D[i,j] = d*D[i,j] + rj
  end
  return D	
end


##################################################################

function (+){F,G}(A::CustomMatrix{F},B::CustomMatrix{G}) 
  assert(size(A)==size(B),"size mismatch")
  #expand to full on flavour with most efficient full method
  elty = promote_type(eltype(A),eltype(B))
  if F==reprow return add!(full(A,elty),B) end 
  if G==reprow || G==repcol return add!(full(B,elty),A) end 
  return add!(full(A,elty),B) 
end


add!(D::Matrix,C::CustomMatrix{repcol},R::CustomMatrix{reprow}) = add!(D,R,C)
function add!(D::Matrix,R::CustomMatrix{reprow},C::CustomMatrix{repcol})
  assert(size(D)==size(R)==size(C),"argument dimensions must match")
  row = R.data
  col = C.data
  m,n = size(D) 
  for j=1:n
    rj = row[j] #makes reprow slightly more efficient than repcolumn
    for i=1:m
      D[i,j] += rj + col[i]
    end
  end
  return D  
end




###################################################################
type repdiag <: SparseFlavour end
repdiag(element::Number,n::Int) = CustomMatrix(repdiag,element,n,n)

square_sz(M::AbstractMatrix) = ((m,n)=size(M);assert(m==n,"argument not square");m)

function add!(D::Matrix,S::CustomMatrix{repdiag})
  n = square_sz(D)
  assert(n==S.n,"argument dimensions must match")
  element = S.data 
  for i=1:n D[i,i] += element end 
  return D  
end

###################################################################
type fulldiag <: SparseFlavour end
fulldiag(diag::Vector) = CustomMatrix(fulldiag,diag,length(diag),length(diag))

function add!(D::Matrix,S::CustomMatrix{fulldiag})
  n = square_sz(D)
  assert(n==S.n,"argument dimensions must match")
  diag = S.data 
  for i=1:n D[i,i] += diag[i] end 
  return D  
end

###################################################################
immutable blocksparse{T<:Number} <: SparseFlavour 
  block::Matrix{T}
  at::(Int,Int)
end
blocksparse{T}(block::Matrix{T},at::(int,Int))
eltype{T}(B::blocksparse{T}) = T
convert{D,S}(::Type{D},B::blocksparse{S}) = D==S?B:blocksparse(convert(D,B.data),B.at)

(+)(A::blocksparse,B::blocksparse) = A.at==B.at?blocksparse(A.block_B.block,A.at):error("size mismatch")

function blocksparse(block::Matrix,at::(Int,Int),sz::(Int,Int)) 
  for i=1:2 assert(1 <= at[i] <= sz[i] - size(block,i) + 1,
    "$(size(block)) block does not fit at $at in $sz matrix") 
    end
  return CustomMatrix(blocksparse,blocksparse(block,at),sz...)
end

function add!(D::Matrix,S::CustomMatrix{blocksparse})
  assert(size(D)==size(S),"argument dimensions must match")
  i0,j0 = S.data.at
  block = S.data.block
  m,n = size(block) 
  atj = j0
  for j=1:n
    ati = i0
    for i=1:m
      D[ati,atj] += block[i,j]
      ati += 1
    end
    atj += 1
  end
  return D  
end

###################################################################


end # module