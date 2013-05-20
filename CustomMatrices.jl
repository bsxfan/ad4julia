module CustomMatrices

importall Base

export CustomMatrix,
       repdiag,repcol,reprow,fulldiag,blocksparse

abstract Flavour
type CustomMatrix{F<:Flavour,E,D} <: AbstractMatrix{E}
    data::D
    m::Int
    n::Int
end
CustomMatrix{D}(F::DataType,data::D,m::Int,n::Int) = CustomMatrix{F,eltype(data),D}(data,m,n)


full{F,E}(M::CustomMatrix{F,E}) = (Z = zeros(E,size(M)); add!(Z,M))
show(io::IO,M::CustomMatrix) = (print(io,typeof(M));println("->");show(io,full(M)))
size(M::CustomMatrix) = (M.m,M.n)
copy{F}(M::CustomMatrix{F}) = CustomMatrix(F,M.data,M.m,M.n) 



###################################################################
type repdiag <: Flavour end
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
type fulldiag <: Flavour end
fulldiag(diag::Vector) = CustomMatrix(fulldiag,diag,length(diag),length(diag))

function add!(D::Matrix,S::CustomMatrix{fulldiag})
  n = square_sz(D)
  assert(n==S.n,"argument dimensions must match")
  diag = S.data 
  for i=1:n D[i,i] += diag[i] end 
  return D  
end


###################################################################
type repcol <: Flavour end
repcol(column::Vector,n::Int) = CustomMatrix(repcol,column,length(column),n)

function add!(D::Matrix,S::CustomMatrix{repcol})
  assert(size(D)==size(S),"argument dimensions must match")
  col = S.data
  m,n = size(S) 
  for j=1:n
  	for i=1:m
  		D[i,j] += col[i]
  	end
  end
  return D	
end

###################################################################
type reprow <: Flavour end
reprow(row::Vector,m::Int) = CustomMatrix(reprow,row,m,length(row))
function add!(D::Matrix,S::CustomMatrix{reprow})
  assert(size(D)==size(S),"argument dimensions must match")
  row = S.data
  m,n = size(S) 
  for j=1:n
  	rj = row[j]
  	for i=1:m
  		D[i,j] += rj
  	end
  end
  return D	
end




###################################################################
type blocksparse <: Flavour 
  block::Matrix
  at::(Int,Int)
end
eltype(B::blocksparse) = eltype(B.block)

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


end