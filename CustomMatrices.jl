module CustomMatrices

importall Base

export DenseFlavour,repcol,reprow,
       SparseFlavour,repdiag,fulldiag,blocksparse

abstract Flavour
abstract DenseFlavour <: Flavour
#reprow
#repcol
abstract SparseFlavour <: Flavour
#blocksparse
abstract DiagFlavour <: SparseFlavour
#repdiag
#fulldiag



immutable CustomMatrix{F<:Flavour,E,D} <: AbstractMatrix{E}
    data::D
    m::Int
    n::Int
end
CustomMatrix{D}(F::DataType,data::D,m::Int,n::Int) = CustomMatrix{F,eltype(data),D}(data,m,n)
size(M::CustomMatrix) = (M.m,M.n)


# If element types allow, do D += C in-place in D.
# else creates new D
function accumulate!{T<:Number,F<:Flavour,E<:Number}(D::Matrix{T},C::CustomMatrix{F,E})
  if (T<:Integer && E<:Union(FloatingPoint,Complex) ) || 
     (T<:FloatingPoint && E<:Complex)
    P = promote_type(T,E)
    N = Array(P,size(D))
    copy!(N,D)
    add()
  else

  end
end



# All flavours have add!(D,S) and add!(D,S,R) methods, which efficiently implements D += S (+ R) in-place, for full
# matrix D and Custom matrices S,R.

# Only dense flavours have an update!(d,D,S) method, in which case add!(D,S) = update!(1,D,S)
add!{F<:DenseFlavour}(D::Matrix,S::CustomMatrix{F}) = update!(1,D,S)  #
add!{F<:DenseFlavour,G<:DenseFlavour}(D::Matrix,L::CustomMatrix{F},R::CustomMatrix{G}) = update!(1,D,L,R) 

function add!{F<:SparseFlavour,G<:SparseFlavour}(D::Matrix,L::CustomMatrix{F},R::CustomMatrix{G})
    if applicable(+,L,R) 
      return add!(D,L+R)
    else
      return add!(add!(D,L),R)
    end
end

add!{F<:SparseFlavour,G<:DenseFlavour}(D::Matrix,L::CustomMatrix{F},R::CustomMatrix{G}) = add!(D,R,L)
add!{F<:DenseFlavour,G<:SparseFlavour}(D::Matrix,L::CustomMatrix{F},R::CustomMatrix{G}) = add!(add!(D,L),R)



full(M::CustomMatrix) = full(M,eltype(M))
full{F<:SparseFlavour}(M::CustomMatrix{F},elty::DataType) = add!(zeros(elty,size(M)),M)
full{F<:DenseFlavour}(M::CustomMatrix{F},elty::DataType) = update!(0,Array(elty,size(M)),M) #faster

show(io::IO,M::CustomMatrix) = (print(io,typeof(M));println("->");show(io,full(M)))

copy{F}(M::CustomMatrix{F}) = CustomMatrix(F,M.data,M.m,M.n) 
copy!{F<:DenseFlavour}(D::Matrix,S::CustomMatrix{F}) = update!(0,D,S)

#custom to full
convert{D<:Number,S<:Number,F<:Flavour}(::Type{Matrix{D}},M::CustomMatrix{F,S}) = full(M,D)



# function (+){a,b,F}(A::Matrix{a},B::CustomMatrix{F,b}) 
#   assert(size(A)==size(B),"size mismatch")
#   D = zeros()
# end
  

###################################################################
type repcol <: DenseFlavour end
repcol(column::Vector,n::Int) = CustomMatrix(repcol,column,length(column),n)
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


###################################################################
type reprow <: DenseFlavour end
reprow(row::Vector,m::Int) = CustomMatrix(reprow,row,m,length(row))
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


############################ reprow and repcol interaction ##########################

#same flavour
update!{F<:DenseFlavour}(d::Number,D::Matrix,
                         L::CustomMatrix{F},R::CustomMatrix{F}) = update!(d::Number,D,L+R) 

#different flavours
update!(D::Matrix,C::CustomMatrix{repcol},R::CustomMatrix{reprow}) = update!(D,R,C)
function update!(d::Number, D::Matrix,R::CustomMatrix{reprow},C::CustomMatrix{repcol})
  assert(size(D)==size(R)==size(C),"argument dimensions must match")
  row = R.data
  col = C.data
  m,n = size(D) 
  for j=1:n
    rj = row[j] 
    for i=1:m
      D[i,j] = d*D[i,j] + rj + col[i]
    end
  end
  return D  
end

function (+)(A::CustomMatrix{F1,E1},B::CustomMatrix{F2,E2})
  assert(size(A)==size(B))
  T = promote_type(E1,E1)
  if F1<:DenseFlavour || F2<:DenseFlavour
    D = Array(T,size(A))
    return update!(0,D,A,B)
  else
    D = zeros(T,size(A))
    return add!(D,A,B)
  end

end 


###################################################################
type repdiag <: DiagFlavour end
repdiag(element::Number,n::Int) = CustomMatrix(repdiag,element,n,n)

square_sz(M::AbstractMatrix) = ((m,n)=size(M);assert(m==n,"argument not square");m)

function update!(d::Number, D::Matrix,S::CustomMatrix{repdiag})
  n = square_sz(D)
  assert(n==S.n,"argument dimensions must match")
  element = S.data 
  for i=1:n D[i,i] = d*D[i,i] + element end 
  return D  
end

###################################################################
type fulldiag <: DiagFlavour end
fulldiag(diag::Vector) = CustomMatrix(fulldiag,diag,length(diag),length(diag))

function update!(d::Number, D::Matrix,S::CustomMatrix{fulldiag})
  n = square_sz(D)
  assert(n==S.n,"argument dimensions must match")
  diag = S.data 
  for i=1:n D[i,i] = d*D[i,i] + diag[i] end 
  return D  
end

###################################################################
immutable blocksparse{T<:Number} <: SparseFlavour 
  block::Matrix{T}
  at::(Int,Int)
end
blocksparse{T}(block::Matrix{T},at::(Int,Int)) = {T}blocksparse(block,at)
eltype{T}(B::blocksparse{T}) = T

(+)(A::blocksparse,B::blocksparse) = 
  A.at==B.at && size(A)==size(B)? blocksparse(A.block+B.block,A.at) : error("block location mismatch")

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

###################################################################



# For efficiency define CustomMatrix + CustomMatrix -> CustomMatrix for some special cases
for (a,b,c) in ( (:repcol,:repcol,:repcol), (:reprow,:reprow,:reprow), 
               (:repdiag,:repdiag,:repdiag), (:fulldiag,:fulldiag,:fulldiag), 
               (:repdiag,fulldiag,:fulldiag), (:fulldiag,:repdiag,:fulldiag) )
  @eval begin
    function (+)(A::CustomMatrix{$a},B::CustomMatrix{$b}) 
      assert(size(A)==size(B),"size mismatch")
      return CustomMatrix($c,A.data+B.data,size(A)...) 
    end
  end
end




end # module