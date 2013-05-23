module CustomMatrices

importall Base

export repvec, diagonal, repel, CustomMatrix,
       DenseFlavour,repcol,reprow,colplusrow,rankone,
       SparseFlavour,repdiag,fulldiag,blocksparse,
       flavour, update!

abstract Flavour
  abstract DenseFlavour <: Flavour
    #colplusrow: reprow + repcol
    #toeplitz t.b.d.
    abstract Rank1Flavour <:DenseFlavour
      #rankone: general case
      #reprow
      #repcol
      #repel: el*ones

  abstract SparseFlavour <: Flavour
    #blocksparse
    abstract DiagFlavour <: SparseFlavour
      #repdiag
      #fulldiag

isinvertible{F<:DiagFlavour}(::Type{F}) = true
isinvertible{F<:Flavour}(::Type{F}) = false




immutable CustomMatrix{F<:Flavour,E,D}
    data::D
    m::Int
    n::Int
end
CustomMatrix{D}(F::DataType,data::D,m::Int,n::Int) = CustomMatrix{F,eltype(data),D}(data,m,n)
flavour{F<:Flavour}(::CustomMatrix{F}) = F

size(M::CustomMatrix) = (M.m,M.n)
size(M::CustomMatrix,i::Int) = 1<=i<=ndims(M)?size(M)[i]:1
length(M::CustomMatrix) = M.m*M.n
eltype{F<:Flavour,E<:Number}(::CustomMatrix{F,E}) = E
ndims(M::CustomMatrix) = 2




# If element types allow, do D += C in-place in D.
# else creates new D
accumulate!{T,N}(D::Vector,C::CustomMatrix) = accumulate!(reshape(D,length(D),1),C)
function accumulate!{T<:Number,F<:Flavour,E<:Number}(D::Matrix{T},C::CustomMatrix{F,E})
  if (T<:Integer && E<:Union(FloatingPoint,Complex) ) || 
     (T<:FloatingPoint && E<:Complex)
    P = promote_type(T,E)
    N = Array(P,size(D))
    copy!(N,D)
    return update!(0,N,C)
  else
    return update!(1,D,C)
  end
end






full(M::CustomMatrix) = full(M,eltype(M))
full{F<:SparseFlavour}(M::CustomMatrix{F},elty::DataType) = update!(0,zeros(elty,size(M)),M)
full{F<:DenseFlavour}(M::CustomMatrix{F},elty::DataType) = update!(0,Array(elty,size(M)),M) 


summary(M::CustomMatrix) = string(Base.dims2string(size(M))," ",typeof(M)) 
function show(io::IO,M::CustomMatrix) 
  #print(io,typeof(M));println("->");show(io,full(M))  
  print(io,summary(M))
  if max(size(M))>20
      println()
  else
      println("->")
      show(io,full(M))
  end
end

copy{F}(M::CustomMatrix{F}) = CustomMatrix(F,M.data,M.m,M.n) 
copy!{F<:DenseFlavour}(D::Matrix,S::CustomMatrix{F}) = update!(0,D,S)

#custom to full
convert{D<:Number,S<:Number,F<:Flavour}(::Type{Matrix{D}},M::CustomMatrix{F,S}) = full(M,D)

#addition is closed within most flavours (some other additions between different flavours are defined below)
function (+){F}(A::CustomMatrix{F},B::CustomMatrix{F})  
    assert(size(A)==size(B),"size mismatch")
    return CustomMatrix(F,A.data + B.data,size(A)...)
end

function (+){F,G}(A::CustomMatrix{F},B::CustomMatrix{G})  
   error("CustomMatrix addition between flavours $F and $G not defined, use full() or update()")
end


(*)(C::CustomMatrix,s::Number) = *(s,C)
function (*){F}(s::Number,C::CustomMatrix{F})
    return CustomMatrix(F,s*C.data,size(C)...)
end

(-){F}(C::CustomMatrix{F}) = CustomMatrix(F,(-1)*C.data,size(C)...)
function (-)(A::CustomMatrix,B::CustomMatrix)  
    return +(A,-B)
end

#transpose is declared below for individual flavours
conj{F}(C::CustomMatrix{F}) = CustomMatrix(F,conj(C.data),C.m,C.n)
ctranspose{F,E<:Real}(C::CustomMatrix{F,E}) = transpose(C)
ctranspose{F,E<:Complex}(C::CustomMatrix{F,E}) = transpose(conj(C))


sum(C::CustomMatrix) = sum(sum(C,1))

###################################################################
include("custommatrices/repvecs.jl")

asvec(v::Vector) = v
function asvec(v::Matrix)
  d,n = size(v)
  if d==1 || n==1 return vec(v) end
  error("argument must have 1 row or 1 column, but is $(size(v))")  
end

include("custommatrices/rankone.jl")
include("custommatrices/repcol.jl")
include("custommatrices/reprow.jl")
include("custommatrices/repel.jl")

include("custommatrices/colplusrow.jl")

include("custommatrices/diagflavour.jl")

include("custommatrices/blocksparse.jl")


###################################################################

transpose(C::RepCol) = reprow(C.n,C.data) 
transpose(C::RepRow) = repcol(C.data,C.m) 

###################################################################


rankone(col::RepVecs,row::VecOrMat) = reprow(length(col),element(col)*asvec(row))
rankone(col::VecOrMat,row::RepVecs) = repcol(element(row)*asvec(col),length(row))
rankone(col::RepVecs,row::RepVecs) = repel(element(col)*element(row),length(col),length(row))

mok(A,B) = (if size(A,2)!=size(B,1) error("mismatched sizes for matrix *") end)
row2(A::CustomMatrix) = (v = row(A); reshape(v,1,length(v)) )
row(M::Matrix) = (sz = size(M); sz[1]==1 ? reshape(M,sz[2]): error("not a row matrix") )
col(M::Matrix) = (sz = size(M); sz[2]==1 ? reshape(M,sz[1]): error("not a column matrix") )

*(c::RepColVec,r::Matrix) = (mok(c,r); rankone(col(c),row(r)) )
*(c::RepColVec,r::RepRowVec) = (mok(c,r); rankone(col(c),row(r)) )
*(c::Vector,r::RepRowVec) = (mok(c,r); rankone(c,row(r)) )


(*){F<:DiagFlavour,G<:Rank1Flavour}(A::CustomMatrix{F}, B::CustomMatrix{G}) = 
   ( mok(A,B); rankone(diag(A).*col(B),row(B)) )

(*){F<:DiagFlavour,G<:DiagFlavour}(A::CustomMatrix{F}, B::CustomMatrix{G}) = 
   ( mok(A,B); diagonal(diag(A).*diag(B)) )

(*){F<:DiagFlavour}(A::CustomMatrix{F}, B::Matrix) = 
   ( mok(A,B); scale(diag(A),B) )

##
(*){F<:Rank1Flavour,G<:Rank1Flavour}(A::CustomMatrix{F},B::CustomMatrix{G})  = 
   (mok(A,B); rankone(dot(row(A),col(B))*col(A),row(B)) )

(*){F<:Rank1Flavour,G<:DiagFlavour}(A::CustomMatrix{F},B::CustomMatrix{G})  = 
   (mok(A,B); rankone(col(A),row(A).*diag(B) ) )

(*){F<:Rank1Flavour}(A::CustomMatrix{F},B::Matrix)  = 
   (mok(A,B); rankone(col(A),row2(A)*B ) )

##
(*){G<:Rank1Flavour}(A::Matrix,B::CustomMatrix{G}) =
   (mok(A,B); rankone(A*col(B), row(B)) )

(*){G<:DiagFlavour}(A::Matrix,B::CustomMatrix{G}) =
   (mok(A,B); scale(A,diag(B)) )


end # module