module CustomMatrices

importall Base

export DenseFlavour,repcol,reprow,colplusrow,rankone,
       SparseFlavour,repdiag,fulldiag,blocksparse,
       flavour

abstract Flavour
  abstract DenseFlavour <: Flavour
    #colplusrow: reprow + repcol
    #toeplitz t.b.d.
    abstract RankOne <:DenseFlavour
      #reprow
      #repcol

  abstract SparseFlavour <: Flavour
    #blocksparse
    abstract DiagFlavour <: SparseFlavour
      #repdiag
      #fulldiag

isinvertible{F<:DiagFlavour}(::Type{F}) = true
isinvertible{F<:Flavour}(::Type{F}) = false




immutable CustomMatrix{F<:Flavour,E,D} <: AbstractMatrix{E}
    data::D
    m::Int
    n::Int
end
CustomMatrix{D}(F::DataType,data::D,m::Int,n::Int) = CustomMatrix{F,eltype(data),D}(data,m,n)
size(M::CustomMatrix) = (M.m,M.n)
flavour{F<:Flavour}(::CustomMatrix{F}) = F
#eltype() is given by AbstractMatrix




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

show(io::IO,M::CustomMatrix) = (print(io,typeof(M));println("->");show(io,full(M)))

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
include("custommatrices/repcol.jl")
include("custommatrices/reprow.jl")
include("custommatrices/colplusrow.jl")
include("custommatrices/rankone.jl")
include("custommatrices/diagflavour.jl")
include("custommatrices/blocksparse.jl")


###################################################################







end # module