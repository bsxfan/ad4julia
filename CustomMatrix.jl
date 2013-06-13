module CustomMatrix
# Lightweight (memory and CPU) representations for various special vectors and matrices.

importall Base

import Base.LinAlg: BlasFloat

using GenUtils, BlasX

export onevec, repvec, zerovec, aszeros, zeropad,
       rankone, rowmat, colmat, reprow, repcol, repel, onemat, zeromat,
       diagmat,
       blocksparse,
       update! , size2,
       procrustean_update!

include("custommatrix/matrixUpdating.jl") #declares update and procrustean_update  


# Custom Array
abstract CArray{E<:Number}
length(A::CArray) = prod(size(A))
ndims(A::CArray) = length(size(A))
size(A::CArray,i::Int) = 1<=i<=ndims(A)?size(A)[i]:1
eltype{E}(A::CArray{E}) = E

ctranspose{E<:Real}(A::CArray{E}) = transpose(A)
ctranspose{E<:Complex}(A::CArray{E}) = transpose(conj(A))
Ac_mul_B{E<:Real}(A::CArray{E},B) = At_mul_B(A,B)
A_mul_Bc{E<:Real}(A,B::CArray{E}) = A_mul_Bt(A,B)
Ac_mul_B{E<:Complex}(A::CArray{E},B) = At_mul_B(conj(A),B)
A_mul_Bc{E<:Complex}(A,B::CArray{E}) = A_mul_Bt(A,conj(B))

# converte: convert element type
converte{E}(::Type{E},A::CArray{E}) = A          # other cases defined below
converte{E}(::Type{E},A::AbstractArray{E}) = A
converte{E,F,N}(::Type{E},A::AbstractArray{F,N}) = convert(Array{E,N},A)

#isdense
isdense(::Array) = true #others defined below

#convert is not wired up yet, do when necessary
convert{T<:CArray}(::Type{T},A::T) = A #trivial case


summary(X::CArray) = string(Base.dims2string(size(X))," ",typeof(X)) 
show(io::IO,X::CArray) = ( println(io,summary(X),"->"); show(full(X)) )
full{E}(X::CArray{E}) = update!(isdense(X)?Array(E,size(X)):zeros(E,size(X)),
                                X)  


#############################################################################
###################   Custom Vectors ########################################

# Custom Vector
abstract CVec{L,E} <: CArray{E}
length{L}(::CVec{L}) = L
size{L}(::CVec{L}) = (L,)
size{L}(::CVec{L},i::Int) = i==1?L:1
ndims(::CVec) = 1

typealias _CVec{E,L} CVec{L,E}
typealias AnyVec{E} Union(_CVec{E},Vector{E})

chkdim(a::AnyVec,b::AnyVec) = length(a)!=length(b)?error("dimension mismatch"):true
chkdim(a::AnyVec,L::Int) = length(a)!=L?error("dimension mismatch"):true
chkdim(L::Int,b::AnyVec) = L!=length(b)?error("dimension mismatch"):true
chkdim(K::Int,L::Int) = K!=L?error("dimension mismatch"):true

###

nzindexrange(v::Vector) = 1:length(v)

###

immutable OneVec{L,E,P} <: CVec{L,E}
  s::E
end
onevec(len::Int, pos::Int, scale = 1.0) = OneVec{len,typeof(scale),pos}(scale)
full{L,E,P}(v::OneVec{L,E,P}) = (f = zeros(E,L); f[P] = v.s; f)
converte{L,E,F,P}(::Type{E},v::OneVec{L,F,P}) = OneVec{L,E,P}(convert(E,v.s))
isdense(::OneVec) = false
*{L,E,P}(x::Number,v::OneVec{L,E,P}) = onevec(L,P,x*v.s)
*{L,E,P}(v::OneVec{L,E,P},x::Number) = onevec(L,P,x*v.s)

nzindexrange{L,E,P}(v::OneVec{L,E,P}) = P
getindex(v::OneVec,ii...) = v.s # valid only for nzindexrange

sum(v::OneVec) = v.s
custom_update!{L,E,P}(D::Array,S::OneVec{L,E,P}) = (D[P] += S.s; D)

###

immutable RepVec{L,E} <: CVec{L,E}
  s::E
end
repvec(len::Int, scale = 1.0) = RepVec{len,typeof(scale)}(scale)

full{L}(v::RepVec{L}) = fill(v.s,L)
isdense(::RepVec) = true
converte{L,E,F}(::Type{E},v::RepVec{L,F}) = RepVec{L,E}(convert(E,v.s))
*{L}(x::Number,v::RepVec{L}) = repvec(L,x*v.s)
*{L}(v::RepVec{L},x::Number) = repvec(L,x*v.s)

nzindexrange{L}(v::RepVec{L}) = 1:L
getindex(v::RepVec,ii...) = v.s # valid only for nzindexrange

sum{L}(v::RepVec{L}) = v.s*L
custom_update!(D::Array,S::RepVec) = update!(D,S.s)

####
typealias _RepVec{E,L} RepVec{L,E}
typealias DenseVec{E} Union(_RepVec{E},Vector{E}) # [] can be used after size checks


###

immutable ZeroVec{L} <: CVec{L,Int}
end
zerovec(len::Int) = ZeroVec{len}()
custom_update!(D::Array,S::ZeroVec) = D
isdense(::ZeroVec) = false

###
immutable ZeropaddedVec{L,E,D,A} <: CVec{L,E}
  data::D
  at:: A
end
ZeropaddedVec{D,A}(L::Int,data::D,at::A) = ZeropaddedVec{L,eltype(data),D,A}(data,at)
isdense(::ZeropaddedVec) = false
zeropad(source::Vector,at) = zeropad(size(source),source[at],at)
zeropad(sz::(Int,),data,at) = ZeropaddedVec(sz...,data,at)
custom_update!{L,E,D,A<:Int}(Dest::Array,S::ZeropaddedVec{L,E,D,A}) = (Dest[S.at] += S.data; Dest)
custom_update!{L,E,D,A}(Dest::Array,S::ZeropaddedVec{L,E,D,A}) = (
  s = 0; data = S.data; at = S.at;
  if       eltype(A)==Int  for i in at Dest[i] += data[s+=1] end 
    elseif eltype(A)==Bool for i=1:length(at) if at[i] Dest[i] += data[s+=1] end end 
    else                   error("can't work with index of type $A") 
  end;
  Dest
  )
###

At_mul_B(x::CVec,y::CVec) = [dot(x,y)]
At_mul_B(x::RepVec,B::Matrix) = x.s*sum(B,1)
At_mul_B{L,E,P}(x::OneVec{L,E,P},B::Matrix) = chkdim(x,B) && x.s*B[P,:]

*{L}(M::Matrix,x::RepVec{L}) = reshape(sum(M,2),L)
*{L,E,P}(M::Matrix,x::OneVec{L,E,P}) =  chkdim(M,x) && x.s*M[:,P]

###


dot{L}(x::RepVec{L},y::RepVec{L}) = L*x.s*y.s
dot{L}(x::RepVec{L},y::OneVec{L}) = x.s*y.s
dot{L}(x::RepVec{L},y::Vector) = chkdim(x,y) && x.s*sum(y)

dot{L}(x::OneVec{L},y::RepVec{L}) = dot(y,x)
dot{L,E,F,P,Q}(x::OneVec{L,E,P},y::OneVec{L,F,Q}) = P==Q?x.s*y.s:zero(x.s*y.s)
dot{L,E,P}(x::OneVec{L,E,P},y::Vector) = chkdim(x,y) && x.s*y[P]

dot(x::Vector,y::RepVec) = dot(y,x)
dot(x::Vector,y::OneVec) = dot(y,x)
###


(.*){L}(x::RepVec{L},y::RepVec{L}) = repvec(L,x.s*y.s)
(.*){L,E,P}(x::RepVec{L},y::OneVec{L,E,P}) = onevec(L,P,x.s*y.s) 
(.*){L}(x::RepVec{L},y::Vector) = chkdim(L,y) && x.s*y

(.*){L}(x::OneVec{L},y::RepVec{L}) = (.*)(y,x)
(.*){L,E,F,P,Q}(x::OneVec{L,E,P},y::OneVec{L,F,Q}) = P==Q?onevec(L,P,x.s*y.s):repvec(L,zero(x.s*y.s))
(.*){L,E,P}(x::OneVec{L,E,P},y::Vector) = chkdim(x,y) && onevec(L,P,x.s*y[P]) 

(.*)(x::Vector,y::RepVec) = (.*)(y,x)
(.*)(x::Vector,y::OneVec) = (.*)(y,x)

###

# (+)

#############################################################################
######################## Custom Matrices ####################################

# Custom matrix
abstract CMat{M,N,E} <: CArray{E}
length{M,N}(::CMat{M,N}) = M*N
size{M,N}(::CMat{M,N}) = (M,N)
size{M,N}(X::CMat{M,N},i::Int) = i<1||i>2?1:(i==1?M:N)
ndims(::CMat) = 2
issquare{M,N}(X::CMat{M,N}) = M==N

typealias _CMat{E,M,N} CMat{M,N,E}
typealias AnyMat{E} Union(_CMat{E},Matrix{E})
typealias AnyVecOrMat{E} Union(AnyMat{E},AnyVec{E})

chkdim(A::AnyMat,B::AnyMat) = size(A,2)!=size(B,1)?error("dimension mismatch"):true
chkdim(a::AnyVec,B::AnyMat) = length(a)!=size(B,1)?error("dimension mismatch"):true
chkdim(A::AnyMat,b::AnyVec) = size(A,2)!=length(b)?error("dimension mismatch"):true


###
immutable RankOne{M,N,E,V,W} <: CMat{M,N,E}
  col::V
  row::W
end
function rankone{V,W}(col::V,row::W)
  M = length(col); N = length(row)
  if ndims(col)>1 && M == max(size(col)) col = reshape(col,M) end
  if ndims(row)>1 && N == max(size(row)) row = reshape(row,N) end
  if ndims(row)!=1 || ndims(col)!=1 error("illegal argument") end 
  T = promote_type(eltype(col),eltype(row));
  col = converte(T,col)
  row = converte(T,row) 
  return RankOne{M,N,T,typeof(col),typeof(row)}(col,row)
end


isdense(R::RankOne) = isdense(R.col) && isdense(R.row)
converte{T,M,N,E}(R::RankOne{M,N,E}) = rankone(converte(T,R.col),converte(T,R.row))

ctranspose{R<:Real}(M::RankOne{R}) = transpose(M)

rowmat{E}(row::CVec{E}) = rankone(repvec(1,one(E)),row)
rowmat(row::Vector) = reshape(row,1,length(row))
colmat{E}(col::CVec{E}) = rankone(col,repvec(1,one(E)))
colmat(col::Vector) = reshape(col,length(col),1)

reprow{E}(m::Int,row::AnyVec{E}) = rankone(repvec(m,one(E)),row)
reprow(m::Int,row::Matrix) = length(row)!=max(size(row))?error("matrix too fat"):reprow(m,vec(row))

repcol{E}(col::AnyVec{E},n::Int) = rankone(col,repvec(n,one(E)))
repcol(col::Matrix,n::Int) = length(col)!=max(size(col))?error("matrix too fat"):repcol(vec(col),n)

repel(m::Int,n::Int, e::Number) = rankone(repvec(m,e),repvec(n,one(e)))

onemat(m::Int, n::Int, e::Number) = rankone(onevec(m,))

sum(A::RankOne) = sum(A.col)*sum(A.row)
function sum{M,N}(A::RankOne,i::Int)
    if i==1 return reshape(sum(A.col)*A.row,1,N) end
    if i==2 return reshape(A.col*sum(A.row),M,1) end
    return full(A)
end  


custom_update!{E<:BlasFloat,M,N,V<:Vector}(D::Matrix{E},R::RankOne{M,N,E,V,V}) = 
    BlasX.ger!(D,one(E),R.col,R.row)

function custom_update!(D::Matrix,R::RankOne)
    col = R.col; row = R.row
    ii = nzindexrange(col); jj = nzindexrange(row)
    for j in jj
      rj = row[j] 
 	    for i in ii D[i,j] = D[i,j] + col[i]*rj end
    end
    return D
end


A_mul_Bt(col::CVec,row::CVec) = rankone(col,row)
A_mul_Bt(col::CVec,row::Vector) = rankone(col,row)
A_mul_Bt(col::Vector,row::CVec) = rankone(col,row)

A_mul_Bc{R<:Real}(col::CVec,row::AnyVec{R}) = rankone(col,row)
A_mul_Bc{R<:Real}(col::Vector,row::CVec{R}) = rankone(col,row)

transpose(M::RankOne) = rankone(M.row,M.col)
transpose(col::CVec) = rowmat(col)

*(R::RankOne,v::AnyVec) = R.col*dot(R.row,v)
At_mul_B(v::AnyVec,R::RankOne) = rankone(repvec(1,dot(v,R.col)),R.row)
Ac_mul_B{R<:Real}(v::AnyVec{R},B::RankOne) = rankone(repvec(1,dot(v,B.col)),B.row)

*(A::RankOne,B::RankOne) = rankone(A.col,dot(A.row,B.col)*B.row)
*(A::RankOne,B::Matrix) = rankone(A.col,A.row.'*B)
*(A::RankOne,s::Number) = rankone(A.col,A.row*s)

*(s::Number,B::RankOne) = rankone(s*B.col,B.row)
*(A::Matrix,B::RankOne) = rankone(A*B.col,B.row)

trace{N}(A::RankOne{N,N}) = dot(A.col,A,row)



#(+)(A::RankOne,B::RankOne) #can give rankone if columns or rows are shared 



###

immutable DiagMat{N,E,V} <: CMat{N,N,E}
  d::V
end
diagmat{E<:Number}(n::Int,s::E=1.0) = ( d = repvec(n,s); DiagMat{n,E,typeof(d)}(d) ) 
diagmat{E}(v::Vector{E}) = DiagMat{length(v),E,Vector{E}}(v) 
diagmat{E}(v::Matrix{E}) = (
    L = length(v);
    if L != max(size(v)) error("bad argument") end;
    DiagMat{length(v),E,Vector{E}}(reshape(v,L)) 
  )

converte{T,N,E}(R::DiagMat{N,E}) = diagmat(converte(T,R.d))
isdense(::DiagMat) = false

transpose(A::DiagMat) = A
sum(A::DiagMat) = sum(A.d) 
sum{N}(A::DiagMat{N},i::Int) = 1<=i<=2?(i=1?reshape(diag(A),N,1):reshape(diag(A),1,N)):full(A) 
diag(A::DiagMat) = full(A.d) 

function custom_update!(D::Matrix,R::DiagMat)
    dg = R.d;
    ii = nzindexrange(dg)
    for i in ii
      D[i,i] = D[i,i] + dg[i] 
    end
    return D
end


*(A::DiagMat,B::Matrix) = scale(full(A.d),B)
*(A::DiagMat,B::DiagMat) = diagmat(A.d .* B.d)
*{L}(A::DiagMat{L},b::AnyVec{L}) = full(A.d).*full(b)
*{N}(A::DiagMat{N},B::RankOne{N}) = rankone(A*B.col,B.row)

*(A::Matrix,B::DiagMat) = scale(A,full(B.d))
*{M,N}(A::RankOne{M,N},B::DiagMat{N}) = rankone(A.col,B*A.row)

##


###

immutable ZeroMat{M,N} <: CMat{M,N,Int}
end
isdense(::ZeroMat) = false
zeromat(m::Int,n::Int) = ZeroMat{m,n}()
custom_update!(D::Array,S::ZeroMat) = D

function aszeros(X)
  if ndims(X)==0 return 0                   end
  if ndims(X)==1 return zerovec(length(X))  end
  if ndims(X)==2 return zeromat(size(X)...) end 
  error("illegal ndims(X)==$(ndims(X)), must be 0, 1, or 2")  
end
###


###
immutable ZeroPaddedMat{M,N,E,D,A} <: CMat{M,N,E}
  data::D
  at:: A
end
ZeroPaddedMat{D,A}(M::Int,N::Int,data::D,at::A) = ZeroPaddedMat{M,N,eltype(data),D,A}(data,at)
zeropad(source::Matrix,at...) = zeropad(size(source),source[at...],at...)
zeropad(sz::(Int,Int),data,at...) = ZeroPaddedMat(sz...,data,at)

isdense(::ZeroPaddedMat) = false

# default: Should work for all types of indexing, but slightly inefficient because RHS 
#          temporary is created and then copied.
custom_update!(D::Array,S::ZeroPaddedMat) = (D[S.at...] += S.data; D)

function custom_update!{M,N,E,D}(Dest::Matrix{E},
                                         S::ZeroPaddedMat{M,N,E,D,(Int,Int)})
  Dest[S.at...] += S.data[1]
  return Dest  
end

function custom_update!{M,N,E,D<:Vector,R<:Ranges}(Dest::Matrix{E},
                                                   S::ZeroPaddedMat{M,N,E,D,(R,Int)})
    (ii,j) = S.at; v = S.data; k=0
    for i in ii Dest[i,j] += v[k+=1] end
    return Dest  
end

function custom_update!{M,N,E,D<:Matrix,R<:Ranges}(Dest::Matrix{E},
                                                   S::ZeroPaddedMat{M,N,E,D,(Int,R)})
    (i,jj) = S.at; v = S.data; k = 0
    @assert length(v) == size(v,2)
    for j in jj Dest[i,j] += v[k+=1] end
    return Dest  
end

function custom_update!{M,N,E,D<:Vector,R<:Ranges,S<:Ranges}(Dest::Matrix{E},
                                                             S::ZeroPaddedMat{M,N,E,D,(R,S)})
    (ii,jj) = S.at; X = S.data; n = 0
    for j in jj
        n+=1; m=0
        for i in ii Dest[i,j] += X[m+=1,n] end
    end
    return Dest  
end



###



end