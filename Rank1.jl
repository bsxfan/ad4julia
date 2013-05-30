module Rank1
# Lightweight (memory and CPU) representations for rank 1 matrices (column * row).
# Includes special representations for repetitions in one or both dimensions, 
# as well as for single non-zero, row, column or element.  

importall Base

export onevec, onesvec, wrap, rankone

abstract CVec{L,E<:Number}
length{L}(::CVec{L}) = L
size{L}(::CVec{L}) = (L,)
size{L}(::CVec{L},i::Int) = i==1?L:1
ndims(::CVec) = 1
summary(v::CVec) = string(Base.dims2string(size(v))," ",typeof(v)) 
show(io::IO,v::CVec) = ( println(io,summary(v),"->"); show(full(v)) )
converte{L,E}(::Type{E},v::CVec{L,E}) = v
ctranspose{L,E<:Real}(v::CVec{L,E}) = transpose(v)

typealias CVecs{E,L} CVec{L,E}
typealias AllVecs{E} Union(CVecs{E},Vector{E})

##############################################################################
immutable WVec{L,E} <: CVec{L,E}
   v:: Vector{E}
end
wrap{E}(v::Vector{E}) = WVec{length(v),E}(v)
wrap(v::Matrix) = size(v,1)==1||size(v,2)==1?wrap(vec(v)):error("argument must have single row or column")
wrap(v::WVec) = v
full(v::WVec) = v.v
converte{L,E,F}(::Type{E},v::WVec{L,F}) = WVec{L,E}(convert(Vector{E},v.v))
*(x::Number,v::WVec) = wrap(x*v.v)
*(v::WVec,x::Number) = wrap(x*v.v)

typealias WVecs{E,L} WVec{L,E}
typealias DenseVec{E} Union(WVecs{E},Vector{E}) 

##############################################################################

immutable OneVec{L,E,P} <: CVec{L,E}
  s::E
end
onevec(pos::Int, len::Int, scale = 1.0) = OneVec{len,typeof(scale),pos}(scale)
full{L,E,P}(v::OneVec{L,E,P}) = (f = zeros(E,L); f[P] = v.s; f)
converte{L,E,F,P}(::Type{E},v::OneVec{L,F,P}) = OneVec{L,E,P}(convert(E,v.s))
#update!{L}(d,D::Vector,v::OneVec{L})
*{L}(x::Number,v::OneVec{L}) = onevec(L,x*v.s)
*{L}(v::OneVec{L},x::Number) = onevec(L,x*v.s)

##############################################################################

immutable OnesVec{L,E} <: CVec{L,E}
  s::E
end
onesvec(len::Int, scale = 1.0) = OnesVec{len,typeof(scale)}(scale)
full{L}(v::OnesVec{L}) = fill(v.s,L)
converte{L,E,F}(::Type{E},v::OnesVec{L,F}) = OnesVec{L,E}(convert(E,v.s))
*{L}(x::Number,v::OnesVec{L}) = onesvec(L,x*v.s)
*{L}(v::OnesVec{L},x::Number) = onesvec(L,x*v.s)

###########################################################################
dot{L}(x::WVec{L},y::WVec{L}) = dot(x.v,y.v)
dot{L}(x::WVec{L},y::OnesVec{L}) = sum(x.v)*y.s
dot{L,E,P}(x::WVec{L},y::OneVec{L,E,P}) = x.v[P]*y.s
dot(x::WVec,y::Vector) = dot(x.v,y)

dot{L}(x::OnesVec{L},y::WVec{L}) = dot(y,x)
dot{L}(x::OnesVec{L},y::OnesVec{L}) = L*x.s*y.s
dot{L}(x::OnesVec{L},y::OneVec{L}) = x.s*y.s
dot{L}(x::OnesVec{L},y::Vector) = L==length(y):x.s*sum(y):error("mismatched dimensions")

dot{L}(x::OneVec{L},y::WVec{L}) = dot(y,x)
dot{L}(x::OneVec{L},y::OnesVec{L}) = dot(y,x)
dot{L,E,F,P,Q}(x::OneVec{L,E,P},y::OneVec{L,F,Q}) = P==Q?x.s*y.s:zero(x.s*y.s)
dot{L,E,P}(x::OneVec{L,E,P},y::Vector) = L==length(y):x.s*y[P]:error("mismatched dimensions")

dot(x::Vector,y::WVec) = dot(y,x)
dot(x::Vector,y::OnesVec) = dot(y,x)
dot(x::Vector,y::OneVec) = dot(y,x)


# (+)

###########################################################################
immutable RankOne{M,N,E}
  col::CVec{M,E} 
  row::CVec{N,E}
end
function rankone{M,N,E,F}(col::CVec{M,E},row::CVec{N,F})
	T = promote_type(E,F)
	col = converte(T,col)
	row = converte(T,row)
	return RankOne{M,N,T}(col,row)
end
rankone(col::CVec,row::Vector) = rankone(col,wrap(row))
rankone(col::Vector,row::CVec) = rankone(wrap(col),row)

full{M,N}(X::RankOne{M,N}) = reshape(full(X.col),M,1) * reshape(full(X.row),1,N)
summary(X::RankOne) = string(Base.dims2string(size(X))," ",typeof(X)) 
show(io::IO,X::RankOne) = ( println(io,summary(X),"->"); show(full(X)) )
length{M,N}(::RankOne{M,N}) = M*N
size{M,N}(::RankOne{M,N}) = (M,N)
size{M,N}(X::RankOne{M,N},i::Int) = 1<=i<=2?size(X,i):1
ndims(::RankOne) = 2

rowmat(row::AllVecs{E}) = rankone(onesvec(1,one{E}),row)
colmat(col::AllVecs{E}) = rankone(col,onesvec(1,one{E}))
reprow(m::Int,row::AllVecs{E}) = rankone(onesvec(m,one{E}),row)
repcol(col::AllVecs{E},n::Int) = rankone(col,onesvec(n,one{E}))



function update!(d,D::Matrix,col::WVec,row::WVec)
end
#and so on for every combination


A_mul_Bt(col::CVec,row::CVec) = rankone(col,row)
A_mul_Bt(col::CVec,row::Vector) = rankone(col,row)
A_mul_Bt(col::Vector,row::CVec) = rankone(col,row)

A_mul_Bc{R<:Real}(col::CVec,row::CVecs{R}) = rankone(col,row)
A_mul_Bc{R<:Real}(col::Vector,row::CVec{R}) = rankone(col,row)
A_mul_Bc{R<:Real}(col::CVec,row::Vector{R}) = rankone(col,row)

transpose(M::RankOne) = rankone(M.row,M.col)
ctranspose{R<:Real}(M::RankOne{R}) = transpose(M)

transpose(col::CVec) = rowmat(col)
ctranspose{R<:Real}(col::CVecs{R}) = transpose(col)

*(R::RankOne,v::AllVecs) = R.col*dot(R.row,v)
At_mul_B(v::AllVecs,R::RankOne) = rankone(onesvec(1,dot(v,R.col)),R.row)
Ac_mul_B{R<:Real}(v::AllVecs{R},B::RankOne) = rankone(onesvec(1,dot(v,B.col)),B.row)




(+)(A::RankOne,B::RankOne) #can give rankone if columns or rows are shared 







end