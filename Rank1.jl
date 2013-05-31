module Rank1
# Lightweight (memory and CPU) representations for rank 1 matrices (column * row).
# Includes special representations for repetitions in one or both dimensions, 
# as well as for single non-zero, row, column or element.  

importall Base

export onevec, onesvec, wrap,
       rankone, rowmat, colmat, reprow, repcol,
       update!

# Custom Array
abstract CArray{E<:Number}
ctranspose{E<:Real}(A::CArray{E}) = transpose(A)
ctranspose{E<:Complex}(A::CArray{E}) = transpose(conj(A))
Ac_mul_B{E<:Real}(A::CArray{E},B) = At_mul_B(A,B)
A_mul_Bc{E<:Real}(A,B::CArray{E}) = A_mul_Bt(A,B)
Ac_mul_B{E<:Complex}(A::CArray{E},B) = At_mul_B(conj(A),B)
A_mul_Bc{E<:Complex}(A,B::CArray{E}) = A_mul_Bt(A,conj(B))

# converte: convert element type, trivial case 
converte{E}(::Type{E},v::CArray{E}) = v

summary(X::CArray) = string(Base.dims2string(size(X))," ",typeof(X)) 
show(io::IO,X::CArray) = ( println(io,summary(X),"->"); show(full(X)) )


######################################################################

# Custom Vector
abstract CVec{L,E} <: CArray{E}
length{L}(::CVec{L}) = L
size{L}(::CVec{L}) = (L,)
size{L}(::CVec{L},i::Int) = i==1?L:1
ndims(::CVec) = 1

typealias CVecs{E,L} CVec{L,E}
typealias AllVecs{E} Union(CVecs{E},Vector{E})

###

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

nzindexrange{L}(v::WVec{L}) = 1:L
getindex(v::WVec,ii...) = getindex(v.v,ii...) # valid only for nzindexrange

typealias WVecs{E,L} WVec{L,E}
typealias DenseVec{E} Union(WVecs{E},Vector{E}) 

###

immutable OneVec{L,E,P} <: CVec{L,E}
  s::E
end
onevec(pos::Int, len::Int, scale = 1.0) = OneVec{len,typeof(scale),pos}(scale)
full{L,E,P}(v::OneVec{L,E,P}) = (f = zeros(E,L); f[P] = v.s; f)
converte{L,E,F,P}(::Type{E},v::OneVec{L,F,P}) = OneVec{L,E,P}(convert(E,v.s))
#update!{L}(d,D::Vector,v::OneVec{L})
*{L}(x::Number,v::OneVec{L}) = onevec(L,x*v.s)
*{L}(v::OneVec{L},x::Number) = onevec(L,x*v.s)

nzindexrange{L,E,P}(v::OneVec{L,E,P}) = P
getindex(v::OneVec,ii...) = v.s # valid only for nzindexrange


###

immutable OnesVec{L,E} <: CVec{L,E}
  s::E
end
onesvec(len::Int, scale = 1.0) = OnesVec{len,typeof(scale)}(scale)
full{L}(v::OnesVec{L}) = fill(v.s,L)
converte{L,E,F}(::Type{E},v::OnesVec{L,F}) = OnesVec{L,E}(convert(E,v.s))
*{L}(x::Number,v::OnesVec{L}) = onesvec(L,x*v.s)
*{L}(v::OnesVec{L},x::Number) = onesvec(L,x*v.s)

nzindexrange{L}(v::OnesVec{L}) = 1:L
getindex(v::OnesVec,ii...) = v.s # valid only for nzindexrange

###

At_mul_B(x::CVec,y::CVec) = [dot(x,y)]

dot{L}(x::WVec{L},y::WVec{L}) = dot(x.v,y.v)
dot{L}(x::WVec{L},y::OnesVec{L}) = sum(x.v)*y.s
dot{L,E,P}(x::WVec{L},y::OneVec{L,E,P}) = x.v[P]*y.s
dot(x::WVec,y::Vector) = dot(x.v,y)
At_mul_B{L}(x::WVec{L},B::Matrix) = reshape(B.'*x.v,1,L)


dot{L}(x::OnesVec{L},y::WVec{L}) = dot(y,x)
dot{L}(x::OnesVec{L},y::OnesVec{L}) = L*x.s*y.s
dot{L}(x::OnesVec{L},y::OneVec{L}) = x.s*y.s
dot{L}(x::OnesVec{L},y::Vector) = L==length(y)?x.s*sum(y):error("mismatched dimensions")
At_mul_B(x::OnesVec,B::Matrix) = x.s*sum(B,1)

dot{L}(x::OneVec{L},y::WVec{L}) = dot(y,x)
dot{L}(x::OneVec{L},y::OnesVec{L}) = dot(y,x)
dot{L,E,F,P,Q}(x::OneVec{L,E,P},y::OneVec{L,F,Q}) = P==Q?x.s*y.s:zero(x.s*y.s)
dot{L,E,P}(x::OneVec{L,E,P},y::Vector) = L==length(y):x.s*y[P]:error("mismatched dimensions")
At_mul_B{L,E,P}(x::OneVec{L,E,P},B::Matrix) = L==size(M,1)?x.s*B[P,:]:error("mismatched dimensions")

dot(x::Vector,y::WVec) = dot(y,x)
dot(x::Vector,y::OnesVec) = dot(y,x)
dot(x::Vector,y::OneVec) = dot(y,x)

*(M::Matrix,x::WVec) = M*x.v
*{L}(M::Matrix,x::OnesVec{L}) = reshape(sum(M,2),L)
*{L,E,P}(M::Matrix,x::OneVec{L,E,P}) =  L==size(M,2)?x.s*M[:,P]:error("mismatched dimensions")


# (+)

###########################################################################

# Custom matrix
abstract CMat{M,N,E} <: CArray{E}
length{M,N}(::CMat{M,N}) = M*N
size{M,N}(::CMat{M,N}) = (M,N)
size{M,N}(X::CMat{M,N},i::Int) = i<1||i>2?1:(i==1:M:N)
ndims(::CMat) = 2


immutable RankOne{M,N,E} <: CMat{M,N,E}
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

ctranspose{R<:Real}(M::RankOne{R}) = transpose(M)

full{M,N}(X::RankOne{M,N}) = reshape(full(X.col),M,1) * reshape(full(X.row),1,N)

rowmat{E}(row::AllVecs{E}) = rankone(onesvec(1,one(E)),row)
colmat{E}(col::AllVecs{E}) = rankone(col,onesvec(1,one(E)))
reprow{E}(m::Int,row::AllVecs{E}) = rankone(onesvec(m,one(E)),row)
repcol{E}(col::AllVecs{E},n::Int) = rankone(col,onesvec(n,one(E)))

# Can create new D to allow conversion --- use the returned D
function update!{E<:Number,F,G}(d::E,D::Matrix{F},R::CArray{G})
    if size(R) != size(D) error("dimension mismatch") end
    T = promote_type(E,F,G)
    d = convert(T,d)
    if (F<:Real && T<:Complex) || (F<:Integer) && T<:Union(Complex,Real,Rational)
      D = convert(Array{T},D)
    end
    return do_update!(d,D,R) 
end

function do_update!{M,N,E}(d::E,D::Matrix{E},R::RankOne{M,N,E})
    col = R.col; row = R.row
    ii = nzindexrange(col); jj = nzindexrange(row)
    for j in jj
      rj = row[j] 
 	    for i in ii D[i,j] = d*D[i,j] + col[i]*rj end
    end
    return D
end


A_mul_Bt(col::CVec,row::CVec) = rankone(col,row)
A_mul_Bt(col::CVec,row::Vector) = rankone(col,row)
A_mul_Bt(col::Vector,row::CVec) = rankone(col,row)

A_mul_Bc{R<:Real}(col::CVec,row::CVecs{R}) = rankone(col,row)
A_mul_Bc{R<:Real}(col::Vector,row::CVec{R}) = rankone(col,row)
A_mul_Bc{R<:Real}(col::CVec,row::Vector{R}) = rankone(col,row)

transpose(M::RankOne) = rankone(M.row,M.col)
transpose(col::CVec) = rowmat(col)

*(R::RankOne,v::AllVecs) = R.col*dot(R.row,v)
At_mul_B(v::AllVecs,R::RankOne) = rankone(onesvec(1,dot(v,R.col)),R.row)
Ac_mul_B{R<:Real}(v::AllVecs{R},B::RankOne) = rankone(onesvec(1,dot(v,B.col)),B.row)

*(A::RankOne,B::RankOne) = rankone(A.col,dot(A.row,B.col)*B.row)
*(A::RankOne,B::Matrix) = rankone(A.col,A.row.'*B)
*(A::RankOne,s::Number) = rankone(A.col,A.row*s)

*(s::Number,B::RankOne) = rankone(s*B.col,B.row)
*(A::Matrix,B::RankOne) = rankone(A*B.col,B.row)


#(+)(A::RankOne,B::RankOne) #can give rankone if columns or rows are shared 







end