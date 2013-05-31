module CustomMatrix
# Lightweight (memory and CPU) representations for various special vectors and matrices.

importall Base

export onevec, repvec, wrap,
       rankone, rowmat, colmat, reprows, repcols,
       diagmat,
       update!

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

# converte: convert element type, trivial case 
converte{E}(::Type{E},v::CArray{E}) = v

#convert is not wired up yet, do when necessary
convert{T<:CArray}(::Type{T},A::T) = A #trivial case


summary(X::CArray) = string(Base.dims2string(size(X))," ",typeof(X)) 
show(io::IO,X::CArray) = ( println(io,summary(X),"->"); show(full(X)) )
full{E}(X::CArray{E}) = update!(0,
                                isdense(X)?zeros(E,size(X)):Array(E,size(X)),
                                X)  

# Can create new D to allow conversion --- use the returned D
function update!{E<:Number,F,G}(d::E,D::Matrix{F},R::CArray{G})
    # Check size and do conversions here, 
    # Then defer to specific implementations of do_update()
    if size(R) != size(D) error("dimension mismatch") end
    T = promote_type(E,F,G)
    d = convert(T,d)
    if (F<:Real && T<:Complex) || (F<:Integer) && T<:Union(Complex,Real,Rational)
      D = convert(Array{T},D)
    end
    return do_update!(d,D,R) 
end



#############################################################################
###################   Custom Vectors ########################################

# Custom Vector
abstract CVec{L,E} <: CArray{E}
length{L}(::CVec{L}) = L
size{L}(::CVec{L}) = (L,)
size{L}(::CVec{L},i::Int) = i==1?L:1
ndims(::CVec) = 1

typealias CVecs{E,L} CVec{L,E}
typealias AnyVec{E} Union(CVecs{E},Vector{E})

###

# wrapped vector, sibling of special vectors
immutable WVec{L,E} <: CVec{L,E}
   v:: Vector{E}
end
wrap{E}(v::Vector{E}) = WVec{length(v),E}(v)
wrap(v::Matrix) = size(v,1)==1||size(v,2)==1?wrap(vec(v)):error("argument must have single row or column")
wrap(v::WVec) = v
full(v::WVec) = v.v
isdense(v::WVec) = true
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
isdense(v::OneVec) = false
*{L}(x::Number,v::OneVec{L}) = onevec(L,x*v.s)
*{L}(v::OneVec{L},x::Number) = onevec(L,x*v.s)

nzindexrange{L,E,P}(v::OneVec{L,E,P}) = P
getindex(v::OneVec,ii...) = v.s # valid only for nzindexrange


###

immutable RepVec{L,E} <: CVec{L,E}
  s::E
end
repvec(len::Int, scale = 1.0) = RepVec{len,typeof(scale)}(scale)

full{L}(v::RepVec{L}) = fill(v.s,L)
isdense(v::RepVec) = true
converte{L,E,F}(::Type{E},v::RepVec{L,F}) = RepVec{L,E}(convert(E,v.s))
*{L}(x::Number,v::RepVec{L}) = repvec(L,x*v.s)
*{L}(v::RepVec{L},x::Number) = repvec(L,x*v.s)

nzindexrange{L}(v::RepVec{L}) = 1:L
getindex(v::RepVec,ii...) = v.s # valid only for nzindexrange

###

At_mul_B(x::CVec,y::CVec) = [dot(x,y)]

dot{L}(x::WVec{L},y::WVec{L}) = dot(x.v,y.v)
dot{L}(x::WVec{L},y::RepVec{L}) = sum(x.v)*y.s
dot{L,E,P}(x::WVec{L},y::OneVec{L,E,P}) = x.v[P]*y.s
dot(x::WVec,y::Vector) = dot(x.v,y)
At_mul_B{L}(x::WVec{L},B::Matrix) = reshape(B.'*x.v,1,L)


dot{L}(x::RepVec{L},y::WVec{L}) = dot(y,x)
dot{L}(x::RepVec{L},y::RepVec{L}) = L*x.s*y.s
dot{L}(x::RepVec{L},y::OneVec{L}) = x.s*y.s
dot{L}(x::RepVec{L},y::Vector) = L==length(y)?x.s*sum(y):error("mismatched dimensions")
At_mul_B(x::RepVec,B::Matrix) = x.s*sum(B,1)

dot{L}(x::OneVec{L},y::WVec{L}) = dot(y,x)
dot{L}(x::OneVec{L},y::RepVec{L}) = dot(y,x)
dot{L,E,F,P,Q}(x::OneVec{L,E,P},y::OneVec{L,F,Q}) = P==Q?x.s*y.s:zero(x.s*y.s)
dot{L,E,P}(x::OneVec{L,E,P},y::Vector) = L==length(y):x.s*y[P]:error("mismatched dimensions")
At_mul_B{L,E,P}(x::OneVec{L,E,P},B::Matrix) = L==size(M,1)?x.s*B[P,:]:error("mismatched dimensions")

dot(x::Vector,y::WVec) = dot(y,x)
dot(x::Vector,y::RepVec) = dot(y,x)
dot(x::Vector,y::OneVec) = dot(y,x)

*(M::Matrix,x::WVec) = M*x.v
*{L}(M::Matrix,x::RepVec{L}) = reshape(sum(M,2),L)
*{L,E,P}(M::Matrix,x::OneVec{L,E,P}) =  L==size(M,2)?x.s*M[:,P]:error("mismatched dimensions")


# (+)

#############################################################################
######################## Custom Matrices ####################################

# Custom matrix
abstract CMat{M,N,E} <: CArray{E}
length{M,N}(::CMat{M,N}) = M*N
size{M,N}(::CMat{M,N}) = (M,N)
size{M,N}(X::CMat{M,N},i::Int) = i<1||i>2?1:(i==1:M:N)
ndims(::CMat) = 2
issquare{M,N}(X::Cmat{M,N}) = M==N

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

isdense(R::RankOne) = isdense(R.col) && isdens(R.row)
converte{T,M,N,E}(R::RankOne{M,N,E}) = rankone(converte(T,R.col),converte(T,R.row))

ctranspose{R<:Real}(M::RankOne{R}) = transpose(M)

rowmat{E}(row::AnyVec{E}) = rankone(repvec(1,one(E)),row)
colmat{E}(col::AnyVec{E}) = rankone(col,repvec(1,one(E)))
reprows{E}(m::Int,row::AnyVec{E}) = rankone(repvec(m,one(E)),row)
repcols{E}(col::AnyVec{E},n::Int) = rankone(col,repvec(n,one(E)))

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

A_mul_Bc{R<:Real}(col::CVec,row::AnyVec{R}) = rankone(col,row)
A_mul_Bc{R<:Real}(col::Vector,row::CVec{R}) = rankone(col,row)

transpose(M::RankOne) = rankone(M.row,M.col)
transpose(col::CVec) = rowmat(col)

*(R::RankOne,v::AllVecs) = R.col*dot(R.row,v)
At_mul_B(v::AllVecs,R::RankOne) = rankone(repvec(1,dot(v,R.col)),R.row)
Ac_mul_B{R<:Real}(v::AllVecs{R},B::RankOne) = rankone(repvec(1,dot(v,B.col)),B.row)

*(A::RankOne,B::RankOne) = rankone(A.col,dot(A.row,B.col)*B.row)
*(A::RankOne,B::Matrix) = rankone(A.col,A.row.'*B)
*(A::RankOne,s::Number) = rankone(A.col,A.row*s)

*(s::Number,B::RankOne) = rankone(s*B.col,B.row)
*(A::Matrix,B::RankOne) = rankone(A*B.col,B.row)


#(+)(A::RankOne,B::RankOne) #can give rankone if columns or rows are shared 



###

immutable DiagMat{N,E} <: CMat{N,N,E}
  d::CVec{N,E} 
end
diagmat{E}(v::Vector{E}) = DiagMat{length(v),E}(wrap(v)) 
diagmat{E<:Number}(n::Int,s::E==1.0) = DiagMat{n,E}(repvec(n,s)) 

converte{T,N,E}(R::DiagMat{N,E}) = diagmat(converte(T,R.d))


function do_update!{N,E}(d::E,D::Matrix{E},R::DiagMat{N,E})
    dg = R.d;
    ii = nzindexrange(dg)
    for i in ii
      # @show D[i,i]
      # @show d*D[i,i]
      # @show 
      D[i,i] = d*D[i,i] + dg[i] 
    end
    return D
end



end