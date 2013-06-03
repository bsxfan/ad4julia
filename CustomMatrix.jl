module CustomMatrix
# Lightweight (memory and CPU) representations for various special vectors and matrices.

importall Base

export onevec, repvec, wrap,
       rankone, rowmat, colmat, reprows, repcols,
       diagmat,
       blocksparse,
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
                                isdense(X)?Array(E,size(X)):zeros(E,size(X)),
                                X)  

# Can create new D to allow conversion --- use the returned D
function update!{E<:Number,F,G}(d::E,D::Matrix{F},R)
    # Check size and do conversions here, 
    # Then defer to specific implementations of custom_update()
    G = eltype(R)
    T = promote_type(E,F,G)
    if !accepts(D,eltype(R))
      D = convert(Array{T},D)
      d = convert(T,d)
    else
      d = convert(F,d)
    end
    if isa(R,Matrix) && 
    return custom_update!(d,D,R) 
end


function full_update(d::Number, D::Matrix, S::Matrix)
    M,N = size(D)
    for j=1:N, i=1:M
      D[i,j] = d*D[i,j] + S[i,j]
    end
    return D
end

function update_broadcast2d(d::Number, D::Matrix, s::Number)
    M,N = size(D)
    for j=1:N, i=1:M
      D[i,j] = d*D[i,j] + s
    end
    return D
end

function update_broadcast_col(d::Number, D::Matrix, s::Array)
    @assert size(s,1) == length(s)
    M,N = size(D)
    for j=1:N, i=1:M
      D[i,j] = d*D[i,j] + s[i]
    end
    return D
end


function update_broadcast_row(d::Number, D::Matrix, s::Array)
    @assert size(s,2) == length(s)
    M,N = size(D)
    for j=1:N
      sj = s[j]
      for i=1:M
          D[i,j] = d*D[i,j] + s[i]
      end
    end
    return D
end


#############################################################################
###################   Custom Vectors ########################################

# Custom Vector
abstract CVec{L,E} <: CArray{E}
length{L}(::CVec{L}) = L
size{L}(::CVec{L}) = (L,)
size{L}(::CVec{L},i::Int) = i==1?L:1
ndims(::CVec) = 1
wrap(v::CVec) = v

typealias _CVec{E,L} CVec{L,E}
typealias AnyVec{E} Union(_CVec{E},Vector{E})

chkdim(a::AnyVec,b::AnyVec) = length(a)!=length(b)?error("dimension mismatch"):true
chkdim(a::AnyVec,L::Int) = length(a)!=L?error("dimension mismatch"):true
chkdim(L::Int,b::AnyVec) = L!=length(b)?error("dimension mismatch"):true
chkdim(K::Int,L::Int) = K!=L?error("dimension mismatch"):true

###

# wrapped vector, sibling of special vectors
immutable WVec{L,E} <: CVec{L,E}
   v:: Vector{E}
end
wrap{E}(v::Vector{E}) = WVec{length(v),E}(v)
wrap(v::Matrix) = size(v,1)==1||size(v,2)==1?wrap(vec(v)):error("argument must have single row or column")
full(v::WVec) = v.v
isdense(::WVec) = true
converte{L,E,F}(::Type{E},v::WVec{L,F}) = WVec{L,E}(convert(Vector{E},v.v))
*(x::Number,v::WVec) = wrap(x*v.v)
*(v::WVec,x::Number) = wrap(x*v.v)

nzindexrange{L}(v::WVec{L}) = 1:L
getindex(v::WVec,ii...) = getindex(v.v,ii...) # valid only for nzindexrange

sum(v::WVec) = sum(v.v)

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

sum(v::WVec) = v.s

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

sum{L}(v::WVec{L}) = v.s*L

####
typealias _WVec{E,L} WVec{L,E}
typealias _RepVec{E,L} RepVec{L,E}
typealias DenseVec{E} Union(_WVec{E},_RepVec{E},Vector{E}) # [] can be used after size checks


###

At_mul_B(x::CVec,y::CVec) = [dot(x,y)]
At_mul_B{L}(x::WVec{L},B::Matrix) = reshape(B.'*x.v,1,L)
At_mul_B(x::RepVec,B::Matrix) = x.s*sum(B,1)
At_mul_B{L,E,P}(x::OneVec{L,E,P},B::Matrix) = chkdim(x,B) && x.s*B[P,:]

*(M::Matrix,x::WVec) = M*x.v
*{L}(M::Matrix,x::RepVec{L}) = reshape(sum(M,2),L)
*{L,E,P}(M::Matrix,x::OneVec{L,E,P}) =  chkdim(M,x) && x.s*M[:,P]

###

dot{L}(x::WVec{L},y::WVec{L}) = dot(x.v,y.v)
dot{L}(x::WVec{L},y::RepVec{L}) = sum(x.v)*y.s
dot{L,E,P}(x::WVec{L},y::OneVec{L,E,P}) = x.v[P]*y.s
dot(x::WVec,y::Vector) = dot(x.v,y)

dot{L}(x::RepVec{L},y::WVec{L}) = dot(y,x)
dot{L}(x::RepVec{L},y::RepVec{L}) = L*x.s*y.s
dot{L}(x::RepVec{L},y::OneVec{L}) = x.s*y.s
dot{L}(x::RepVec{L},y::Vector) = chkdim(x,y) && x.s*sum(y)

dot{L}(x::OneVec{L},y::WVec{L}) = dot(y,x)
dot{L}(x::OneVec{L},y::RepVec{L}) = dot(y,x)
dot{L,E,F,P,Q}(x::OneVec{L,E,P},y::OneVec{L,F,Q}) = P==Q?x.s*y.s:zero(x.s*y.s)
dot{L,E,P}(x::OneVec{L,E,P},y::Vector) = chkdim(x,y) && x.s*y[P]

dot(x::Vector,y::WVec) = dot(y,x)
dot(x::Vector,y::RepVec) = dot(y,x)
dot(x::Vector,y::OneVec) = dot(y,x)
###

(.*){L}(x::WVec{L},y::WVec{L}) = full(x) .* full(y) 
(.*){L}(x::WVec{L},y::RepVec{L}) = x * y.s
(.*){L,E,P}(x::WVec{L},y::OneVec{L,E,P}) = onevec(L,P,x[P]*y.s)
(.*)(x::WVec,y::Vector) = full(x) .* y

(.*){L}(x::RepVec{L},y::WVec{L}) = .*(y,x) 
(.*){L}(x::RepVec{L},y::RepVec{L}) = repvec(L,x.s*y.s)
(.*){L,E,P}(x::RepVec{L},y::OneVec{L,E,P}) = onevec(L,P,x.s*y.s) 
(.*){L}(x::RepVec{L},y::Vector) = chkdim(L,y) && x.s*y

(.*){L}(x::OneVec{L},y::WVec{L}) = (.*)(y,x)
(.*){L}(x::OneVec{L},y::RepVec{L}) = (.*)(y,x)
(.*){L,E,F,P,Q}(x::OneVec{L,E,P},y::OneVec{L,F,Q}) = P==Q?onevec(L,P,x.s*y.s):repvec(L,zero(x.s*y.s))
(.*){L,E,P}(x::OneVec{L,E,P},y::Vector) = chkdim(x,y) && onevec(L,P,x.s*y[P]) 

(.*)(x::Vector,y::WVec) = (.*)(y,x) 
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
size{M,N}(X::CMat{M,N},i::Int) = i<1||i>2?1:(i==1:M:N)
ndims(::CMat) = 2
issquare{M,N}(X::CMat{M,N}) = M==N

typealias _CMat{E,M,N} CMat{M,N,E}
typealias AnyMat{E} Union(_CMat{E},Matrix{E})
typealias AnyVecOrMat{E} Union(AnyMat{E},AnyVec{E})

chkdim(A::AnyMat,B::AnyMat) = size(A,2)!=size(B,1)?error("dimension mismatch"):true
chkdim(a::AnyVec,B::AnyMat) = length(a)!=size(B,1)?error("dimension mismatch"):true
chkdim(A::AnyMat,b::AnyVec) = size(A,2)!=length(b)?error("dimension mismatch"):true


###
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
rankone(col::AnyVecOrMat,row::AnyVecOrMat) = rankone(wrap(col),wrap(row))

isdense(R::RankOne) = isdense(R.col) && isdense(R.row)
converte{T,M,N,E}(R::RankOne{M,N,E}) = rankone(converte(T,R.col),converte(T,R.row))

ctranspose{R<:Real}(M::RankOne{R}) = transpose(M)

rowmat{E}(row::AnyVec{E}) = rankone(repvec(1,one(E)),row)
colmat{E}(col::AnyVec{E}) = rankone(col,repvec(1,one(E)))
reprows{E}(m::Int,row::AnyVec{E}) = rankone(repvec(m,one(E)),row)
repcols{E}(col::AnyVec{E},n::Int) = rankone(col,repvec(n,one(E)))

sum(A::RankOne) = sum(A.col)*sum(A.row)
function sum{M,N}(A::RankOne,i::Int)
    if i==1 return reshape(sum(A.col)*A.row,1,N) end
    if i==2 return reshape(A.col*sum(A.row),M,1) end
    return full(A)
end  




function custom_update!(d::Number,D::Matrix,R::RankOne)
    # t.b.d.: some cases could be deferred to BlasX.ger
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

immutable DiagMat{N,E} <: CMat{N,N,E}
  d::CVec{N,E} 
end
diagmat{E}(v::Vector{E}) = DiagMat{length(v),E}(wrap(v)) 
diagmat{E<:Number}(n::Int,s::E=1.0) = DiagMat{n,E}(repvec(n,s)) 

converte{T,N,E}(R::DiagMat{N,E}) = diagmat(converte(T,R.d))
isdense(::DiagMat) = false

transpose(A::DiagMat) = A
sum(A::DiagMat) = sum(A.d) 
sum{N}(A::DiagMat{N},i::Int) = 1<=i<=2?(i=1?reshape(diag(A),N,1):reshape(diag(A),1,N)):full(A) 
diag(A::DiagMat) = full(A.d) 

function custom_update!(d::Number,D::Matrix,R::DiagMat)
    dg = R.d;
    ii = nzindexrange(dg)
    for i in ii
      D[i,i] = d*D[i,i] + dg[i] 
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

immutable BlockSparse{M,N,E,P,Q,R,S} <: CMat{M,N,E}
    # block lives in M,N matrix at P<=i<=R && Q<=j<=S
    block::Matrix{E}
end
typealias IRC Union(Int,Range1,Colon)
function blocksparse{E}(sz::(Int,Int),at::(IRC,IRC),block::Matrix{E}) 
  at = map( i->isa(at[i],Colon)?(1:sz[i]):at[i], (1,2) ) #expand colons to ranges
  if map(length,at) != size(block) error("block does not fit") end 
  M,N = sz; P,Q = map(first,at); R,S = map(last,at) 
  if P<1 || Q<1 || R>M || S>N error("index out of range") end 
  if P==R && Q==1 && S==N return rankone(onevec(M,P),block) end  # single non-zero row
  if Q==S && P==1 && R==M return rankone(block,onevec(N,Q)) end # single non-zero column
  return BlockSparse{M,N,E,P,Q,R,S}(block)
end

transpose{M,N,E,P,Q,R,S}(B::BlockSparse{M,N,E,P,Q,R,S}) = BlockSparse{N,M,E,Q,P,S,R}(B.block.')
sum(B::BlockSparse) = sum(B.block)
function sum{M,N,E,P,Q,R,S}(B::BlockSparse{M,N,E,P,Q,R,S},i::Int) 
    if i==1 return full(BlockSparse{1,N,E,1,Q,1,S}(sum(B.block,i))) end
    if i==2 return full(BlockSparse{M,1,E,P,1,R,1}(sum(B.block,i))) end
end

isdense(::BlockSparse) = false

getindex{M,N,E,P,Q,R,S}(B::BlockSparse{M,N,E,P,Q,R,S},i::Int,j::Int) =
  (P<=i<=R && Q<=j<=S) ? B.block[i-P+1,j-Q+1] : 
  (1<=i<=M&&1<=j<=N)   ? zero(E) : 
                         error("index out of bounds") 

function custom_update!{M,N,E,P,Q,R,S}(d::Number,D::Matrix,B::BlockSparse{M,N,E,P,Q,R,S})
    block = B.block;
    ii = P:R; jj = Q:S; P1 = P-1; Q1 = Q-1;
    for j in jj, i in ii
      D[i,j] = d*D[i,j] + block[i-P1,j-Q1] 
    end
    return D
end



end