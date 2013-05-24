module Rank1

importall Base

export onevec, onesvec, wrap

abstract CVec{L,E<:Number}
length{L}(::CVec{L}) = L
size{L}(::CVec{L}) = (L,)
size{L}(::CVec{L},i::Int) = i==1?L:1
ndims(::CVec) = 1
summary(v::CVec) = string(Base.dims2string(size(v))," ",typeof(v)) 
show(io::IO,v::CVec) = ( println(io,summary(v),"->"); show(full(v)) )
converte{L,E}(::Type{E},v::CVec{L,E}) = v
ctranspose{L,E<:Real}(v::CVec{L,E}) = transpose(v)

immutable FullVec{L,E} <: CVec{L,E}
   v:: Vector{E}
end
wrap{E}(v::Vector{E}) = FullVec{length(v),E}(v)
wrap(v::Matrix) = size(v,1)==1||size(v,2)==1?wrap(vec(v)):error("argument must have single row or column")
full(v::FullVec) = v.v
#convert{L,E,F}(::Type{CVec{L,E}},v::FullVec{L,F}) = FullVec{L,E}(convert(Vector{E,v.v}))
converte{L,E,F}(::Type{E},v::FullVec{L,F}) = FullVec{L,E}(convert(Vector{E},v.v))

immutable OneVec{L,E,P} <: CVec{L,E}
  s::E
end
onevec(pos::Int, len::Int, scale = 1.0) = OneVec{len,typeof(scale),pos}(scale)
full{L,E,P}(v::OneVec{L,E,P}) = (f = zeros(E,L); f[P] = v.s; f)
converte{L,E,F,P}(::Type{E},v::OneVec{L,F,P}) = OneVec{L,E,P}(convert(E,v.s))
#update!{L}(d,D::Vector,v::OneVec{L})


immutable OnesVec{L,E} <: CVec{L,E}
  s::E
end
onesvec(len::Int, scale = 1.0) = OnesVec{len,typeof(scale)}(scale)
full{L}(v::OnesVec{L}) = fill(v.s,L)
converte{L,E,F}(::Type{E},v::OnesVec{L,F}) = OnesVec{L,E}(convert(E,v.s))

###########################################################################
dot{L}(x::FullVec{L},y::FullVec{L}) = dot(x.v,y.v)
dot{L}(x::FullVec{L},y::OnesVec{L}) = sum(x.v)*y.s
dot{L,E,P}(x::FullVec{L},y::OneVec{L,E,P}) = x.v[P]*y.s
dot(x::FullVec,y::Vector) = dot(x.v,y)

dot{L}(x::OnesVec{L},y::FullVec{L}) = dot(y,x)
dot{L}(x::OnesVec{L},y::OnesVec{L}) = L*x.s*y.s
dot{L}(x::OnesVec{L},y::OneVec{L}) = x.s*y.s
dot{L}(x::OnesVec{L},y::Vector) = L==length(y):x.s*sum(y):error("mismatched dimensions")

dot{L}(x::OneVec{L},y::FullVec{L}) = dot(y,x)
dot{L}(x::OneVec{L},y::OnesVec{L}) = dot(y,x)
dot{L,E,F,P,Q}(x::OneVec{L,E,P},y::OneVec{L,F,Q}) = P==Q?x.s*y.s:zero(x.s*y.s)
dot{L,E,P}(x::OneVec{L,E,P},y::Vector) = L==length(y):x.s*y[P]:error("mismatched dimensions")

dot(x::Vector,y::FullVec) = dot(y,x)
dot(x::Vector,y::OnesVec) = dot(y,x)
dot(x::Vector,y::OneVec) = dot(y,x)


# (+)

###########################################################################
immutable RankOne{M,N,E}
  col::CVec{M,E} 
  row::CVec{N,E}
end
function At_mul_B{M,N,E,F}(col::CVec{M,E},row::CVec{N,F})
	T = promote_type(E,F)
	col = converte(T,col)
	row = converte(T,row)
	return RankOne{M,N,T}(col,row)
end
Ac_mul_B{M,E<:Real}(col::CVec{M,E},row::CVec) = At_mul_B(col,row)

full{M,N}(X::RankOne{M,N}) = reshape(full(X.col),M,1) * reshape(full(X.row),1,N)
summary(X::RankOne) = string(Base.dims2string(size(X))," ",typeof(X)) 
show(io::IO,X::RankOne) = ( println(io,summary(X),"->"); show(full(X)) )
length{M,N}(::RankOne{M,N}) = M*N
size{M,N}(::RankOne{M,N}) = (M,N)
size{M,N}(X::RankOne{M,N},i::Int) = 1<=i<=2?size(X,i):1
ndims(::RankOne) = 2




function update!(d,D::Matrix,col::FullVec,row::FullVec)
end
#and so on for every combination

(+)(A::RankOne,B::RankOne) #can give rankone if columns or rows are shared 







end