module DualNumbers

importall Base

#export DualNum,du,dualnum,spart,dpart,dual2complex,complex2dual


FloatScalar = Union(Float64, Complex{Float64}, Float32, Complex{Float32})
FloatVector = Union(Array{Float64,1}, Array{Complex{Float64},1}, Array{Float32,1}, Array{Complex{Float32},1})
# strictly 2-dim matrix
FloatMatrix2 = Union(Array{Float64,2}, Array{Complex{Float64},2}, Array{Float32,2}, Array{Complex{Float32},2})
FloatMatrix = Union(FloatMatrix2, FloatVector)
FloatComponent = Union(FloatScalar, FloatVector, FloatMatrix)


immutable DualNum{T<:FloatComponent} 
  st::T # standard part
  di::T # differential part
  function DualNum(s::T,d::T)
    n=ndims(s)
	assert(n==ndims(d)<=2,"dimension mismatch")
	for i=1:n;
	  assert(size(s,i)==size(d,i),"size mismatch in dimension $i")
	end
	return new(s,d)
  end
  DualNum(s::T) = new(s,zero(s))
  end
DualNum{T<:FloatComponent}(s::T,d::T) = DualNum{T}(s,d)
DualNum{T<:FloatComponent}(s::T) = DualNum{T}(s)

# DualNum(s::FloatComponent,d::FloatComponent) = DualNum(promote(s,d)...) # make sure parts match
dualnum(s::FloatComponent,d::FloatComponent) = DualNum(s,d)
dualnum(n::FloatComponent) = DualNum(n)

spart(n::DualNum) = n.st
dpart(n::DualNum) = n.di


# to and from Complex types, to facilitate comparison with complex step differentiation
dual2complex{T<:FloatingPoint}(x::DualNum{T}) = complex(x.st,x.di*1e-20)  
complex2dual{T<:FloatingPoint}(z::Complex{T}) = dualnum(real(z),imag(z)*1e20)

# show 
function show(io::IO,x::DualNum)
  print("standard part: ")
  show(io,x.st)  
  print("\ndifferential part: ")
  show(io,x.di)  
end



# some 0's and 1's
zero{T<:FloatScalar}(x::DualNum{T})=DualNum(zero(T),zero(T)) 
one{T<:FloatScalar}(x::DualNum{T})=DualNum(one(T),zero(T)) 
zero{T<:FloatMatrix}(x::DualNum{T})=DualNum(zero(x.st),zero(x.di))
one{T<:FloatMatrix2}(x::DualNum{T})=DualNum(one(x.st),zero(x.di))
zeros{T<:FloatScalar}(::Type{DualNum{T}},ii...) = fill(dualnum(zero(T)),ii...)
ones{T<:FloatScalar}(::Type{DualNum{T}},ii...) = fill(dualnum(one(T)),ii...)
eye{T<:FloatScalar}(::Type{DualNum{T}},ii...) = (I=eye(T,ii...);DualNum(I,zero(I)))


########## promotion and conversion (may not be used that much if operators do their job) #############
# trivial conversion
#convert{T<:FloatComponent}(::Type{DualNum{T}}, z::DualNum{T}) = z 
# conversion from one DualNum flavour to another
#convert{T<:FloatComponent}(::Type{DualNum{T}}, z::DualNum) = DualNum{T}(convert(T,z.st),convert(T,z.di))
# conversion from non-Dual to DualNum
#convert{T<:FloatComponent}(::Type{DualNum{T}}, x::FloatComponent) = DualNum{T}(convert(T,x), zero(x))
# reverse conversion
#convert{T<:FloatComponent}(::Type{T},::DualNum) = (Error("can't convert from DualNum to $T"))


#promote_rule{T<:FloatComponent}(::Type{DualNum{T}}, ::Type{T}) = DualNum{T}
#promote_rule{T<:FloatComponent,S<:FloatComponent}(::Type{DualNum{T}}, ::Type{S}) =
#    DualNum{promote_type(T,S)}
#promote_rule{T<:FloatComponent,S<:FloatComponent}(::Type{DualNum{T}}, ::Type{DualNum{S}}) =
#    DualNum{promote_type(T,S)}
#####################################################################################################


# some general matrix wiring
length(x::DualNum) = length(x.st)
endof(x::DualNum) = endof(x.st)
size(x::DualNum,ii...) = size(x.st,ii...)
getindex(x::DualNum,ii...) = DualNum(getindex(x.st,ii...),getindex(x.di,ii...)) 
ndims(x::DualNum) = ndims(x.st)	
reshape{T<:FloatMatrix}(x::DualNum{T},ii...) = DualNum(reshape(x.st,ii...),reshape(x.di,ii...)) 
vec(x::DualNum) = DualNum(vec(x.st),vec(x.di))
==(x::DualNum,y::DualNum) = (x.st==y.st) && (x.di==y.di) 
isequal(x::DualNum,y::DualNum) = isequal(x.st,y.st) && isequal(x.di,y.di) 
copy(x::DualNum) = DualNum(copy(x.st),copy(x.di))
cat(k::Integer,x::DualNum,y::DualNum) = DualNum(cat(k,x.st,y.st),cat(k,x.di,y.di))
cat(k::Integer,x::DualNum,y::DualNum,z::DualNum) = DualNum(cat(k,x.st,y.st,z.st),cat(k,x.di,y.di,z,di))
fill!{D,S}(d::DualNum{D},s::DualNum{S}) = (fill!(d.st,s,st);fill!(d.di,s.di);d)
fill{V<:FloatScalar}(v::DualNum{V},ii...) = DualNum(fill(v.st,ii...),fill(v.di,ii...))

setindex!{T1<:FloatMatrix,T2<:FloatComponent}(D::DualNum{T1},S::DualNum{T2},ii...) = 
    (setindex!(D.st,S.st,ii...);setindex!(D.di,S.di,ii...);D)


#bsxfun



############ operator library ###################

#unary 
+(x::DualNum) = X
-(x::DualNum) = DualNum(-x.st,-x.di)
ctranspose(x::DualNum) = DualNum(x.st',x.di')
transpose(x::DualNum) = DualNum(x.st.',x.di.')

+(x::DualNum,y::DualNum) = DualNum(x.st+y.st, x.di+y.di)
+(x::DualNum,y::FloatComponent) = DualNum(x.st+y, x.di)
+(x::FloatComponent,y::DualNum) = DualNum(x+y.st, y.di)

-(x::DualNum,y::DualNum) = DualNum(x.st-y.st, x.di-y.di)
-(x::DualNum,y::FloatComponent) = DualNum(x.st-y, x.di)
-(x::FloatComponent,y::DualNum) = DualNum(x-y.st, -y.di)

.*(x::DualNum,y::DualNum) = DualNum(x.st.*y.st, x.st.*y.di + x.di.*y.st)
.*(x::DualNum,y::FloatComponent) = DualNum(x.st.*y, x.di.*y)
.*(x::FloatComponent,y::DualNum) = DualNum(x.*y.st, x.*y.di)

*(x::DualNum,y::DualNum) = DualNum(x.st*y.st, x.st*y.di + x.di*y.st)
*(x::DualNum,y::FloatComponent) = DualNum(x.st*y, x.di*y)
*(x::FloatComponent,y::DualNum) = DualNum(x*y.st, x*y.di)

/(a::DualNum,b::DualNum) = (y=a.st/b.st;DualNum(y, (a.di - y*b.di)/b.st))
/(a::DualNum,b::FloatComponent) = (y=a.st/b;DualNum(y, a.di /b))
/(a::FloatComponent,b::DualNum) = (y=a/b.st;DualNum(y, - y*b.di/b.st))

\(a::DualNum,b::DualNum) = (y=a.st\b.st;DualNum(y, a.st\(b.di - a.di*y)))
\(a::DualNum,b::FloatComponent) = (y=a.st\b;DualNum(y, -a.st\a.di*y))
\(a::FloatComponent,b::DualNum) = (y=a\b.st;DualNum(y, a.st\b.di))



./(a::DualNum,b::DualNum) = (y=a.st./b.st;DualNum(y, (a.di - y.*b.di)./b.st))
./(a::DualNum,b::FloatComponent) = (y=a.st./b;DualNum(y, a.di./b))
./(a::FloatComponent,b::DualNum) = (y=a./b.st;DualNum(y, - y.*b.di./b.st))

function .^(a::DualNum,b::DualNum)
  y = a.st.^b.st
  dyda = b.st.*a.st.^(b.st-1) # derivative of a^b wrt a
  dydb = y.*log(a.st) # derivative of a^b wrt b
  return DualNum(y, a.di.*dyda + b.di.*dydb)
end
function .^(a::DualNum,b::FloatComponent)
  y = a.st.^b
  dyda = b.*a.st.^(b-1) # derivative of a^b wrt a
  return DualNum(y, a.di.*dyda )
end
function .^(a::FloatComponent,b::DualNum)
  y = a.^b.st
  dydb = y.*log(a) # derivative of a^b wrt b
  return DualNum(y, b.di.*dydb)
end

^{A<:FloatScalar,B<:FloatScalar}(a::DualNum{A},b::DualNum{B}) = a.^b
^{A<:FloatScalar,B<:FloatScalar}(a::DualNum{A},b::B) = a.^b
^{A<:FloatScalar,B<:FloatScalar}(a::A,b::DualNum{B}) = a.^b

const du = DualNum(0.0,1.0) # differential unit


######## Function Library #######################
sum(x::DualNum,ii...) = DualNum(sum(x.st,ii...),sum(x.di,ii...))
trace(x::DualNum,ii...) = DualNum(trace(x.st,ii...),trace(x.di,ii...))
diag{T<:FloatMatrix}(x::DualNum{T},k...) = DualNum(diag(x.st,k...),diag(x.di,k...))
diagm{T<:FloatMatrix}(x::DualNum{T},k...) = DualNum(diagm(x.st,k...),diagm(x.di,k...))

diagmm{X<:FloatMatrix2,Y<:FloatVector}(x::DualNum{X},y::DualNum{Y}) = 
    DualNum(diagmm(x.st,y.st),diagmm(x.di,y.st)+diagmm(x.st,y.di))
diagmm{X<:FloatMatrix2,Y<:FloatVector}(x::DualNum{X},y::Y) = 
    DualNum(diagmm(x.st,y),diagmm(x.di,y))
diagmm{X<:FloatMatrix2,Y<:FloatVector}(x::X,y::DualNum{Y}) = 
    DualNum(diagmm(x,y.st),diagmm(x,y.di))

diagmm{X<:FloatVector,Y<:FloatMatrix2}(x::DualNum{X},y::DualNum{Y}) = 
    DualNum(diagmm(x.st,y.st),diagmm(x.di,y.st)+diagmm(x.st,y.di))
diagmm{X<:FloatVector,Y<:FloatMatrix2}(x::DualNum{X},y::Y) = 
    DualNum(diagmm(x.st,y),diagmm(x.di,y))
diagmm{X<:FloatVector,Y<:FloatMatrix2}(x::X,y::DualNum{Y}) = 
    DualNum(diagmm(x,y.st),diagmm(x,y.di))

inv{T<:FloatMatrix2}(x::DualNum{T}) = (y=inv(x.st);DualNum(y,-y*x.di*y))
det{T<:FloatMatrix2}(x::DualNum{T}) = (LU=lufact(x.st);y=det(LU);DualNum(y,y*dot(vec(inv(LU)),vec(x.di.'))))


#chol
function logdet{T}(C::Cholesky{T})
    dd = zero(T)
    for i in 1:size(C.UL,1) dd += log(C.UL[i,i]) end
    2*dd
end

#lu



log(x::DualNum) = DualNum(log(x.st),x.di./x.st)
exp(x::DualNum) = (y=exp(x.st);DualNum(y,x.di.*y))

sin(x::DualNum) = DualNum(sin(x.st),x.di.*cos(x.st))
cos(x::DualNum) = DualNum(cos(x.st),-x.di.*sin(x.st))
tan(x::DualNum) = (y=tan(x.st);DualNum(y,x.di.*(1+y.^2)))

sqrt(x::DualNum) = (y=sqrt(x);DualNum(y,0.5./y))

end # DualNumbers

DualNum = DualNumbers.DualNum
dualnum = DualNumbers.dualnum
du = DualNumbers.du
spart = DualNumbers.spart
dpart = DualNumbers.dpart
dual2complex = DualNumbers.dual2complex
complex2dual = DualNumbers.complex2dual



