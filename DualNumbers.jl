module DualNumbers
importall Base
export DualNum,du,dualnum,spart,dpart,dual2complex,complex2dual

#some new type sets -- 'Num' should be understood as 'numeric', which can be scalar, vector or matrix
typealias FloatScalar Union(Float64, Complex128, Float32, Complex64)  # identical to Linalg.BlasFloat
typealias FloatVector{T<:FloatScalar} Array{T,1}
typealias FloatMatrix{T<:FloatScalar} Array{T,2}
typealias FloatArray Union(FloatMatrix, FloatVector)
typealias FloatNum Union(FloatScalar, FloatArray)

typealias FixReal Union(Integer,Rational)
typealias FixComplex{T<:FixReal} Complex{T}
typealias FixScalar Union(FixReal,FixComplex)
typealias FixVector{T<:FixScalar} Array{T,1}
typealias FixMatrix{T<:FixScalar} Array{T,2}
typealias FixArray Union(FixMatrix,FixVector)
typealias FixNum Union(FixScalar,FixArray)

typealias Numeric Union(FloatNum,FixNum) 

typealias Scalar Union(FloatScalar,FixScalar)
vec(x::Scalar) = [x]

#some new promotion rules for vectors and matrices
promote_rule{A<:FloatScalar,B<:FloatScalar}(::Type{Vector{A}},::Type{Vector{B}}) = Array{promote_type(A,B),1}
promote_rule{A<:FloatScalar,B<:FloatScalar}(::Type{Matrix{A}},::Type{Vector{B}}) = Array{promote_type(A,B),2}
promote_rule{A<:FloatScalar,B<:FloatScalar}(::Type{Matrix{A}},::Type{Matrix{B}}) = Array{promote_type(A,B),2}
# and an associated conversion so vectors can be promoted to matrices
convert{A<:FloatScalar,B<:FloatScalar}(::Type{Matrix{A}},v::Vector{B}) = reshape(convert(Vector{A},v),length(v),1) 


# This is the main new type declared here. The two components can be scalar, vector or matrix, 
# but must agree in type and shape.
immutable DualNum{T<:FloatNum} 
  st::T # standard part
  di::T # differential part
  function DualNum(s::T,d::T)
    #println("in inner constructor: $T")
    n=ndims(s)
	assert(n==ndims(d)<=2,"dimension mismatch")
	for i=1:n;
	  assert(size(s,i)==size(d,i),"size mismatch in dimension $i")
	end
	return new(s,d)
  end
end
function dualnum{T<:FloatNum}(s::T,d::T) 
  #println("in 1st outer constructor")
  return DualNum{T}(s,d)  # construction given float types that agree
end

function dualnum{S<:Numeric,D<:Numeric}(s::S,d::D) # otherwise convert to floats and force match
  #println("here: $S and $D")
  if S<:FloatNum && S==D
    return dualnum(s,d)
  elseif  S<:FloatNum && D<:FloatNum
    #println("here2: $S and $D")
	s,d = promote(s,d)
	if typeof(s) == typeof(d)
      #println("here3: $(typeof(s)) and $(typeof(d))")
      return dualnum(s,d)
	else
	  error("cannot promote $(typeof(s)) and $(typeof(d)) to same type")
	end
  else
    return dualnum(promote(1.0*s,1.0*d)...) #to Float64 or Complex128 and then match (if neccesary)
  end	
end
dualnum{T<:FloatNum}(s::T) = dualnum(s,zero(s))
dualnum{T<:FixNum}(s::T) = dualnum(1.0*s)


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
zero{T<:FloatScalar}(x::DualNum{T})=dualnum(zero(T),zero(T)) 
one{T<:FloatScalar}(x::DualNum{T})=dualnum(one(T),zero(T)) 
zero{T<:FloatArray}(x::DualNum{T})=dualnum(zero(x.st),zero(x.di))
one{T<:FloatMatrix}(x::DualNum{T})=dualnum(one(x.st),zero(x.di))
zeros{T<:FloatScalar}(::Type{DualNum{T}},ii...) = fill(dualnum(zero(T)),ii...)
ones{T<:FloatScalar}(::Type{DualNum{T}},ii...) = fill(dualnum(one(T)),ii...)
eye{T<:FloatScalar}(::Type{DualNum{T}},ii...) = (I=eye(T,ii...);dualnum(I,zero(I)))


const du = dualnum(0.0,1.0) # differential unit



########## promotion and conversion (may not be used that much if operators do their job) #############
# trivial conversion
convert{T<:FloatNum}(::Type{DualNum{T}}, z::DualNum{T}) = z 
# conversion from one DualNum flavour to another
convert{T<:FloatNum}(::Type{DualNum{T}}, z::DualNum) = dualnum(convert(T,z.st),convert(T,z.di))
# conversion from non-Dual to DualNum
convert{T<:FloatNum}(::Type{DualNum{T}}, x::FloatNum) = dualnum(convert(T,x))
# reverse conversion
convert{T<:FloatNum}(::Type{T},::DualNum) = (Error("can't convert from DualNum to $T"))


promote_rule{T<:FloatNum}(::Type{DualNum{T}}, ::Type{T}) = DualNum{T}
promote_rule{T<:FloatNum,S<:FloatNum}(::Type{DualNum{T}}, ::Type{S}) =
    DualNum{promote_type(T,S)}
promote_rule{T<:FloatNum,S<:FloatNum}(::Type{DualNum{T}}, ::Type{DualNum{S}}) =
    DualNum{promote_type(T,S)}
#####################################################################################################


# some general matrix wiring
length(x::DualNum) = length(x.st)
endof(x::DualNum) = endof(x.st)
size(x::DualNum,ii...) = size(x.st,ii...)
getindex(x::DualNum,ii...) = dualnum(getindex(x.st,ii...),getindex(x.di,ii...)) 
ndims(x::DualNum) = ndims(x.st)	
reshape{T<:FloatArray}(x::DualNum{T},ii...) = dualnum(reshape(x.st,ii...),reshape(x.di,ii...)) 
vec(x::DualNum) = dualnum(vec(x.st),vec(x.di))
==(x::DualNum,y::DualNum) = (x.st==y.st) && (x.di==y.di) 
isequal(x::DualNum,y::DualNum) = isequal(x.st,y.st) && isequal(x.di,y.di) 
copy(x::DualNum) = dualnum(copy(x.st),copy(x.di))

vcat(x::DualNum,y::DualNum) = dualnum([x.st, y.st],[x.di, y.di])
hcat(x::DualNum,y::DualNum) = dualnum([x.st  y.st],[x.di  y.di])



fill!(d::DualNum,s::DualNum) = (fill!(d.st,s.st);fill!(d.di,s.di);d)
fill!(d::DualNum,s::Scalar) = (fill!(d.st,s);fill!(d.di,0);d)
fill{V<:FloatScalar}(v::DualNum{V},ii...) = dualnum(fill(v.st,ii...),fill(v.di,ii...))
fill(v::Scalar,ii...) = dualnum(fill(v,ii...))

setindex!{T1<:FloatArray,T2<:FloatNum}(D::DualNum{T1},S::DualNum{T2},ii...) = 
    (setindex!(D.st,S.st,ii...);setindex!(D.di,S.di,ii...);D)
setindex!{T1<:FloatArray,T2<:Numeric}(D::DualNum{T1},S::T2,ii...) = 
    (setindex!(D.st,S,ii...);setindex!(D.di,0,ii...);D)


#bsxfun



############ operator library ###################

#unary 
+(x::DualNum) = X
-(x::DualNum) = dualnum(-x.st,-x.di)
ctranspose(x::DualNum) = dualnum(x.st',x.di')
transpose(x::DualNum) = dualnum(x.st.',x.di.')

+(x::DualNum,y::DualNum) = dualnum(x.st+y.st, x.di+y.di)
+(x::DualNum,y::FloatNum) = dualnum(x.st+y, x.di)
+(x::FloatNum,y::DualNum) = dualnum(x+y.st, y.di)

-(x::DualNum,y::DualNum) = dualnum(x.st-y.st, x.di-y.di)
-(x::DualNum,y::FloatNum) = dualnum(x.st-y, x.di)
-(x::FloatNum,y::DualNum) = dualnum(x-y.st, -y.di)

.*(x::DualNum,y::DualNum) = dualnum(x.st.*y.st, x.st.*y.di + x.di.*y.st)
.*(x::DualNum,y::FloatNum) = dualnum(x.st.*y, x.di.*y)
.*(x::FloatNum,y::DualNum) = dualnum(x.*y.st, x.*y.di)

*(x::DualNum,y::DualNum) = dualnum(x.st*y.st, x.st*y.di + x.di*y.st)
*(x::DualNum,y::FloatNum) = dualnum(x.st*y, x.di*y)
*(x::FloatNum,y::DualNum) = dualnum(x*y.st, x*y.di)

/(a::DualNum,b::DualNum) = (y=a.st/b.st;dualnum(y, (a.di - y*b.di)/b.st))
/(a::DualNum,b::FloatNum) = (y=a.st/b;dualnum(y, a.di /b))
/(a::FloatNum,b::DualNum) = (y=a/b.st;dualnum(y, - y*b.di/b.st))

\(a::DualNum,b::DualNum) = (y=a.st\b.st;dualnum(y, a.st\(b.di - a.di*y)))
\(a::DualNum,b::FloatNum) = (y=a.st\b;dualnum(y, -a.st\a.di*y))
\(a::FloatNum,b::DualNum) = (y=a\b.st;dualnum(y, a.st\b.di))



./(a::DualNum,b::DualNum) = (y=a.st./b.st;dualnum(y, (a.di - y.*b.di)./b.st))
./(a::DualNum,b::FloatNum) = (y=a.st./b;dualnum(y, a.di./b))
./(a::FloatNum,b::DualNum) = (y=a./b.st;dualnum(y, - y.*b.di./b.st))

function .^(a::DualNum,b::DualNum)
  y = a.st.^b.st
  dyda = b.st.*a.st.^(b.st-1) # derivative of a^b wrt a
  dydb = y.*log(a.st) # derivative of a^b wrt b
  return dualnum(y, a.di.*dyda + b.di.*dydb)
end
function .^(a::DualNum,b::FloatNum)
  y = a.st.^b
  dyda = b.*a.st.^(b-1) # derivative of a^b wrt a
  return dualnum(y, a.di.*dyda )
end
function .^(a::FloatNum,b::DualNum)
  y = a.^b.st
  dydb = y.*log(a) # derivative of a^b wrt b
  return dualnum(y, b.di.*dydb)
end

^{A<:FloatScalar,B<:FloatScalar}(a::DualNum{A},b::DualNum{B}) = a.^b
^{A<:FloatScalar,B<:FloatScalar}(a::DualNum{A},b::B) = a.^b
^{A<:FloatScalar,B<:FloatScalar}(a::A,b::DualNum{B}) = a.^b



######## Matrix Function Library #######################
include("MatrixFunctionLib.jl")

######## Vectorized Scalar Function Library #######################
include("VectorizedScalarFunctionLib.jl")



include("TestTools.jl")


end # DualNumbers



# DualNum = DualNumbers.DualNum
# dualnum = DualNumbers.dualnum
# du = DualNumbers.du
# spart = DualNumbers.spart
# dpart = DualNumbers.dpart
# dual2complex = DualNumbers.dual2complex
# complex2dual = DualNumbers.complex2dual



