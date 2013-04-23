module DualNumbers

importall Base

export DualNum,du,dualnum,spart,dpart,dual2complex,complex2dual


FloatScalar = Union(Float64, Complex{Float64}, Float32, Complex{Float32})
FloatVector = Union(Array{Float64,1}, Array{Complex{Float64,1}}, Array{Float32,1}, Array{Complex{Float32,1}})
FloatMatrix = Union(Array{Float64,2}, Array{Complex{Float64,2}}, Array{Float32,2}, Array{Complex{Float32,2}}, FloatVector)
FloatComponent = Union{FloatScalar, FloatVector, FloatMatrix}


immutable DualNum{T<:FloatComponent} 
  st::T # standard part
  di::T # differential part
  DualNum(s::T,d::T)=(for i=1:2;assert(size(s,i)==size(d,i);end;new(s,d))
end

DualNum(s::FloatComponent,d::FloatComponent) = DualNum(promote(s,d)...) # make sure parts match
dualnum(s::FloatComponent,d::FloatComponent) = DualNum(s,d)
DualNum(s::FloatComponent)=DualNum(s,zero(s)) # convenience constructor---omitted differential assumed zero
dualnum(n::FloatComponent) = DualNum(n)

spart(n::DualNum) = n.st
dpart(n::DualNum) = n.di


# to and from Complex types, to facilitate comparison with complex step differentiation
dual2complex{T<:FloatingPoint}(x::DualNum{T}) = complex(x.st,x.di*1e-20)  
complex2dual{T<:FloatingPoint}(z::Complex{T}) = dualnum(real(z),imag(z)*1e20)

# show needs to print both matrices row by row, next to each other

# return DualNum representing 0 or 1 of same flavour as x 
zero{R}(x::DualNum{R})=DualNum{R}(zero(R),zero(R)) 
one{R}(x::DualNum{R})=DualNum{R}(one(R),zero(R)) 
zero{R}(::Type(DualNum{R}))=DualNum{R}(zero(R),zero(R)) 
one{R}(::Type(DualNum{R}))=DualNum{R}(one(R),zero(R)) 



# trivial conversion
convert{T<:FloatComponent}(::Type{DualNum{T}}, z::DualNum{T}) = z 
# conversion from one DualNum flavour to another
convert{T<:FloatComponent}(::Type{DualNum{T}}, z::DualNum) = DualNum{T}(convert(T,z.st),convert(T,z.di))
# conversion from non-Dual to DualNum
convert{T<:FloatComponent}(::Type{DualNum{T}}, x::FloatComponent) = DualNum{T}(convert(T,x), zero(x))
# reverse conversion
convert{T<:FloatComponent}(::Type{T},::DualNum) = (Error("can't convert from DualNum to $T"))


promote_rule{T<:FloatComponent}(::Type{DualNum{T}}, ::Type{T}) = DualNum{T}
promote_rule{T<:FloatComponent,S<:FloatComponent}(::Type{DualNum{T}}, ::Type{S}) =
    DualNum{promote_type(T,S)}
promote_rule{T<:FloatComponent,S<:FloatComponent}(::Type{DualNum{T}}, ::Type{DualNum{S}}) =
    DualNum{promote_type(T,S)}



#unary plus and minus
+(x::DualNum) = X
-(x::DualNum) = DualNum(-x.st,-x.di)

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

#/(a::DualNum,z::Complex) = invoke(/,(DualNum,FloatComponent),a,z)  # to handle ambiguity elsewhere
/(a::DualNum,b::DualNum) = (y=a.st/b.st;DualNum(y, (a.di - y*b.di)/b.st))
/(a::DualNum,b::FloatComponent) = (y=a.st/b;DualNum(y, a.di /b))
/(a::FloatComponent,b::DualNum) = (y=a/b.st;DualNum(y, - y*b.di/b.st))

\(a::DualNum,b::DualNum) = (y=a.st\b.st;DualNum(y, a.st\(b.di - a.di*y)))
\(a::DualNum,b::FloatComponent) = (y=a.st\b;DualNum(y, -a.st\a.di*y))
\(a::FloatComponent,b::DualNum) = (y=a\b.st;DualNum(y, a.st\b.di))



./(a::DualNum,b::DualNum) = (y=a.st./b.st;DualNum(y, (a.di - y.*b.di)./b.st))
./(a::DualNum,b::FloatComponent) = (y=a.st./b;DualNum(y, a.di./b))
./(a::FloatComponent,b::DualNum) = (y=a./b.st;DualNum(y, - y.*b.di./b.st))

#^(a::DualNum,z::Rational) = invoke(^,(DualNum,FloatComponent),a,z)  # to handle ambiguity elsewhere
#^(a::DualNum,z::Integer) = invoke(^,(DualNum,FloatComponent),a,z)  # to handle ambiguity elsewhere
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

function ^{A<:FloatScalar,B<:FloatScalar}(a::DualNum{A},b::DualNum{B}) = a.^b
function ^{A<:FloatScalar,B<:FloatScalar}(a::DualNum{A},b::B) = a.^b
function ^{A<:FloatScalar,B<:FloatScalar}(a::A,b::DualNum{B}) = a.^b

const du = DualNum(0.0,1.0) # differential unit


######## Function Library #######################
log(x::DualNum) = DualNum(log(x.st),x.di./x.st)
exp(x::DualNum) = (y=exp(x.st);DualNum(y,x.di.*y))

sin(x::DualNum) = DualNum(sin(x.st),x.di.*cos(x.st))
cos(x::DualNum) = DualNum(cos(x.st),-x.di.*sin(x.st))
tan(x::DualNum) = (y=tan(x.st);DualNum(y,x.di.*(1+y.^2)))

sqrt(x::DualNum) = (y=sqrt(x);DualNum(y,0.5./y))


end # DualNumbers