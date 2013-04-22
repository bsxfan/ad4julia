module DualNumbers

importall Base

export DualNumber,du,dual,value,differential


immutable DualNumber{T<:Number} <: Number
  va::T # value part
  di::T # differential part
end

DualNumber(v::Number,d::Number) = DualNumber(promote(v,d)...) # make sure parts match
DualNumber(v::Number)=DualNumber(v,zero(v)) # convenience constructor---omitted differential assumed zero
dual(n::Number) = DualNumber(n)

value(n::DualNumber) = n.va
value(n::Number) = n
differential(n::DualNumber) = n.di
differential(n::Number) = zero(n)


# to and from complex for comparison with complex step differentiation
complex{T<:Real}(x::DualNumber{T}) = Complex{Float64}(convert(Float64,x.va),x.di*1e-20)
dual(z::Complex) = DualNumber{Float64}(real(z),imag(z)*1e20)


show(io::IO,n::DualNumber)=(show(n.va);print(io," | ");print(io,n.di))

# return DualNumber representing 0 or 1 of same flavour as x 
zero{R}(x::DualNumber{R})=DualNumber{R}(zero(R),zero(R)) 
one{R}(x::DualNumber{R})=DualNumber{R}(one(R),zero(R)) 

# trivial conversion
convert{T<:Number}(::Type{DualNumber{T}}, z::DualNumber{T}) = z 
# conversion from one DualNumber flavour to another
convert{T<:Number}(::Type{DualNumber{T}}, z::DualNumber) = DualNumber{T}(convert(T,z.va),convert(T,z.di))
# conversion from non-Dual to DualNumber
convert{T<:Number}(::Type{DualNumber{T}}, x::Number) = DualNumber{T}(convert(T,x), convert(T,0))
# reverse conversion
convert{T<:Number}(::Type{T},::DualNumber) = (Error("can't convert from DualNumber to $T"))


promote_rule{T<:Number}(::Type{DualNumber{T}}, ::Type{T}) = DualNumber{T}
promote_rule{T<:Number,S<:Number}(::Type{DualNumber{T}}, ::Type{S}) =
    DualNumber{promote_type(T,S)}
promote_rule{T<:Number,S<:Number}(::Type{DualNumber{T}}, ::Type{DualNumber{S}}) =
    DualNumber{promote_type(T,S)}



+(x::DualNumber,y::DualNumber) = DualNumber(x.va+y.va, x.di+y.di)
+(x::DualNumber,y::Number) = DualNumber(x.va+y, x.di)
+(x::Number,y::DualNumber) = DualNumber(x+y.va, y.di)

-(x::DualNumber) = DualNumber(-x.va,-x.di)
-(x::DualNumber,y::DualNumber) = DualNumber(x.va-y.va, x.di-y.di)
-(x::DualNumber,y::Number) = DualNumber(x.va-y, x.di)
-(x::Number,y::DualNumber) = DualNumber(x-y.va, -y.di)

*(x::DualNumber,y::DualNumber) = DualNumber(x.va*y.va, x.va*y.di+x.di*y.va)
*(x::DualNumber,y::Number) = DualNumber(x.va*y, x.di*y)
*(x::Number,y::DualNumber) = DualNumber(x*y.va, x*y.di)

/(a::DualNumber,z::Complex) = invoke(/,(DualNumber,Number),a,z)  # to handle ambiguity elsewhere
/(a::DualNumber,b::DualNumber) = (y=a.va/b.va;DualNumber(y, (a.di - y*b.di)/b.va))
/(a::DualNumber,b::Number) = (y=a.va/b;DualNumber(y, a.di /b))
/(a::Number,b::DualNumber) = (y=a/b.va;DualNumber(y, - y*b.di/b.va))


^(a::DualNumber,z::Rational) = invoke(^,(DualNumber,Number),a,z)  # to handle ambiguity elsewhere
^(a::DualNumber,z::Integer) = invoke(^,(DualNumber,Number),a,z)  # to handle ambiguity elsewhere
function ^(a::DualNumber,b::DualNumber)
  y = a.va^b.va
  dyda = b.va*a.va^(b.va-1) # derivative of a^b wrt a
  dydb = y*log(a.va) # derivative of a^b wrt b
  return DualNumber(y, a.di*dyda + b.di*dydb)
end
function ^(a::DualNumber,b::Number)
  y = a.va^b
  dyda = b*a.va^(b-1) # derivative of a^b wrt a
  return DualNumber(y, a.di*dyda )
end
function ^(a::Number,b::DualNumber)
  y = a^b.va
  dydb = y*log(a) # derivative of a^b wrt b
  return DualNumber(y, b.di*dydb)
end

# differential unit
const du = DualNumber(0,1) # differential unit
###################################################
# type DifferentialUnit <: Number end
# const du = DifferentialUnit()
# value(::DifferentialUnit) = 0
# differential(::DifferentialUnit) = 1
# *(::DifferentialUnit,::DifferentialUnit) = 0 
# *(x::Real,::DifferentialUnit) = DualNumber(0.0,x) 
# *(::DifferentialUnit,y::Real) = DualNumber(0.0,x) 
# +(x::Real,::DifferentialUnit) = DualNumber(x,1.0)
# +(::DifferentialUnit,y::Real,) = DualNumber(y,1.0)
# -(x::Real,::DifferentialUnit) = DualNumber(x,-1.0)
# -(::DifferentialUnit,y::Real,) = DualNumber(-y,1.0)
###################################################


######## Function Library #######################
log(x::DualNumber) = DualNumber(log(x.va),x.di/x.va)
exp(x::DualNumber) = (y=exp(x.va);DualNumber(y,x.di*y))

sin(x::DualNumber) = DualNumber(sin(x.va),x.di*cos(x.va))
cos(x::DualNumber) = DualNumber(cos(x.va),-x.di*sin(x.va))
tan(x::DualNumber) = (y=tan(x.va);DualNumber(y,x.di*(1+y^2)))

sqrt(x::DualNumber) = (y=sqrt(x);DualNumber(y,0.5/y)




end #module DualNumbers



