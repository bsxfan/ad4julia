module MatrixPromotion

export single,double

# In Julia 0.2.0, arrays are not promoted: promote_type() defaults to typejoin(), 
# which just returns Array, so that convert(promote_type(typeof(A),typeof(B)),A) does nothing when a 
# and B are arrays. This is not always desired. Currenctly it causes for example matrix mulltiplication 
# to be slow for mixed-precision arguments. Forcing a promotion to a common element type prior to matrix 
# arithmetic could speed things up. 
#
# Notes:
# 1. This implementation promotes mixtures of single and double precision to double. By comparison,
#    MATLAB seems to do the opposite before doing matrix arithmetic. This would also be easy to do.
#

importall Base

# The promotion mechanisms below are only defined for real and complex float types, 
# ---could perhaps be generalized to Number? 
Reals = Union(Float32,Float64)
Complexes = Union(Complex64,Complex128)
Floats = Union(Reals,Complexes)

#utility functions
single{F<:Reals}(f::F) = convert(Float32,f)
single{Z<:Complexes}(z::Z) = convert(Complex64,z)
single{F<:Reals,N}(f::Array{F,N}) = convert(Matrix{Float32,N},f)
single{Z<:Complexes,N}(z::Array{Z,N}) = convert(Matrix{Complex64,N},z)
double{F<:Reals}(f::F) = convert(Float64,f)
double{Z<:Complexes}(z::Z) = convert(Complex128,z)
double{F<:Reals,N}(f::Array{F,N}) = convert(Matrix{Float64,N},f)
double{Z<:Complexes,N}(z::Array{Z,N}) = convert(Matrix{Complex128,N},z)

# promotion rules
promote_rule{A<:Floats,B<:Floats}(::Type{Vector{A}},::Type{Vector{B}}) = Array{promote_type(A,B),1}
promote_rule{A<:Floats,B<:Floats}(::Type{Matrix{A}},::Type{Vector{B}}) = Array{promote_type(A,B),2}
promote_rule{A<:Floats,B<:Floats}(::Type{Matrix{A}},::Type{Matrix{B}}) = Array{promote_type(A,B),2}

# conversions
convert{A<:Floats,B<:Floats}(::Type{Matrix{A}},v::Vector{B}) = reshape(convert(v),length(v),1) 
# Convert from matrix to vector is already implemented by vec() and is probably not desired as a conversion
# it would lose matrix structure. 
# convert{A,B}(::Type{Vector{A}},m::Matrix{B}) = reshape(v,length(v))  

# operators
for i in (:*,) #(:*, :/, :\)
  @eval begin
    ($i){A<:Floats}(a::Matrix{A},b::Matrix{A}) = ($i)(a,b)
    ($i){A<:Floats,B<:Floats}(a::Matrix{A},b::Matrix{B}) = ($i)(promote(a,b)...)
    ($i){A<:Floats,B<:Floats}(a::Vector{A},b::Matrix{B}) = ($i)(promote(a,b)...)
    ($i){A<:Floats,B<:Floats}(a::Matrix{A},b::Vector{B}) = ($i)(promote(a,b)...)
  end
end

end