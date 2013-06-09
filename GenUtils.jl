module GenUtils

importall Base

export eq,eqsize,eqlength, 
       diagonal, Centering, 
       check, prevent,
       argumentsmatch,
       @elapsedloop,
       promote_eltype, accepts, isscalar,
       randw,
       max1,max2, logsumexp

######### Patches #####################################
# fixed: https://github.com/JuliaLang/julia/issues/3202
# convert(::Type{Rational},x::Integer) = convert(Rational{typeof(x)},x)

# fixed: https://github.com/JuliaLang/julia/issues/3246
# import Base.power_by_squaring
# ^{T<:FloatingPoint}(z::Complex{T}, n::Bool) = n ? z : one(z)
# ^{T<:Rational}(z::Complex{T}, n::Bool) = n ? z : one(z)
# ^{T<:Integer}(z::Complex{T}, n::Bool) = n ? z : one(z)

# ^{T<:FloatingPoint}(z::Complex{T}, n::Integer) = n>=0?power_by_squaring(z,n):power_by_squaring(inv(z),-n)
# ^{T<:Rational}(z::Complex{T}, n::Integer) = n>=0?power_by_squaring(z,n):power_by_squaring(inv(z),-n)
# ^{T<:Integer}(z::Complex{T}, n::Integer) = power_by_squaring(z,n) # DomainError for n<0


#######################################################
abstract TagType 
# Below we will overload some operators to accept TagTypes as arguments, to do new things.


#######################################################
#base has: |(x,f::Function) = f(x)
|(args::NTuple, f::Function) = f(args...)

########################################################

eq{T}(msg::String,a1::T,args::T...) = all(map(x->x==a1,args))?a1:error(msg)
eq{N}(sz1::NTuple{N,Int},sizes::NTuple{N,Int}...) = eq("size mismatch",sz1,sizes...)
eq(d1::Int,dims::Int...) = eq("dimension mismatch",d1,dims...)
eqsize(args...) = eq(map(size,args)...)
eqlength(args...) = eq("length mismatch",map(length,args)...)

########################################################

isscalar(X) = ndims(X) == 0

promote_eltype(args::AbstractArray...) = promote_type(map(eltype,args)...)

# predicts which conversions will not throw inexact error or similar
# Note, things like 
typealias IntFlavours{T<:Integer} Union(T,Complex{T})
typealias RatFlavours{T<:Rational} Union(T,Complex{T})
typealias FloatFlavours{T<:FloatingPoint} Union(T,Complex{T})
willconvert{D<:Number,S<:Number}(::Type{D},::Type{S}) = true
willconvert{D<:Real,S<:Complex}(::Type{D},::Type{S}) = false
willconvert{D<:IntFlavours,S<:RatFlavours}(::Type{D},::Type{S}) = false
willconvert{D<:IntFlavours,S<:FloatFlavours}(::Type{D},::Type{S}) = false

accepts{D<:Number,S<:Number}(A::Array{D},::Type{S}) = willconvert(D,S)
accepts{D<:Number,S<:Number}(A::Array{D},::S) = willconvert(D,S)


############### precondition checking #######################
check(ok::Bool, msg="assertion failed") = ok?true:error(msg)
prevent(notok::Bool, msg="assertion failed") = notok?error(msg):true
# Follow check or prevent by &&
#  - for assignment, use brackets: check(condition)&&(x=5), or check(condition)&&(x=5;true)

# Also provide a user-definable check for user-defined operators. 
# Users can supply argumentsmatch
check(op::Function,A,B) = check(argumentsmatch(op,A,B)) 
#default argumentsmatch for matrix arguments
function argumentsmatch(f::Function,A,B)
	if   ( contains({+,-},f)   && size(A) != size(B)     )  ||  
		 ( contains({*,/,\},f) && size(A,2) != size(B,1) ) 
        error("arguments do not match: $(summary(A)) $(f) $(summary(B))")
    else
        return true
    end
end

# just use assert() to check postconditions, or if you want to chain further, use check/prevent
#lest(notok::Function) = x-> notok(x)?error("assertion failed"):x #latin "ne"
#sothat(ok::Function) = x-> ok(x)?x:error("assertion failed") #latin "ut"

########################################################
# getindex(A::Array, f::Function) = f(A)
# setindex!(A::Array, X, f::Function) = f(A,X)


########################################################
immutable diagonal <: TagType
  k::Int 
end

getindex(M::Matrix,::Type{diagonal}) = M[diagind(M)]
setindex!(M::Matrix,X,::Type{diagonal}) = setindex!(M,X,diagind(M))

getindex(M::Matrix,d::diagonal) = M[diagind(M,d.k)]
setindex!(M::Matrix,X,d::diagonal) = setindex!(M,X,diagind(M,d.k))


########################################################

immutable Centering <: TagType
  n::Int
end
summary(C::Centering) = "Centering($(C.n))"
full(C::Centering) = (n = C.n; F = fill(-1/n,n,n); F[diagonal] = (n-1)/n; F )
full(::Type{Rational},C::Centering) = (n = C.n; F = fill(-1//n,n,n); F[diagonal] = (n-1)//n; F )

size(C::Centering) = (C.n,C.n)
size(C::Centering,i::Int) = 1<=i<=2?C.n:1
length(C::Centering) = C.n^2
ndims(C::Centering) = 2
getindex(C::Centering,i::Int,j::Int) = (n=C.n; all(1.<[i,j].<n)?(i==j?(n-1)/n:-1/n):error("index out of bounds") )


*(::Type{Centering},v::Vector) = v - mean(v)
*(::Type{Centering},M::Matrix) = M .- mean(M,1)
*(M::Matrix,::Type{Centering}) = M .- mean(M,2)

*(C::Centering,v::Vector) = check(*,C,v)&& v - mean(v) 
*(C::Centering,M::Matrix) = check(*,C,M)&& M .- mean(M,1) 
*(M::Matrix,C::Centering) = check(*,M,C)&& M .- mean(M,2) 

*(::Type{Centering},::Type{Centering}) = Centering
*(C::Centering,::Type{Centering}) = C
*(::Type{Centering},C::Centering) = C
*(A::Centering,B::Centering) = Centering(eq(A.n,B.n))


########################################################

randw(d,n) = (R = randn(d,n); R*R')
randw(d) = randw(d,d+1)

########################################################

macro elapsedloop(n,ex)
    quote
        local s = 0.0
        for i=1:$(esc(n))
          local t0 = time_ns()
          local val = $(esc(ex))
          s += (time_ns()-t0)/1e9
      end    
      s
    end
end

########################################################

max1{T}(A::Matrix{T})=(
    (m,n) = size(A); @assert m>0 && n>0;
    s = Array(T,n);
    for j=1:n 
        t=A[1,j]
        for i=2:m e = A[i,j]; if e>t t = e end end 
        s[j] = t 
    end; 
    reshape(s,1,n) 
)

max2(A::Matrix)=(
    (m,n)=size(A); @assert m>0 && n>0;
    s = A[:,1];
    for j=2:n, i=1:m 
      si = s[i]; e = A[i,j]
      if e > si s[i] = e end
    end;
    reshape(s,m,1)
)

########################################################


function logsumexp{E}(X::Matrix{E})
# Mathematically the same as y=log(sum(exp(x),1)), 
# but guards against numerical overflow of exp(x).
    m,n = size(X)
    y = Array(E,n)
    for j=1:n
        mx = X[1,j]
        for i=2:m e = X[i,j]; if e>mx mx = e end end
        s = 0 
        for i=1:m s += exp(X[i,j]-mx) end
        y[j] = mx + log(s);
    end
    return reshape(y,1,n)
end





##########################################################



end