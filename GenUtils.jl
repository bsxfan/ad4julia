module GenUtils

importall Base

export eq, 
       diagonal, Centering, 
       check, prevent,
       argumentsmatch,
       @elapsedloop


######### Patches #####################################
# https://github.com/JuliaLang/julia/issues/3202
convert(::Type{Rational},x::Integer) = convert(Rational{typeof(x)},x)

#######################################################
abstract TagType 
# Below we will overload some operators to accept TagTypes as arguments, to do new things.


#######################################################
#base has: |(x,f::Function) = f(x)
|(args::NTuple, f::Function) = f(args...)

########################################################

eq(m::Int,n::Int) = m==n?n:error("dimension mismatch")
eq{N}(sz1::NTuple{N,Int},sz2::NTuple{N,Int}) = sz1==sz2?sz1:error("dimension mismatch")

############### precondition checking #######################
check(ok::Bool, msg="assertion failed") = ok?true:error(msg)
prevent(notok::Bool, msg="assertion failed") = notok?error(msg):true
# Follow check or prevent by &&
#  - for assignment, use brackets: check(condition)&&(x=5), or check(condition)&&(x=5;true)

# Also provide a user-definable check for user-defined operators. 
# Users can suooly argumentsmatch
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
getindex(A::Array, f::Function) = f(A)
setindex!(A::Array, X, f::Function) = f(A,X)


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


end