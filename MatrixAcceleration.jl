module MatrixAcceleration

importall Base
import Base.LinAlg: BLAS, BlasFloat

using GenUtils

export loopaddarrays, loopaddarrays!,
       sum1,sum2,sum1blas,sum2blas


####################### Bye to goddamn subnormals ##########################

if !ccall(:jl_zero_subnormals, Bool, (Bool,), true)
    error("ccall jl_zero_subnormals failed")
end
assert(exp(-710)==0,"subnormals are unfortunately still with us")
println("hooray: subnormals disabled!")



############ Accelerate A+B+C, A+B+C+D, etc.  ##########################

# These loops are much faster as separate little functions.
loop3(E,n,A,B,C) = for i=1:n E[i]=A[i]+B[i]+C[i] end
loop4(E,n,A,B,C,D) = for i=1:n E[i]=A[i]+B[i]+C[i]+D[i] end
loop5(F,n,A,B,C,D,E) = for i=1:n F[i]=A[i]+B[i]+C[i]+D[i]+E[i] end
const loops = {nothing,nothing,loop3,loop4,loop5}
function loopaddarrays(args::AbstractArray...)
    m = length(args)
    if m==2 return +(args[1],args[2]) end
    T = promote_eltype(args...)    
    sz = eqsize(args...)
    D = Array(T,sz)
    n = prod(sz);
    if 3<=m<=length(loops)
        loops[m](D,n,args...) 
    else
        error("not implemented for $m arguments") 
    end
    return D
end

# install
+(A::Matrix,B::Matrix,C::Matrix) = loopaddarrays(A,B,C)
+(A::Matrix,B::Matrix,C::Matrix,D::Matrix) = loopaddarrays(A,B,C,D)
+(A::Matrix,B::Matrix,C::Matrix,D::Matrix,E::Matrix) = loopaddarrays(A,B,C,D,E)

######################################################################

loop1!(F,n,A) = for i=1:n F[i] += A[i] end
loop2!(F,n,A,B) = for i=1:n F[i] += A[i]+B[i] end
loop3!(F,n,A,B,C) = for i=1:n F[i] += A[i]+B[i]+C[i] end
loop4!(F,n,A,B,C,D) = for i=1:n F[i] += A[i]+B[i]+C[i]+D[i] end
loop5!(F,n,A,B,C,D,E) = for i=1:n F[i] += A[i]+B[i]+C[i]+D[i]+E[i] end
const loops! = {loop1!,loop2!,loop3!,loop4!,loop5!}
function loopaddarrays!(D::AbstractArray,args::AbstractArray...)
    m = length(args)
    n = eqlength(args...)
    if 1<=m<=length(loops!)
        loops![m](D,n,args...)
    else
        error("not implemented yet for more than 5 arguments") 
    end
    return D
end

# install: A[:+] = B --- does A += B, but faster. 
#   note B is returned, not A+B. 
function setindex!{T<:BlasFloat}(Y::Matrix{T},X::Array{T},s::Symbol)
    if is(s,:+)
        a = one(T)
        return BLAS.axpy!(a,X,Y)
    else
        error("setindex! not implemented for symbol $s")
    end
end


# install: A[:-] = B --- does A -= B, but faster. 
#   note B is returned, not A-B. 
function setindex!{T<:BlasFloat}(Y::Matrix{T},X::Array{T},s::Symbol)
    if is(s,:+)
        a = -one(T)
        return BLAS.axpy!(a,X,Y)
    else
        error("setindex! not implemented for symbol $s")
    end
end



# install: A[:+] = B,C --- up to 5 array on rhs --- adds everything (including A) in-place in A. 
#   note RHs tuple is returned, not sum. 
function setindex!(Y::Array,X::NTuple{Array},s::Symbol)
    if is(s,:+)
        loopaddarrays!(Y,X...)  
        return X # return RHS, rather than Y
    else
        error("setindex! not implemented for symbol $s")
    end
end


######################################################################

# about the same speed as sum(A,1)
sum1{T}(A::Matrix{T})=(
    (m,n) = size(A);
    s = Array(T,n);
    for j=1:n t=zero(T)
        for i=1:m t += A[i,j] end 
        s[j] = t 
    end; 
    reshape(s,1,n) 
)

sum1blas{T<:BlasFloat}(A::Matrix{T}) = reshape(A.'*ones(T,size(A,1)),1,size(A,2))
sum2blas{T<:BlasFloat}(A::Matrix{T}) = A*ones(T,size(A,2))


# currently about 3x faster than sum(A,2) 
sum2{T}(A::Matrix{T})=(
    (m,n)=size(A);
    s = zeros(T,m);
    for j=1:n, i=1:m s[i] += A[i,j] end;
    s
)

sum{T<:BlasFloat}(A::Matrix{T},i::Int) = (
    assert(1<=i<=2);
    if length(A) < 10_000 # this heuristic must be refined
       return i=1?sum1(A):sum2(A)
    else
       return i=1?sum1blas(A):sum2blas(A)
   end
)

######################################################################


end