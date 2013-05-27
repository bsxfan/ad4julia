module MatrixAcceleration

importall Base

using GenUtils

export loopaddarrays, loopaddarrays!


############ Accelerate A+B+C, A+B+C+D, etc.  ##########################

# These loops are much, much faster as separate little functions.
loop3(E,n,A,B,C) = for i=1:n E[i]=A[i]+B[i]+C[i] end
loop4(E,n,A,B,C,D) = for i=1:n E[i]=A[i]+B[i]+C[i]+D[i] end
loop5(F,n,A,B,C,D,E) = for i=1:n F[i]=A[i]+B[i]+C[i]+D[i]+E[i] end
const loops = {nothing,nothing,loop3,loop4,loop5}
function loopaddarrays(args::AbstractArray...)
    m = length(args)
    if m==2 return args[1] + args[2] end
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

loop1!(F,n,A) = for i=1:n F[i] += A[i] end
loop2!(F,n,A,B) = for i=1:n F[i] += A[i]+B[i] end
loop3!(F,n,A,B,C) = for i=1:n F[i] += A[i]+B[i]+C[i] end
loop4!(F,n,A,B,C,D) = for i=1:n F[i] += A[i]+B[i]+C[i]+D[i] end
loop5!(F,n,A,B,C,D,E) = for i=1:n F[i] += A[i]+B[i]+C[i]+D[i]+E[i] end
const loops! = {loop1!,loop2!,loop3!,loop4!,loop5!}
function loopaddarrays!(D,args::AbstractArray...)
    m = length(args)
    n = eqlength(args...)
    if 1<=m<=length(loops!)
        loops![m](D,n,args...)
    else
        error("not implemented yet for more than 5 arguments") 
    end
    return D
end


######################################################################






end