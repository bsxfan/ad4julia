module NewUtils

importall Base

using GenUtils

export loopaddarrays4, loopaddarrays3, loopaddarrays, loop4



function loopaddarrays(args::AbstractArray...)
    m = length(args)
    if m==2 return args[1] + args[2] end
    T = promote_eltype(args...)    
    sz = eqsize(args...)
    D = Array(T,sz)
    n = prod(sz);
    if m==3 
        loop3(D,n,args...) 
    elseif m==4
        loop4(D,n,args...) 
    elseif m==5
        loop5(D,n,args...) 
    else
        error("not implemented yet for more than 5 arguments") 
    end
    return D
end


loop3(E,n,A,B,C) = for i=1:n E[i]=A[i]+B[i]+C[i] end
loop4(E,n,A,B,C,D) = for i=1:n E[i]=A[i]+B[i]+C[i]+D[i] end
loop5(F,n,A,B,C,D,E) = for i=1:n F[i]=A[i]+B[i]+C[i]+D[i]+E[i] end






end