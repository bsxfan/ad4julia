module BlasX

const libblas = Base.libblas_name

import Base.LinAlg: BlasFloat, BlasChar, BlasInt, blas_int, DimensionMismatch


# (GE) general rank 1 update A = A + alpha * x * y.' for matrix A, vectors x,y
for ( ger, elty) in
    ((:dger_,:Float64),
     (:sger_,:Float32),
     (:zgeru_,:Complex128),  #does transpose (conjugate transpose would be gerc) 
     (:cgeru_,:Complex64))   #does transpose (conjugate transpose would be gerc) 
   @eval begin
       #SUBROUTINE GER ( M, N, ALPHA, X, INCX, Y, INCY, A, LDA )
       function ger!( A::StridedMatrix{$elty}, alpha::($elty), 
                      X::StridedVector{$elty},
                      Y::StridedVector{$elty})
           assert(stride(A,1)==1,"A must be column dense")
           assert(length(X) == size(A,1) && length(Y) == size(A,2),"dimension mismatch")
           ccall(($(string(ger)),libblas), Void,
                 (Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$elty}, 
                  Ptr{$elty}, Ptr{BlasInt}, Ptr{$elty}, Ptr{BlasInt},
                  Ptr{$elty}, Ptr{BlasInt}),
                 &size(A,1), &size(A,2), &alpha, 
                 X, &stride(X,1), Y, &stride(Y,1),
                 A, &stride(A,2))
           return A
       end
   end
end




end