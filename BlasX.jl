module BlasX

const libblas = Base.libblas_name

import Base.LinAlg: BlasFloat, BlasChar, BlasInt, blas_int, DimensionMismatch

export symmunpack, symmpack

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
# extend ger! to A = beta*A + alpha * x * y.'
function ger!{T<:BlasFloat}(beta::T,A::StridedMatrix{T},alpha::T,x::StridedVector{T},y::StridedVector{T})
    if beta!=1 scale!(A,beta) end
    return ger!(A,alpha,x,y)
end

# symmetric packed rank 1 update: A = A + alpha * x * x.'  (A is packed symmetric)
for ( spr, elty) in
    ((:dspr_,:Float64),
     (:sspr_,:Float32)) #complex functions only have conjugated version. Use ger instead.
   @eval begin
       #SUBROUTINE SPR ( uplo, N, ALPHA, X, INCX, AP )
       function spr!( uplo::BlasChar, AP::Vector{$elty}, alpha::($elty), 
                      X::StridedVector{$elty})
           assert(length(AP)==spsize(length(X)),"dimension mismatch")
           ccall(($(string(spr)),libblas), Void,
                 (Ptr{Uint8},Ptr{BlasInt}, Ptr{$elty}, 
                  Ptr{$elty}, Ptr{BlasInt}, 
                  Ptr{$elty}),
                 &uplo, &length(X), &alpha, 
                 X, &stride(X,1), 
                 AP)
           return AP
       end
   end
end
#make a plan for complex
function spr!{T<:Complex}(uplo::BlasChar, AP::Vector{T},alpha::T,x::StridedVector{T})
    return symmpack(ger!(uplo,symmpack(A,uplo),alpha,x),uplo)
end

# extend spr! to A = beta*A + alpha * x * y.'
function spr!{T<:BlasFloat}(uplo::BlasChar, beta::T,AP::Vector{T},alpha::T,x::StridedVector{T})
    if beta!=1 scale!(AP,beta) end
    return spr!(uplo,AP,alpha,x)
end


spsize(n::Int) = div(n*(n+1),2)
fullsize(k) = iround(sqrt(8*k+1)-1)>>1
function symmpack{T<:BlasFloat}(A::StridedMatrix{T},uplo::BlasChar='U')
  m,n = size(A)
  assert(m==n,"A must be square")
  sz = spsize(n);
  AP = Array(T,sz)
  if uplo=='U'
      k = 1; 
      for j=1:n, i=1:j AP[k] = A[i,j]; k += 1 end
      assert(k==sz+1)  
  elseif uplo=='L'
      k = 1; 
      for j=1:n, i=j:m AP[k] = A[i,j]; k += 1 end
      assert(k==sz+1)  
  else
    error("uplo must be U or L")
  end
  return AP
end

function symmunpack{T<:BlasFloat}(AP::Vector{T},uplo::BlasChar='U')
  sz = length(AP)
  n = fullsize(sz)
  A = Array(T,n,n)
  if uplo=='U'
      k = 1; 
      for j=1:n 
        for i=1:j-1
          A[i,j] = A[j,i] = AP[k]
          k += 1 
        end
        A[j,j] = AP[k]
        k += 1 
      end
      assert(k==sz+1)  
  elseif uplo=='L'
      k = 1; 
      for j=1:n 
        A[j,j] = AP[k]
        k += 1 
        for i=j+1:n
          A[i,j] = A[j,i] = AP[k]
          k += 1 
        end
      end
      assert(k==sz+1)  
  else
    error("uplo must be U or L")
  end
  return A
end


end