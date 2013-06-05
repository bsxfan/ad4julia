module TransposeWrapper

importall Base
import Base.LinAlg: BLAS, BlasFloat

export transposewrap

immutable TransposedMatrix{T}
    M::T
end
transposewrap{T}(M::T) = TransposedMatrix{T}(M)

show(io::IO,A::TransposedMatrix) = println(io,summary(A)," -->")show(io,unwrap(A))

transpose(A::TransposedMatrix) = A.M
unwrap(A::TransposedMatrix) = A.M.'

-(A::TransposedMatrix) = transposewrap(-A.M)
+(A::TransposedMatrix) = A

*(A::TransposedMatrix,B::TransposedMatrix) = At_mul_Bt(transpose(A),transpose(B))
*(A::TransposedMatrix,B) = At_mul_B(transpose(A),B)
*(A,B::TransposedMatrix) = A_mul_Bt(A,transpose(B))

\(A::TransposedMatrix,B::TransposedMatrix) = unwrap(A) \ unwrap(B)
\(A::TransposedMatrix,B) = unwrap(A) \ B
\(A,B::TransposedMatrix) = A \ unwrap(B)

/(A::TransposedMatrix,B::TransposedMatrix) = transposewrap(transpose(A) \ transpose(B) )
/(A::TransposedMatrix,B) = transposewrap(transpose(A) \ transpose(B) )
/\(A,B::TransposedMatrix) = transposewrap(transpose(A) \ transpose(B) )

+(A::TransposedMatrix,B::TransposedMatrix) = transposewrap(transpose(A) + transpose(B))
+(A::TransposedMatrix,B) = unwrap(A) + B
+(A,B::TransposedMatrix) = A + unwrap(B)

-(A::TransposedMatrix,B::TransposedMatrix) = transposewrap(transpose(A) - transpose(B))
-(A::TransposedMatrix,B) = unwrap(A) - B
-(A,B::TransposedMatrix) = A - unwrap(B)

function (\){T<:BlasFloat}(A::TransposedMatrix{LU{T}}, B::StridedVecOrMat{T})
    M = A.M
    if M.info > 0; throw(SingularException(M.info)); end
    LAPACK.getrs!('T', M.factors, M.ipiv, copy(B))
end


end