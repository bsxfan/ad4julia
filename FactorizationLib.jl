
#(factorization element type, RHS element type) => target type for conversion of RHS
const facttypemap = ((DataType,DataType)=>DataType)[

  (Float64,Float64)=>Float64,
  (Float32,Float32)=>Float32,
  (Complex128,Complex128)=>Complex128,
  (Complex64,Complex64)=>Complex64,

  (Float64,Float32)=>Float64,
  (Float64,Int32)=>Float64,
  (Float64,Int64)=>Float64,
  (Float64,Rational{Int32})=>Float64,
  (Float64,Rational{Int64})=>Float64,

  (Complex128,Complex64)=>Complex128,
  (Complex128,Float32)=>Complex128,
  (Complex128,Float64)=>Complex128,
  (Complex128,Int32)=>Complex128,
  (Complex128,Int64)=>Complex128,
  (Complex128,Rational{Int32})=>Complex128,
  (Complex128,Rational{Int64})=>Complex128

]


function copy_convert_or_transpose{T<:Number}(X::StridedVecOrMat{T},trans::Bool,R::DataType)
    if !trans 
      return (T==R)? copy(X) : convert(Array{R,ndims(X)},X)
    else
      X = X.'
      return (T==R)? X: convert(Array{R,ndims(X)},X)
    end
end

############################# Cholesky  #####################################

transpose{T<:Union(Complex64,Complex128)}(C::Cholesky{T}) = Cholesky{T}(conj(C.UL),C.uplo)
transpose{T<:Union(Float64,Float32)}(C::Cholesky{T}) = C
ctranspose(C::Cholesky) = C


function ldivide_chol{S<:Number,T<:Number}(C::Cholesky{S}, transC, 
                                           B::StridedVecOrMat{T}, transB::Bool, 
                                           R::DataType) 
  if transC; C = C'; end
  B = copy_convert_or_transpose(B,transB,R)
  return LinAlg.LAPACK.potrs!(C.uplo, C.UL, B)
end


function rdivide_chol{S<:Number,T<:Number}(B::StridedVecOrMat{T}, transB::Bool, 
                                           C::Cholesky{S}, transC::Bool,
                                           R::DataType) 
  if size(B,1)==1
    if ~transC; C = C.'; end
    B = copy_convert_or_transpose(B,~transB,R)
    return (LinAlg.LAPACK.potrs!(C.uplo, C.UL, B)) .'   #'
  else
      if transC; C = C.'; end
      if T!=R; B = convert(Array{R,ndims(B)},B); end
    return transB? B.' * inv(C) : B * inv(C)
  end
end

for (S,T) in keys(facttypemap)
  R =  facttypemap[S,T]
  @eval (\)(C::Cholesky{$S},B::StridedVecOrMat{$T}) = ldivide_chol(C,false,B,false,$R)
  @eval (/)(B::StridedVecOrMat{$T},C::Cholesky{$S}) = rdivide_chol(B,false,C,false,$R)
end

############################# LU #####################################


function ldivide_chol{S<:Number,T<:Number}(C::LU{S}, transC, 
                                           B::StridedVecOrMat{T}, transB::Bool, 
                                           R::DataType) 
  if C.info > 0; throw(SingularException(C.info)); end
  B = copy_convert_or_transpose(B,transB,R)
  return LinAlg.LAPACK.getrs!(transC?'T':'N', C.factors, C.ipiv, B)
end


function rdivide_chol{S<:Number,T<:Number}(B::StridedVecOrMat{T}, transB::Bool, 
                                           C::LU{S}, transC::Bool,
                                           R::DataType) 
  if C.info > 0; throw(SingularException(C.info)); end
  if size(B,1)==1
    B = copy_convert_or_transpose(B,~transB,R)
    return (LinAlg.LAPACK.getrs!(transC?'T':'N', C.factors, C.ipiv, B)) .'   #'
  else
    if transC; C = C.'; end
    if T!=R; B = convert(Array{R,ndims(B)},B); end
    return transB? B.' * inv(C) : B * inv(C)
  end
end

for (S,T) in keys(facttypemap)
  R =  facttypemap[S,T]
  @eval (\)(C::LU{$S},B::StridedVecOrMat{$T}) = ldivide_LU(C,false,B,false,$R)
  @eval (/)(B::StridedVecOrMat{$T},C::LU{$S}) = rdivide_LU(B,false,C,false,$R)
end



















function ldivide_LU{S<:Number,T<:Number}(A::LU{S}, B::StridedVecOrMat{T}, R::DataType) 
  if A.info > 0; throw(SingularException(A.info)); end
  Bc = T==R? copy(B) : convert(Array{R,ndims(B)},B) 
  return LinAlg.LAPACK.getrs!('N', A.factors, A.ipiv, Bc)
end

function rdivide_LU{S<:Number,T<:Number}(B::StridedVecOrMat{T}, A::LU{S}, R::DataType) 
    if A.info > 0; throw(SingularException(A.info)); end
    if size(B,1)==1
      Bt = B.'
      if T != R; Bt = convert(Array{R,ndims(Bt)},Bt); end
      return (LinAlg.LAPACK.getrs!('T', A.factors, A.ipiv, Bt)) .'   #'
    else
      return (T==R? B : convert(Array{R,ndims(B)},B)) * inv(A)
    end
end

for (S,T) in keys(facttypemap)
  R =  facttypemap[S,T]
  @eval (\)(C::LU{$S},B::StridedVecOrMat{$T}) = ldivide_LU(C,B,$R)
  @eval (/)(B::StridedVecOrMat{$T},C::LU{$S}) = rdivide_LU(B,C,$R)
end



