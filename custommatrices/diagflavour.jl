type repdiag <: DiagFlavour end
repdiag(element::Number,n::Int) = CustomMatrix(repdiag,element,n,n)

square_sz(M::AbstractMatrix) = ((m,n)=size(M);assert(m==n,"argument not square");m)

function update!(d::Number, D::Matrix,S::CustomMatrix{repdiag})
  n = square_sz(D)
  assert(n==S.n,"argument dimensions must match")
  element = S.data 
  for i=1:n D[i,i] = d*D[i,i] + element end 
  return D  
end

transpose(C::CustomMatrix{repdiag}) = C

function sum(C::CustomMatrix{repdiag},i::Int) 
    if i==1
      return fill(C.data,C.m,1)
    elseif i==2
      return fill(C.data,1,C.n)
    else
      return full(C)
    end
end


*(M::Matrix, C::CustomMatrix{repdiag}) = C.data*M
*(C::CustomMatrix{repdiag}, M::Matrix) = C.data*M

*(A::CustomMatrix{repdiag}, B::CustomMatrix{repdiag}) = CustomMatrix(repdiag,A.data*B.data,size(A)...) 



###################################################################
type fulldiag <: DiagFlavour end
fulldiag(diag::Vector) = CustomMatrix(fulldiag,diag,length(diag),length(diag))
fulldiag(diag::Matrix) = fulldiag(asvec(diag))

function update!(d::Number, D::Matrix,S::CustomMatrix{fulldiag})
  n = square_sz(D)
  assert(n==S.n,"argument dimensions must match")
  diag = S.data 
  for i=1:n D[i,i] = d*D[i,i] + diag[i] end 
  return D  
end

for (L,R) in { (:repdiag,fulldiag), (:fulldiag,:repdiag) }
  @eval begin
    function (+)(A::CustomMatrix{$L},B::CustomMatrix{$R}) 
      assert(size(A)==size(B),"size mismatch")
      return CustomMatrix(fulldiag,A.data+B.data,size(A)...) 
    end
  end
end

transpose(C::CustomMatrix{fulldiag}) = C

function sum(C::CustomMatrix{fulldiag},i::Int) 
    if i==1
      return reshape(copy(C.data),1,C.n)
    elseif i==2
      return reshape(copy(C.data),C.m,1)
    else
      return full(C)
    end
end


*(M::Matrix, C::CustomMatrix{fulldiag}) = scale(M,C.data)
*(C::CustomMatrix{fulldiag}, M::Matrix) = scale(C.data,M)

*(A::CustomMatrix{fulldiag}, B::CustomMatrix{fulldiag}) = CustomMatrix(fulldiag,A.data.*B.data,size(A)...) 


