diagonal(element::Number,n::Int) = repdiag(element,n)
diagonal(v::VecOrMat) = fulldiag(asvec(v))
diagonal(v::RepVec) = repdiag(element(v),length(v))


type repdiag <: DiagFlavour end
repdiag(element::Number,n::Int) = CustomMatrix(repdiag,element,n,n)

typealias RepDiag{E:<Number} CustomMatrix{repdiag,E}

diag(C::RepDiag) = repvec(C.data,C.n)
element(C::RepDiag) = C.data


square_sz(M::AbstractMatrix) = ((m,n)=size(M);assert(m==n,"argument not square");m)

function update!(d::Number, D::Matrix,S::RepDiag)
  n = square_sz(D)
  assert(n==S.n,"argument dimensions must match")
  element = S.data 
  for i=1:n D[i,i] = d*D[i,i] + element end 
  return D  
end

transpose(C::RepDiag) = C

function sum(C::RepDiag,i::Int) 
    if i==1
      return fill(C.data,C.m,1)
    elseif i==2
      return fill(C.data,1,C.n)
    else
      return full(C)
    end
end






###################################################################
type fulldiag <: DiagFlavour end
fulldiag(diag::Vector) = CustomMatrix(fulldiag,diag,length(diag),length(diag))
fulldiag(diag::Matrix) = fulldiag(asvec(diag))

typealias FullDiag{E:<Number} CustomMatrix{fulldiag,E}

diag(C::FullDiag) = C.data

function update!(d::Number, D::Matrix,S::FullDiag)
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

transpose(C::FullDiag) = C

function sum(C::FullDiag,i::Int) 
    if i==1
      return reshape(copy(C.data),1,C.n)
    elseif i==2
      return reshape(copy(C.data),C.m,1)
    else
      return full(C)
    end
end





