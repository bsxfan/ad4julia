type colplusrow <: DenseFlavour end

############################################################################

immutable colrowdata{T<:Number} 
  col::Vector{T}
  row::Vector{T}
end
function colrowdata{R,C}(col::Vector{R}, row::Vector{C} ) 
    T = promote_type(R,C)
    col = convert(Vector{T},col)
    row = convert(Vector{T},row)
    return colrowdata{T}(col,row)
end
eltype{T}(::colrowdata{T}) = T

(*)(s::Number,data::colrowdata) = colrowdata(s*data.col,s*data.row)
(*)(data::colrowdata,s::Number) = *(s,data)

transpose(data::colrowdata) = colrowdata(data.row,data.col)
conj(data::colrowdata) = colrowdata(conj(data.col),conj(data.row))
ctranspose(data::colrowdata) = colrowdata(conj(data.row),conj(data.col))

(+)(A::colrowdata,B::colrowdata) = colrowdata(A.col+B.col,A.row+B.row)
(+)(A::colrowdata, row::Vector) = colrowdata(A.col,A.row+row) #not cummutative
(+)(col::Vector,B::colrowdata) = colrowdata(col+B.col,B.row) #not cummutative

############################################################################


function colplusrow(col::Vector,row::Vector) 
  return CustomMatrix(colplusrow,colrowdata(col,row),length(col),length(row))
end
colplusrow(col::VecOrMat, row::VecOrMat) = colplusrow(asvec(col),asvec(row))

typealias ColPlusRow{E<:Number} CustomMatrix{colplusrow,E}

col(C::ColPlusRow) = C.data.col
row(C::ColPlusRow) = C.data.row


getindex(M::ColPlusRow,i::Int,j::Int) = M.data.col[i] + M.data.row[j]
getindex(M::ColPlusRow,k::Int) = getindex(M,1+(i-1)%M.m,1+div(i-1,M.m))

function update!(d::Number, D::Matrix, S::ColPlusRow)
  assert(size(D)==size(S),"argument dimensions must match")
  col = S.data.col
  row = S.data.row
  m,n = size(S) 
  for j=1:n
    rj = row[j]
    for i=1:m 
      D[i,j] = d*D[i,j] + col[i] + rj
    end
  end 
  return D  
end

# define some additions for mismatched flavours
for (L,R) in { (:colplusrow,:reprow) ,(:repcol,:colplusrow) }
  @eval begin
    function (+)(A::CustomMatrix{$L}, B::CustomMatrix{$R})  
        assert(size(A)==size(B),"size mismatch")
        return CustomMatrix(colplusrow, +(A.data,B.data),size(A)...)  # + here is not commutative
    end
  end
end
(+)(A::CustomMatrix{reprow}, B::ColPlusRow) = +(B,A) 
(+)(A::ColPlusRow, B::CustomMatrix{repcol}) = +(B,A) 


function (+)(A::CustomMatrix{repcol}, B::CustomMatrix{reprow})  
    assert(size(A)==size(B),"size mismatch")
    return CustomMatrix(colplusrow, colrowdata(A.data,B.data),size(A)...)  
end
(+)(A::CustomMatrix{reprow}, B::CustomMatrix{repcol}) = +(B,A) 


#CustomMatrix(colplusrow,C.data.',C.n,C.m)
transpose(C::ColPlusRow) = colplusrow(C.data.row,C.data.col)

function sum(C::ColPlusRow,i::Int) 
    col = C.data.col
    row = C.data.row
    m = C.m
    n = C.n
    if i==1
      return reshape(m*row + sum(col), 1, n) 
    elseif i==2
      return reshape(n*col + sum(row), m, 1)
    else
      return full(C)
    end
end

