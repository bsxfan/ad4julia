immutable colplusrow{T<:Number} <: DenseFlavour 
  col::Vector{T}
  row::Vector{T}
  colplusrow(data::( Vector{T}, Vector{T}) ) = new(data[1],data[2])
end
function construct_colplusrow{R,C}(col::Vector{R}, row::Vector{C} ) 
    T = promote_type(R,C)
    col = convert(Vector{T},col)
    row = convert(Vector{T},row)
    return colplusrow{T}( (col,row) )
end
eltype{T}(::colplusrow{T}) = T

(+)(A::colplusrow,B::colplusrow) = construct_colplusrow(A.col+B.col,A.row+B.row)
(+)(A::colplusrow, row::Vector) = construct_colplusrow(A.col,A.row+row) #not cummutative
(+)(col::Vector,B::colplusrow) = construct_colplusrow(col+B.col,B.row) #not cummutative

(*)(s::Number,data::colplusrow) = construct_colplusrow(s*data.col,s*data.row)
(*)(data::colplusrow,s::Number) = *(s,data)


transpose{T}(data::colplusrow{T}) = colplusrow{T}((data.row,data.col))
conj{T}(data::colplusrow{T}) = colplusrow{T}((conj(data.col),conj(data.row)))
ctranspose{T}(data::colplusrow{T}) = colplusrow{T}((conj(data.row),conj(data.col)))


function colplusrow(col::Vector,row::Vector) 
  return CustomMatrix(colplusrow,construct_colplusrow(col,row),length(col),length(row))
end

getindex(M::CustomMatrix{colplusrow},i::Int,j::Int) = M.data.col[i] + M.data.row[j]
getindex(M::CustomMatrix{colplusrow},k::Int) = getindex(M,1+(i-1)%M.m,1+div(i-1,M.m))

function update!(d::Number, D::Matrix, S::CustomMatrix{colplusrow})
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
(+)(A::CustomMatrix{reprow}, B::CustomMatrix{colplusrow}) = +(B,A) 
(+)(A::CustomMatrix{colplusrow}, B::CustomMatrix{repcol}) = +(B,A) 


function (+)(A::CustomMatrix{repcol}, B::CustomMatrix{reprow})  
    assert(size(A)==size(B),"size mismatch")
    return CustomMatrix(colplusrow, construct_colplusrow(A.data,B.data),size(A)...)  
end
(+)(A::CustomMatrix{reprow}, B::CustomMatrix{repcol}) = +(B,A) 


transpose(C::CustomMatrix{colplusrow}) = CustomMatrix(colplusrow,C.data.',C.n,C.m)

