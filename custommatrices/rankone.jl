immutable rankone{T<:Number} <: RankOne
  col::Vector{T}
  row::Vector{T}
  rankone(data::( Vector{T}, Vector{T}) ) = new(data[1],data[2])
end
function construct_rankone{R,C}(col::Vector{R}, row::Vector{C} ) 
    T = promote_type(R,C)
    col = convert(Vector{T},col)
    row = convert(Vector{T},row)
    return rankone{T}( (col,row) )
end
eltype{T}(::rankone{T}) = T

(*)(s::Number,data::rankone) = construct_rankone(s*data.col,s*data.row)
(*)(data::rankone,s::Number) = *(s,data)


transpose(data::rankone) = construct_rankone(data.row,data.col)
conj(data::rankone) = construct_rankone(conj(data.col), conj(data.row) )
ctranspose(data::rankone) = construct_rankone(conj(data.row), conj(data.col) )



function rankone(col::Vector,row::Vector) 
  return CustomMatrix(rankone,construct_rankone(col,row),length(col),length(row))
end

getindex(M::CustomMatrix{rankone},i::Int,j::Int) = M.data.col[i] * M.data.row[j]
getindex(M::CustomMatrix{rankone},k::Int) = getindex(M,1+(i-1)%M.m,1+div(i-1,M.m))

function update!(d::Number, D::Matrix, S::CustomMatrix{rankone})
  assert(size(D)==size(S),"argument dimensions must match")
  col = S.data.col
  row = S.data.row
  m,n = size(S) 
  for j=1:n
    rj = row[j]
    for i=1:m 
      D[i,j] = d*D[i,j] + col[i] * rj
    end
  end 
  return D  
end

function (+)(A::CustomMatrix{rankone},B::CustomMatrix{rankone})  
   error("CustomMatrix addition between rankone flavours not defined, use full() or update()")
end

function (*)(A::CustomMatrix{rankone},B::CustomMatrix{rankone})  
  s = dot(A.data.row,B.data.col)
  return CustomMatrix(rankone,construct_rankone(s*A.data.col,B.data.row),
                      length(A.data.col),length(B.data.row))
end


transpose(C::CustomMatrix{rankone}) = CustomMatrix(rankone,C.data.',C.n,C.m)

function sum(C::CustomMatrix{rankone},i::Int) 
    if i==1
      return reshape(sum(C.data.col)*C.data.row,1,C.n) 
    elseif i==2
      return reshape(sum(C.data.row)*C.data.col,C.m,1)
    else
      return full(C)
    end
end
