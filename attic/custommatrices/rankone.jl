type rankone <: Rank1Flavour end

############################################################################

immutable rank1data{T<:Number} 
  col::Vector{T}
  row::Vector{T}
end
function rank1data{R,C}(col::Vector{R}, row::Vector{C} ) 
    T = promote_type(R,C)
    col = convert(Vector{T},col)
    row = convert(Vector{T},row)
    return rank1data{T}(col,row)
end
eltype{T}(::rank1data{T}) = T

(*)(s::Number,data::rank1data) = rank1data(s*data.col,s*data.row)
(*)(data::rank1data,s::Number) = *(s,data)

transpose(data::rank1data) = rank1data(data.row,data.col)
conj(data::rank1data) = rank1data(conj(data.col), conj(data.row) )
ctranspose(data::rank1data) = rank1data(conj(data.row), conj(data.col) )

############################################################################


function rankone(col::Vector,row::Vector) 
  return CustomMatrix(rankone,rank1data(col,row),length(col),length(row))
end
rankone(col::VecOrMat, row::VecOrMat) = rankone(asvec(col),asvec(row))

typealias RankOne{E<:Number} CustomMatrix{rankone,E}

col(C::RankOne) = C.data.col
row(C::RankOne) = C.data.row

getindex(M::RankOne,i::Int,j::Int) = M.data.col[i] * M.data.row[j]
getindex(M::RankOne,k::Int) = getindex(M,1+(i-1)%M.m,1+div(i-1,M.m))

function update!(d::Number, D::Matrix, S::RankOne)
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

function (+)(A::RankOne,B::RankOne)  
   error("CustomMatrix addition between rankone flavours not defined, use full() or update()")
end


transpose(C::RankOne) = rankone(C.data.row,C.data.col) #CustomMatrix(rankone,C.data.',C.n,C.m)


function sum(C::RankOne,i::Int) 
    if i==1
      return reshape(sum(C.data.col)*C.data.row,1,C.n) 
    elseif i==2
      return reshape(sum(C.data.row)*C.data.col,C.m,1)
    else
      return full(C)
    end
end

