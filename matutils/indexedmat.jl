typealias IntRange Ranges{Int}
typealias RangeTuple{N} NTuple{N,IntRange}

type IndexedMat{T<:Number,N} <: AbstractMatrix{T}  
	data:: Vector{T}
    rd:: RangeTuple{N}
    sz:: (Int,Int) # size of the whole matrix
    ri:: IntRange
    rj:: IntRange
end
function IndexedMat{T,N}(data::Vector{T}, rd::RangeTuple{N}, 
                         sz::(Int,Int), 
                         ri::IntRange, rj::IntRange) 
  return IndexedMat{T,N}(data,rd,sz,ri,rj)  
end

function IndexedMat(data::Vector, rd:: IntRange, sz:: (Int,Int),
                    ri:: IntRange, rj:: IntRange)
    return IndexedMat(data,(rd,),sz,ri,rj)  
end

function IndexedMat(data::Vector, rd, sz:: (Int,Int) )
    return IndexedMat(data,rd,sz,1:sz[1],1:sz[2])  
end

size(M::IndexedMat) = M.sz

function full{T}(M::IndexedMat{T,1}) 
	F = zeros(T,M.sz)
    rd,ri,rj = (M.rd[1],M.ri,M.rj)
    @assert length(rd) == length(ri) == length(rj)
    i,j = (first(ri),first(rj))
    si,sj = (step(ri),step(rj))
    for d in rd
        F[i,j] = M.data[d]
        i += si
        j += sj
    end
    return F
end

function full{T}(M::IndexedMat{T,2}) 
    F = zeros(T,M.sz)
    rdi,rdj = (M.rd[1],M.rd[2])
    @assert length(rdj)==length(M.rj)  
    @assert length(rdi)==length(M.ri)  
    dj = first(rdj)
    di0 = first(rdi)
    dis = step(rdi)
    djs = step(rdj)
    for j in M.rj
        di = di0
        for i in M.ri
            #println("$(di+dj) -> $i,$j")
            F[i,j] = M.data[di+dj]
            di += dis
        end
        dj += djs
    end
    return F
end

show(io::IO, M::IndexedMat) = (println(io,"IndexedMat -> "); show(io,full(M)) )

rep(k,n) = Range(k,0,n) # n repetitions of k

repdiagm(x::Number,n::Int) = IndexedMat([x],rep(1,n),(n,n))

function fulldiagm(v::Vector) 
    n = length(v)
    sz = (n,n)
    IndexedMat(v,1:n,sz) 
end

function reprows(row::Vector,m::Int) 
     n = length(row)
     sz = (m,n)
     rdi = rep(0,m) 
     rdj = 1:n 
     IndexedMat(row, (rdi,rdj), sz) 
end

function repcolumns(col::Vector,n::Int) 
    m = length(col) 
    sz = (m,n) 
    rdi = 1:m 
    rdj = rep(0,n)
    IndexedMat(col, (rdi,rdj), sz)
end

function column_at(col::Vector,j::Int,n::Int) 
    m = length(col); 
    sz = (m,n)
    rd = 1:m
    ri = 1:m
    rj = rep(j,m) 
    IndexedMat(col,rd,sz,ri,rj)
end

function row_at(row::Vector,i::Int,m::Int) 
    n = length(row) 
    sz = (m,n) 
    rd = 1:n 
    ri = rep(i,n) 
    rj = 1:n 
    IndexedMat(row,rd,sz,ri,rj)
end

function element_at(e::Number,i::Int,j::Int,sz::(Int,Int)) 
    IndexedMat([e],1:1,sz,i:i,j:j)
end

function slice_at(M::Matrix,ri::IntRange,rj::IntRange,sz::(Int,Int)) 
    m,n = size(M) 
    rdi = 1:m
    rdj = Range(0,m,n) 
    IndexedMat(vec(M),(rdi,rdj),sz,ri,rj)
end





