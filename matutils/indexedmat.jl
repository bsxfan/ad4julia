type IndexedMat{T} <: AbstractMatrix{T}
	data: Vector{T}
	i0:: Int
	j0:: Int
	istride:: Int
	jstride:: Int
	dstride:: Int 
	m:: Int
	n:: Int
end

size(M::IndexedMat) = (M.m,M.n)
size(M::IndexedMat,k::Int) = ( (0<k<3)||error("dimension out of range"); k=1?m:n )
length(M::IndexedMat) = m*n

full(M::IndexedMat)
