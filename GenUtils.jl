module GenUtils

export eq

eq(m::Int,n::Int) = m==n?n:error("dimension mismatch")

end