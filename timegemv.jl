

function gemmMV(y::Vector{Float64}, A::Matrix{Float64}, x::Vector{Float64})
	LinAlg.BLAS.gemm!('N','N',1.0,A,reshape(x,length(x),1),0.0,reshape(y,length(y),1))
	return y
end

function gemvMV(y::Vector{Float64}, A::Matrix{Float64}, x::Vector{Float64})
	LinAlg.BLAS.gemv!('N', 1.0,A,x,0.0,y)
	return y
end

function loopMV(y::Vector{Float64}, A::Matrix{Float64}, x::Vector{Float64})
     m, n = size(A)
     fill!(y,0.0)
     for j = 1:n, i=1:m  y[i] += A[i, j] * x[j] end; 
     return y	
end


function test(mv::Function, n::Int, count::Int, t::Int)
	Base.openblas_set_num_threads(t)
    A = randn(n,n)
    x = randn(n)
    y = zeros(n)
    tic()
    for i = 1 : count
        mv(y, A, x)
    end
    toq()
end


nn = [4,5,8,9,16,17,32,33,64,65,128,129,512,513,1024,1025,2048,2049]

for (i,n) in enumerate(nn) 
	nrm=10_000_000/(n*n);
	a = test(loopMV,n,100,1) * nrm 
	b = test(gemvMV,n,100,1) * nrm 
	c = test(gemmMV,n,100,1) * nrm 
	d = test(gemvMV,n,100,2) * nrm 
	e = test(gemmMV,n,100,2) * nrm 
	@printf("    %4d %10.5f %10.5f %10.5f %10.5f %10.5f\n",n,a,b,c,d,e) 
end  