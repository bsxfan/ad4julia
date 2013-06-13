
export unpackparams!,packparams, radobjective

unpackparams!(ofs::Int,w::Vector,params) = (
	for (i,P) in enumerate(params)
		if ndims(P)==0
			sz = 1
			params[i] = w[ofs+(1:sz)]
		else
			sz = length(P)
			P[:] = w[ofs+(1:sz)]
		end
		ofs += sz
	end
	)
unpackparams!(w::Vector,params) = unpackparams!(0,w,params)

vec(x::Number) = x
packparams(params) = [map(vec,params)...]


function radobjective(f::Function, args, flags)
    flags = bool(flags)
    function obj(w::Vector{Float64}) 
    	unpackparams!(w,args[flags])
    	y,g = radeval(f,args,flags)
    	return y,packparams(g(1.0))
    end
    return obj
end
