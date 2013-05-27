module NewUtils

importall Base

export @elapsedloop

macro elapsedloop(ex, n)
    quote
        local s = 0.0
        for i=1:$(esc(n))
	        local t0 = time_ns()
	        local val = $(esc(ex))
	        s += (time_ns()-t0)/1e9
	    end    
	    s
    end
end

end