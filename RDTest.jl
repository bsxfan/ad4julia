module RDTest

export gen_reducedim_func

function make_loop_nest(vars, ranges, body)
    otherbodies = cell(length(vars),2)
    #println(vars)
    for i = 1:2*length(vars)
        otherbodies[i] = nothing
    end
    make_loop_nest(vars, ranges, body, otherbodies)
end

function make_loop_nest(vars, ranges, body, otherbodies)
    expr = body
    len = size(otherbodies)[1]
    for i=1:length(vars)
        v = vars[i]
        r = ranges[i]
        l = otherbodies[i]
        j = otherbodies[i+len]
        expr = quote
            $l
            for ($v) = ($r)
                $expr
            end
            $j
        end
    end
    expr
end

function gen_reducedim_func(n, f)
    ivars = { symbol(string("i",i)) for i=1:n }
    # limits and vars for reduction loop
    lo    = { symbol(string("lo",i)) for i=1:n }
    hi    = { symbol(string("hi",i)) for i=1:n }
    rvars = { symbol(string("r",i)) for i=1:n }
    setlims = { quote
        # each dim of reduction is either 1:sizeA or ivar:ivar
        if contains(region,$i)
            $(lo[i]) = 1
            $(hi[i]) = size(A,$i)
        else
            $(lo[i]) = $(hi[i]) = $(ivars[i])
        end
               end for i=1:n }
    rranges = { :( $(lo[i]):$(hi[i]) ) for i=1:n }  # lo:hi for all dims
    body =
    quote
        _tot = v0
        $(setlims...)
        $(make_loop_nest(rvars, rranges,
                         :(_tot = ($f)(_tot, A[$(rvars...)]))))
        R[_ind] = _tot
        _ind += 1
    end
    quote
        local _F_
        function _F_(f, A, region, R, v0)
            _ind = 1
            $(make_loop_nest(ivars, { :(1:size(R,$i)) for i=1:n }, body))
        end
        _F_
    end
end


quote  # C:\Users\nbrummer\Documents\GitHub\ad4julia\RDTest.jl, line 59:
    local _F_ # line 60:
    function _F_(f,A,region,R,v0) 
        _ind = 1 
        for i2 = 1:size(R,2) 
            for i1 = 1:size(R,1) 
            _tot = v0 
            if contains(region,1)
                lo1 = 1 
                hi1 = size(A,1)
            else  
                lo1 = hi1 = i1
            end
            if contains(region,2) 
                lo2 = 1 
                hi2 = size(A,2)
            else  
                lo2 = hi2 = i2
            end
            for r2 = lo2:hi2 
                begin  
                    for r1 = lo1:hi1 # line 25:
                        _tot = +(_tot,A[r1,r2])
                    end # line 27:
                end
            end # line 27:
            R[_ind] = _tot # line 56:
            _ind += 1
        end # line 27:
    end # line 64:
    _F_
end


end 


