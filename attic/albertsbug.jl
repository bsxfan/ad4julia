
    importall Base

    # =====================================================================
    # Stripped-down copy of bsxfan's DualNum
    # =====================================================================
    # A convenient taxonomy of numeric types 
    # 'Num' should be understood as 'numeric', which can be scalar, vector or matrix.
    typealias FloatReal Union(Float64,Float32)
    typealias FloatComplex Union(Complex128,Complex64)
    typealias FloatScalar Union(FloatReal, FloatComplex)  # identical to Linalg.BlasFloat
    typealias FloatVector{T<:FloatScalar} Array{T,1}
    typealias FloatMatrix{T<:FloatScalar} Array{T,2}
    typealias FloatArray{T<:FloatScalar} Union(FloatMatrix{T}, FloatVector{T})
    typealias FloatNum{T<:FloatScalar} Union(T, FloatArray{T})

    typealias FixReal Union(Integer,Rational)
    typealias FixComplex{T<:FixReal} Complex{T}
    typealias FixScalar Union(FixReal,FixComplex)
    typealias FixVector{T<:FixScalar} Array{T,1}
    typealias FixMatrix{T<:FixScalar} Array{T,2}
    typealias FixArray{T<:FixScalar} Union(FixMatrix{T},FixVector{T})
    typealias FixNum{T<:FixScalar} Union(T,FixArray{T})

    typealias Numeric Union(FloatNum,FixNum) 

    # =====================================================================
    # This is the main new type declared here. The two components can be scalar,
    # vector or matrix, but must agree in type and shape.
    immutable DualNum{T<:FloatNum} 
      st::T # standard part
      di::T # differential part
      function DualNum(s::T,d::T)
        n=ndims(s)
    	assert(n==ndims(d)<=2,"dimension mismatch")
    	for i=1:n;
    	  assert(size(s,i)==size(d,i),"size mismatch in dimension $i")
    	end
    	return new(s,d)
      end
    end

    function dualnum{T<:FloatNum}(s::T,d::T) 
      return DualNum{T}(s,d)  # construction given float types that agree
    end

    # show 
    function show(io::IO,x::DualNum)
      print(io,"standard part: ")
      show(io,x.st)  
      print(io,"\ndifferential part: ")
      show(io,x.di)  
    end

    # binary operators
    .+(x::DualNum,y::DualNum) = dualnum(x.st.+y.st, x.di.+y.di)
    *(x::DualNum,y::DualNum) = dualnum(x.st*y.st, x.st*y.di + x.di*y.st)
    *(x::DualNum,y::Numeric) = dualnum(x.st*y, x.di*y)
    *(x::Numeric,y::DualNum) = dualnum(x*y.st, x*y.di)

    # =====================================================================
    # =====================================================================
    # Some functions to illustrate various buggy behaviours
    function bugfun1()
        # returns nothing
        A = dualnum(randn(2,2),randn(2,2))
        B = dualnum(randn(2,2),randn(2,2))

        C = A .+ B
        y = C*1

        println("returning:\n$y")
        return y
    end 

    function bugfun2(ignore=None)
        # may return nothing... depending on what typeof(ignore) is
        A = dualnum(randn(2,2),randn(2,2))
        B = dualnum(randn(2,2),randn(2,2))

        C = A .+ B
        y = C*1

        println("returning:\n$y")
        return y
    end 

    function bugfun3()
        # multiplying by a float here produces a 1x1 DualNum, returns correctly
        A = dualnum(randn(2,2),randn(2,2))
        B = dualnum(randn(2,2),randn(2,2))

        C = A .+ B
        y = C*1.0 # scalar!?
        println("Should be returning:\n$C")
        println("returning:\n$y")
        return y
    end 

    # =====================================================================
    # =====================================================================

    println("\nCorrect behaviour outside function =============")
    A = dualnum(randn(2,2),randn(2,2))
    B = dualnum(randn(2,2),randn(2,2))
    C = A .+ B
    y = C*1
    println("\nNo bug here: C*1 = \n$y")

    y = C*1.0
    println("\nNo bug here: C*1.0 = \n$y")

    println("\nexample 1: (bug) returned value vanishes ===========================")
    z = bugfun1()
    println("\nReturned:\n$z")

    println("\nexample 2: (bug) returned value vanishes ===========================")
    z = bugfun2(2)
    println("\nReturned:\n$z")

    println("\ncounterexample 3: (OK) now returned value is there ===========================")
    z = bugfun2(randn(2))
    println("\nReturned:\n$z")

    println("\nexample3: (bug) inexplicable collapse of matrix to scalar =============")
    z = bugfun3()
    println("\nReturned:\n$z")



    # Correct behaviour outside function =============

    # No bug here: C*1 =
    # standard part: 2x2 Float64 Array:
    #   0.552178   0.585758
    #  -0.329195  -0.0420111
    # differential part: 2x2 Float64 Array:
    #  0.974976   -2.65792
    #  0.0991329   0.0728872

    # No bug here: C*1.0 =
    # standard part: 2x2 Float64 Array:
    #   0.552178   0.585758
    #  -0.329195  -0.0420111
    # differential part: 2x2 Float64 Array:
    #  0.974976   -2.65792
    #  0.0991329   0.0728872

    # example 1: (bug) returned value vanishes ===========================
    # returning:
    # standard part: 2x2 Float64 Array:
    #  -0.269862  0.164337
    #  -0.041986  2.46685
    # differential part: 2x2 Float64 Array:
    #  0.143382  -0.795063
    #  0.868825   0.64295

    # Returned:
    # nothing

    # example 2: (bug) returned value vanishes ===========================
    # returning:
    # standard part: 2x2 Float64 Array:
    #  -0.994362   0.798359
    #   0.528806  -0.285397
    # differential part: 2x2 Float64 Array:
    #  -1.38295   -1.03693
    #   0.832376  -1.79867

    # Returned:
    # nothing

    # counterexample 3: (OK) now returned value is there ===========================
    # returning:
    # standard part: 2x2 Float64 Array:
    #  -0.300047  -1.31415
    #   2.95492   -1.43682
    # differential part: 2x2 Float64 Array:
    #  -0.0603585  -1.48603
    #  -0.105336    1.31012

    # Returned:
    # standard part: 2x2 Float64 Array:
    #  -0.300047  -1.31415
    #   2.95492   -1.43682
    # differential part: 2x2 Float64 Array:
    #  -0.0603585  -1.48603
    #  -0.105336    1.31012

    # example3: (bug) inexplicable collapse of matrix to scalar =============
    # Should be returning:
    # standard part: 2x2 Float64 Array:
    #   2.65351   -0.580244
    #  -0.593985   1.44685
    # differential part: 2x2 Float64 Array:
    #  -0.063097  -0.263272
    #   0.542484   2.71743
    # returning:
    # standard part: 1.09782604e-315
    # differential part: 1.097826434e-315

    # Returned:
    # standard part: 1.09782604e-315
    # differential part: 1.097826434e-315

