module NewUtils

importall Base

using GenUtils

printc(color::Symbol, msg::String...) = Base.print_with_color(color,msg...)

examples = {Int32 => 2, Int64 => 2, Float32 => float32(2.5), Float64 => 2.5, 
           Complex64 => complex64(2.5,2.5), Complex128 => complex(2.5,2.5),
           Rational{Int32} => int32(1)//int32(2), Rational{Int64} => 1//2,
           Complex{Int} => complex(2,2), Complex{Rational{Int}} => complex(1//2,1//3) }

# typealias IntFlavours{T<:Integer} Union(T,Complex{T})
# typealias RatFlavours{T<:Rational} Union(T,Complex{T})
# willconvert{D<:Number,S<:Number}(::Type{D},::Type{S}) = true
# willconvert{D<:Real,S<:Complex}(::Type{D},::Type{S}) = false
# willconvert{D<:IntFlavours,S<:RatFlavours}(::Type{D},::Type{S}) = false



for (st,ex) in examples, dt in keys(examples)
  D = Array(dt,1)
  print(dt," <- ",st," ")
  try D[1] = ex; 
    if willconvert(dt,st) println("OK") 
    else printc(:red,"false alarm\n") end
  catch E
    if willconvert(dt,st) printc(:red,string(E),"\n") 
    else printc(:blue,string(E),"\n") end
  end
end 



end