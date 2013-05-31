typealias IntRange Union(Int, Ranges{Int})

type RangeCartesian{N}
  ranges::Vector{IntRange}
  len::Int
end
function cartesian(ranges::IntRange...)
  N = length(ranges)
  ranges = IntRange[ranges...]
  len = prod(length,ranges)
  return RangeCartesian{N}(ranges,len)
end



start(RC::RangeCartesian) = (Int[],{start(r) for r in RC.ranges}) #(values,states)
done{N}(RC::RangeCartesian{N},state) = RC.len==0 || all([done(RC.ranges[i],state[2][i]) for i=1:N]) 
length(RC::RangeCartesian) = RC.len

function next{N}(RC::RangeCartesian{N},state)
  values,states = state
  if isempty(values)
      values = Array(Int,N)
      for i=1:N values[i],states[i] = next(RC.ranges[i],states[i]) end
      return tuple(values...), (values,states)
  end  
  for i=N:-1:1
    if !done(RC.ranges[i],states[i])
      values[i],states[i] = next(RC.ranges[i],states[i])
      return tuple(values...), (values,states)
    else
      @assert N>1
      states[i] = start(RC.ranges[i])
      values[i],states[i] = next(RC.ranges[i],states[i])
    end  
  end
end

