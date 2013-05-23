abstract RepVecs{E<:Number}

ndims(r::RepVecs) = length(size(r)) 
length(r::RepVecs) = r.n
eltype{E}(r::RepVecs{E}) = E
element(r::RepVecs) = r.data
full(r::RepVecs) = fill(r.data,size(r))
sum(r::RepVecs) = r.data*r.n
size(r::RepVecs,i::Int) = 1<=i<=ndims(r)?size(r)[i]:1

szstr(sz::(Int,)) = "$(sz[1])-fold"
szstr(sz::(Int,Int)) = "$(sz[1])x$(sz[2])"


show(io::IO,v::RepVecs) = (
  println(io,"$(typeof(v)): $(szstr(size(v))) repetion of $(element(v)) -->");
  show(io,full(v)) 
) 

ctranspose{E<:Real}(r::RepVecs{E}) = transpose(r)


##############################################################

immutable RepVec{E<:Number} <: RepVecs{E}
  data::E
  n::Int
end
repvec{E}(e::E,n::Int) = RepVec{E}(e,n)
size(r::RepVec) = (r.n,)
sum(r::RepVec,i::Int) = i==1?[sum(r)]:r
copy(r::RepVec) = repvec(r.data,r.n)

dot(v::Vector,r::RepVec) = length(v)==r.n ? r.data*sum(v) : error("mismatched sizes")
dot(r::RepVec,v::Vector) = dot(v,r)
dot(a::RepVec,b::RepVec) = a.data*b.data

(.*)(v::Vector,r::RepVec) = r.data*v
(.*)(r::RepVec,v::Vector) = r.data*v
(.*)(a::RepVec,b::RepVec) = a.n==b.n ? repvec(a.data*b.data,a.n) : error("mismatched sizes")

+(v::Vector,r::RepVec) = r.data+v
+(r::RepVec,v::Vector) = r.data+v

-(v::Vector,r::RepVec) = v-r.data
-(r::RepVec,v::Vector) = r.data-v

*(r::RepVec,s::Number) = repvec(s*r.data,r.n)
*(s::Number,r::RepVec) = repvec(s*r.data,r.n)

-(r::RepVec) = repvec(-r.data,r.n)

scale(r::RepVec,M::Matrix) = r.data*M
scale(M::Matrix,r::RepVec) = r.data*M


##############################################################

immutable RepRowVec{E<:Number} <: RepVecs{E}
  data::E
  n::Int
end
reprowvec{E}(e::E,n::Int) = RepRowVec{E}(e,n)
size(r::RepRowVec) = (1,r.n)
sum(r::RepRowVec,i::Int) = i==2?[sum(r)]:r
row(r::RepRowVec) = repvec(r.data,r.n)
copy(r::RepRowVec) = reprowvec(r.data,r.n)

*(r::RepRowVec,s::Number) = reprowvec(s*r.data,r.n)
*(s::Number,r::RepRowVec) = reprowvec(s*r.data,r.n)

##############################################################

immutable RepColVec{E<:Number} <: RepVecs{E}
  data::E
  n::Int
end
repcolvec{E}(e::E,n::Int) = RepColVec{E}(e,n)
size(r::RepColVec) = (r.n,1)
sum(r::RepColVec,i::Int) = i==1?[sum(r)]:r
col(r::RepColVec) = repvec(r.data,r.n)
copy(r::RepColVec) = repcolvec(r.data,r.n)

*(r::RepColVec,s::Number) = repcolvec(s*r.data,r.n)
*(s::Number,r::RepColVec) = repcolvec(s*r.data,r.n)

##############################################################


function *(M::Matrix,r::RepVec) 
  if size(M,2)!= r.n error("mismatched sizes") end
  return reshape(sum(M,2)*r.data,size(M,1)) 
end

function *(M::Matrix,r::RepColVec)  
  if size(M,2)!=r.n error("mismatched sizes") end
  return r.data*sum(M,2)
end

function *(r::RepRowVec,M::VecOrMat) 
  if size(M,1)!= r.n error("mismatched sizes") end
  return sum(M,1)*r.data
end	


function *(r::RepRowVec,c::RepColVec) 
  if r.n!=c.n error("mismatched sizes") end
  return fill(r.data*c.data,1,1)
end	

function *(r::RepRowVec,c::RepVec) 
  if r.n!=c.n error("mismatched sizes") end
  return fill(r.data*c.data,1)
end	

*(r::RepVec,M::Matrix) = repcolvec(r.data,r.n) * M
*(r::RepVec,M::RepRowVec) = repcolvec(r.data,r.n) * M



transpose(r::RepVec) = reprowvec(r.data,r.n)
transpose(r::RepRowVec) = repcolvec(r.data,r.n)
transpose(r::RepColVec) = reprowvec(r.data,r.n)

