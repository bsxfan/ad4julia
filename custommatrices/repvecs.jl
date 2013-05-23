abstract RepVecs{E:<Number}
ndims(r::RepVecs) = length(size(r)) 
length(r::RepVecs) = r.n
eltype{E}(r::RepVecs{E}) = E
element(r::RepVecs) = r.data
copy(r::RepVecs) = r
full(r::RepVecs) = fill(r.data,size(r))
sum(r::RepVecs) = r.data*r.n

##############################################################

immutable RepVec{E:<Number} <: RepVecs{E}
  data: E
  n::Int
end
repvec{E}(e:E,n::Int) = {E}RepVec(e,n)
size(r::RepVec) = r.n
sum(r::RepVec,i::Int) = i==1?[sum(r)]:r

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

immutable RepRowVec{E:<Number} <: RepVecs{E}
  data: E
  n::Int
end
reprowvec{E}(e:E,n::Int) = {E}RepRowVec(e,n)
size(r::RepRow) = (1,r.n)
sum(r::RepRowVec,i::Int) = i==2?[sum(r)]:r
row(r::RepRowVec) = repvec(r.data,r.n)

##############################################################

immutable RepColVec{E:<Number} <: RepVecs{E}
  data: E
  n::Int
end
repcolvec{E}(e:E,n::Int) = {E}RepColVec(e,n)
size(r::RepRow) = (r.n,1)
sum(r::RepColVec,i::Int) = i==1?[sum(r)]:r
col(r::RepRowVec) = repvec(r.data,r.n)


##############################################################


*(M::Matrix,r::RepVec) = size(M,2)==r.n ? reshape(sum(M,2),r.n) : error("mismatched sizes")
*(M::Matrix,r::RepColVec) = size(M,2)==r.n ? sum(M,2),r.n : error("mismatched sizes")
*(r::RepRowVec,M::Matrix) = size(M,1)==r.n ? sum(M,1) : error("mismatched sizes")

*(r::RepRowVec,c::RepColVec) r.n==c.n ? fill(r.data*c.data,1,1) : error("mismatched sizes")
*(r::RepRowVec,c::RepVec) r.n==c.n ? fill(r.data*c.data,1) : error("mismatched sizes")

*(r::RepVec,M::Matrix) = *(repcolvec(r.data,r.n),M)
*(r::RepVec,M::RepRowVec) = *(repcolvec(r.data,r.n),M)



transpose(r::RepVec) = reprowvec(r.data,r.n)
transpose(r::RepRowVec) = repcolvec(r.data,r.n)
transpose(r::RepColVec) = reprowvec(r.data,r.n)

