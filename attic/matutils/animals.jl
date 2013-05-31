abstract Blood
abstract warm <: Blood
abstract cold <: Blood


abstract Skin
abstract hair <: Skin
abstract feathers <: Skin
abstract scales <: Skin
abstract slimy <: Skin

abstract Reproduction
abstract eggs <: Reproduction
abstract womb <: Reproduction

abstract Respiration
abstract gills <: Respiration
abstract lungs <: Respiration

type Vertebrate{B<:Blood,S<:Skin,Rp<:Reproduction,Rs<:Respiration}
    species::String
end


typealias Reptile Vertebrate{cold,scales,eggs,lungs}
typealias Mammal Vertebrate{warm,hair,womb,lungs}
typealias Bird Vertebrate{warm,feathers,eggs,lungs}
typealias Amphibian Vertebrate{cold,slimy,eggs,Union(gills,lungs)}
typealias Fish Vertebrate{cold,scales,eggs,gills}

typealias Warmblooded{S<:Skin,Rp<:Reproduction} Vertebrate{warm,S,Rp,lungs}
typealias Coldblooded{S<:Skin,Rs<:Respiration} Vertebrate{cold,S,eggs,Rs}
typealias Egglayer{B<:Blood,S<:Skin,Rs<:Respiration} Vertebrate{B,S,eggs,Rs}

function classify(animal::Vertebrate)
  symbols = names(Main)
  for s in symbols
  	class = @eval $s
  	#println("$s: $(typeof(class))")
  	if isa(class,Union(DataType,TypeConstructor)) && isa(animal,class)
  		println(s)
  	end
  end
end

attributes{B,S,Rp,Rs}(animal::Vertebrate{B,S,Rp,Rs}) = Type[B,S,Rp,Rs]


goldfish = Vertebrate{cold,scales,eggs,gills}("goldfish")
dove = Vertebrate{warm,feathers,eggs,lungs}("dove")
frog = Vertebrate{cold,slimy,eggs,Union(lungs,gills)}("frog")
cat = Vertebrate{warm,hair,womb,lungs}("cat")



