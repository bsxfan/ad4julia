######## Vectorized Scalar Function Library #######################
# add new function here as you need them 
log(x::DualNum) = dualnum(log(x.st),x.di./x.st)
exp(x::DualNum) = (y=exp(x.st);dualnum(y,x.di.*y))

sin(x::DualNum) = dualnum(sin(x.st),x.di.*cos(x.st))
cos(x::DualNum) = dualnum(cos(x.st),-x.di.*sin(x.st))
tan(x::DualNum) = (y=tan(x.st);dualnum(y,x.di.*(1+y.^2)))

sqrt(x::DualNum) = (y=sqrt(x);dualnum(y,0.5./y))
