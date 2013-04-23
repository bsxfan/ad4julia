using DualNumbers


K = randn(2,5)
K = K*K'  #make K positive definite 2-by-2 

f(X) = log(det(X*K*X.'))


