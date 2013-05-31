module MatUtils
importall Base

include("matutils/IndexedMat.jl")
export IndexedMat,repdiagm,fulldiagm,reprows,repcolumns,column_at,row_at,slice_at

end