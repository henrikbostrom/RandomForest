function Base.names(a::SparseData)
    return a.header
end

Base.size(a::SparseData) = (length(a.labels), length(a.header))
function Base.size(a::SparseData, i::Int)
    if i == 1
        length(a.labels)
    elseif i == 2
        length(a.header)
    else
        throw(ArgumentError("SparseData only have two dimensions"))
    end
end

function DataFrames.eltypes(a::SparseData)
    types = map(eltype, a.data)
    push!(types, eltype(a.labels))
    push!(types, eltype(a.weights))
    return types
end

# a[SingleColumnIndex] --> SparseVector
function Base.getindex(a::SparseData, col::Int)
    return (col == length(a.data)+1) ? a.labels : ((col == length(a.data)+2) ? a.weights : a.data[col])
end
function Base.getindex(a::SparseData, col::Symbol)
    colIndex = findfirst(a.header, col)
    if (colIndex == 0)
        throw(ArgumentError("$col not found"))
    end
    return a[colIndex]
end

# a[MultiColumnIndex] --> SparseData
function Base.getindex(a::SparseData, cols::AbstractVector{Int})
    columns = [a[c] for c in cols]
    return SparseData(a.header[cols], columns, a.labels, a.weights)
end

# a[:] => self
Base.getindex(a::SparseData, cols::Colon) = a

# a[SingleRowIndex, SingleColumnIndex] => element type
function Base.getindex(a::SparseData, row::Int, col::Int)
    return a[col][row]
end
function Base.getindex(a::SparseData, row::Int, col::Symbol)
    return a[col][row]
end

# a[SingleRowIndex, MultiColumnIndex] => Array{Any,1}
function Base.getindex(a::SparseData, row::Int, cols::AbstractVector{Int})
    return [a[col][row] for c in cols]
end

# a[MultiRowIndex, SingleColumnIndex] => SparseVector
function Base.getindex(a::SparseData, rows::AbstractVector{Int}, col::Int)
    return a[col][rows]
end

# a[MultiRowIndex, MultiColumnIndex] => SparseData
function Base.getindex(a::SparseData, rows::AbstractVector{Int}, cols::AbstractVector{Int})
    newData = [a[c][rows] for c in cols]
    return SparseData(a.header[cols], newData, a.labels[rows], a.weights[rows])
end

# a[:, SingleColumnIndex] => a[SingleColumnIndex]
# a[:, MultiColumnIndex] => a[MultiColumnIndex]
Base.getindex(a::SparseData, rows::Colon, cols) = a[cols]    

# a[SingleRowIndex, :] => SparseData
# a[MultiRowIndex, :] => SparseData
function Base.getindex(a::SparseData, rows, cols::Colon)
    newData = [a.data[c][rows] for c in 1:length(a.data)]
    return SparseData(a.header, newData, a.labels[rows], a.weights[rows])
end

# a[:, :] => self
Base.getindex(a::SparseData, rows::Colon, cols::Colon) = a
