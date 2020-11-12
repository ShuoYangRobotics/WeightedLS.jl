module WeightedLS

using LinearAlgebra

greet() = print("Hello World!")

function normalLS(A::Array{Float64,2}, b::Array{Float64,1})
    return A\b
end

# On the modified Gram-Schmidt algorithm for weighted and constrained linear least squares problems
# https://github.com/borglab/gtsam/blob/develop/gtsam/linear/NoiseModel.cpp#L445
# solve min (b-Ax)^TW(b-Ax)  
function weightedLS(A::Array{Float64,2}, b::Array{Float64,1}, W::Array{Float64,1})
    # TODO: input size check 

    # get size, A is mxn, b is nx1, W is nx1
    m = size(A,1)
    n = size(A,2)
    max_rank = max(m,n)
    Ab = [A b]
    weights = W;
    outweights = zeros(n, 1)
    Rd = zeros(m, n+1)
    rd_row = 1
    rd = zeros(1, n+1)
    for j=1:n
        a = Ab[1:m,j:j]
        constraint_row = checkConstraint(dropdims(a,dims=2),W,m)
        if constraint_row > 0
            rd = Ab[constraint_row:constraint_row,:]
            Rd[rd_row,:] = rd
            outweights[rd_row] = weights[constraint_row]
            rd_row += 1
            if rd_row -1 > max_rank
                break
            end

            m -= 1

            if constraint_row != m-1
                Ab[constraint_row,:] = Ab[m+1,:]
                weights[constraint_row,:] = weights[m+1,:]
            end
            a_reduced = Ab[1:m,j:j]
            
            a_reduced = a_reduced .* 1.0/rd[1,j]
            Ab[1:m,j+1:n+1] -= a_reduced * rd[1:1,j+1:n+1]
        else
            precision = 0
            pseudo = zeros(m, 1)
            for i=1:m
                ai = a[i]
                if (abs(ai) > 1e-9)
                    pseudo[i,1] = weights[i]*ai
                    precision += pseudo[i]*ai
                else
                    pseudo[i,1] = 0
                end
            end
            if precision > 1e-8
                pseudo = pseudo ./ precision
                rd = zeros(1, n+1)
                rd[1,j] = 1.0
                rd[1:1,j+1:n+1] = pseudo' * Ab[1:m,j+1:n+1]
                Rd[rd_row,:] = rd
                outweights[rd_row] = precision
                rd_row += 1
            else
                continue
            end
            if rd_row -1 > max_rank
                break
            end
            Ab[1:m,j+1:n+1] -= a * rd[1:1,j+1:n+1]

        end
    end
    return Rd, outweights
end

function checkConstraint(a::Array{Float64,1}, W::Array{Float64,1}, m::Integer)
    max_element = 1e-9
    constraint_row = -1
    for i = 1:m
        if W[i] != Inf
            continue
        end
        abs_ai = abs(a[i])
        if abs_ai > max_element
            max_element = abs_ai
            constraint_row = i
        end
    end
    return constraint_row
end


end # module
