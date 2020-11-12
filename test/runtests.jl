using WeightedLS
using LinearAlgebra
using Test

@testset "normal LS" begin
    A = Float64[1 0; 1 -2]
    B = Float64[32; -4]
    x = WeightedLS.normalLS(A,B)

    @test isapprox(norm(x-Float64[32.0; 18.0]), 0.0; atol = 1e-5)

    A = Float64[10 12;20 22;21 0]
    B = Float64[1;2;56]
    x = WeightedLS.normalLS(A,B)
    @test isapprox(norm(x-Float64[2.662964939354169; -2.285446442736200]), 0.0; atol = 1e-5)
end

@testset "weighted LS" begin

    a = Float64[1; 1; 1; 1; 1]
    m = size(a,1)
    W = Float64[1; Inf; Inf; 1; 1]

    @test WeightedLS.checkConstraint(a,W,m) == 2

    a = Float64[1; 0.5; 1; 1; 1]
    m = size(a,1)
    W = Float64[1; Inf; Inf; 1; 1]

    @test WeightedLS.checkConstraint(a,W,m) == 3

    a = Float64[1; 1; 1; 1; 1]
    m = size(a,1)
    W = Float64[1; 1; 1; 1; 1]

    @test WeightedLS.checkConstraint(a,W,m) == -1

    A = Float64[1 0 0 0 0 1;
                0 0 0 0 1 0;
                0 0 1 1 0 0;
                0 1 0 0 0 0;
                0 0 0 0 0 1]
    B = Float64[0;0;0;0;0]
    W = Float64[0;1;0;1;1]
    W = 1 ./W
    # println(W)
    Rd, outweights = WeightedLS.weightedLS(A,B,W)
    # println(outweights)
    # show(stdout, "text/plain", Rd)

    expected_Rd = Float64[1  0  0  0  0  1  0;   
                          0  1  0  0  0  0  0;   
                          0  0  1  1  0  0  0;   
                          0  0  0  0  1  0  0;   
                          0  0  0  0  0  1  0;]
    @test isapprox(norm(Rd-expected_Rd), 0.0; atol = 1e-5)


    A = Float64[    1 0 0;    
                    0 1 0;    
                    0 0 1;    
                   -1 0 1;     
                    1 0 0;    
                    0 1 0;    
                    0 0 1;    
                    0 -1 1;     
                    1 0 0;    
                    0 1 0;    
                    0 0 1]
    B = zeros(11,1)
    W = ones(11,1)
    W[4] = Inf; W[8] = Inf;
    # println(W)
    Rd, outweights = WeightedLS.weightedLS(A,dropdims(B,dims=2),dropdims(W,dims=2))
    # println(outweights)
    # show(stdout, "text/plain", Rd)

    expected_Rd = Float64[-1  0 1 0;
                           0 -1 1 0;
                           0  0 1 0;]
    @test isapprox(norm(Rd[1:3,:]-expected_Rd), 0.0; atol = 1e-5)



    A = Float64[1 0; 1 -2]
    B = Float64[32; -4]
    W = Float64[1; Inf]
    Rd, outweights = WeightedLS.weightedLS(A,B,W)
    R = Rd[1:2,1:2]
    d = Rd[1:2,3]
    # println(R)
    # println(d)
    x = R\d

    @test isapprox(norm(x-Float64[32.0; 18.0]), 0.0; atol = 1e-5)
end