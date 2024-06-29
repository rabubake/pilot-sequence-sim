using CUDA, DSP, Random, Distributions, Bessels, Permutations, LinearAlgebra, FFTW, JLD2;


function generate_m_sequence(coeff, initial_state)
    # coeff = [c0 c1 c2 c3 ... ]
    # initial_state = [x0 x1 x2 x3 ... ]
    # output = [y0 y1 y2 y3..]
    
    output = zeros(2^length(coeff)-1)
    curr_state = initial_state
    
    for n=1:lastindex(output)
        output[n] = curr_state[1]
        fb = sum(coeff.*curr_state) % 2;
        curr_state = circshift(curr_state, -1);
        curr_state[length(curr_state)] = fb;
    end

    return output
end

function generate_psk_GCP_sequence(m, b_q)
    #Output sequence is length 2^m
    #b_q selects if it is bpsk(0) or qpsk(1) generated 

    a = zeros(ComplexF32, 2^m);
    b = zeros(ComplexF32, 2^m);
    pi = randperm(m);

    H = 2^(b_q+1) 
    c = rand((0:H), m); 
    c_prime = rand(0:H); 
    for x =1:2^m
        a[x] = (1im)^f(binary_decomp(x, m), c, b_q, pi, H);
        b[x] = (1im)^((f(binary_decomp(x, m), c, b_q, pi, H)+2^(b_q)*binary_decomp(x, m)[pi[b_q == 0 ? 1 : m]]+c_prime) % H);
    end

    return a, b
end

function f(x, c, b_q, pi, H)
    out = ((2^b_q)*h(x, pi, H)) % H;
    m = length(x);
    for j=1:m
        out = (out+c[j]*x[j]) % H;
    end

    return out
end

function h(x, pi, H)
    m = length(x);
    out = 0;

    for j=1:(m-1)
        out = (out+x[pi[j]]*x[pi[j+1]]) % H;
    end
    return out
end

function binary_decomp(x, m)
    x_bin = zeros(m);
    for j=1:m
        x_bin[j] = x%2;
        x = x>>1;
    end
    return x_bin
end

function generate_Milewski_seq(k)
    L = 2^k; #Period of FZC-sequence
    M = 2^(2*k); 
    N = L*M; #Length of total sequence
    
    ω_m = exp(1im*2*pi/M); 
    ω_2L = exp(1im*2*pi/(2*L)); 
    
    #row = ω_2L.^((0:(M-1)).^2);
    row = zeros(ComplexF32, M);
    base_seq = ω_2L.^((0:(L-1)).^2)
    for m=1:(M/L)
        row[Int32((m-1)*L+1):Int32(m*L)] = base_seq;
    end
    
    u = zeros(ComplexF32, L, M);
    #u = zeros(ComplexF32, N);
        
    for i=0:(L-1)
        # for j=0:(M-1)
        #     u[i+1, j+1] = ω_2L^((j)^2)*ω_m^(i*j);
        #     #u[i*M+j+1] =  exp(1im*pi*j^2/L)*exp(1im*2*pi*i*j/M);#ω_2L^((j)^2)*ω_m^(i*j);
        # end

        u[i+1,:] = row.*ω_m.^(i.*(0:(M-1)));
    end

    return u
end

function generate_generalized_Milewski_seq(k, q)
    L = 2^k; #Period of FZC-sequence
    M = L*q; 
    
    ω_m = exp(1im*2*pi/M); 
    ω_2L = exp(1im*2*pi/(2*L)); 
    
    #row = ω_2L.^((0:(M-1)).^2);
    row = zeros(ComplexF32, M);
    base_seq = ω_2L.^((0:(L-1)).^2)
    for m=1:q
        row[Int32((m-1)*L+1):Int32(m*L)] = base_seq;
    end
    
    u = zeros(ComplexF32, q, M);
    #u = zeros(ComplexF32, N);
        
    for i=0:(q-1)
        # for j=0:(M-1)
        #     u[i+1, j+1] = ω_2L^((j)^2)*ω_m^(i*j);
        #     #u[i*M+j+1] =  exp(1im*pi*j^2/L)*exp(1im*2*pi*i*j/M);#ω_2L^((j)^2)*ω_m^(i*j);
        # end

        u[i+1,:] = row.*ω_m.^(i.*(0:(M-1)));
    end

    return u
end

function seq2matrix(seq, L, M)
    seq_mat = zeros(ComplexF32, L, M);
    for i=0:(L-1)
        for j=0:(M-1)
            seq_mat[i+1, j+1] = seq[i*M+j+1];
        end
    end

    return seq_mat
end

function matrix2seq(seq_mat, L, M)
    seq = zeros(ComplexF32, L*M);
    for i=0:(L-1)
        for j =0:(M-1)
            seq[i*M+j+1] = seq_mat[i+1,j+1];
        end
    end

    return seq
end

#Aperiodic Autocorrelation Function
function AACF(seq_in) 
    N = length(seq_in)
    seq_autocorr = zeros(typeof(seq_in[1]), 2*N-1);
    out = zeros(typeof(seq_in[1]), N)'
    seq_autocorr[(length(seq_in)):(2*length(seq_in)-1)] = seq_in

    for n=1:(N)
        out[n] = conj(seq_autocorr)'*circshift(seq_autocorr,(n-1))
    end

    return out'
end

function ms_autocorr(seq_in)
    out = zeros(typeof(seq_in[1]), length(seq_in))

    for n=1:lastindex(out)
        for m=1:lastindex(seq_in)
            if seq_in[m] == circshift(seq_in,(n-1))[m]
                out[n]+=1;
            else
                out[n]-=1;
            end
        end
    end

    return out
end
