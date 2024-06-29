function interpolate_sig(sig_in, r)
    #sig_in: Input signal to interpolate
    #r: interpolation rate
       
    return real.(ifft([fft(real.(sig_in))[1:Int32(length(sig_in)/2)];zeros(ComplexF32, Int32((r-1)*length(sig_in))); fft(real.(sig_in))[Int32(length(sig_in)-length(sig_in)/2+1):length(sig_in)]]))+1im*real.((ifft([fft(imag.(sig_in))[1:Int32(length(sig_in)/2)];zeros(ComplexF32, Int32((r-1)*length(sig_in))); fft(imag.(sig_in))[Int32(length(sig_in)-length(sig_in)/2+1):length(sig_in)]])))
end

function decimate_sig(sig_in, r)

    return real.(ifft([fft(real.(sig_in))[1:Int32(length(sig_in)/r/2)];fft(real.(sig_in))[Int32(length(sig_in)-length(sig_in)/r/2+1):length(sig_in)]]))+1im*real.((ifft([fft(imag.(sig_in))[1:Int32(length(sig_in)/r/2)]; fft(imag.(sig_in))[Int32(length(sig_in)-length(sig_in)/r/2+1):length(sig_in)]])))
end

function sub_sample(sig_in, r)
    out = zeros(typeof(sig_in[1]), Int64(floor(length(sig_in)/r)));
    for n=0:(lastindex(out)-1)
        out[n+1] = sig_in[r*n+1];  
    end
    return out
end

function raised_cosine(n, T, β)
    output = zeros(typeof(n[1]),length(n));

    for i in 1:lastindex(n)
        #println(n[i])
        if abs(n[i]) <= (1-β)/(2*T)
            output[i] = 1;
        elseif abs(n[i]) > (1-β)/(2*T) && abs(n[i]) <= (1+β)/(2*T)
            output[i] = 1/2*(cos(pi*T/β*(abs(n[i]) - (1-β)/(2*T)))+1)
        end
    end
 
    # output[abs.(n) .<= (1-β)/(2*T)] .= 1;
    # println(1/2*(cos.(pi*T/β*(n[abs.(n) .> (1-β)/(2*T) .* abs.(n) .<= (1+β)/(2*T)]).-(1-β)/(2*T)) .+ 1))
    # output[abs.(n) .> (1-β)/(2*T) .* abs.(n) .<= (1+β)/(2*T)] .= 1/2*(cos.(pi*T/β*(n[abs.(n) .> (1-β)/(2*T) .* abs.(n) .<= (1+β)/(2*T)]).-(1-β)/(2*T)) .+ 1);
     
    return output
end

function ccor(vec_1, vec_2)
    N_1 = length(vec_1);
    N_2 = length(vec_2);

    in_1 = zeros(ComplexF32, N_1+N_2-1);
    in_1[1:N_1] = vec_1;
    in_2 = zeros(ComplexF32, N_1+N_2-1);
    in_2[1:N_2] = vec_2;

    return ifft(fft(in_1).*conj.(fft(in_2)))
end

function fine_seq_pilot_sync(R, pilot_seq)
#R is the input vector

    avg_peak = 0;
    peak_cnt = 0;
    for i =1:lastindex(R,1)
        if(sum(abs.(R[i,:]))!=0)
            avg_peak = avg_peak+findmax(abs.(ccor(abs.(R[i,:]), abs.(pilot_seq)))[1:Int32(size(R,2)/2)])[1];
            peak_cnt = peak_cnt+1;
        end
    end
    avg_peak = avg_peak/peak_cnt;

    R_pilots = zeros(typeof(R[1,1]), size(R,1), length(pilot_seq));
    pilot_cnt = 0;

    for i =1:lastindex(R,1)
        curr_max_val, curr_max_index = findmax(abs.(ccor(abs.(R[i,:]), abs.(pilot_seq)))[1:Int32(size(R,2)/2)]);
        if(sum(abs.(R[i,:]))!=0 &&  curr_max_val >= 0.5*avg_peak)

            pilot_cnt = pilot_cnt+1;
            R_pilots[pilot_cnt,:] = R[i,curr_max_index:(curr_max_index+length(pilot_seq)-1)];
            
        
        else
            println(i)
        end
    end

    return R_pilots[1:pilot_cnt,:];
end

#Delta Function
delta(n) = n == 0 ? 1 : 0;

function test_inter(x, r)
    y = zeros(typeof(x[1]),length(x)*(r+1)-1)
    for n=1:lastindex(x)
        for m=1:lastindex(x)
            y[n] += x[n]*delta(m-(n-1)*(r+1))
        end
    end
    return y
end

function pilot_fixed_phase_removal(R, pilot_seq)

end