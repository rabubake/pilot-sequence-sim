using DSP, Random, Distributions, Bessels, Permutations, LinearAlgebra, FFTW, JLD2, Revise;
includet("sequences.jl")
includet("signal_processing.jl")
includet("transceiver.jl")

function seq_simulation(num_pilots, buffer_size, gen_seq_func, seq_args, inter_rate, deci_rate, SNR)
    #SNR is in dB

    seq_mat = transpose(gen_seq_func(seq_args...))
    #seq = gen_qpsk_ofdm_sig(16, 16) #naive sequence
    seq = matrix2seq(seq_mat, size(seq_mat, 1), size(seq_mat, 2)) 
    pilot_seq = interpolate_sig(seq, inter_rate);

    noise_pwr = 10^(-SNR/10)*mean(abs.(pilot_seq).^2);

    rx = Transceiver(rx_buffer_size = buffer_size);

    all_buffers = zeros(ComplexF32, num_pilots, 2*buffer_size);

    pilots_sent = 0;
    false_pos_cnt = 0;
    false_neg_cnt = 0;
    total_empty_buffers = 0;

    while pilots_sent < num_pilots
        pilot_sent = false
        curr_pilot_cnt = rx.pilot_cnt

        if rand() <0.1
            pilot_sent = true
            pilots_sent = pilots_sent+1;
            pilot_pos = rand(1:rx.rx_buffer_size);
            fill_buffer(rx, pilot_pos, pilot_seq, noise_pwr);
            all_buffers[pilots_sent,:] = [rx.rx_buffer1 ; rx.rx_buffer2]';     
        else
            fill_buffer(rx, 1, zeros(ComplexF32, length(pilot_seq)), noise_pwr);
            total_empty_buffers+=1;
        end

        coarse_avg_pwr_sync(rx, deci_rate); #Detect using the Avg. Pwr. Coarse Synchronization if a packect was sent or received

        #If a pilot was sent, but it was not detected, increase false negative count
        if pilot_sent && rx.pilot_cnt == curr_pilot_cnt
            false_neg_cnt += 1
        end

        #If a pilot was not sent, but one was detected, increase false positive count
        if !pilot_sent && rx.pilot_cnt > curr_pilot_cnt
            false_pos_cnt += 1
        end

    end
    
    #Return the rates of false positives and false negatives
    return false_pos_cnt/total_empty_buffers, false_neg_cnt/num_pilots
end


function main()
    num_pilots = 100000;
    buffer_size = 2048;

    for deci_rate in [30, 50, 70]
        sim_results = Matrix{Any}(["Sequence" "SNR = 0 dB" "SNR = 2.5 dB" "SNR = 5 dB" "SNR = 7.5 dB" "SNR = 10 dB" "SNR = 12.5 dB" "SNR = 15 dB" "SNR = 17.5 dB" "SNR = 20 dB" "SNR = 22.5 dB" "SNR = 25 dB" "SNR = 27.5 dB" "SNR = 30 dB"])
        
        # experi_1 = Matrix{Any}(["Naive QPSK, L=4, MC" 0 0 0 0 0 0 0 0 0 0 0 0 0])
        # Threads.@threads for SNR=0:2.5:30
        #     println("Experiment, SNR: ", SNR)
        #     fpr, fnr = seq_simulation(num_pilots, buffer_size, generate_Milewski_seq, [generate_FZC, 2], 16, 70, SNR);
        #     experi_1[Int64(SNR/2.5+2)] = fnr
        # end
        # sim_results = vcat(sim_results, experi_1)

        experi_1 = Matrix{Any}(["FZC, L=4, MC" 0 0 0 0 0 0 0 0 0 0 0 0 0])
        Threads.@threads for SNR=0:2.5:30
            println("Experiment 1, SNR: ", SNR)
            fpr, fnr = seq_simulation(num_pilots, buffer_size, generate_Milewski_seq, [generate_FZC, 2], 4, deci_rate, SNR);
            experi_1[Int64(SNR/2.5+2)] = fnr
        end
        sim_results = vcat(sim_results, experi_1)

        experi_2 = Matrix{Any}(["GCP-BPSK, L=4, MC" 0 0 0 0 0 0 0 0 0 0 0 0 0])
        Threads.@threads for SNR=0:2.5:30
            println("Experiment 2, SNR: ", SNR)
            fpr, fnr = seq_simulation(num_pilots, buffer_size, generate_Milewski_seq, [generate_psk_GCP_sequence, [2, 0]], 4, deci_rate, SNR);
            experi_2[Int64(SNR/2.5+2)] = fnr
        end
        sim_results = vcat(sim_results, experi_2)

        experi_3 = Matrix{Any}(["GCP-QPSK, L=4, MC" 0 0 0 0 0 0 0 0 0 0 0 0 0])
        Threads.@threads for SNR=0:2.5:30
            println("Experiment 3, SNR: ", SNR)
            fpr, fnr = seq_simulation(num_pilots, buffer_size, generate_Milewski_seq, [generate_psk_GCP_sequence, [2, 1]], 4, deci_rate, SNR);
            experi_3[Int64(SNR/2.5+2)] = fnr
        end
        sim_results = vcat(sim_results, experi_3)
        
        jldsave("deci_"*string(deci_rate)*"_L=4_MC_experiment_results.jld2"; sim_results)
        #jldsave("naive_QPSK_sig_deci_"*string(deci_rate)*"_experiment_results.jld2"; sim_results)
    end
    return sim_results;
end


