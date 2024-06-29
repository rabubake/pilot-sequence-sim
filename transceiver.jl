using DSP, Random, Distributions, Bessels, Parameters, LinearAlgebra, FFTW, JLD2, Revise;

Base.@kwdef mutable struct Transceiver
    rx_buffer_size::Int64
    rx_buffer1::Vector{ComplexF32} = zeros(ComplexF32, rx_buffer_size);
    rx_buffer2::Vector{ComplexF32} = zeros(ComplexF32, rx_buffer_size);
    est_noise_power::Float32 = 0;
    est_noise_cnt::Float32 = 0;
    pilot_cnt::Int64 = 0;
end

function fill_buffer(txrx::Transceiver, rx_pos, tx_sig, noise_power)
    txrx.rx_buffer1 = zeros(ComplexF32, txrx.rx_buffer_size);
    txrx.rx_buffer2 = zeros(ComplexF32, txrx.rx_buffer_size);

    #The transmitted signal is in some random position in the buffer
    if rx_pos > txrx.rx_buffer_size-length(tx_sig)
        txrx.rx_buffer1[rx_pos:txrx.rx_buffer_size] = tx_sig[1:(txrx.rx_buffer_size-rx_pos+1)];
        txrx.rx_buffer2[1:(length(tx_sig)-(txrx.rx_buffer_size-rx_pos+1))] = tx_sig[(txrx.rx_buffer_size-rx_pos+2):length(tx_sig)];
    else
        txrx.rx_buffer1[rx_pos:(rx_pos+length(tx_sig)-1)] = tx_sig;
    end  

    #Add noise to rx buffers
    txrx.rx_buffer1 = txrx.rx_buffer1 + sqrt(noise_power/2)*(randn(txrx.rx_buffer_size)+im*randn(txrx.rx_buffer_size));
    txrx.rx_buffer2 = txrx.rx_buffer2 + sqrt(noise_power/2)*(randn(txrx.rx_buffer_size)+im*randn(txrx.rx_buffer_size));
end


function coarse_avg_pwr_sync(txrx::Transceiver, rate)
    avg_pwr_l = 0 
    avg_pwr_r = 0 

    #Calculate the average power in left and half of 1st buffer
    for n=1:rate:Int64(length(txrx.rx_buffer1)/2)
        avg_pwr_l = avg_pwr_l + abs(txrx.rx_buffer1[n])^2;
        avg_pwr_r = avg_pwr_r + abs(txrx.rx_buffer1[length(txrx.rx_buffer1)-n+1])^2;
    end

    avg_pwr_l = avg_pwr_l/ceil((length(txrx.rx_buffer1)/2)/rate);
    avg_pwr_r = avg_pwr_r/ceil((length(txrx.rx_buffer1)/2)/rate);

    #Calibrate Noise Power
    txrx.est_noise_cnt = txrx.est_noise_cnt + 1;
    txrx.est_noise_power = (txrx.est_noise_cnt-1)*txrx.est_noise_power/txrx.est_noise_cnt+((avg_pwr_l+avg_pwr_r)/2)/txrx.est_noise_cnt;
    
    #Detect if a pilot was received or not by thresholding the power
    if avg_pwr_l> 1.5*txrx.est_noise_power || avg_pwr_r> 1.5*txrx.est_noise_power
        txrx.pilot_cnt = txrx.pilot_cnt+1
        return
    end

    #Repeat for second buffer if nothing detected in the first
    avg_pwr_l = 0 
    avg_pwr_r = 0 

    for n=1:rate:Int64(length(txrx.rx_buffer2)/2)
        avg_pwr_l = avg_pwr_l + abs(txrx.rx_buffer2[n])^2;
        avg_pwr_r = avg_pwr_r + abs(txrx.rx_buffer2[length(txrx.rx_buffer2)-n+1])^2;
    end
    avg_pwr_l = avg_pwr_l/ceil((length(txrx.rx_buffer2)/2)/rate);
    avg_pwr_r = avg_pwr_r/ceil((length(txrx.rx_buffer2)/2)/rate);
    
    #Continue to calibrate Noise Power again
    txrx.est_noise_cnt = txrx.est_noise_cnt + 1;
    txrx.est_noise_power = (txrx.est_noise_cnt-1)*txrx.est_noise_power/txrx.est_noise_cnt+((avg_pwr_l+avg_pwr_r)/2)/txrx.est_noise_cnt;

    if avg_pwr_l> 2*txrx.est_noise_power || avg_pwr_r> 2*txrx.est_noise_power
        txrx.pilot_cnt = txrx.pilot_cnt+1
        return
    end
end
