clear,close all
snr_list=4:0.5:19.5;
ser_list = zeros(size(snr_list));
ber_list=zeros(size(snr_list));
for idx=1:length(snr_list)
    snr=snr_list(idx);
    xdata = randi([0,15],[1000000,1]);
    bits=zeros(16,4);
    for i=0:15
        tmp=i;
        for j=1:4
            bits(i+1,j)=mod(tmp,2);
            tmp=floor(tmp/2);
        end
    end
    bits=fliplr(bits);
    x_bits = bits(xdata+1,:);
    y = qammod(xdata,16,'UnitAveragePower',true);

    noise_sigma = sqrt(0.5/10^(snr/10));
    noise = noise_sigma*randn(length(xdata),1)+noise_sigma*1j*randn(length(xdata),1);
    rx = y+noise;
    demod = qamdemod(rx,16,'UnitAveragePower',true);
    ser = mean(demod~=xdata);
    demod_bits = bits(demod+1,:);
    ber = mean(demod_bits~=x_bits,'all');
    ser_list(idx)=ser;
    ber_list(idx)=ber;
end
figure;semilogy(snr_list,ser_list,'-*',snr_list,ber_list,'-o')
figure;semilogy(snr_list,ber_list,'-o');hold on;grid on