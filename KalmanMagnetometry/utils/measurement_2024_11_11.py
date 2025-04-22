import numpy as np
import pathlib
from scipy import signal, optimize
from cmath import inf, pi
import matplotlib.pyplot as plt


class Measurement:

    def __init__(self, data:np.ndarray = None, sr:float = None):
        self.data = data
        self.sr = sr
        self.best_frequency = None # filled with value by find_best_frequency
        self.sincos_fit = None # filled with values by make_sincos_fit
    
    def make_sincos_fit(self, analysis_frequency, block_duration):
        self.sincos_fit = sincos_Fit(self, analysis_frequency, block_duration)

    def make_demod(self, analysis_frequency, block_duration, os):
        self.demod = Demod(self, analysis_frequency, block_duration, os)

    def find_best_frequency(self, start_frequency):
        
        (f, PSD) = signal.welch(self.data, self.sr, window="hann", scaling="density", nperseg=np.floor(len(self.data) / 1))

        # set frequency range for finding peak
        range_for_search = 2  # in Hz
        freq_min = start_frequency - range_for_search
        freq_max = start_frequency + range_for_search

        index_f_min = (np.absolute(f - freq_min)).argmin()
        index_f_max = (np.absolute(f - freq_max)).argmin()

        peak_indices, peak_properties = signal.find_peaks(
            PSD[index_f_min:index_f_max], distance=len(self.data)
        )  # due to the big disatance only one point in dataset should be found
        print(f"peak indices: {peak_indices}")

        peak_indices = peak_indices + index_f_min  # adjust index to complete data set, not just peak finding range

        # create a figure and axes
        fig, ax = plt.subplots()

        # setting title to graph
        ax.set_title("PSD")
        ax.loglog()

        # label x-axis and y-axis
        ax.set_ylabel("PSD ((nT)^2/Hz)")
        ax.set_xlabel("f (Hz)")

        # function to plot and show graph
        ax.plot(f, PSD)
        for index in peak_indices:
            ax.vlines(x=f[index], ymin=0, ymax=PSD[index], color="r")
            ax.annotate(f" f = {f[index]:.2f} Hz\n B = {PSD[index]:.2f} nT^2/Hz", (f[index], PSD[index]), (f[index], PSD[index]))
        plt.show()

        for i in peak_indices:
            f_max_peak = f[i]

        self.best_frequency = f_max_peak

        


class sincos_Fit():

    def __init__(self, Measurement, analysis_frequency = None, block_duration = None):
        self.input_freq = analysis_frequency
        self.input_block_duration = block_duration
        self.phase = None
        self.phase_std = None
        self.amp = None
        self.amp_std = None
        self.time = None
        self.frequency = None
        self.frequency_std = None
        self.real_blocklength = None
        self.residuals = None
        self.residuals_time = None
        self.freq_mean = None
        self.freq_mean_std = None
        self.freq_weighted_mean = None
        self.freq_weighted_mean_std = None

        #x_time =  np.linspace(0, np.size(Measurement.data)/Measurement.sr, np.size(Measurement.data))
        x_time = np.arange(0, np.size(Measurement.data)) / Measurement.sr # error found by Jasper!!! change in other versions
        para, para_std, cov, blk_actual_length = self._Block_fit(x_time, Measurement.data, block_duration, analysis_frequency)

        self.phase, self.phase_std = self._Phase_determination(para,para_std,cov,blk_actual_length)
        blk_t = np.arange(blk_actual_length/2,len(self.phase)*blk_actual_length,blk_actual_length)
        # temp, cov_l = np.polyfit(blk_t, self.phase , 1, w=np.reciprocal(self.phase_std),cov=True)
        # perr = np.sqrt(np.diag(cov_l))
        # print(f"Estimated Lamour freq. is {temp[0]/2/np.pi:.6f} ± {perr[0]/2/np.pi:.2e}")

        # jasper
        # self.est_init_freq = temp[0]/2/np.pi
        # self.est_init_freq_std = perr[0]/2/np.pi

        self.amp = np.sqrt(para[1]**2+para[2]**2)
        self.amp_std=np.divide(np.sqrt((para[1]*para_std[1])**2+(para[2]*para_std[2])**2),self.amp)
        blk_t = np.arange(blk_actual_length/2,len(self.amp)*blk_actual_length,blk_actual_length)
        self.time = blk_t
        self.frequency = para[0]
        self.frequency_std = para_std[0]

        # jasper
        self.est_init_amp = self.amp[0]
        self.est_init_amp_std = self.amp_std
        

        #create residuals
        A = para[1]
        B = para[2]
        amp_offset = para[3]
        block_data_points = int(blk_actual_length*Measurement.sr)

        created_signal2 = self._signal_for_all_blocks2(x_time, self.frequency, A, B, Measurement.sr, amp_offset, block_data_points)
        residual2 = Measurement.data[0:np.size(created_signal2)] -created_signal2
        self.residuals = residual2 
        self.residuals_time = x_time[0:np.size(created_signal2)]

        self.freq_mean = np.mean(self.frequency)
        self.freq_mean_std = np.std(self.frequency)
        sincos_popt, sincos_pcov= optimize.curve_fit(constant,self.time, self.frequency, p0=[self.input_freq], sigma = self.frequency_std)
        sincos_perr = np.sqrt(np.diag(sincos_pcov))
        self.freq_weighted_mean = np.average(sincos_popt)
        self.freq_weighted_mean_std = sincos_perr[0] 


    def _Block_fit(self,t,y,blk_duration,f_given_RF):
        # cuts data in blocks and makes VP fit for every block
        f_samp=1/(t[1]-t[0])
        block_data_points=int(np.floor(blk_duration*f_samp))
        block_number=int(np.floor((t[-1]-t[0])/blk_duration))
        blk_actual_length=block_data_points/f_samp
        self.real_blocklength = blk_actual_length
        para=np.zeros((4,block_number))
        para_std=np.zeros((4,block_number))
        cov=np.zeros(block_number)
        for i in range(block_number):
            t_block=t[block_data_points*i:block_data_points*(i+1)]    # not include the last point
            y_block=y[block_data_points*i:block_data_points*(i+1)]
            para[:,i],para_std[:,i],cov[i]=self._VP_fit(t_block,y_block,f_given_RF)
            f_given_RF = para[0,i] # WP use fitted frequency for next block
        return para, para_std, cov, blk_actual_length
    
    def _VP_fit(self,t,y,ini_freq):
        t=t-np.mean(t)   
        opt_freq=ini_freq
        D=np.column_stack((np.sin(t*pi*2*ini_freq),np.cos(t*pi*2*ini_freq),np.ones(t.size))) # ist hier erste guess Amplitude noch 1? WP
        P= np.linalg.lstsq(D, y, rcond=None) # P[0] contains initial amplitudes and offset value WP
        opt_freq_old=opt_freq
        inst = sincos_fit_only_freq()
        Error_diff=10
        Error_buff=-0.1
        while Error_diff>0.1:
            inst.para=P[0]
            popt_freq, pcov_freq = optimize.curve_fit(inst.sincos,t,y,opt_freq_old)   # curve fit starts every iteration with guess frequency
            #opt_freq_old = popt_freq # WP start frequency from prevoius optimization
            D=np.column_stack((np.sin(t*pi*2*popt_freq),np.cos(t*pi*2*popt_freq),np.ones(t.size)))
            P= np.linalg.lstsq(D, y, rcond=None)         
            error=np.linalg.norm(np.matmul(D,P[0])-y) # multiply D with parameters (A,B,offset)
            # error=np.mean((np.matmul(D,P[0])-y)**2)**0.5 # multiply D with parameters (A,B,offset)
            Error_diff=error-Error_buff
            Error_buff=error 
        Fisher=np.linalg.inv(np.matmul(D.T,D))     
        dof=len(t)-len(P[0]) # WP changed 11.11.2024 to calcuate the degrees of freedom, the number of arameters storedin P[0] is needed, just P was most likely a typing error
        cov_matrix=Fisher*(error**2/dof) # WP changed 11.11.2024 error has to be squared here, since the previously calculated error was the norm of the residuals, aso in the sincos-matblab-code norm of the residuals squared is used (called srs)
        para_std=np.sqrt(np.diag(cov_matrix))  # The standard deviation of 3 para (6 when fitted for 2 frequencies) 

        opt_para=np.concatenate((popt_freq,P[0])) # Total 4 parameters: fXe,Axe,Bxe,y-offset
        cov=cov_matrix[0,1]
        opt_para_std=np.concatenate((np.sqrt(pcov_freq[0]), para_std))
        # Fitting quality checking
        # Noise_SQUID=freq*0.4*(6e-3)^2
        # Reduced_Kai_square=error/dof/Noise_SQUID
        return opt_para,opt_para_std,cov
    
    def _Phase_determination(self,opt_para,opt_para_std,cov,blk_actual_length):
        Block_size = np.size(opt_para[0,:])
        # The inital unwrapped phase 
        Phase_he=np.ones(Block_size)
        Phase_check_he=np.ones(Block_size)
        Phase_he[0]=np.angle(opt_para[1,0]+1j*opt_para[2,0])  # arctan2 # phase for time = 0 (middle of block)
        Phase_check_he[0]= Phase_he[0]
        for i in np.arange(1,Block_size,1):
            Phase_check_he[i]=Phase_he[i-1]+opt_para[0,i-1]*2*pi*blk_actual_length
            Phase_he[i]=np.angle(opt_para[1,i]+1j*opt_para[2,i])+ Phase_check_he[i]-np.mod(Phase_check_he[i],2*pi)
            if np.abs(Phase_he[i]-Phase_check_he[i])>pi:
                Phase_he[i]=Phase_he[i]-2*pi*np.sign(Phase_he[i]-Phase_check_he[i])
        std_Phase_He=np.divide(np.sqrt((opt_para[2,:]*opt_para_std[1,:])**2+(opt_para[1,:]*opt_para_std[2,:])**2+2*opt_para[1,:]*opt_para[2,:]*cov),(opt_para[1,:]**2+opt_para[2,:]**2))
        return Phase_he,std_Phase_He
    
    def _create_signal_one_block2(self, time_in_block,A,B,frequency,yoffset):
        time_in_block = time_in_block-np.mean(time_in_block)
        created_signal = A*np.sin(time_in_block*2*np.pi*frequency)+B*np.cos(time_in_block*2*np.pi*frequency)+yoffset
        return created_signal

    def _signal_for_all_blocks2(self, x_time, frequency_sincos,A,B,sr,yoffset,block_data_points):
        #time_in_block = np.arange(0,time_sincos[1]-time_sincos[0],1/sr)
        signal_all = np.empty(0)
        for i in range(np.size(frequency_sincos)):
            time_in_block=x_time[block_data_points*i:block_data_points*(i+1)] 
            signal = self._create_signal_one_block2(time_in_block, A[i], B[i], frequency_sincos[i], yoffset[i])
            signal_all = np.append(signal_all,signal)
        return signal_all


class sincos_fit_only_freq:
    def __init__(self):
        self.para=[0,0]
        pass
    def sincos(self,t,freq):
        y=self.para[0]*np.sin(t*2*pi*freq)+self.para[1]*np.cos(t*2*pi*freq)+self.para[2]
        return y  


class Demod():

    def __init__(self, Measurement, analysis_frequency = None, block_duration = None, os: float = 1):
        self.phase = None
        self.amp = None
        self.time_freq = None
        self.time_phase = None
        self.frequency = None
        self.freq_mean = None
        self.freq_mean_std = None
        self.real_blocklength = None

        best_frequency = analysis_frequency
        sampling_rate = Measurement.sr
        oversampling = os
        data_array = Measurement.data



        f_sig = np.cdouble(complex(best_frequency, 0))  # make frequency complex
        dt = 1 / sampling_rate

        # substract offset
        data_array = data_array - np.mean(data_array)

        # calculation for block size and number of points in block #cahnged to choose nearest odd number on 02.07.2024
        block_data_points_floor=int(np.floor(block_duration * sampling_rate)) # number of data points in one block
        block_data_points_ceil=int(np.ceil(block_duration * sampling_rate))
        if block_data_points_floor % 2 > 0: # to ensure we have an odd number of points per block
                block_data_points = block_data_points_floor # Even 
        elif block_data_points_ceil % 2 > 0: # to ensure we have an odd number of points per block
                block_data_points = block_data_points_ceil # Even
        else:
                block_data_points = block_data_points_ceil-1

        blk_actual_length = block_data_points / sampling_rate  # time of block calculated from block points

        points_in_kernel = block_data_points

        i_time_in_kernel = np.arange(points_in_kernel)
        x2 = (2 * i_time_in_kernel - (points_in_kernel - 1)) / (points_in_kernel)
        result = self._window(x2)
        integral_window = np.sum(result)
        fmix_0 = np.exp(complex(0, 2) * np.pi * f_sig * i_time_in_kernel * dt)
        fmix_0 = fmix_0*result


        array_blocks = np.lib.stride_tricks.sliding_window_view(data_array,points_in_kernel)[::int(np.floor(points_in_kernel/oversampling)), :] # here adjust so it gives an array of all blocks
        array_time = np.linspace(0,(np.size(data_array)-1)*dt,np.size(data_array))
        # print(array_time)
        array_time_blocks = np.lib.stride_tricks.sliding_window_view(array_time,points_in_kernel)[::int(np.floor(points_in_kernel/oversampling)), :] # time for every datapoint, same structure as array_blocks
        time_blockstart = array_time_blocks[:,0] # first timepoint of every block
        time_blockend = array_time_blocks[:,-1] # last timepoint of every block


        chelp0_array = np.cdouble(array_blocks*fmix_0) # multiply every element in block with associated element of fmix_0 (kernel)
        chelp0_array = np.sum(chelp0_array,1) # sum up for every block
        calc_result = 2 * chelp0_array * np.exp(complex(0, 2) * np.pi * f_sig * time_blockstart) / integral_window


        phase = -np.angle(calc_result)
        phase = phase - phase[0]  # start phase at 0
        phase = np.unwrap(phase)  # unwrap phase
        magnitude = np.abs(calc_result)
        delta_frequency = np.diff(
            phase / (2 * np.pi * (blk_actual_length) / oversampling) #blk_actual_length
        )  # y = np.diff(np.unwrap(phi)/(2*np.pi*df)) # before: np.diff(phase/(2*np.pi*blk_actual_length/oversampling))
        frequency = np.real(f_sig) + delta_frequency  #

        time_demod = time_blockstart+((time_blockstart[0]+time_blockend[0]+dt)/2) #np.linspace(block_duration / 2, block_duration / 2 + (block_duration / oversampling) * (used_block_number - 1), used_block_number)  # stop point before: len(data_array)/sampling_rate-block_duration/2
        first_point_time_freq = (time_demod[1] + time_demod[0]) / 2
        last_point_time_freq = (time_demod[-1] + time_demod[-2]) / 2
        time_freq = np.linspace(
            first_point_time_freq, last_point_time_freq, (len(time_demod) - 1)
        )  # np.linspace(time_demod[1]-time_demod[0],(time_demod[1]-time_demod[0])*(len(time_demod)-1),(len(time_demod)-1))

        self.time_freq = time_freq
        self.time_phase = time_demod
        self.frequency = frequency
        self.phase = phase
        self.amp = magnitude
        self.real_blocklength = blk_actual_length
        self.freq_mean = np.mean(self.frequency)
        self.freq_mean_std = np.std(self.frequency)

    def _window(self,x):
        #calculates window
        np.seterr(all='warn') # Jans settings say to raise all floating point errors, but for the window function this rounding seems to be ok and necessary
        window = np.where(abs(x)<1,np.where(x == 0,np.exp(-1 / (1 - (x**2))) * 2.037 * np.pi, np.exp(-1 / (1 - (x**2))) * np.sin(2.037 * np.pi * x) / x),0)
        return window

def constant(x,c):
    return c


if __name__ == "__main__":

    data_block = fa.read_Python_npz(pathlib.Path(r"n:\XenonEDM\Experimental_setup\PTB HeXe-Setup\Polarizer_MEOP\measurements\OMG_Python_measurements\longtime_05_06_2024_30mm_10torr_30spumping\2024-06-05_17-30-16_30mm10torr_30spumping80mVpumping_longtime"))
    #convert to pT
    data_block = data_block*1000 # to convert to pT
    data_block = np.array(data_block, dtype='float64')
    #data_array = np.array([1,1,1,1,1,1])
    sr = 500
    measurement_check = Measurement(data_block, sr)
    print(measurement_check.sr)
    print(measurement_check.data)

    measurement_check.sincos_fit(84.6,20)
    print(measurement_check.sincos_fit.time)
    #print(measurement_check.result)
