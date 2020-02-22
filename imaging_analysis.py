import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import datajoint as dj
import os
dj.conn()
from pipeline import pipeline_tools
from pipeline import lab, experiment, ephys_patch, ephysanal, imaging, imaging_gt
import matplotlib.pyplot as plt
import pandas as pd
import time
from plot_imaging.plot_main import *
font = {'size'   : 16}

matplotlib.rc('font', **font)
from pathlib import Path

homefolder = '/nrs/svoboda/rozsam'
#homefolder = str(Path.home())

#%% Precision and recall of spike times
roi_type = 'Spikepursuit'#'VolPy_denoised'#'SpikePursuit'#'VolPy_dexpF0'#'VolPy'#'SpikePursuit_dexpF0'#'VolPy_dexpF0'#''Spikepursuit'#'VolPy'#
key = {'roi_type':roi_type}
gtdata = pd.DataFrame((imaging_gt.GroundTruthROI()&key))
cells = gtdata.groupby(['session', 'subject_id','cell_number','motion_correction_method','roi_type']).size().reset_index(name='Freq')
snratio = list()
for cell in cells.iterrows():
    cell = cell[1]
    key_cell = dict(cell)    
    del key_cell['Freq']
    snratios = (imaging_gt.ROIAPWave()&key_cell).fetch('apwave_snratio')
    #print(np.mean(snratios[:100]))    
    snratio.append(np.mean(snratios[:50]))
cells['SN']=snratio
print(cells)
#%%
fig=plt.figure()
ax_hist = fig.add_axes([0,0,1,1])
ax_hist.hist(snratio)
ax_hist.set_xlabel('S/N ratio of first 50 spikes')
ax_hist.set_ylabel('# of cells')
#%%

cell_index = 1
binwidth = 30 #s
firing_rate_window = 1 #s
frbinwidth = .01

cell = cells.iloc[cell_index]
key_cell = dict(cell)    
del key_cell['Freq']
if imaging.Movie&key_cell:#&'movie_frame_rate>800':
    #%
    plot_precision_recall(key_cell,binwidth =  30,frbinwidth = 0.01,firing_rate_window = 3)    
    print(cell)
#%%    #%%
data = plot_ephys_ophys_trace(key_cell,time_to_plot=None,trace_window = 5,show_e_ap_peaks = True,show_o_ap_peaks = True)
    #%%
data = plot_ephys_ophys_trace(key_cell,time_to_plot=50,trace_window = 110,show_stimulus = True,show_e_ap_peaks = True,show_o_ap_peaks = True)
    #%%

    

#%%        plot time offset between best ROI and real elphys
min_corr_coeff = .5
subject_ids = ephys_patch.Cell().fetch('subject_id')
sessions = ephys_patch.Cell().fetch('session')
cellnums = ephys_patch.Cell().fetch('cell_number')
roi_types = ['Spikepursuit','VolPy']
for roi_type_idx ,roi_type in enumerate(roi_types):
    fig=plt.figure()
    axs_delay_sweep = list()
    axs_delay_coeff = list()
    axs_coeff_sweep = list()
    axs_delay_sweep.append(fig.add_axes([0,-roi_type_idx,.8,.8]))
    axs_coeff_sweep.append(fig.add_axes([1,-roi_type_idx,.8,.8]))
    axs_delay_coeff.append(fig.add_axes([2,-roi_type_idx,.8,.8]))
    for subject_id,session,cellnum in zip(subject_ids,sessions,cellnums):
        key = { 'subject_id': subject_id, 'session':session,'cell_number':cellnum,'roi_type':roi_type}
        roi_numbers,corrcoeffs = (imaging_gt.ROIEphysCorrelation()&key).fetch('roi_number','corr_coeff')
        if len(corrcoeffs)>0:
            if np.max(np.abs(corrcoeffs))>min_corr_coeff:
                roi_number = np.min(roi_numbers)#roi_numbers[np.argmax(np.abs(corrcoeffs))]
                key['roi_number'] = roi_number
                sweep_number,lag,corrcoeffs = (imaging.Movie()*imaging_gt.ROIEphysCorrelation()&key&'movie_frame_rate>100').fetch('sweep_number','time_lag','corr_coeff')
                needed = np.abs(corrcoeffs)>min_corr_coeff
                #print(lag[needed])
                #print(corrcoeffs[needed])
                axs_delay_sweep[-1].plot(lag[needed],'o-')#sweep_number[needed]-sweep_number[needed][0],
                axs_delay_sweep[-1].set_title(roi_type)
                axs_delay_sweep[-1].set_xlabel('sweep number from imaging start')
                axs_delay_sweep[-1].set_ylabel('time offset (ephys-ophys, ms)')
                axs_delay_sweep[-1].set_ylim([-10,0])
                
                axs_delay_coeff[-1].plot(np.abs(corrcoeffs[needed]),lag[needed],'o')
                axs_delay_coeff[-1].set_title(roi_type)
                axs_delay_coeff[-1].set_xlabel('correlation coefficient')
                axs_delay_coeff[-1].set_ylabel('time offset (ephys-ophys, ms)')
                
                axs_coeff_sweep[-1].plot(np.abs(corrcoeffs[needed]),'-o') #sweep_number[needed]-sweep_number[needed][0]
                axs_coeff_sweep[-1].set_title(roi_type)
                axs_coeff_sweep[-1].set_xlabel('sweep number from imaging start')
                axs_coeff_sweep[-1].set_ylabel('correlation coefficient')
#%% photocurrent
window = 3 #seconds
roi_type = 'Spikepursuit'
key = {'roi_type':roi_type}
gtrois = (imaging_gt.GroundTruthROI()&key).fetch('subject_id','session','cell_number','movie_number','motion_correction_method','roi_type','roi_number',as_dict=True) 
for roi in gtrois:
    session_time = (experiment.Session()&roi).fetch('session_time')[0]
    cell_time = (ephys_patch.Cell()&roi).fetch('cell_recording_start')[0]
    movie_start_time = float((imaging.Movie()&roi).fetch1('movie_start_time'))
    movie_start_time = session_time.total_seconds() + movie_start_time - cell_time.total_seconds()
    
    
    
    sweeps = (imaging_gt.ROIEphysCorrelation()&roi).fetch('sweep_number')
    sweep_now = ephys_patch.Sweep()&roi&'sweep_number = '+str(sweeps[0])
    trace,sr = ((ephys_patch.SweepResponse()*ephys_patch.SweepMetadata())&sweep_now).fetch1('response_trace','sample_rate')
    sweep_start_time = float(sweep_now.fetch1('sweep_start_time'))
    trace_time = np.arange(len(trace))/sr+sweep_start_time
    neededidx = (trace_time>movie_start_time-window) & (trace_time<movie_start_time)
    fig=plt.figure()
    ax = fig.add_axes([0,0,.8,.8])
    ax.plot(trace_time[neededidx],trace[neededidx])
    print(roi)
    
    #fig.show()


# =============================================================================
#         print(key_cell)
#         print('waiting')
#         time.sleep(3)
# =============================================================================
    #%%
        
        
#%% plot average spike waveform

subject_ids = ephys_patch.Cell().fetch('subject_id')
sessions = ephys_patch.Cell().fetch('session')
cellnums = ephys_patch.Cell().fetch('cell_number')
for subject_id,session,cellnum in zip(subject_ids,sessions,cellnums):
    key = { 'subject_id': subject_id, 'session':session,'cell_number':cellnum}
    roi_types = np.unique((imaging_gt.ROIEphysCorrelation()&key).fetch('roi_type'))
    for roi_type in roi_types:
        key['roi_type'] = roi_type
        roi_numbers,corrcoeffs = (imaging_gt.ROIEphysCorrelation()&key).fetch('roi_number','corr_coeff')
        if len(corrcoeffs)>0:
            if np.max(np.abs(corrcoeffs))>.7:
                roi_number = roi_numbers[np.argmax(np.abs(corrcoeffs))]
                key['roi_number'] = roi_number
                sweep_number,lag,corrcoeffs = (imaging_gt.ROIEphysCorrelation()&key).fetch('sweep_number','time_lag','corr_coeff')
                needed = np.abs(corrcoeffs)>.7
                neededsweeps = sweep_number[needed]
                roi_numbers,movie_numbers,motion_correction_methods,roi_types,sweep_numbers,apwavetimes,apwaves,famerates,snratio = ((imaging.Movie()*imaging_gt.ROIAPWave())&key).fetch('roi_number','movie_number','motion_correction_method','roi_type','sweep_number','apwave_time','apwave_dff','movie_frame_rate','apwave_snratio')
                framerates_round = np.round(famerates)
                uniqueframerates = np.unique(framerates_round)
                for framerate_now in uniqueframerates:
                    
                    fig=plt.figure()
                    ax_raw=fig.add_axes([0,0,2,.8])
                    ax_raw.set_title([roi_type,framerate_now])
                    ax_bin=fig.add_axes([0,-1,2,.8])
                    ax_snratio=fig.add_axes([0,2,2,.8])
                    ax_snratio_time=fig.add_axes([0,1,2,.8])
                    aps_now = framerates_round == framerate_now
                    ax_snratio.hist(snratio[aps_now],100)
                    histyvals = ax_snratio.get_ylim()
                    medsn = np.median(snratio[aps_now])
                    ax_snratio.plot([medsn,medsn],histyvals)
                    ax_snratio_time.plot(snratio[aps_now])
                    #aps_now = (framerates_round == framerate_now) & (snratio>medsn)
                    
                    apwavetimes_conc = np.concatenate(apwavetimes[aps_now])
                    apwaves_conc = np.concatenate(apwaves[aps_now])
                    if len(roi_numbers)>0:
                        for apwavetime,apwave in zip(apwavetimes[aps_now],apwaves[aps_now]):
                            ax_raw.plot(apwavetime,apwave,'o')
                        #break
                    #%
                    bin_step = .00001
                    bin_size = .0002
                    bin_centers = np.arange(np.min(apwavetime),np.max(apwavetime),bin_step)
                    
                    bin_mean = list()
                    for bin_center in bin_centers:
                        bin_mean.append(np.mean(apwaves_conc[(apwavetimes_conc>bin_center-bin_size/2) & (apwavetimes_conc<bin_center+bin_size/2)]))
                    ax_bin.plot(bin_centers,bin_mean)
                    
                    imaging_gt.ROIEphysCorrelation()
                    print(lag[needed])
                    print(corrcoeffs[needed])
                    print(key)
                    
                #break
                #ax_bin.plot(sweep_number[needed]-sweep_number[needed][0],lag[needed],'o')
#sample movie
#%% save movie - IR + voltron
from pathlib import Path
import os
import caiman as cm
import numpy as np
#%

def moving_average(a, n=3) :
    ret = np.cumsum(a,axis = 0, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
#%%
sample_movie_dir = homefolder+'/Data/Voltage_imaging/sample_movie/'
files = np.sort(os.listdir(sample_movie_dir))
fnames_full = list()
for fname in files:
    fnames_full.append(os.path.join(sample_movie_dir,fname))
#%
m_orig = cm.load(fnames_full[0:10])
#%
minvals = np.percentile(m_orig,5,(1,2))
maxvals = np.percentile(m_orig,95,(1,2)) - minvals 
#%
m_now = (m_orig-minvals[:,np.newaxis,np.newaxis])/maxvals[:,np.newaxis,np.newaxis]
#%
m_now.play(fr=20, magnification=.5,save_movie = True)  # press q to exit
#%%
#%% save movie - motion correction, denoising, etc
key =  {'session': 1,
 'subject_id': 462149,
 'cell_number': 1,
 'motion_correction_method': 'VolPy',
 'roi_type': 'VolPy'}
# =============================================================================
# key = {'session': 1,
#  'subject_id': 456462,
#  'cell_number': 3,
#  'motion_correction_method': 'Matlab',
#  'roi_type': 'SpikePursuit'}
# =============================================================================
movie_nums = (imaging_gt.GroundTruthROI()*imaging.Movie()&key).fetch('movie_number')
movie_name = (imaging_gt.GroundTruthROI()*imaging.Movie()&key &'movie_number = {}'.format(min(movie_nums))).fetch1('movie_name')
fnames,dirs = (imaging.MovieFile()*imaging_gt.GroundTruthROI()*imaging.Movie()&key &'movie_number = {}'.format(min(movie_nums))).fetch('movie_file_name','movie_file_directory')
allfnames = list()

for fname,directory in zip(fnames,dirs):
    allfnames.append(os.path.join(homefolder,directory,fname))
    
originaldir = os.path.join(homefolder,directory)
puthere = originaldir.find('Voltage_imaging')+len('Voltage_imaging')
denoiseddir = os.path.join(originaldir[:puthere],'denoised_volpy',originaldir[puthere+1:],movie_name)
volpydir = os.path.join(originaldir[:puthere],'VolPy',originaldir[puthere+1:],movie_name)
files = os.listdir(denoiseddir)
for file in files:
    if movie_name in file and file[-4:] == 'mmap':
        denoised_file = file
    if 'memmap_' in file and file[-4:] == 'mmap':
        motioncorrected_denoised_file = file
files = os.listdir(volpydir)
for file in files:
    if 'memmap_' in file and file[-4:] == 'mmap':
        volpy_file = file        

m_orig = cm.load(allfnames[:5])
m_denoised = cm.load(os.path.join(denoiseddir,denoised_file))
m_mocorr_denoised = cm.load(os.path.join(denoiseddir,motioncorrected_denoised_file))
m_volpy = cm.load(os.path.join(volpydir,volpy_file))
#%%
framenum = 1000
baseline_window = 200
offset = int(np.round(baseline_window/2))

m_orig_now = m_orig[:framenum,:,:].copy()
#m_orig_now =(m_orig_now - np.mean(m_orig_now , axis=0))
#m_orig_now =np.diff(m_orig_now,axis = 0)
m_orig_baseline = moving_average(m_orig_now, n=baseline_window)
m_orig_now = m_orig_now[offset:m_orig_baseline.shape[0]+offset,:,:]/m_orig_baseline
#%%
m_volpy_now = m_volpy[:framenum,:,:].copy()
#m_volpy_now=(m_volpy_now- np.mean(m_volpy_now, axis=0))
#m_volpy_now =np.diff(m_volpy_now,axis = 0)
#m_volpy_now  = (m_volpy_now - np.mean(m_volpy_now, axis=(1,2))[:,np.newaxis,np.newaxis])
m_volpy_baseline = moving_average(m_volpy_now, n=baseline_window)
m_volpy_now = m_volpy_now[offset:m_volpy_baseline.shape[0]+offset,:,:]/m_volpy_baseline
#%
m_denoised_now = m_denoised[:framenum,:,:]#.copy()
#m_denoised_now=(m_denoised_now- np.mean(m_denoised_now, axis=0))
#m_denoised_now =np.diff(m_denoised_now,axis = 0)
#m_denoised_now  = (m_denoised_now - np.mean(m_denoised_now, axis=(1,2))[:,np.newaxis,np.newaxis])
m_denoised_baseline = moving_average(m_denoised_now, n=baseline_window)
m_denoised_now = m_denoised_now[offset:m_denoised_baseline.shape[0]+offset,:,:]/m_denoised_baseline 

m_mocorr_denoised_now = m_mocorr_denoised[:framenum,:,:].copy()
#m_mocorr_denoised_now=(m_mocorr_denoised_now- np.mean(m_mocorr_denoised_now, axis=0))
#m_mocorr_denoised_now =np.diff(m_mocorr_denoised_now,axis = 0)
m_mocorr_denoised_baseline = moving_average(m_mocorr_denoised_now, n=baseline_window)
m_mocorr_denoised_now = m_mocorr_denoised_now[offset:m_mocorr_denoised_baseline.shape[0]+offset,:,:]/m_mocorr_denoised_baseline 
#%%

m_now =  cm.concatenate([m_orig_now,m_volpy_now, m_denoised_now,m_mocorr_denoised_now], axis=1)
#%%
m_now.play(fr=400, magnification=2,q_max=99.9, q_min=0.1,save_movie = True)
#m_orig = cm.load(allfnames[0:3])
#m_volpy_now.play(fr=400, magnification=1,q_max=99.5, q_min=0.5,save_movie = False)
#%% Szar van a palacsintaban
subject_ids,movie_names,frame_times,sessions,movie_numbers = (imaging.Movie*imaging.MovieFrameTimes()).fetch('subject_id','movie_name','frame_times','session','movie_number')
for subject_id,movie_name,frame_time,session,movie_number in zip(subject_ids,movie_names,frame_times,sessions,movie_numbers):
    frametimediff = np.diff(frame_time)
    if np.min(frametimediff)<.5*np.median(frametimediff):
        key = {'subject_id':subject_id,'movie_number':movie_number,'session':session}
        fig=plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.plot(np.diff(frame_time))
        ax.set_title([subject_id,movie_name])
        
        #(imaging.Movie()&key).delete()
    
    
  #%% comparing denoising to original motion corrected movie with caiman
# =============================================================================
# import caiman as cm
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from skimage import io as imio
# basedir ='/groups/svoboda/home/rozsam/Data/'
# 
# #data_dir = '/home/rozmar/Data/Voltage_imaging/VolPy/Voltage_rig_1P/rozsam/20191201/40x_patch1' #data_dir = sys.argv[1]
# out_dir = os.path.join(basedir,'Voltage_imaging/sgpmd-nmf/Voltage_rig_1P/rozsam/20191201/40x_patch1')#out_dir = sys.argv[3]
# data_dir = out_dir
# mov_in  = 'memmap__d1_128_d2_512_d3_1_order_C_frames_80000_.mmap'#mov_in = sys.argv[2]
# denoised = 'denoised.tif'
# trend = 'trend.tif'
# dtrend_nnorm = 'detr_nnorm.tif'
# sn_im = 'Sn_image.tif'
# #%%
# i_sn = imio.imread(os.path.join(out_dir,sn_im))[:,:,0]
# m_orig = cm.load(os.path.join(out_dir,mov_in))
# m_denoised = cm.load(os.path.join(out_dir,denoised)).transpose(2,0,1)
# m_trend = cm.load(os.path.join(out_dir,trend)).transpose(2,0,1)
# #m_denoised_w_trend = m_denoised + m_trend
# #m_dtrend_nnorm = cm.load(os.path.join(out_dir,dtrend_nnorm)).transpose(2,0,1)
# #m_noise_substracted = m_orig[:m_denoised.shape[0]]-(m_dtrend_nnorm-m_denoised)*i_sn
# #%%
# #m_pwrig = cm.load(mc.mmap_file)
# ds_ratio = 0.2
# moviehandle = cm.concatenate([m_orig[:m_denoised.shape[0]].resize(1, 1, ds_ratio),
#                               m_noise_substracted.resize(1, 1, ds_ratio)], axis=2)
# 
# moviehandle.play(fr=60, q_max=99.5, magnification=2)  # press q to exit
# #%%
# # % movie subtracted from the mean
# m_orig2 = (m_orig[:m_denoised.shape[0]] - np.mean(m_orig[:m_denoised.shape[0]], axis=0))
# m_denoised2 = (m_noise_substracted - np.mean(m_noise_substracted, axis=0))
# #%%
# ds_ratio = 0.2
# moviehandle1 = cm.concatenate([m_orig2.resize(1, 1, ds_ratio),
#                                m_denoised2.resize(1, 1, ds_ratio),], axis=2)
# moviehandle1.play(fr=60, q_max=99.5, magnification=2)  
# =============================================================================
