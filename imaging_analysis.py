import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import datajoint as dj
import os
dj.conn()
from pipeline import pipeline_tools
from pipeline import lab, experiment, ephys_patch, ephysanal, imaging, imaging_gt
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import time
from plot_imaging.plot_main import *
font = {'size'   : 16}

matplotlib.rc('font', **font)
from pathlib import Path

homefolder = dj.config['locations.mr_share']
#homefolder = str(Path.home())
def moving_average(a, n=3) : # moving average 
    if n>2:
        begn = int(np.ceil(n/2))
        endn = int(n-begn)-1
        a = np.concatenate([a[begn::-1],a,a[:-endn:-1]])
    ret = np.cumsum(a,axis = 0, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
#%%
fig=plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(cells_sp['SN'],cells_volpy['SN'],'ko')
ax.set_xlabel('S/N SpikePursuit')
ax.set_ylabel('S/N VolPy')
fig.savefig('./figures/SN_compared.png', bbox_inches = 'tight')
#%% potential roi types
roi_types = imaging.ROIType().fetch()
print('potential roi types: {}'.format(roi_types))
#%% Show S/N ratios
holding_min = -200 #pA
v0_max = -35 #mV
roi_type = 'SpikePursuit'#'Spikepursuit'#'VolPy_denoised'#'SpikePursuit'#'VolPy_dexpF0'#'VolPy'#'SpikePursuit_dexpF0'#'VolPy_dexpF0'#''Spikepursuit'#'VolPy'#
key = {'roi_type':roi_type}
gtdata = pd.DataFrame((imaging_gt.GroundTruthROI()&key))
cells = gtdata.groupby(['session', 'subject_id','cell_number','motion_correction_method','roi_type']).size().reset_index(name='Freq')
snratio = list()
v0s = list()
holdings = list()
rss = list()
for cell in cells.iterrows():
    cell = cell[1]
    key_cell = dict(cell)    
    del key_cell['Freq']
    snratios = (imaging_gt.GroundTruthROI()*imaging_gt.ROIAPWave()&key_cell).fetch('apwave_snratio')
    sweep = (imaging_gt.GroundTruthROI()*imaging_gt.ROIAPWave()&key_cell).fetch('sweep_number')[0]
    trace = (ephys_patch.SweepResponse()*imaging_gt.GroundTruthROI()&key_cell&'sweep_number = {}'.format(sweep)).fetch('response_trace')
    trace =trace[0]
    stimulus = (ephys_patch.SweepStimulus()*imaging_gt.GroundTruthROI()&key_cell&'sweep_number = {}'.format(sweep)).fetch('stimulus_trace')
    stimulus =stimulus[0]
    
    RS = (ephysanal.SweepSeriesResistance()*imaging_gt.GroundTruthROI()&key_cell&'sweep_number = {}'.format(sweep)).fetch('series_resistance')
    RS =RS[0]
    
    medianvoltage = np.median(trace)*1000
    holding = np.median(stimulus)*10**12
    #print(np.mean(snratios[:100]))    
    snratio.append(np.mean(snratios[:50]))
    v0s.append(medianvoltage)
    holdings.append(holding)
    rss.append(RS)
#%
cells['SN']=snratio
cells['V0']=v0s
cells['holding']=holdings
cells['RS']=np.asarray(rss,float)
print(cells)
cells = cells[cells['V0']<v0_max]
cells = cells[cells['holding']>holding_min]
print(cells)
#% S/N ratio histogram
fig=plt.figure()
ax_hist = fig.add_axes([0,0,1,1])
ax_hist.hist(cells['SN'].values)
ax_hist.set_xlabel('S/N ratio of first 50 spikes')
ax_hist.set_ylabel('# of cells')
ax_hist.set_title(roi_type.replace('_',' '))
ax_hist.set_xlim([0,15])
fig.savefig('./figures/SN_hist_{}.png'.format(roi_type), bbox_inches = 'tight')
#%% ephys recording histograms

fig=plt.figure()
ax_hist = fig.add_axes([0,0,1,1])
ax_hist.hist(cells['RS'])
ax_hist.set_xlabel('Access resistance (MOhm)')
ax_hist.set_ylabel('# of cells')
fig.savefig('./figures/RS_hist.png'.format(roi_type), bbox_inches = 'tight')

fig=plt.figure()
ax_hist = fig.add_axes([0,0,1,1])
ax_hist.hist(cells['V0'])
ax_hist.set_xlabel('Resting membrane potential during movie (mV)')
ax_hist.set_ylabel('# of cells')
fig.savefig('./figures/V0_hist.png'.format(roi_type), bbox_inches = 'tight')
                   
fig=plt.figure()
ax_hist = fig.add_axes([0,0,1,1])
ax_hist.hist(cells['holding'])
ax_hist.set_xlabel('Injected current (pA)')
ax_hist.set_ylabel('# of cells')
fig.savefig('./figures/holding_hist.png'.format(roi_type), bbox_inches = 'tight')                 
#%% IVs
for cell in cells.iterrows():
    ivnum = 0
    try:
        fig = plot_IV(subject_id = cell[1]['subject_id'], cellnum = cell[1]['cell_number'], ivnum = ivnum,IVsweepstoplot = None)
        fig.savefig('./figures/IV_{}_{}_iv{}.png'.format(cell[1]['subject_id'],cell[1]['cell_number'],ivnum), bbox_inches = 'tight')         
    except:
        print('no such IV')
    #break
    #print(cell)

          
#%% PLOT AP waveforms

session = 1
subject_id = 456462
cell_number = 3
roi_type = 'Spikepursuit'#'Spikepursuit'#'VolPy_denoised'#'SpikePursuit'#'VolPy_dexpF0'#'VolPy'#'SpikePursuit_dexpF0'#'VolPy_dexpF0'#''Spikepursuit'#'VolPy'#

binwidth = 30 #s
firing_rate_window = 1 #s
frbinwidth = .01

AP_tlimits = [-.005,.01] #s

#%
key = {'session':session,'subject_id':subject_id,'cell_number':cell_number,'roi_type':roi_type }
gtdata = pd.DataFrame((imaging_gt.GroundTruthROI()&key))
key_cell = gtdata.groupby(['session', 'subject_id','cell_number','motion_correction_method','roi_type']).size().reset_index(name='Freq')
for key_cell in key_cell.iterrows():key_cell = dict(key_cell[1]); del key_cell['Freq']
plot_AP_waveforms(key_cell,AP_tlimits)
#%%
plot_precision_recall(key_cell,binwidth =  binwidth ,frbinwidth = frbinwidth,firing_rate_window =  firing_rate_window)    
# =============================================================================
# data = plot_ephys_ophys_trace(key_cell,time_to_plot=15,trace_window = 5,show_stimulus = True,show_e_ap_peaks = True,show_o_ap_peaks = True)
# data['figure_handle'].savefig('./figures/{}_cell_{}_roi_type_{}_short.png'.format(key_cell['subject_id'],key_cell['cell_number'],key_cell['roi_type']), bbox_inches = 'tight')
# =============================================================================
#%%
data = plot_ephys_ophys_trace(key_cell,time_to_plot=None,trace_window = 50,show_stimulus = True,show_e_ap_peaks = True,show_o_ap_peaks = True)
data['figure_handle'].savefig('./figures/{}_cell_{}_roi_type_{}_long.png'.format(key_cell['subject_id'],key_cell['cell_number'],key_cell['roi_type']), bbox_inches = 'tight')
#%%
plot_precision_recall(key_cell,binwidth =  binwidth ,frbinwidth = frbinwidth,firing_rate_window =  firing_rate_window)    
#%%
plot_precision_recall(key_cell,binwidth =  30,frbinwidth = 0.001,firing_rate_window = 1)    
#%%
plot_ephys_ophys_trace(key_cell,time_to_plot=250,trace_window = 100,show_stimulus = True,show_e_ap_peaks = True,show_o_ap_peaks = True)
#%% plot everything..
binwidth = 30 #s
firing_rate_window = 1 #s
frbinwidth = .01

#cell_index = 15
for cell_index in range(len(cells)):

    cell = cells.iloc[cell_index]
    key_cell = dict(cell)    
    del key_cell['Freq']
    if imaging.Movie&key_cell:#&'movie_frame_rate>800':
        #%
        plot_precision_recall(key_cell,binwidth =  binwidth ,frbinwidth = frbinwidth,firing_rate_window =  firing_rate_window)    
# =============================================================================
#         data = plot_ephys_ophys_trace(key_cell,time_to_plot=None,trace_window = 50,show_stimulus = True,show_e_ap_peaks = True,show_o_ap_peaks = True)
#         data['figure_handle'].savefig('./figures/{}_cell_{}_roi_type_{}_long.png'.format(key_cell['subject_id'],key_cell['cell_number'],key_cell['roi_type']), bbox_inches = 'tight')
#         print(cell)
# =============================================================================
#%%    #%%
data = plot_ephys_ophys_trace(key_cell,time_to_plot=25,trace_window = 1,show_e_ap_peaks = True,show_o_ap_peaks = True)
    #%%
data = plot_ephys_ophys_trace(key_cell,time_to_plot=None,trace_window = 50,show_stimulus = True,show_e_ap_peaks = True,show_o_ap_peaks = True)
    #%%

    

#%%plot time offset between 1st ROI and real elphys
min_corr_coeff = .1
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
#%% subthreshold correlations
convolve_voltron_kinetics = True
tau_1_on = .64/1000
tau_2_on = 4.1/1000
tau_1_ratio_on =  .61
tau_1_off = .78/1000
tau_2_off = 3.9/1000
tau_1_ratio_off = 55

movingaverage_windows = [0,.01,.02,.03,.04,.05,.1]    

session = 1
subject_id = 456462
cell_number = 3
roi_type = 'VolPy_denoised_raw'#'Spikepursuit_dexpF0'#'VolPy_denoised'#'SpikePursuit'#'VolPy_dexpF0'#'VolPy'#'SpikePursuit_dexpF0'#'VolPy_dexpF0'#''Spikepursuit'#'VolPy'#
key = {'session':session,'subject_id':subject_id,'cell_number':cell_number,'roi_type':roi_type }
gtdata = pd.DataFrame((imaging_gt.GroundTruthROI()&key))
key_cell = gtdata.groupby(['session', 'subject_id','cell_number','motion_correction_method','roi_type']).size().reset_index(name='Freq')
for key_cell in key_cell.iterrows():key_cell = dict(key_cell[1]); del key_cell['Freq']

movie_numbers,famerates = ((imaging_gt.GroundTruthROI()*imaging.Movie())&key).fetch('movie_number','movie_frame_rate')
session_time = (experiment.Session()&key_cell).fetch('session_time')[0]
cell_time = (ephys_patch.Cell()&key_cell).fetch('cell_recording_start')[0]
for movie_number in movie_numbers:
    
    frame_rate = ((imaging.Movie())&key_cell & 'movie_number = '+str(movie_number)).fetch('movie_frame_rate')[0]
    #frame_num = ((imaging.Movie())&key_cell & 'movie_number = '+str(movie_number)).fetch('movie_frame_num')[0]
    movie_start_time = float(((imaging.Movie())&key_cell & 'movie_number = '+str(movie_number)).fetch('movie_start_time')[0])
    movie_start_time = session_time.total_seconds() + movie_start_time - cell_time.total_seconds()
    movie_time = (imaging.MovieFrameTimes()&key_cell & 'movie_number = '+str(movie_number)).fetch('frame_times')[0] -cell_time.total_seconds() +session_time.total_seconds()
    movie_end_time = movie_time[-1]

    sweeps_needed = ephys_patch.Sweep()&key_cell&'sweep_start_time < '+str(movie_end_time) & 'sweep_end_time > '+str(movie_start_time)
    sweep_start_ts, sweep_end_ts, traces,sweep_nums, sample_rates= (sweeps_needed*ephys_patch.SweepResponse()*ephys_patch.SweepMetadata()).fetch('sweep_start_time','sweep_end_time','response_trace','sweep_number','sample_rate')
    trace_times = list()
    for sweep_start_t, sweep_end_t, trace,sample_rate in zip(sweep_start_ts, sweep_end_ts, traces,sample_rates):
        trace_times.append(np.arange(float(sweep_start_t), float(sweep_end_t)+1/sample_rate,1/sample_rate))#np.arange(len(trace))/sample_rate+float(sweep_start_t)
    #%
    dff = (imaging.ROI()*imaging_gt.GroundTruthROI()&key_cell&'movie_number = {}'.format(movie_number)).fetch1('roi_dff')
    for trace,tracetime,sweep_number,sample_rate in zip(traces,trace_times,sweep_nums,sample_rates):  
        #%
        
        start_t = tracetime[0]
        start_t = movie_time[np.argmax(movie_time>=start_t)]
        end_t = np.min([tracetime[-1],movie_time[-1]])
        t = np.arange(0,.01,1/sample_rate)
        
        if convolve_voltron_kinetics:
            f_on = tau_1_ratio_on*np.exp(t/tau_1_on) + (1-tau_1_ratio_on)*np.exp(-t/tau_2_on)
            f_off = tau_1_ratio_off*np.exp(t[::-1]/tau_1_off) + (1-tau_1_ratio_off)*np.exp(-t[::-1]/tau_2_off)
            f_on = f_on/np.max(f_on)
            f_off = f_off/np.max(f_off)
            kernel = np.concatenate([f_on,np.zeros(len(f_off))])[::-1]
            kernel  = kernel /sum(kernel )
            trace_conv = np.convolve(trace,kernel,mode = 'same') 
        else:
            trace_conv = trace
        
        kernel = np.ones(int(np.round(sample_rate/frame_rate)))
        kernel  = kernel /sum(kernel )
        trace_conv = np.convolve(trace_conv,kernel,mode = 'same') 
        trace_conv_time   = tracetime#[down_idxes]
        
        f_idx_now = (movie_time>=start_t) & (movie_time<=end_t)
        dff_time_now = movie_time[f_idx_now]
        e_idx_original = list()
        for t in dff_time_now:
            e_idx_original.append(np.argmin(trace_conv_time<t))
        #timelag = imaging_gt.GroundTruthROI()*imaging_gt.ROIEphysCorrelation()&key_cell&'sweep_number = {}'.format(sweep_number)
        
            #%
        e_vals = trace[e_idx_original]
        f_vals = dff[f_idx_now]
        #%
        for movingaverage_window in movingaverage_windows:
            if movingaverage_window >1/frame_rate:
                e_vals_filt = moving_average(e_vals,int(np.round(movingaverage_window/(1/frame_rate))))
                f_vals_filt = moving_average(f_vals,int(np.round(movingaverage_window/(1/frame_rate))))
                
             #%
            
            fig=plt.figure()
            ax = fig.add_axes([0,0,1.2,1.2])
            if movingaverage_window >1/frame_rate:
                ax.hist2d(e_vals_filt*1000,f_vals_filt,150,[[np.min(e_vals_filt)*1000,-20],[np.min(f_vals_filt),np.max(f_vals_filt)]],norm=colors.PowerNorm(gamma=0.3),cmap =  plt.get_cmap('jet'))
            else:
                ax.hist2d(e_vals*1000,f_vals,150,[[np.min(e_vals)*1000,-20],[np.min(f_vals),np.max(f_vals)]],norm=colors.PowerNorm(gamma=0.3),cmap =  plt.get_cmap('jet'))
            ax.set_xlabel('mV')
            ax.set_ylabel('dF/F')
            ax.invert_yaxis()
            if movingaverage_window >1/frame_rate:
                ax.set_title('subject: {} cell: {} movie: {} sweep: {} moving average: {} ms'.format(key_cell['subject_id'],key_cell['cell_number'],movie_number,sweep_number,movingaverage_window*1000))
                fig.savefig('./figures/subthreshold_{}_cell{}_movie{}_sweep{}_{}_averaging_{}ms.png'.format(key_cell['subject_id'],key_cell['cell_number'],movie_number,sweep_number,roi_type,int(np.round(movingaverage_window*1000))), bbox_inches = 'tight')
            else:
                ax.set_title('subject: {} cell: {} movie: {} sweep: {}'.format(key_cell['subject_id'],key_cell['cell_number'],movie_number,sweep_number))
                fig.savefig('./figures/subthreshold_{}_cell{}_movie{}_sweep{}_{}.png'.format(key_cell['subject_id'],key_cell['cell_number'],movie_number,sweep_number,roi_type), bbox_inches = 'tight')
        #%
        #break
    #%%

    
        
        
        




#%% save movie - IR + voltron
from pathlib import Path
import os
import caiman as cm
import numpy as np
#%

def moving_average(a, n=3) : # moving average 
    if n>2:
        begn = int(np.ceil(n/2))
        endn = int(n-begn)-1
        a = np.concatenate([a[begn::-1],a,a[:-endn:-1]])
    ret = np.cumsum(a,axis = 0, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
#%% patching movie
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
key = {'session': 1,
 'subject_id': 456462,
 'cell_number': 3,
 'motion_correction_method': 'Matlab',
 'roi_type': 'SpikePursuit'}
movie_nums = (imaging_gt.GroundTruthROI()*imaging.Movie()&key).fetch('movie_number')
movie_name = (imaging_gt.GroundTruthROI()*imaging.Movie()&key &'movie_number = {}'.format(min(movie_nums))).fetch1('movie_name')
session_date = str((experiment.Session()*imaging_gt.GroundTruthROI()*imaging.Movie()&key &'movie_number = {}'.format(min(movie_nums))).fetch1('session_date'))
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
startframe = 9900
framenum = 200
baseline_window = 200
baselinesubtract = False
#offset = int(np.round(baseline_window/2))

m_orig_now = m_orig[startframe:startframe+framenum,:,:]
m_orig_now = m_orig_now.copy()
#m_orig_now =(m_orig_now - np.mean(m_orig_now , axis=0))
#m_orig_now =np.diff(m_orig_now,axis = 0)
if baselinesubtract :
    m_orig_baseline = moving_average(m_orig_now, n=baseline_window)
    m_orig_now = m_orig_now/m_orig_baseline
#%
m_volpy_now = m_volpy[startframe:startframe+framenum,:,:]
m_volpy_now = m_volpy_now.copy()
#m_volpy_now=(m_volpy_now- np.mean(m_volpy_now, axis=0))
#m_volpy_now =np.diff(m_volpy_now,axis = 0)
#m_volpy_now  = (m_volpy_now - np.mean(m_volpy_now, axis=(1,2))[:,np.newaxis,np.newaxis])
if baselinesubtract :
    m_volpy_baseline = moving_average(m_volpy_now, n=baseline_window)
    m_volpy_now = m_volpy_now/m_volpy_baseline#[offset:m_volpy_baseline.shape[0]+offset,:,:]
#%%
m_denoised_now = m_denoised[startframe:startframe+framenum,:,:]
m_denoised_now = m_denoised_now.copy()
#m_denoised_now=(m_denoised_now- np.mean(m_denoised_now, axis=0))
#m_denoised_now =np.diff(m_denoised_now,axis = 0)
#m_denoised_now  = (m_denoised_now - np.mean(m_denoised_now, axis=(1,2))[:,np.newaxis,np.newaxis])
if baselinesubtract :
    m_denoised_baseline = moving_average(m_denoised_now, n=baseline_window)
    m_denoised_now = m_denoised_now/m_denoised_baseline 
#%
m_mocorr_denoised_now = m_mocorr_denoised[startframe:startframe+framenum,:,:]#.copy()
#m_mocorr_denoised_now=(m_mocorr_denoised_now- np.mean(m_mocorr_denoised_now, axis=0))
#m_mocorr_denoised_now =np.diff(m_mocorr_denoised_now,axis = 0)
if baselinesubtract :
    m_mocorr_denoised_baseline = moving_average(m_mocorr_denoised_now, n=baseline_window)
    m_mocorr_denoised_now = m_mocorr_denoised_now/m_mocorr_denoised_baseline 
#%

m_now =  cm.concatenate([m_orig_now,m_volpy_now, m_denoised_now,m_mocorr_denoised_now], axis=1)#
#%%
#%%
m_now =  cm.concatenate([m_orig_now,m_volpy_now], axis=1)#
#%%
m_now.play(fr=900, magnification=2,q_max=99.9, q_min=0.1,save_movie = True)
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
