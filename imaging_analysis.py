import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import datajoint as dj
dj.conn()
from pipeline import pipeline_tools
from pipeline import lab, experiment, ephys_patch, ephysanal, imaging, imaging_gt
import matplotlib.pyplot as plt
import pandas as pd
import time
from plot_imaging.plot_main import *
font = {'size'   : 16}

matplotlib.rc('font', **font)
#%% Precision and recall of spike times
binwidth = 30 #s
firing_rate_window = 3 #s
frbinwidth = .01

roi_type = 'VolPy'#'VolPy_dexpF0'#'SpikePursuit_dexpF0'#'VolPy'#'Spikepursuit'#
key = {'roi_type':roi_type}
gtdata = pd.DataFrame((imaging_gt.GroundTruthROI()&key))
#%
cells = gtdata.groupby(['session', 'subject_id','cell_number','motion_correction_method','roi_type']).size().reset_index(name='Freq')
for cell in cells.iterrows():
    cell = cell[1]
    key_cell = dict(cell)    
    del key_cell['Freq']
    if imaging.Movie&key_cell:#&'movie_frame_rate>800':
        #%
        plot_precision_recall(key_cell,binwidth =  30,frbinwidth = 0.01,firing_rate_window = 3)
        
        print(cell)
        time.sleep(3)
        #%
#%%
data = plot_ephys_ophys_trace(key_cell,time_to_plot=None,trace_window = 100)
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
    
    
    