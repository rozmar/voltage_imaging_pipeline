from utils.metaarray import * # to import the very first recording...
import h5py as h5
import pickle
from utils import configfile
import matplotlib.pyplot as plt
from pathlib import Path
import datetime
import time as timer
import pandas as pd
import numpy as np
import datajoint as dj
from PIL import Image
from scipy.io import loadmat
import h5py
import scipy
from skimage.feature import register_translation
import scipy.signal as signal
import re 
import os
import shutil
dj.conn()
from pipeline import pipeline_tools
from pipeline import lab, experiment, ephys_patch, ephysanal, imaging, imaging_gt
import ray
import time

homefolder = dj.config['locations.mr_share']
#%%

def moving_average(a, n=3) : # moving average 
    if n>2:
        begn = int(np.ceil(n/2))
        endn = int(n-begn)-1
        a = np.concatenate([a[begn::-1],a,a[:-endn:-1]])
    ret = np.cumsum(a,axis = 0, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
#%%
@ray.remote
def apply_motion_correction(movie,shifts):
    movie_corrected=list()
    for frame,shift in zip(movie,shifts):
        movie_corrected.append(scipy.ndimage.shift(frame,shift))
    return movie_corrected

@ray.remote
def populatemytables_gt_core_paralel(arguments,runround):
    if runround ==1:
        imaging_gt.GroundTruthROI().populate(**arguments)
        
def populatemytables_gt_core(arguments,runround):
    if runround ==1:
        imaging_gt.GroundTruthROI().populate(**arguments)

def populatemytables_gt(paralel = True, cores = 3):

    ray.init(num_cpus = cores)
    for runround in [1]:
        arguments = {'display_progress' : False, 'reserve_jobs' : True,'order' : 'random'}
        print('round '+str(runround)+' of populate')
        if paralel and cores>1:
            result_ids = []
            for coreidx in range(cores):
                result_ids.append(populatemytables_gt_core_paralel.remote(arguments,runround))        
            ray.get(result_ids)
            arguments = {'display_progress' : True, 'reserve_jobs' : False}
        populatemytables_gt_core(arguments,runround)
    ray.shutdown()

#%%
def read_tiff(path,ROI_coordinates=None, n_images=100000):
    """
    path - Path to the multipage-tiff file
    n_images - Number of pages in the tiff file
    """
    img = Image.open(path)

    images = []
    #dimensions = np.diff(ROI_coordinates.T).T[0]+1
    for i in range(10000000):
        try:
            img.seek(i)
            img.getpixel((1, 1))
            imarray = np.array(img)
            if ROI_coordinates:
                slice_ = imarray[ROI_coordinates[0][1]:ROI_coordinates[1][1]+1,ROI_coordinates[0][0]:ROI_coordinates[1][0]+1]
                images.append(slice_)
            else:
                images.append(imarray)
            
           # break
        except EOFError:
            # Not enough frames in img
            break

    return np.array(images)
#%%
def upload_spike_pursuit_matlab_data(key_now,movie_dir_now):
    #%%
    files = os.listdir(movie_dir_now)
    movie_x_size, movie_y_size = (imaging.Movie()&key_now).fetch1('movie_x_size','movie_y_size')
    if 'ROIs.mat' in files:
        print(['starting',key_now])
        roifile =  loadmat(os.path.join(movie_dir_now, 'ROIs.mat'))
        try:
            cellnum = np.size(roifile['ROIs'],2)
        except:
            cellnum = 1
            print('only one ROI')
        
        if len(imaging.ROI()&key_now &'roi_type = "SpikePursuit"'&'roi_number = '+str(cellnum)) == 0:

            for cellid in np.arange(cellnum)+1:
                #%
                blocknum = 1
                spikeidxes = list()
                dff = list()
                f0 = list()
                framenums = list()
                motion_corr_vectors = list()
                meanimages = list()
                ROI_coordinates = list()
                spatial_filters = list()
                shifts = list()
                fs = list()
                while 'datasetblock'+str(blocknum)+'Cell'+str(cellid)+'.mat' in files:
                    print('datasetblock'+str(blocknum)+'Cell'+str(cellid)+'.mat')
                    #%
                    try:
                        with h5py.File(os.path.join(movie_dir_now, 'datasetblock'+str(blocknum)+'.mat'), 'r') as f:
                            alignshifts = f['alignShifts'][:]
                    except:
                        alignshifts = None
                    meanimage = loadmat(os.path.join(movie_dir_now, 'datasetblock'+str(blocknum)+'_meanimage.mat'))
                    meanimage = meanimage['data_mean']
                    celldata =  loadmat(os.path.join(movie_dir_now, 'datasetblock'+str(blocknum)+'Cell'+str(cellid)+'.mat'))
                    celldata = celldata['output'][0][0]
                    ROI_coordinate = celldata['ROI']-1
                    if len(motion_corr_vectors) == 0:
                        motion_corr_vectors = [alignshifts]
                        meanimages = [meanimage]
                        spikeidxes = [np.asarray(celldata['spikeTimes'][0],dtype = int)+sum(framenums)]
                        ROI_coordinates = [celldata['ROI']-1]
                        dff = [celldata['dFF'].T[0]]
                        f0 = [celldata['F0'].T[0]]
                        spatial_filters = [celldata['spatialFilter']*-1]
                    else:                                    
                        shift, error, diffphase = register_translation(meanimages[-1], meanimage, 10)
                        shifts.append(shift)
                        motion_corr_vectors.append(alignshifts+shift)
                        meanimages.append(meanimage)
                        spikeidxes.append(np.asarray(celldata['spikeTimes'][0],dtype = int)+sum(framenums))
                        ROI_coordinates.append(celldata['ROI']-1)
                        dff.append(celldata['dFF'].T[0])
                        f0.append(celldata['F0'][0])
                        spatial_filters.append(celldata['spatialFilter']*-1)
                    framenums.append(len(dff[-1]))
                    with h5py.File(os.path.join(movie_dir_now, 'datasetblock'+str(blocknum)+'.mat'), 'r') as f:
                        fl = f['data'][:,ROI_coordinate[0][0]:ROI_coordinate[1][0]+1,ROI_coordinate[0][1]:ROI_coordinate[1][1]+1]
                    spatial_filter = spatial_filters[-1]/sum(sum(spatial_filters[-1]))
                    filtered = np.multiply(fl, spatial_filter.T[np.newaxis, :,:])
                    f = np.mean(filtered,axis = (1,2))
                    fs.append(f)
                    blocknum += 1
                if len(dff)>0:
                    if len(imaging.RegisteredMovie()&key_now &'motion_correction_method = "Matlab"')==0:
                        key_reg_movie = key_now.copy()
                        key_reg_movie['motion_correction_method'] = 'Matlab'
                        key_reg_movie['registered_movie_mean_image'] = np.mean(np.stack(meanimages),axis = 0)
                        imaging.RegisteredMovie().insert1(key_reg_movie, allow_direct_insert=True)
                        key_motioncorr = key_now.copy()
                        key_motioncorr['motion_correction_method'] = 'Matlab'
                        key_motioncorr['motion_correction_id'] = 0
                        key_motioncorr['motion_corr_description'] = 'rigid motion correction done with dftregistration'
                        key_motioncorr['motion_corr_vectors'] = np.concatenate(motion_corr_vectors)
                        imaging.MotionCorrection().insert1(key_motioncorr, allow_direct_insert=True)
                    while not type(spikeidxes[0]) == np.int64:
                        spikeidxes = np.concatenate(spikeidxes)
                        dff = np.concatenate(dff)
                        f0 = np.concatenate(f0)
                    f = np.concatenate(fs)*np.mean(meanimages[0])
                    mask = np.zeros([int(movie_y_size),int(movie_x_size)])
                    ROI_coordinate = ROI_coordinates[0]
                    mask[ROI_coordinate[0][1]:ROI_coordinate[1][1]+1,ROI_coordinate[0][0]:ROI_coordinate[1][0]+1] = spatial_filters[0]
                    #%
                    key_roi = key_now.copy()
                    key_roi['motion_correction_method'] = 'Matlab'
                    key_roi['roi_number'] = cellid
                    key_roi['roi_type'] = 'SpikePursuit'
                    key_roi['roi_dff'] = dff
                    key_roi['roi_f0'] = f0
                    key_roi['roi_spike_indices'] = spikeidxes
                    key_roi['roi_centroid_x'] = np.mean(ROI_coordinates[0],0)[0]
                    key_roi['roi_centroid_y'] = np.mean(ROI_coordinates[0],0)[1]
                    key_roi['roi_mask'] = mask
                    try:
                        imaging.ROI().insert1(key_roi, allow_direct_insert=True)
                    except:
                        print('original spikepursuit ROI already uploaded?')
                        #%
                    try:   
                        t = np.arange(len(f))
                        out = scipy.optimize.curve_fit(lambda t,a,b,c,aa,bb: a*np.exp(-t/b) + c + aa*np.exp(-t/bb),  t,  f,maxfev=20000)#,bounds=(0, [np.inf,np.inf,np.inf])
                        a = out[0][0]
                        b = out[0][1]
                        c = out[0][2]
                        aa = out[0][3]
                        bb = out[0][4]                #break
                        f0 = a*np.exp(-t/b) + c + aa*np.exp(-t/bb)
                        dff = (f-f0)/f0
                        key_roi['roi_dff'] = dff
                        key_roi['roi_f0'] = f0
                        key_roi['roi_type'] = 'SpikePursuit_dexpF0'
                        imaging.ROI().insert1(key_roi, allow_direct_insert=True)
                    except:
                        print('couldn''t fit double exponential? ')
                        #%
    else:
        print(['skipping',print(key_now)])
                    
#%%
        #print(movie_name)      

@ray.remote
def ROIEphysCorrelation_ROIAPwave_populate(key,movie_number,cellnum):
    #%%
# =============================================================================
#     key = {'subject_id': 463291, 'session': 1}
#     movie_number = 5
#     cellnum = 1
# =============================================================================
    
    
    
    plotit = False
    convolve_voltron_kinetics = False
    tau_1_on = .64/1000
    tau_2_on = 4.1/1000
    tau_1_ratio_on =  .61
    tau_1_off = .78/1000
    tau_2_off = 3.9/1000
    tau_1_ratio_off = 55
    session_time = (experiment.Session()&key).fetch('session_time')[0]
    cell_time = (ephys_patch.Cell()&key&'cell_number = '+str(cellnum)).fetch('cell_recording_start')[0]
    
    
    motion_corr_methods = (imaging.ROI()&key & 'movie_number = '+str(movie_number)).fetch('motion_correction_method')
    roi_numbers = (imaging.ROI()&key & 'movie_number = '+str(movie_number)).fetch('roi_number')
    roi_types = (imaging.ROI()&key & 'movie_number = '+str(movie_number)).fetch('roi_type')
    frame_rate = ((imaging.Movie())&key & 'movie_number = '+str(movie_number)).fetch('movie_frame_rate')[0]
    #frame_num = ((imaging.Movie())&key & 'movie_number = '+str(movie_number)).fetch('movie_frame_num')[0]
    movie_start_time = float(((imaging.Movie())&key & 'movie_number = '+str(movie_number)).fetch('movie_start_time')[0])
    movie_start_time = session_time.total_seconds() + movie_start_time - cell_time.total_seconds()
    movie_time = (imaging.MovieFrameTimes()&key & 'movie_number = '+str(movie_number)).fetch('frame_times')[0] -cell_time.total_seconds() +session_time.total_seconds()
    movie_end_time = movie_time[-1]
    sweeps_needed = ephys_patch.Sweep()&key&'cell_number = '+str(cellnum)&'sweep_start_time < '+str(movie_end_time) & 'sweep_end_time > '+str(movie_start_time)
    sweep_start_ts, sweep_end_ts, traces,sweep_nums, sample_rates= (sweeps_needed*ephys_patch.SweepResponse()*ephys_patch.SweepMetadata()).fetch('sweep_start_time','sweep_end_time','response_trace','sweep_number','sample_rate')
    trace_times = list()
    for sweep_start_t, sweep_end_t, trace,sample_rate in zip(sweep_start_ts, sweep_end_ts, traces,sample_rates):
        trace_times.append(np.arange(float(sweep_start_t), float(sweep_end_t)+1/sample_rate,1/sample_rate))#np.arange(len(trace))/sample_rate+float(sweep_start_t)
    for trace,tracetime,sweep_number,sample_rate in zip(traces,trace_times,sweep_nums,sample_rates):
        convolvingdone = False
        for roi_number,motion_correction_method,roi_type in zip(roi_numbers,motion_corr_methods,roi_types):
            #%
            try:
                roikey = key.copy()
                roikey['movie_number'] = movie_number
                roikey['roi_number'] = roi_number
                roikey['motion_correction_method'] = motion_correction_method
                roikey['roi_type'] = roi_type
                roikey['sweep_number'] = sweep_number
                if len(imaging_gt.ROIEphysCorrelation()&roikey) ==0 or len(imaging_gt.ROIAPWave()&roikey) ==0:
                    if not convolvingdone:
                        start_t = tracetime[0]
                        start_t = movie_time[np.argmax(movie_time>=start_t)]
                        end_t = np.min([tracetime[-1],movie_time[-1]])
                        apmaxtimes, apmaxidxes = ((sweeps_needed*ephysanal.ActionPotential())&'sweep_number = '+str(sweep_number)).fetch('ap_max_time','ap_max_index')
                        if plotit:
                            fig=plt.figure()
                            ax_v=fig.add_axes([0,0,2,.8])
                            ax_v.plot(tracetime,trace*1000,'k-')
                            ax_v.plot(tracetime[list(apmaxidxes)],trace[list(apmaxidxes)]*1000,'ro')
                            #ax_v.set_xlim([end_t-3,end_t])#end_t
                        
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
                        convolvingdone = True
                            
                    
    
                    dff = (imaging.ROI()&roikey).fetch('roi_dff')[0]
    # =============================================================================
    #                 if len(f_idx_now)>len(dff) or len(e_idx_original)>len(dff):
    #                     print('movie length not correct.. cutting indexes')
    #                     f_idx_now = f_idx_now[:len(dff)]
    #                     e_idx_original = e_idx_original[:len(dff)]
    # =============================================================================
                    t = movie_time-movie_time[0]
                    sos = signal.butter(5, 10, 'hp', fs=frame_rate, output='sos')
                    dff_filt = signal.sosfilt(sos, dff)   
                    try: #TODO debug me! roikey = {}?????
                        dff_now = dff[f_idx_now]
                    except:
                        print(roikey)
                        dff_now = dff[f_idx_now]
                    dff_filt_now = dff_filt[f_idx_now]
                    spiketimes = (imaging.ROI()&key & 'movie_number = '+str(movie_number)&'roi_number = '+str(roi_number)).fetch('roi_spike_indices')[0]
                    if plotit:
                        ax_i=fig.add_axes([0,-roi_number,2,.8])
                        ax_i.plot(movie_time,dff*-1)
                        ax_i.plot(movie_time[list(spiketimes-1)],dff[list(spiketimes-1)]*-1,'ro')
                        ax_i.set_xlim([start_t,end_t])#start_t,end_t
                        ax_i.set_ylim([-.01,.02])#start_t,end_t
                        ax_v.set_xlim([start_t,end_t])#start_t,end_t
                        ax_i.set_title('frame rate: '+str(round(frame_rate)))
            
                    mean_value = 0
                    for range_end,step in zip([1000000,100000,10000,1000,100],[50000,5000,500,50,5]):
                        corr_values = list()
                        tdiffs = np.arange(-range_end,range_end,step)+mean_value
                        for tdiff in tdiffs:     
                            #%
                            e_idx_now = np.asarray(e_idx_original)+tdiff
                            e_idx_now[e_idx_now>=len(trace_conv)] = len(trace_conv)-1
                            e_idx_now[e_idx_now<0] = 0
                            #e_idx_now = np.unique(e_idx_now)
                            e_now = trace_conv[e_idx_now]
                            #%
                            if len(e_now) != len(dff_now):
                                print([key,movie_number,cellnum])
                                
                            corr = np.corrcoef(e_now,dff_now)
                            corr_values.append(corr[0][1])
                        mean_value = np.arange(-range_end,range_end,step)[np.argmin(corr_values)]+mean_value
                        
                    time_offset = (tdiffs/sample_rate*1000)[np.argmin(corr_values)]
                    corr_value = np.min(corr_values)
                    key_roi_ephys = key.copy()
                    key_roi_ephys['movie_number'] = movie_number
                    key_roi_ephys['motion_correction_method'] = motion_correction_method
                    key_roi_ephys['roi_type'] = roi_type
                    key_roi_ephys['roi_number'] = roi_number
                    key_roi_ephys['cell_number'] = cellnum
                    key_roi_ephys['sweep_number'] = sweep_number
                    key_roi_ephys['time_lag'] = time_offset
                    key_roi_ephys['corr_coeff'] = corr_value
                    print(key_roi_ephys)
                    try:
                        imaging_gt.ROIEphysCorrelation().insert1(key_roi_ephys, allow_direct_insert=True)
                    except:
                        print('correlation already filled in.. hm...')
                    if plotit:
                        plt.plot(tdiffs/sample_rate*1000,corr_values)
                    
                if len(imaging_gt.ROIAPWave()&roikey) ==0:    
                    if not convolvingdone:
                        start_t = tracetime[0]
                        start_t = movie_time[np.argmax(movie_time>=start_t)]
                        end_t = np.min([tracetime[-1],movie_time[-1]])
                        apmaxtimes, apmaxidxes = ((sweeps_needed*ephysanal.ActionPotential())&'sweep_number = '+str(sweep_number)).fetch('ap_max_time','ap_max_index')
                        if plotit:
                            fig=plt.figure()
                            ax_v=fig.add_axes([0,0,2,.8])
                            ax_v.plot(tracetime,trace*1000,'k-')
                            ax_v.plot(tracetime[list(apmaxidxes)],trace[list(apmaxidxes)]*1000,'ro')
                            #ax_v.set_xlim([end_t-3,end_t])#end_t
                        
                        
                        
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
                        
# =============================================================================
#                         f_on = tau_1_ratio_on*np.exp(t/tau_1_on) + (1-tau_1_ratio_on)*np.exp(-t/tau_2_on)
#                         f_off = tau_1_ratio_off*np.exp(t[::-1]/tau_1_off) + (1-tau_1_ratio_off)*np.exp(-t[::-1]/tau_2_off)
#                         f_on = f_on/np.max(f_on)
#                         f_off = f_off/np.max(f_off)
#                         
#                         kernel = np.concatenate([f_on,np.zeros(len(f_off))])[::-1]
#                         kernel  = kernel /sum(kernel )
#                         trace_conv = np.convolve(trace,kernel,mode = 'same') 
# =============================================================================
                        
                        kernel = np.ones(int(np.round(sample_rate/frame_rate)))
                        kernel  = kernel /sum(kernel )
                        trace_conv = np.convolve(trace_conv,kernel,mode = 'same') 
                        trace_conv_time   = tracetime#[down_idxes]
                        
                        f_idx_now = (movie_time>=start_t) & (movie_time<=end_t)
                        dff_time_now = movie_time[f_idx_now]
                        e_idx_original = list()
                        for t in dff_time_now:
                            e_idx_original.append(np.argmin(trace_conv_time<t))
                        convolvingdone = True
                    ap_nums,ap_max_times = (ephysanal.ActionPotential()&key&'cell_number ='+str(cellnum) &'sweep_number = '+str(sweep_number)).fetch('ap_num','ap_max_time')
                    #%
                    #snratios = list()
                    time_back = .02
                    baseline_time = .3
                    baseline_time_end = .1
                    time_forward = .02
                    step_back = int(np.round(time_back*frame_rate))
                    step_forward = int(np.round(time_forward*frame_rate))
                    step_baseline = int(np.round(baseline_time*frame_rate))
                    step_end_baseline = int(np.round(baseline_time_end*frame_rate))
                    #%
                    apkeys = list()
                    apkey = key.copy()
                    apkey['movie_number'] = movie_number
                    apkey['motion_correction_method'] = motion_correction_method
                    apkey['roi_type'] = roi_type
                    apkey['roi_number'] = roi_number
                    apkey['cell_number'] = cellnum
                    apkey['sweep_number'] = sweep_number
                    ap_max_times = np.asarray(ap_max_times,float)
                    #%
                    for ap_num,ap_max_time in zip(ap_nums, ap_max_times):
                       f_max_idx = np.argmax(dff_time_now>ap_max_time)
                       if f_max_idx >step_back and f_max_idx+step_forward<len(dff_now):
                           apkey_now = apkey.copy()
                           #f_ap_idxes = np.arange(-step_back,step_forward)+f_max_idx
                           ap_dff = dff_now[f_max_idx-step_back:f_max_idx+step_forward]
                           ap_dff_filt = dff_filt_now[f_max_idx-step_back:f_max_idx+step_forward]
                           ap_time = dff_time_now[f_max_idx-step_back:f_max_idx+step_forward]-ap_max_time
                           if ap_num>1:
                               baseline_max_idx = np.argmax(dff_time_now>ap_max_times[ap_num-1-np.argmax(np.diff(ap_max_times[:ap_num])[::-1]>baseline_time)])
                           else:
                               baseline_max_idx  = f_max_idx
                           baseline_dff = dff_filt_now[baseline_max_idx-step_baseline:baseline_max_idx-step_end_baseline]
                           if len(baseline_dff)<5:
                               baseline_dff = dff_filt_now[f_max_idx-step_back:f_max_idx-2]
                           apkey_now['ap_num'] = ap_num
                           apkey_now['apwave_time'] = ap_time
                           apkey_now['apwave_dff'] = ap_dff
                           
                           if (np.diff(ap_dff[step_back-1::-1])<0).any:
                               apstartidx = step_back - (np.diff(ap_dff[step_back-1::-1])<0).argmax()-1
                           else:
                               apstartidx = 0
                           ap_peak_amplitude = ap_dff[apstartidx]-np.min(ap_dff[step_back-1:step_back+3])
                           
# =============================================================================
#                            idxap,temp = scipy.signal.find_peaks(ap_dff_filt*-1)
#                            neededap = idxap[idxap>=step_back-1]
#                            neededap = neededap[0]
#                            ap_peak_amplitude = ap_dff_filt[neededap]*-1
# =============================================================================
                           apkey_now['apwave_snratio'] = ap_peak_amplitude/np.std(baseline_dff)
                           apkey_now['apwave_peak_amplitude'] = ap_peak_amplitude#np.abs(np.min(ap_dff[step_back-step_end_baseline:step_back+step_end_baseline])-np.mean(baseline_dff))
                           apkey_now['apwave_noise'] = np.std(baseline_dff)
                           apkeys.append(apkey_now)
                           #snratios.append(apkey_now['apwave_snratio'] )
# =============================================================================
#                            if apkey_now['apwave_snratio']<5:
#                                break
# =============================================================================
                           #break
                       #%
                    if len(ap_nums)>0:
                        imaging_gt.ROIAPWave().insert(apkeys,allow_direct_insert=True,skip_duplicates=True)            
            except:
                print('couldn''t do this one:')
                print(roikey)
                #asdasd
                #time.sleep(1000)
                

#%%
def upload_movie_metadata():   
#%%                 
    copy_tiff_locally_first = False
    tempdir = '/home/rozmar/temp/datajoint'
    repository = 'mr_share'
    if repository == 'mr_share':  
        repodir = '/home/rozmar/Network/slab_share_mr'
        repodir = '/home/rozmar'
        basedir = os.path.join(repodir, 'Data/Voltage_imaging/Voltage_rig_1P/rozsam')
    #basedir = '/home/rozmar/Network/Voltage_imaging_rig_1p_imaging/rozsam'
    dirs = os.listdir(basedir )
    sessiondates = experiment.Session().fetch('session_date')
    for session_now in dirs:
        if session_now.isnumeric():
            session_date = datetime.datetime.strptime(session_now,'%Y%m%d').date()
            if session_date in sessiondates:
                subject_id = (experiment.Session() & 'session_date = "'+str(session_date)+'"').fetch('subject_id')[0]
                session = (experiment.Session() & 'session_date = "'+str(session_date)+'"').fetch('session')[0]            
                key = {'subject_id':subject_id,'session':session}
                
                session_dir_now = os.path.join(basedir, session_now)
                files = os.listdir(session_dir_now)
                
                basenames = list()
                fileindexes = list()
                ismulti = list()
                for filename in files:
                    sepidx = filename[::-1].find('_')
                    dotidx = filename.find('.')
                    basenames.append(filename[:-sepidx-1])
                    fileindexes.append(filename[-sepidx:dotidx])
                    ismulti.append(filename[dotidx:] == '.tif' and fileindexes[-1].isnumeric())
                files = np.asarray(files)[ismulti]
                basenames = np.asarray(basenames)[ismulti]
                fileindexes = np.asarray(fileindexes)[ismulti]
                uniquebasenames = np.unique(basenames)
                movinamesuploaded_sofar = (imaging.Movie()&key).fetch('movie_name')
                print(session_dir_now)
                for movie_idx, basename in enumerate(uniquebasenames):
                    if basename not in movinamesuploaded_sofar:
                        print(basename)
                        sepidxes = np.asarray([i for i, ltr in enumerate(basename) if ltr == '_' or ltr == '-'])
                        if basename.find('x_')>-1:
                            magnification = int(basename[:basename.find('x_')])
                        else:
                            magnification = 40. #default value
                        if 'tube' in basename.lower():
                            if '_tube' in basename.lower():
                                camera_tube_magn= float(basename[sepidxes[np.argmax(sepidxes>basename.lower().find('_tube'))-2]+1:basename.lower().find('_tube')-1])
                            else:
                                camera_tube_magn= float(basename[sepidxes[np.argmax(sepidxes>basename.lower().find('tube'))-1]+1:basename.lower().find('tube')-1])
                                    
                        else: 
                            camera_tube_magn = 1. # default value
                        
                        
                        files_now = files[basenames == basename]
                        indexes_now = fileindexes[basenames == basename]
                        order = np.argsort(indexes_now)
                        files_now = files_now[order]
                        frame_count = list()
                        freqs = list()
                        first_frame_times = list()
                        for file_now in files_now:
                            if copy_tiff_locally_first:
                                shutil.copyfile(os.path.join(session_dir_now,file_now),os.path.join(tempdir,'temp.tif'))
                                tifffile = os.path.join(tempdir,'temp.tif')
                            else:
                                tifffile = os.path.join(session_dir_now,file_now)
                            with Image.open(tifffile) as im:#with Image.open(os.path.join(session_dir_now,file_now)) as im:
                                frame_count.append(im.n_frames)
                                width = im.tag[256][0]
                                height = im.tag[257][0]
                                tagstring = im.tag[270][0]
                                datetimematch = re.search(r'\d{4} \d{2}:\d{2}:\d{2}', tagstring)
                                movie_time = datetime.datetime.strptime(tagstring[tagstring.find('\r\n')+2:datetimematch.end()], '%a, %d %b %Y %H:%M:%S')
                                string_now = tagstring[tagstring.find('Time_From_Start =')+len('Time_From_Start = '):]
                                string_now = string_now[:string_now.find('\r\n')]
                                a = datetime.datetime.strptime(string_now, '%H:%M:%S.%f').time()#
                                string_now = tagstring[tagstring.find('Exposure1 = ')+len('Exposure1 = '):]
                                string_now = string_now[:string_now.find('s')]
                                freq = 1/float(string_now)
                                freqs.append(freq)
                                time_to_first_frame = datetime.timedelta(hours = a.hour,minutes = a.minute,seconds = a.second, microseconds = a.microsecond)
                                total_seconds = time_to_first_frame.total_seconds()
                                first_frame_time = movie_time + time_to_first_frame
                                first_frame_times.append(first_frame_time)
                                string_now = tagstring[tagstring.find('Binning =')+len('Binning = '):]
                                string_now = string_now[:string_now.find('\r\n')]
                                binning = int(string_now)
                                #print(total_seconds)
                                print(file_now)
                                #print(movie_time)
                                #print(first_frame_time)
                                #print(freq)
                        
                        try:
                            movie_length = (first_frame_times[-1] - movie_time).total_seconds() + frame_count[-1]/freqs[-1]
                        except:
                            movie_length = 0
                        
                        
                        #% find the time offset between the two computers
                        
                        session_datetime = (datetime.datetime.combine((experiment.Session()&key).fetch('session_date')[0], datetime.time.min )+ (experiment.Session()&key).fetch('session_time')[0])
                        celldatetimes = (datetime.datetime.combine((experiment.Session()&key).fetch('session_date')[0], datetime.time.min ) + (ephys_patch.Cell()&key).fetch('cell_recording_start'))
                        potential_cell_number = (ephys_patch.Cell()&key).fetch('cell_number')[(celldatetimes<movie_time).argmin()-1]     
                        key_cell = key.copy()
                        key_cell['cell_number'] = potential_cell_number
                        
                        
                        cell_datetime = (datetime.datetime.combine((experiment.Session()&key_cell).fetch('session_date')[0], datetime.time.min ) + (ephys_patch.Cell()&key_cell).fetch('cell_recording_start'))[0]
                        movie_start_timediff = (movie_time - cell_datetime).total_seconds()-movie_length
                        #movie_end_timediff = (first_frame_times[-1] - cell_datetime).total_seconds() + frame_count[-1]/freqs[-1] - movie_length
                        
                        sweep_nums, sweep_start_times, frametimes, frame_sweeptimes = (ephys_patch.Sweep()*ephysanal.SweepFrameTimes() & key_cell).fetch('sweep_number','sweep_start_time','frame_time','frame_sweep_time')
                        if len(sweep_nums)>0:
                            #%
                            frametimes_all = np.concatenate(frametimes)
                            frame_sweeptimes_all = np.concatenate(frame_sweeptimes)
                            frametimes_diff_start = np.concatenate([[np.inf],np.diff(frametimes_all)])
                            frametimes_diff_end = np.concatenate([np.diff(frametimes_all),[np.inf]])
                            #%
                            if len(frametimes_all[(frametimes_diff_start>1) & (frame_sweeptimes_all>1)])>0:
                                #%
                                try:#%
                                    #%
                                    time_offset_idx = np.argmin(np.abs(frametimes_all[(frametimes_diff_start>.5) & (frame_sweeptimes_all>1) & (np.round(frametimes_diff_end/(1/freqs[0])) == 1)] - movie_start_timediff))
                                    residual_time_offset = (frametimes_all[(frametimes_diff_start>.5) & (frame_sweeptimes_all>1)& (np.round(frametimes_diff_end/(1/freqs[0])) == 1)] - movie_start_timediff)[time_offset_idx]
                                    movie_start_time = cell_datetime + datetime.timedelta(seconds = (frametimes_all[(frametimes_diff_start>.5) & (frame_sweeptimes_all>1) & (np.round(frametimes_diff_end/(1/freqs[0])) == 1)])[time_offset_idx])
                                    #%
                                except:
                                    time_offset_idx=0
                                    residual_time_offset = np.inf
                                    movie_start_time = cell_datetime
                                
                                pixel_size = 6.5/(1.11*magnification*camera_tube_magn)*binning
                                
                                movie_start_time_from_session_start = movie_start_time - session_datetime
                                
                                print(['movie start time from session start ', movie_start_time_from_session_start])
                                print(['framerate ', np.mean(freqs[1:])])
                                print(['frame count ', sum(frame_count)])
                                print(['time offset between computers ', residual_time_offset])
                                print(residual_time_offset)
                                if np.abs(residual_time_offset)<100:

    
                                    moviedata = {'subject_id':subject_id,'session':session}
                                    moviedata['movie_number'] = movie_idx
                                    moviedata['movie_name'] = basename
                                    moviedata['movie_x_size'] = width
                                    moviedata['movie_y_size'] = height                                
                                    moviedata['movie_frame_rate'] = np.mean(freqs)
                                    moviedata['movie_frame_num'] = sum(frame_count)
                                    moviedata['movie_start_time'] = movie_start_time_from_session_start.total_seconds()
                                    moviedata['movie_pixel_size'] = pixel_size

# =============================================================================
#                                     print('waiting')
#                                     timer.sleep(1000)
# =============================================================================
                                    imaging.Movie().insert1(moviedata, allow_direct_insert=True)
    
                                    moviefiledata = list()
                                    for movie_file_number, (movie_file_name,movie_frame_count) in enumerate(zip(files_now.tolist(),frame_count)):
                                        moviefiledata_now  = {'subject_id':subject_id,'session':session}
                                        moviefiledata_now['movie_number'] = movie_idx
                                        moviefiledata_now['movie_file_number'] = movie_file_number
                                        moviefiledata_now['movie_file_repository']= repository
                                        moviefiledata_now['movie_file_directory']= session_dir_now[len(repodir)+1:]
                                        moviefiledata_now['movie_file_name'] = movie_file_name
                                        moviefiledata_now['movie_file_start_frame'] = 0
                                        moviefiledata_now['movie_file_end_frame']= movie_frame_count
                                        moviefiledata.append(moviefiledata_now)
                                    imaging.MovieFile().insert(moviefiledata, allow_direct_insert=True)

#%% search for the exposition times
def calculate_exposition_times():
    #%%
    subject_ids,sessions,movie_numbers = imaging.Movie().fetch('subject_id','session','movie_number')
    for subject_id,session,movie_number in zip(subject_ids,sessions,movie_numbers):
        moviekey = {'subject_id':subject_id,'session':session,'movie_number':movie_number}
        sessionkey = {'subject_id':subject_id,'session':session}
        moviestarttime = float((imaging.Movie()&moviekey).fetch('movie_start_time')[0])
        movieframenum = float((imaging.Movie()&moviekey).fetch('movie_frame_num')[0])
        movieframerate = float((imaging.Movie()&moviekey).fetch('movie_frame_rate')[0])
        if movieframerate>0 and  len(imaging.MovieFrameTimes()&moviekey) == 0:#TODO HOTFIXlen(imaging.ROI()&moviekey)>0 and
            sessiontime = (experiment.Session()&sessionkey).fetch('session_time')[0]
            cellstarttimes = (ephys_patch.Cell()&sessionkey).fetch('cell_recording_start')-sessiontime
            (cellstarttimes < datetime.timedelta(seconds = moviestarttime))
            if sum(cellstarttimes < datetime.timedelta(seconds = moviestarttime)) == len(cellstarttimes):
                cellindex =  len(cellstarttimes)-1
            elif sum(cellstarttimes < datetime.timedelta(seconds = moviestarttime)) == 0:
                print('video recorded before all cells.. waiting')
                timer.sleep(1000)
            else:
                cellindex = np.argmin((cellstarttimes < datetime.timedelta(seconds = moviestarttime))) - 1
            cell_number = (ephys_patch.Cell()&sessionkey).fetch('cell_number')[cellindex]
            cellkey = {'subject_id':subject_id,'session':session,'cell_number':cell_number}
            cell_start_time = (ephys_patch.Cell()&cellkey).fetch('cell_recording_start')[0].total_seconds()
            sweep_start_times = np.asarray((ephys_patch.Sweep()&cellkey).fetch('sweep_start_time'),float)
            sweep_end_times = np.asarray((ephys_patch.Sweep()&cellkey).fetch('sweep_end_time'),float)
            sweepidx_start = np.argmin(sweep_start_times+cell_start_time - sessiontime.total_seconds() < moviestarttime)-1
            sweepidx_end = np.argmax(sweep_end_times+cell_start_time - sessiontime.total_seconds() > moviestarttime+movieframenum/movieframerate)
            sweep_number_start = (ephys_patch.Sweep()&cellkey).fetch('sweep_number')[sweepidx_start]
            sweep_number_end = (ephys_patch.Sweep()&cellkey).fetch('sweep_number')[sweepidx_end]
            exptimes = (ephysanal.SweepFrameTimes()&cellkey&'sweep_number>='+str(sweep_number_start)).fetch('frame_time')
            exptimes_session = exptimes +cell_start_time - sessiontime.total_seconds() # seconds from session start
            exptimes_all = np.concatenate(exptimes_session)
            startindex = np.argmin(np.abs(exptimes_all - moviestarttime))
    # =============================================================================
    #         print('waiting')
    #         timer.sleep(3000)
    # =============================================================================
            exptimes_all  = exptimes_all[startindex:]
            exptimes_diff = np.diff(exptimes_all) 
            gapidxes = np.concatenate([np.where((exptimes_diff>2/movieframerate))[0],[-1]])#&(exptimes_diff<3)
            
            #%
            frame_times = list()
            gapidx_now = -1
            frames_recorded_start_idx = list()
            gap_start_idx = list()
            gap_end_idx = list()
            while len(frame_times)<movieframenum and gapidx_now<len(gapidxes)-1:
                gapidx_now += 1
                if gapidx_now ==0:
                    startidx = 0
                else:
                    startidx = gapidxes[gapidx_now-1]+1
                endidx = gapidxes[gapidx_now]
                if startidx<endidx:
# =============================================================================
#                     if len(frame_times)>0: # this part removes a frame time if the imaging framerate would jump too high.. probably not needed...
#                         if (exptimes_all[startidx]-frame_times[-1])/np.diff(exptimes_all[startidx:startidx+2])[0]<.5:
#                             frame_times= frame_times[:-1]
#                         if 1/np.diff(frame_times)[0]>900: #HOTFIX .. please think of something else.. in the high frequency movies, there is an extra frame between sweeps....
#                             frame_times= frame_times[:-1]
# =============================================================================
                    frames_recorded_start_idx.append(len(frame_times))
                    frame_times.extend(exptimes_all[startidx:endidx])
                    
                    if len(frame_times)<movieframenum:
                        ifi_mode = scipy.stats.mode(np.diff(exptimes_all[startidx:endidx]))[0][0]
                        ifi_mean = np.mean(np.diff(exptimes_all[startidx:endidx]))
                        timediff = exptimes_diff[gapidxes[gapidx_now]]
                        #frames_in_gap = timediff/ifi_mean
                        #gapsequence = np.arange(np.floor(frames_in_gap))*ifi_mean+ifi_mean + frame_times[-1]
                        frames_in_gap = np.argmax(frame_times-frame_times[0]>timediff)+1
                        gapsequence = frame_times[1:frames_in_gap]-frame_times[0]+frame_times[-1]
                        gap_start_idx.append(len(frame_times))
                        frame_times.extend(gapsequence)
                        gap_end_idx.append(len(frame_times))
                        #plt.plot(exptimes_all[startidx:endidx])
                        #print(len(frame_times))
            #print([len(frame_times),'vs',movieframenum])
            #print(gap_start_idx)
           # timer.sleep(3)
            if len(frame_times)<movieframenum:
                missingframes = movieframenum - len(frame_times)
                missing_part = frame_times[1:int(missingframes)+1]-frame_times[0]+frame_times[-1]
                frame_times = np.concatenate([frame_times,missing_part])
                
            frame_times = frame_times[:int(movieframenum)]
            frametimedata = moviekey.copy()
            frametimedata['frame_times']=np.asarray(frame_times)
            print(len(frame_times))
            print(movieframenum)
            print(frametimedata)


            imaging.MovieFrameTimes().insert1(frametimedata, allow_direct_insert=True)
    


#%% save matlab spikepursuit image registration and ROIs
def save_spikepursuit_pipeline():
    #repodir = '/home/rozmar'
    #%%
    spikepursuitfolder = homefolder+'/Data/Voltage_imaging/Spikepursuit/Voltage_rig_1P/rozsam'
    dirs = os.listdir(spikepursuitfolder )
    sessiondates = experiment.Session().fetch('session_date')
    for session_now in dirs:
        if session_now.isnumeric():
            session_date = datetime.datetime.strptime(session_now,'%Y%m%d').date()
            if session_date in sessiondates:
                subject_id = (experiment.Session() & 'session_date = "'+str(session_date)+'"').fetch('subject_id')[0]
                session = (experiment.Session() & 'session_date = "'+str(session_date)+'"').fetch('session')[0]            
                key = {'subject_id':subject_id,'session':session}
                session_dir_now = os.path.join(spikepursuitfolder, session_now)
                movies = os.listdir(session_dir_now)
                movinames_dj= (imaging.Movie()&key).fetch('movie_name')
    
                for movie_name in movies:
                    if movie_name in movinames_dj:
                        key_now = key.copy()
                        key_now['movie_name'] = movie_name
                        movie_number = (imaging.Movie()&key_now).fetch('movie_number')[0]
                        key_now['movie_number'] = movie_number
                        del key_now['movie_name']
                        movie_dir_now = os.path.join(session_dir_now, movie_name)
                        upload_spike_pursuit_matlab_data(key_now,movie_dir_now)
                        print(key_now)
                        
                                                                                             
#%%
#%% 
def save_volpy_pipeline(roitype = 'VolPy',motion_corr = 'VolPy'):
    #repodir = '/home/rozmar'
    #%%
    if roitype == 'VolPy' or roitype ==  'VolPy_raw':
        volpyfolder = homefolder +'/Data/Voltage_imaging/VolPy/Voltage_rig_1P/rozsam'
    elif roitype == 'VolPy_denoised' or roitype == 'VolPy_denoised_raw':
        volpyfolder = homefolder +'/Data/Voltage_imaging/denoised_volpy/Voltage_rig_1P/rozsam'
    dirs = os.listdir(volpyfolder )
    sessiondates = experiment.Session().fetch('session_date')
    for session_now in dirs:
        if session_now.isnumeric():
            session_date = datetime.datetime.strptime(session_now,'%Y%m%d').date()
            if session_date in sessiondates:
                subject_id = (experiment.Session() & 'session_date = "'+str(session_date)+'"').fetch('subject_id')[0]
                session = (experiment.Session() & 'session_date = "'+str(session_date)+'"').fetch('session')[0]            
                key = {'subject_id':subject_id,'session':session}
                session_dir_now = os.path.join(volpyfolder, session_now)
                movies = os.listdir(session_dir_now)
                movinames_dj= (imaging.Movie()&key).fetch('movie_name')
                for movie_name in movies:
                    if movie_name in movinames_dj:
                        key_now = key.copy()
                        key_now['movie_name'] = movie_name
                        movie_number = (imaging.Movie()&key_now).fetch('movie_number')[0]
                        movie_x_size, movie_y_size = (imaging.Movie()&key_now).fetch1('movie_x_size','movie_y_size')
                        movie_dir_now = os.path.join(session_dir_now, movie_name)
                        files = os.listdir(movie_dir_now)
                        if 'spikepursuit.pickle' in files:
                            spikepursuit = pickle.load(open(os.path.join(movie_dir_now, 'spikepursuit.pickle'), 'rb'))
                            motioncorr_0 = pickle.load(open(os.path.join(movie_dir_now, 'motion_corr_0.pickle'), 'rb'))
                            motioncorr_1 = pickle.load(open(os.path.join(movie_dir_now, 'motion_corr_1.pickle'), 'rb'))
                            cellnum = spikepursuit['estimates']['cellN'][-1]
                            if len(imaging.ROI()&key&'movie_number = '+str(movie_number) &'roi_type = "'+roitype+'"'&'roi_number = '+str(cellnum)) == 0:
    
                                for cellid in spikepursuit['estimates']['cellN']:
                                    #%
                                    spikeidxes = spikepursuit['estimates']['spikeTimes'][cellid]
                                    
                                    dff = spikepursuit['estimates']['dFF'][cellid]
                                    f0 = spikepursuit['estimates']['F0'][cellid]
                                    motion_corr_vectors_0 = motioncorr_0['shifts_rig']
                                    if 'x_shifts_els' in motioncorr_0.keys():
                                        motion_corr_vectors_1 =dict()
                                        motion_corr_vectors_1 =dict()
                                        motion_corr_vectors_1['x_shifts'] = motioncorr_1['x_shifts_els']
                                        motion_corr_vectors_1['y_shifts'] = motioncorr_1['y_shifts_els']
                                    else:
                                        motion_corr_vectors_1 = motioncorr_1['shifts_rig']
                                    meanimage = motioncorr_1['templates_rig'][-1]
                                    #%
                                    ROI_centroid = scipy.ndimage.measurements.center_of_mass(spikepursuit['estimates']['bwexp'][cellid])
                                    if len(dff)>0:
                                        if len(imaging.RegisteredMovie()&key&'movie_number = '+str(movie_number)&'motion_correction_method = "' +motion_corr + '"')==0:
                                            
                                            key_reg_movie = key.copy()
                                            key_reg_movie['movie_number'] = movie_number
                                            key_reg_movie['motion_correction_method'] = motion_corr
                                            key_reg_movie['registered_movie_mean_image'] = meanimage
                                            imaging.RegisteredMovie().insert1(key_reg_movie, allow_direct_insert=True)
                                            mcorrid = 0
                                            if motion_corr=='VolPy_denoised':
                                                print('NOT FINISHED') #TODO finish it
                                                #time.sleep(1000) # add the two previous motion corrections
                                                        
                                            key_motioncorr = key.copy()
                                            key_motioncorr['movie_number'] = movie_number
                                            key_motioncorr['motion_correction_method'] = motion_corr
                                            key_motioncorr['motion_correction_id'] = mcorrid
                                            key_motioncorr['motion_corr_description'] = 'rigid motion correction done with VolPy'
                                            key_motioncorr['motion_corr_vectors'] = motion_corr_vectors_0
                                            imaging.MotionCorrection().insert1(key_motioncorr, allow_direct_insert=True)
                                            mcorrid =+ 1
                                            key_motioncorr = key.copy()
                                            key_motioncorr['movie_number'] = movie_number
                                            key_motioncorr['motion_correction_method'] = motion_corr
                                            key_motioncorr['motion_correction_id'] = mcorrid
                                            key_motioncorr['motion_corr_description'] = 'pairwise rigid motion correction done with VolPy'
                                            key_motioncorr['motion_corr_vectors'] = motion_corr_vectors_1
                                            imaging.MotionCorrection().insert1(key_motioncorr, allow_direct_insert=True)
                                            mcorrid =+ 1
                                        f = dff*f0+f0
                                        #%
                                        mask = np.asarray(spikepursuit['estimates']['bwexp'][cellid],float)
                                        wherex = np.where(spikepursuit['estimates']['bwexp'][cellid])[0]
                                        wherey = np.where(spikepursuit['estimates']['bwexp'][cellid])[1]
                                        mask[wherex[0]:wherex[-1]+1,wherey[0]:wherey[-1]+1] = spikepursuit['estimates']['spatialFilter'][cellid]
    
                                        #
# =============================================================================
#                                         print('waiting')
#                                         time.sleep(1000)
# =============================================================================
                                        #%
                                        if 'raw' in roitype: # a very primitive measure ..
                                            #%
                                            files = os.listdir(movie_dir_now)
                                            for file in files:
                                                if 'memmap_' in file and file[-4:] == 'mmap':
                                                    m_file = file
                                            print(os.path.join(movie_dir_now,m_file))
                                            try:
                                                movie = cm.load(os.path.join(movie_dir_now,m_file))
                                            except:
                                                print('caiman was not loaded, loading now')
                                                import caiman as cm
                                                movie = cm.load(os.path.join(movie_dir_now,m_file))
                                                #%
                                            
                                            ROI = spikepursuit['params']['ROIs'][cellid]
                                            f = np.mean(movie[:,np.asarray(ROI,bool)],1)
                                            #%
                                            fr = spikepursuit['params']['fr']
                                            f0 = moving_average(f,int(fr*.05))
                                            dff_raw = (f-f0)/f0
                                            idxap,apdict = scipy.signal.find_peaks(dff_raw*-1,height = (0,np.inf))
                                            idxneg,nonapdict = scipy.signal.find_peaks(dff_raw,height = (0,np.inf))
                                            medianvalue = np.median(nonapdict['peak_heights'])
                                            peakvals = nonapdict['peak_heights'][nonapdict['peak_heights']>medianvalue]-medianvalue
                                            peakvals = np.concatenate([peakvals,peakvals*-1])
                                            sd = np.std(peakvals)
                                            cutoff = medianvalue+3*sd
                                            aps =  apdict['peak_heights']>cutoff
                                            spikeidxes = idxap[aps]+1
                                            f0 = moving_average(f,int(fr*4))
                                            dff = (f-f0)/f0
                                            mask = ROI
                                            
                                            #ITT TARTOK!!!
                                          #%%
    #%
                                        key_roi = key.copy()
                                        key_roi['movie_number'] = movie_number
                                        key_roi['motion_correction_method'] = motion_corr
                                        key_roi['roi_number'] = cellid
                                        key_roi['roi_type'] = roitype
                                        key_roi['roi_dff'] = dff
                                        key_roi['roi_f0'] = f0
                                        key_roi['roi_spike_indices'] = spikeidxes
                                        key_roi['roi_centroid_x'] = ROI_centroid[1]
                                        key_roi['roi_centroid_y'] = ROI_centroid[0]
                                        key_roi['roi_mask'] = mask
                                        #%%
                                        try:
                                            imaging.ROI().insert1(key_roi, allow_direct_insert=True)
                                        except:
                                            print('original spikepursuit ROI already uploaded?')
                                            #%
                                        try:   
                                            #%%
                                            t = np.arange(len(f))
                                            out = scipy.optimize.curve_fit(lambda t,a,b,c,aa,bb: a*np.exp(-t/b) + c + aa*np.exp(-t/bb),  t,  f,maxfev=20000)#,bounds=(0, [np.inf,np.inf,np.inf])
                                            a = out[0][0]
                                            b = out[0][1]
                                            c = out[0][2]
                                            aa = out[0][3]
                                            bb = out[0][4]                #break
                                            f0 = a*np.exp(-t/b) + c + aa*np.exp(-t/bb)
                                            dff = (f-f0)/f0
                                            key_roi['roi_dff'] = dff
                                            key_roi['roi_f0'] = f0
                                            key_roi['roi_type'] = roitype+'_dexpF0'
                                            #%%
                                            imaging.ROI().insert1(key_roi, allow_direct_insert=True)
                                            #print('I could fit the double exponential')
                                        except:
                                            print('couldn''t fit double exponential? ')
                                            #%
                                        
    
                            print(movie_name)                                                                               
    
    


#%% Xcorrelation for each sweep  and ROIAPwaves
def upload_gt_correlations_apwaves(cores = 3):
    subject_ids = ephys_patch.Cell().fetch('subject_id')
    sessions = ephys_patch.Cell().fetch('session')
    cellnums = ephys_patch.Cell().fetch('cell_number')
    ray.init(num_cpus = cores)
    result_ids = []
    for subject_id,session,cellnum in zip(subject_ids,sessions,cellnums):
        key = { 'subject_id': subject_id, 'session':session}
        print(key)
        # =============================================================================
        # key = { 'subject_id': 462149, 'session':1}
        # cellnum = 1
        # 
        # =============================================================================
        #(imaging.Movie()*imaging.MovieDetails()*imaging.ROI())&key
        
        (experiment.Session()*ephys_patch.Cell())&key&'cell_number = '+str(cellnum)
        session_time = (experiment.Session()&key).fetch('session_time')[0]
        cell_time = (ephys_patch.Cell()&key&'cell_number = '+str(cellnum)).fetch('cell_recording_start')[0]
        cell_sweep_start_times =  (ephys_patch.Sweep()&key&'cell_number = '+str(cellnum)).fetch('sweep_start_time')
        cell_sweep_end_times =  (ephys_patch.Sweep()&key&'cell_number = '+str(cellnum)).fetch('sweep_end_time')
        time_start = float(np.min(cell_sweep_start_times))+ cell_time.total_seconds() - session_time.total_seconds()
        time_end = float(np.max(cell_sweep_end_times))+ cell_time.total_seconds() - session_time.total_seconds()
        movies_now = (imaging.Movie())&key & 'movie_start_time > '+str(time_start) & 'movie_start_time < '+str(time_end)
        movie_nums = movies_now.fetch('movie_number')
        
        
        for movie_number in movie_nums:
            print(['movie_number:',movie_number])
            #ROIEphysCorrelation_ROIAPwave_populate(key,movie_number,cellnum)
            result_ids.append(ROIEphysCorrelation_ROIAPwave_populate.remote(key.copy(),movie_number,cellnum))
    ray.get(result_ids)
    ray.shutdown()
            


