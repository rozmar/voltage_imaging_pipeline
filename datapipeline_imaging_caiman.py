#!/usr/bin/env python
"""
RUN in terminal before running python:
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
"""
import pickle
import os
import cv2
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import h5py

try:
    cv2.setNumThreads(0)
except:
    pass

try:
    if __IPYTHON__:
        # this is used for debugging purposes only. allows to reload classes
        # when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass

import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.utils.utils import download_demo, download_model
from caiman.source_extraction.volpy.volparams import volparams
from caiman.source_extraction.volpy.volpy import VOLPY
from caiman.source_extraction.volpy.mrcnn import visualize, neurons
import caiman.source_extraction.volpy.mrcnn.model as modellib
from caiman.paths import caiman_datadir



import datajoint as dj
dj.conn()
from pipeline import pipeline_tools
from pipeline import lab, experiment, ephys_patch, ephysanal, imaging
import os
import shutil
import time
import pathlib
import pickle
from scipy.io import loadmat
from skimage import io as imio
logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]" \
                    "[%(process)d] %(message)s",
                    level=logging.INFO)

    
#%%    



        
 # %%  Load demo movie and ROIs
#fnames = download_demo('demo_voltage_imaging.hdf5', 'volpy')  # file path to movie file (will download if not present)
#path_ROIs = download_demo('demo_voltage_imaging_ROIs.hdf5', 'volpy')  # file path to ROIs file (will download if not present)

 # %% Setup some parameters for data and motion correction
 # dataset parameters
#fr = fr                  # sample rate of the movie
def run_caiman_pipeline(movie,fr,fnames,savedir,usematlabroi):
    #%%
    
    
    cpu_num = 7
    cpu_num_spikepursuit = 2
    #gsig_filt_micron = (4, 4)  
    #max_shifts_micron = (6,6) 
    #strides_micron = (60,60)
    #overlaps_micron = (30, 30)   
    
    gsig_filt_micron = (4, 4)  
    max_shifts_micron = (6,6) 
    strides_micron = (30,30)
    overlaps_micron = (15, 15)   
    
    max_deviation_rigid_micron = 4
    
    
    pixel_size = movie['movie_pixel_size']
    
    ROIs = None                                     # Region of interests
    index = None                                    # index of neurons
    weights = None                                  # reuse spatial weights by 
                                                   # opts.change_params(params_dict={'weights':vpy.estimates['weights']})
    # motion correction parameters
    pw_rigid = False                                # flag for pw-rigid motion correction
    gSig_filt =   tuple(np.asarray(np.round(np.asarray(gsig_filt_micron)/float(pixel_size)),int))                          # size of filter, in general gSig (see below),
                                                   # change this one if algorithm does not work
    max_shifts = tuple(np.asarray(np.round(np.asarray(max_shifts_micron)/float(pixel_size)),int))
    strides = tuple(np.asarray(np.round(np.asarray(strides_micron)/float(pixel_size)),int))    # start a new patch for pw-rigid motion correction every x pixels
    overlaps =  tuple(np.asarray(np.round(np.asarray(overlaps_micron)/float(pixel_size)),int))    # start a new patch for pw-rigid motion correction every x pixels  
                                                                                                    # overlap between pathes (size of patch strides+overlaps)
    max_deviation_rigid = int(round(max_deviation_rigid_micron/pixel_size))                        # maximum deviation allowed for patch with respect to rigid shifts
    border_nan = 'copy'
    opts_dict = {
        'fnames': fnames,
        'fr': fr,
        'index': index,
        'ROIs': ROIs,
        'weights': weights,
        'pw_rigid': pw_rigid,
        'max_shifts': max_shifts,
        'gSig_filt': gSig_filt,
        'strides': strides,
        'overlaps': overlaps,
        'max_deviation_rigid': max_deviation_rigid,
        'border_nan': border_nan
    }
    opts = volparams(params_dict=opts_dict)
    
    # %% play the movie (optional)
    # playing the movie using opencv. It requires loading the movie in memory.
    # To close the video press q
    display_images = False
    
    if display_images:
        m_orig = cm.load(fnames)
        ds_ratio = 0.2
        moviehandle = m_orig.resize(1, 1, ds_ratio)
        moviehandle.play(q_max=99.5, fr=60, magnification=2)
    
     # %% start a cluster for parallel processing
    
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=cpu_num , single_thread=False)
    
    # % MOTION CORRECTION
    # Create a motion correction object with the specified parameters
    mcrig = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
    # Run piecewise rigid motion correction
    #%
    mcrig.motion_correct(save_movie=True)
    dview.terminate()
    
    # % MOTION CORRECTION2
    opts.change_params({'pw_rigid': True})
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=cpu_num , single_thread=False)
    # Create a motion correction object with the specified parameters
    mc = MotionCorrect(mcrig.mmap_file, dview=dview, **opts.get_group('motion'))
    # Run piecewise rigid motion correction
    mc.motion_correct(save_movie=True)
    dview.terminate()
    
    # %% motion correction compared with original movie
    display_images = False
    if display_images:
        m_orig = cm.load(fnames)
        m_rig = cm.load(mcrig.mmap_file)
        m_pwrig = cm.load(mc.mmap_file)
        ds_ratio = 0.2
        moviehandle = cm.concatenate([m_orig.resize(1, 1, ds_ratio) - mc.min_mov * mc.nonneg_movie,
                                      m_rig.resize(1, 1, ds_ratio),m_pwrig.resize(1, 1, ds_ratio)], axis=2)
        moviehandle.play(fr=60, q_max=99.5, magnification=2)  # press q to exit
    # % movie subtracted from the mean
        m_orig2 = (m_orig - np.mean(m_orig, axis=0))
        m_rig2 = (m_rig - np.mean(m_rig, axis=0))
        m_pwrig2 = (m_pwrig - np.mean(m_pwrig, axis=0))
        moviehandle1 = cm.concatenate([m_orig2.resize(1, 1, ds_ratio),
                                       m_rig2.resize(1, 1, ds_ratio),m_pwrig2.resize(1, 1, ds_ratio)], axis=2)
        moviehandle1.play(fr=60, q_max=99.5, magnification=2)
    
    # %% Memory Mapping
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=cpu_num , single_thread=False)
    border_to_0 = 0 if mc.border_nan == 'copy' else mc.border_to_0
    fname_new = cm.save_memmap_join(mc.mmap_file, base_name='memmap_',
                               add_to_mov=border_to_0, dview=dview, n_chunks=10)
    dview.terminate()
    
     # %% change fnames to the new motion corrected one
    opts.change_params(params_dict={'fnames': fname_new})
    
    # %% SEGMENTATION
    
    roidir = savedir[:savedir.find('VolPy')] + 'Spikepursuit' + savedir[savedir.find('VolPy')+len('Volpy'):]
    try:
        files = os.listdir(roidir)
    except:
        files= []
    if usematlabroi  and 'ROIs.mat' in files:        
        ROIs =  loadmat(os.path.join(roidir, 'ROIs.mat'))['ROIs']
        if len(np.shape(ROIs))==3:
            ROIs  = np.moveaxis(np.asarray(ROIs,bool),2,0)
        else:
            ROIs = np.asarray([ROIs])
        all_rois = ROIs
        opts.change_params(params_dict={'ROIs':ROIs,
                                            'index':list(range(ROIs.shape[0])),
                                            'method':'SpikePursuit'})
        
    else:
        #%
        print('WTF')
        # Create mean and correlation image
        use_maskrcnn = True  # set to True to predict the ROIs using the mask R-CNN
        if not use_maskrcnn:                 # use manual annotations
            with h5py.File(path_ROIs, 'r') as fl:
                ROIs = fl['mov'][()]  # load ROIs
            opts.change_params(params_dict={'ROIs': ROIs,
                                            'index': list(range(ROIs.shape[0])),
                                            'method': 'SpikePursuit'})
        else:
            try:
                m = cm.load(mc.mmap_file[0], subindices=slice(0, 20000))
            except:
                m = cm.load('/home/rozmar/Data/Voltage_imaging/Voltage_rig_1P/rozsam/20200120/40x_1xtube_10A_7_000_rig__d1_128_d2_512_d3_1_order_F_frames_2273_._els__d1_128_d2_512_d3_1_order_F_frames_2273_.mmap', subindices=slice(0, 20000))
            m.fr = fr
            img = m.mean(axis=0)
            img = (img-np.mean(img))/np.std(img)
            m1 = m.computeDFF(secsWindow=1, in_place=True)[0]
            m = m - m1
            Cn = m.local_correlations(swap_dim=False, eight_neighbours=True)
            img_corr = (Cn-np.mean(Cn))/np.std(Cn)
            summary_image = np.stack([img, img, img_corr], axis=2).astype(np.float32)
            del m
            del m1
        
            # %
            # Mask R-CNN
            config = neurons.NeuronsConfig()
            class InferenceConfig(config.__class__):
                # Run detection on one image at a time
                GPU_COUNT = 1
                IMAGES_PER_GPU = 1
                DETECTION_MIN_CONFIDENCE = 0.7
                IMAGE_RESIZE_MODE = "pad64"
                IMAGE_MAX_DIM = 512
                RPN_NMS_THRESHOLD = 0.7
                POST_NMS_ROIS_INFERENCE = 1000
            config = InferenceConfig()
            config.display()
            model_dir = os.path.join(caiman_datadir(), 'model')
            DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
            with tf.device(DEVICE):
                model = modellib.MaskRCNN(mode="inference", model_dir=model_dir,
                                          config=config)
            weights_path = download_model('mask_rcnn')
            model.load_weights(weights_path, by_name=True)
            results = model.detect([summary_image], verbose=1)
            r = results[0]
            ROIs_mrcnn = r['masks'].transpose([2, 0, 1])
        
        # %% visualize the result
            display_result = False
            if display_result:
                _, ax = plt.subplots(1,1, figsize=(16,16))
                visualize.display_instances(summary_image, r['rois'], r['masks'], r['class_ids'], 
                                        ['BG', 'neurons'], r['scores'], ax=ax,
                                        title="Predictions")
        # %% set rois
            opts.change_params(params_dict={'ROIs':ROIs_mrcnn,
                                            'index':list(range(ROIs_mrcnn.shape[0])),
                                            'method':'SpikePursuit'})
            #all_rois = ROIs_mrcnn
    
    
    
    # %% Trace Denoising and Spike Extraction
    
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=cpu_num_spikepursuit , single_thread=False, maxtasksperchild=1)
    #dview=None
    vpy = VOLPY(n_processes=n_processes, dview=dview, params=opts)
    vpy.fit(n_processes=n_processes, dview=dview)
        
   
    #%%
    print('saving parameters')
    parameters = dict()
    parameters['motion'] = opts.motion
    parameters['data'] = opts.data
    parameters['volspike'] = opts.volspike
    with open(os.path.join(savedir,'parameters.pickle'), 'wb') as outfile:
        pickle.dump(parameters, outfile)
    #%%    
    volspikedata = dict()
    volspikedata['estimates'] = vpy.estimates
    volspikedata['params'] = vpy.params.data
    with open(os.path.join(savedir,'spikepursuit.pickle'), 'wb') as outfile:
        pickle.dump(volspikedata, outfile)
    #%%
    
    for mcidx, mc_now in enumerate([mcrig,mc]):
        motioncorr = dict()
        motioncorr['fname'] = mc_now.fname
        motioncorr['fname_tot_rig'] = mc_now.fname_tot_rig
        motioncorr['mmap_file'] = mc_now.mmap_file
        motioncorr['min_mov'] = mc_now.min_mov
        motioncorr['shifts_rig'] = mc_now.shifts_rig
        motioncorr['shifts_opencv'] = mc_now.shifts_opencv
        motioncorr['niter_rig'] = mc_now.niter_rig
        motioncorr['min_mov'] = mc_now.min_mov
        motioncorr['templates_rig'] = mc_now.templates_rig
        motioncorr['total_template_rig'] = mc_now.total_template_rig
        try:
            motioncorr['x_shifts_els'] = mc_now.x_shifts_els
            motioncorr['y_shifts_els'] = mc_now.y_shifts_els
        except:
            pass 
        with open(os.path.join(savedir,'motion_corr_'+str(mcidx)+'.pickle'), 'wb') as outfile:
            pickle.dump(motioncorr, outfile)
     #%% saving stuff
    print('moving files')
    for mmap_file in mcrig.mmap_file:
        fname = pathlib.Path(mmap_file).name
        os.remove(mmap_file)
        #shutil.move(mmap_file, os.path.join(savedir,fname))
    for mmap_file in mc.mmap_file:
        fname = pathlib.Path(mmap_file).name
        os.remove(mmap_file)
        #shutil.move(mmap_file, os.path.join(savedir,fname))    
        
    fname = pathlib.Path(fname_new).name
    shutil.move(fname_new, os.path.join(savedir,fname))
    #print('waiting')
    #time.sleep(1000)
    # %% some visualization
    plotstuff = False
    if plotstuff:
        print(np.where(vpy.estimates['passedLocalityTest'])[0])    # neurons that pass locality test
        n = 0
        
        # Processed signal and spikes of neurons
        plt.figure()
        plt.plot(vpy.estimates['trace'][n])
        plt.plot(vpy.estimates['spikeTimes'][n],
                 np.max(vpy.estimates['trace'][n]) * np.ones(vpy.estimates['spikeTimes'][n].shape),
                 color='g', marker='o', fillstyle='none', linestyle='none')
        plt.title('signal and spike times')
        plt.show()
        # Location of neurons by Mask R-CNN or manual annotation
        plt.figure()
        if use_maskrcnn:
            plt.imshow(ROIs_mrcnn[n])
        else:
            plt.imshow(ROIs[n])
        mv = cm.load(fname_new)
        plt.imshow(mv.mean(axis=0),alpha=0.5)
        
        # Spatial filter created by algorithm
        plt.figure()
        plt.imshow(vpy.estimates['spatialFilter'][n])
        plt.colorbar()
        plt.title('spatial filter')
        plt.show()
     
     
    
    # %% STOP CLUSTER and clean up log files

    cm.stop_server(dview=dview)
    log_files = glob.glob('*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)
    
    
    
#%%
def populatevolpy():
    volpy_basedir = str(pathlib.Path.home())+'/Data/Voltage_imaging/VolPy/'        
    usematlabroi = True
    movies = imaging.Movie().fetch(as_dict=True)    
    for movie in movies:#[::-1]:
        moviefiles = imaging.MovieFile()&movie
        filenames,dirs,basedirs = moviefiles.fetch('movie_file_name','movie_file_directory','movie_file_repository')
        fnames = list()
        for filename,dir_now,basedir in zip(filenames,dirs,basedirs):
            fnames.append(os.path.join(dj.config['locations.'+basedir],dir_now,filename))
        fr = movie['movie_frame_rate']
        savedir = os.path.join(volpy_basedir,dir_now[dir_now.find('Voltage_imaging')+len('Voltage_imaging')+1:],movie['movie_name'])
    
        pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)
        print(movie)
        #time.sleep(5)
        if len(os.listdir(savedir))>0:
            print('already done.. skipped')
        else:
            roidir = savedir[:savedir.find('VolPy')] + 'Spikepursuit' + savedir[savedir.find('VolPy')+len('Volpy'):]
            try:
                files = os.listdir(roidir)
            except:
                files= []
            if usematlabroi  and 'ROIs.mat' not in files:
                print('no matlab ROIs found')
            else:
                #print('waiting')
                #time.sleep(1000)
                if movie['movie_frame_num']>500:
                    run_caiman_pipeline(movie,fr,fnames,savedir,usematlabroi)
     

def merge_denoised_tiff_files(movie,loaddir,savedir):
    #%%
    cpu_num = 2
    #cpu_num_spikepursuit = 1
    
    filenames = os.listdir(loaddir)
    counter = 0
    filenames_final = list()
    residualnames = list()
    while 'denoised_{}.tif'.format(counter) in filenames:
        m_new_denoised = cm.load(os.path.join(loaddir,'denoised_{}.tif'.format(counter))).transpose(2,0,1)
        i_new_sn = imio.imread(os.path.join(loaddir,'Sn_image_{}.tif'.format(counter)))[:,:,0]
        m_new_trend = cm.load(os.path.join(loaddir,'trend_{}.tif'.format(counter))).transpose(2,0,1)
        movief = m_new_denoised*i_new_sn+m_new_trend
        movief.save(os.path.join(loaddir,'movie{}.tif'.format(counter)))
        filenames_final.append(os.path.join(loaddir,'movie{}.tif'.format(counter)))
        residualnames.append(os.path.join(loaddir,'PMD_residual_{}.tif'.format(counter)))
        counter += 1
        print(counter)
    

    #%%
    residuals_movie = cm.load_movie_chain(residualnames)
    residuals_movie.save(os.path.join(savedir,'PMD_residuals.tif'))
    #movie_big = cm.load_movie_chain(filenames_final)
    # %% Memory Mapping
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=cpu_num , single_thread=False)
    fname_new = cm.save_memmap(filenames_final, base_name=movie['movie_name'], dview=dview, n_chunks=10, order='C')
    dview.terminate()
    fname = pathlib.Path(fname_new).name
    shutil.move(fname_new, os.path.join(savedir,fname))
    print('done')
#%%






       
def run_denoised_caiman_pipeline(movie,fr,loaddir,savedir,usematlabroi,onlyspikepursuit = False):
    #%%
    filenames = os.listdir(loaddir)
    
    for filename in filenames: # findinf mmap file
        if filename[-4:] == 'mmap':
            break
    filename = os.path.join(loaddir,filename)
    cpu_num = 4
    cpu_num_spikepursuit = 2
    #%%
    
    
    gsig_filt_micron = (4, 4)  
    max_shifts_micron = (6,6) 
    strides_micron = (50,50)
    overlaps_micron = (25, 25)   
    
    max_deviation_rigid_micron = 4
    
    
    pixel_size = movie['movie_pixel_size']
    
    ROIs = None                                     # Region of interests
    index = None                                    # index of neurons
    weights = None                                  # reuse spatial weights by 
                                                   # opts.change_params(params_dict={'weights':vpy.estimates['weights']})
    # motion correction parameters
    pw_rigid = False                                # flag for pw-rigid motion correction
    gSig_filt =   tuple(np.asarray(np.round(np.asarray(gsig_filt_micron)/float(pixel_size)),int))                          # size of filter, in general gSig (see below),
                                                   # change this one if algorithm does not work
    max_shifts = tuple(np.asarray(np.round(np.asarray(max_shifts_micron)/float(pixel_size)),int))
    strides = tuple(np.asarray(np.round(np.asarray(strides_micron)/float(pixel_size)),int))    # start a new patch for pw-rigid motion correction every x pixels
    overlaps =  tuple(np.asarray(np.round(np.asarray(overlaps_micron)/float(pixel_size)),int))    # start a new patch for pw-rigid motion correction every x pixels  
                                                                                                    # overlap between pathes (size of patch strides+overlaps)
    max_deviation_rigid = int(round(max_deviation_rigid_micron/pixel_size))                        # maximum deviation allowed for patch with respect to rigid shifts
    border_nan = 'copy'
    opts_dict = {
        'fnames': filename,
        'fr': fr,
        'index': index,
        'ROIs': ROIs,
        'weights': weights,
        'pw_rigid': pw_rigid,
        'max_shifts': max_shifts,
        'gSig_filt': gSig_filt,
        'strides': strides,
        'overlaps': overlaps,
        'max_deviation_rigid': max_deviation_rigid,
        'border_nan': border_nan
    }
    opts = volparams(params_dict=opts_dict)
        
     # %% start a cluster for parallel processing
    if onlyspikepursuit:
        finishedfilenames = os.listdir(savedir)
        for fnow in finishedfilenames :
            if fnow[:6] == 'memmap':
                opts.change_params(params_dict={'fnames': os.path.join(savedir,fnow)})
                break
        
        #%%
        
    else:
        c, dview, n_processes = cm.cluster.setup_cluster(
            backend='local', n_processes=cpu_num , single_thread=False)
        
        # % MOTION CORRECTION
        # Create a motion correction object with the specified parameters
        mcrig = MotionCorrect(filename, dview=dview, **opts.get_group('motion'))
        # Run piecewise rigid motion correction
        #%
        mcrig.motion_correct(save_movie=True)
        dview.terminate()
        
        # % MOTION CORRECTION2
        opts.change_params({'pw_rigid': True})
        c, dview, n_processes = cm.cluster.setup_cluster(
            backend='local', n_processes=cpu_num , single_thread=False)
        # Create a motion correction object with the specified parameters
        mc = MotionCorrect(mcrig.mmap_file, dview=dview, **opts.get_group('motion'))
        # Run piecewise rigid motion correction
        mc.motion_correct(save_movie=True)
        dview.terminate()
        
        # %% motion correction compared with original movie
        display_images = False
        if display_images:
            m_orig = cm.load(filename)
            m_rig = cm.load(mcrig.mmap_file)
            m_pwrig = cm.load(mc.mmap_file)
            ds_ratio = 0.2
            moviehandle = cm.concatenate([m_orig.resize(1, 1, ds_ratio) - mc.min_mov * mc.nonneg_movie,
                                          m_rig.resize(1, 1, ds_ratio),m_pwrig.resize(1, 1, ds_ratio)], axis=2)
            moviehandle.play(fr=60, q_max=99.5, magnification=2)  # press q to exit
        # % movie subtracted from the mean
            m_orig2 = (m_orig - np.mean(m_orig, axis=0))
            m_rig2 = (m_rig - np.mean(m_rig, axis=0))
            m_pwrig2 = (m_pwrig - np.mean(m_pwrig, axis=0))
            moviehandle1 = cm.concatenate([m_orig2.resize(1, 1, ds_ratio),
                                           m_rig2.resize(1, 1, ds_ratio),m_pwrig2.resize(1, 1, ds_ratio)], axis=2)
            moviehandle1.play(fr=60, q_max=99.5, magnification=2)
        
        # %% Memory Mapping
        c, dview, n_processes = cm.cluster.setup_cluster(
            backend='local', n_processes=cpu_num , single_thread=False)
        border_to_0 = 0 if mc.border_nan == 'copy' else mc.border_to_0
        fname_new = cm.save_memmap_join(mc.mmap_file, base_name='memmap_',
                                   add_to_mov=border_to_0, dview=dview, n_chunks=10)
        dview.terminate()
        
         # %% change fnames to the new motion corrected one
        opts.change_params(params_dict={'fnames': fname_new})
        
    # %% SEGMENTATION
    
    roidir = savedir[:savedir.find('denoised_volpy')] + 'Spikepursuit' + savedir[savedir.find('denoised_volpy')+len('denoised_volpy'):]
    try:
        files = os.listdir(roidir)
    except:
        files= []
    if usematlabroi  and 'ROIs.mat' in files:        
        ROIs =  loadmat(os.path.join(roidir, 'ROIs.mat'))['ROIs']
        if len(np.shape(ROIs))==3:
            ROIs  = np.moveaxis(np.asarray(ROIs,bool),2,0)
        else:
            ROIs = np.asarray([ROIs])
        all_rois = ROIs
        opts.change_params(params_dict={'ROIs':ROIs,
                                            'index':list(range(ROIs.shape[0])),
                                            'method':'SpikePursuit'})
        
    else:
        #%
        print('WTF')
        # Create mean and correlation image
        use_maskrcnn = True  # set to True to predict the ROIs using the mask R-CNN
        if not use_maskrcnn:                 # use manual annotations
            with h5py.File(path_ROIs, 'r') as fl:
                ROIs = fl['mov'][()]  # load ROIs
            opts.change_params(params_dict={'ROIs': ROIs,
                                            'index': list(range(ROIs.shape[0])),
                                            'method': 'SpikePursuit'})
        else:
            try:
                m = cm.load(mc.mmap_file[0], subindices=slice(0, 20000))
            except:
                m = cm.load('/home/rozmar/Data/Voltage_imaging/Voltage_rig_1P/rozsam/20200120/40x_1xtube_10A_7_000_rig__d1_128_d2_512_d3_1_order_F_frames_2273_._els__d1_128_d2_512_d3_1_order_F_frames_2273_.mmap', subindices=slice(0, 20000))
            m.fr = fr
            img = m.mean(axis=0)
            img = (img-np.mean(img))/np.std(img)
            m1 = m.computeDFF(secsWindow=1, in_place=True)[0]
            m = m - m1
            Cn = m.local_correlations(swap_dim=False, eight_neighbours=True)
            img_corr = (Cn-np.mean(Cn))/np.std(Cn)
            summary_image = np.stack([img, img, img_corr], axis=2).astype(np.float32)
            del m
            del m1
        
            # %
            # Mask R-CNN
            config = neurons.NeuronsConfig()
            class InferenceConfig(config.__class__):
                # Run detection on one image at a time
                GPU_COUNT = 1
                IMAGES_PER_GPU = 1
                DETECTION_MIN_CONFIDENCE = 0.7
                IMAGE_RESIZE_MODE = "pad64"
                IMAGE_MAX_DIM = 512
                RPN_NMS_THRESHOLD = 0.7
                POST_NMS_ROIS_INFERENCE = 1000
            config = InferenceConfig()
            config.display()
            model_dir = os.path.join(caiman_datadir(), 'model')
            DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
            with tf.device(DEVICE):
                model = modellib.MaskRCNN(mode="inference", model_dir=model_dir,
                                          config=config)
            weights_path = download_model('mask_rcnn')
            model.load_weights(weights_path, by_name=True)
            results = model.detect([summary_image], verbose=1)
            r = results[0]
            ROIs_mrcnn = r['masks'].transpose([2, 0, 1])
        
        # %visualize the result
            display_result = False
            if display_result:
                _, ax = plt.subplots(1,1, figsize=(16,16))
                visualize.display_instances(summary_image, r['rois'], r['masks'], r['class_ids'], 
                                        ['BG', 'neurons'], r['scores'], ax=ax,
                                        title="Predictions")
        # %set rois
            opts.change_params(params_dict={'ROIs':ROIs_mrcnn,
                                            'index':list(range(ROIs_mrcnn.shape[0])),
                                            'method':'SpikePursuit'})
            #all_rois = ROIs_mrcnn
    
    
    
    # %% Trace Denoising and Spike Extraction
    
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=cpu_num_spikepursuit , single_thread=False, maxtasksperchild=1)
    #dview=None
    vpy = VOLPY(n_processes=n_processes, dview=dview, params=opts)
    vpy.fit(n_processes=n_processes, dview=dview)
        
   
    #%%
    if not onlyspikepursuit:
        print('saving parameters')
        parameters = dict()
        parameters['motion'] = opts.motion
        parameters['data'] = opts.data
        parameters['volspike'] = opts.volspike
        with open(os.path.join(savedir,'parameters.pickle'), 'wb') as outfile:
            pickle.dump(parameters, outfile)
        
        for mcidx, mc_now in enumerate([mcrig,mc]):
            motioncorr = dict()
            motioncorr['fname'] = mc_now.fname
            motioncorr['fname_tot_rig'] = mc_now.fname_tot_rig
            motioncorr['mmap_file'] = mc_now.mmap_file
            motioncorr['min_mov'] = mc_now.min_mov
            motioncorr['shifts_rig'] = mc_now.shifts_rig
            motioncorr['shifts_opencv'] = mc_now.shifts_opencv
            motioncorr['niter_rig'] = mc_now.niter_rig
            motioncorr['min_mov'] = mc_now.min_mov
            motioncorr['templates_rig'] = mc_now.templates_rig
            motioncorr['total_template_rig'] = mc_now.total_template_rig
            try:
                motioncorr['x_shifts_els'] = mc_now.x_shifts_els
                motioncorr['y_shifts_els'] = mc_now.y_shifts_els
            except:
                pass        
            with open(os.path.join(savedir,'motion_corr_'+str(mcidx)+'.pickle'), 'wb') as outfile:
                pickle.dump(motioncorr, outfile)
             #%% saving stuff
        #os.remove(str(mcrig.fname))
        print('moving files')
        for mmap_file in mcrig.mmap_file:
            fname = pathlib.Path(mmap_file).name
            os.remove(mmap_file)
            #shutil.move(mmap_file, os.path.join(savedir,fname))
        for mmap_file in mc.mmap_file:
            fname = pathlib.Path(mmap_file).name
            os.remove(mmap_file)
            #shutil.move(mmap_file, os.path.join(savedir,fname))  
    #%%    
    volspikedata = dict()
    volspikedata['estimates'] = vpy.estimates
    volspikedata['params'] = vpy.params.data
    with open(os.path.join(savedir,'spikepursuit.pickle'), 'wb') as outfile:
        pickle.dump(volspikedata, outfile)
    #%%
    
    
      
    #%%    
    #fname = pathlib.Path(fname_new).name
    #shutil.move(fname_new, os.path.join(savedir,fname))
    #print('waiting')
    #time.sleep(1000)
    # %% some visualization
    plotstuff = False
    if plotstuff:
        print(np.where(vpy.estimates['passedLocalityTest'])[0])    # neurons that pass locality test
        n = 0
        
        # Processed signal and spikes of neurons
        plt.figure()
        plt.plot(vpy.estimates['trace'][n])
        plt.plot(vpy.estimates['spikeTimes'][n],
                 np.max(vpy.estimates['trace'][n]) * np.ones(vpy.estimates['spikeTimes'][n].shape),
                 color='g', marker='o', fillstyle='none', linestyle='none')
        plt.title('signal and spike times')
        plt.show()
        # Location of neurons by Mask R-CNN or manual annotation
        plt.figure()
        if use_maskrcnn:
            plt.imshow(ROIs_mrcnn[n])
        else:
            plt.imshow(ROIs[n])
        mv = cm.load(fname_new)
        plt.imshow(mv.mean(axis=0),alpha=0.5)
        
        # Spatial filter created by algorithm
        plt.figure()
        plt.imshow(vpy.estimates['spatialFilter'][n])
        plt.colorbar()
        plt.title('spatial filter')
        plt.show()
     
     
    
    # %% STOP CLUSTER and clean up log files

    cm.stop_server(dview=dview)
    log_files = glob.glob('*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)
            
        

# =============================================================================
# #%% merging files
#        
# denoised_caiman_basedir = str(pathlib.Path.home())+'/Data/Voltage_imaging/sgpmd-nmf/'        
# denoised_savedir = str(pathlib.Path.home())+'/Data/Voltage_imaging/denoised_volpy/'        
# usematlabroi = True
# movies = imaging.Movie().fetch(as_dict=True)    
# for movie in movies:#[::-1]:
#     moviefiles = imaging.MovieFile()&movie
#     filenames,dirs,basedirs = moviefiles.fetch('movie_file_name','movie_file_directory','movie_file_repository')
#     fnames = list()
#     for filename,dir_now,basedir in zip(filenames,dirs,basedirs):
#         fnames.append(os.path.join(dj.config['locations.'+basedir],dir_now,filename))
#     fr = movie['movie_frame_rate']
#     savedir = os.path.join(denoised_savedir,dir_now[dir_now.find('Voltage_imaging')+len('Voltage_imaging')+1:],movie['movie_name'])
#     loaddir = os.path.join(denoised_caiman_basedir,dir_now[dir_now.find('Voltage_imaging')+len('Voltage_imaging')+1:],movie['movie_name'])
#     pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)
#     print(movie)
#     #time.sleep(5)
#     if len(os.listdir(savedir))>0:
#         print('already done.. skipped')
#     else:
#         roidir = savedir[:savedir.find('denoised_volpy')] + 'Spikepursuit' + savedir[savedir.find('denoised_volpy')+len('denoised_volpy'):]
#         try:
#             files = os.listdir(roidir)
#         except:
#             files= []
#         if usematlabroi  and 'ROIs.mat' not in files:
#             print('no matlab ROIs found')
#         else:
#             #print('waiting')
#             #time.sleep(1000)
#             if movie['movie_frame_num']>500:
#                 #print('starting')
#                 #time.sleep(1000)
#                 merge_denoised_tiff_files(movie,loaddir,savedir)
#                 #run_denoised_caiman_pipeline(movie,fr,loaddir,savedir,usematlabroi)
# =============================================================================
                
                
#%% running spikepursuit
denoised_caiman_basedir = str(pathlib.Path.home())+'/Data/Voltage_imaging/denoised_volpy/'              
usematlabroi = True
movies = imaging.Movie().fetch(as_dict=True)    
for movie in movies:#[::-1]:
    moviefiles = imaging.MovieFile()&movie
    filenames,dirs,basedirs = moviefiles.fetch('movie_file_name','movie_file_directory','movie_file_repository')
    fnames = list()
    for filename,dir_now,basedir in zip(filenames,dirs,basedirs):
        fnames.append(os.path.join(dj.config['locations.'+basedir],dir_now,filename))
    fr = movie['movie_frame_rate']
    loaddir = os.path.join(denoised_caiman_basedir,dir_now[dir_now.find('Voltage_imaging')+len('Voltage_imaging')+1:],movie['movie_name'])
    savedir = loaddir
    print(movie)
    #time.sleep(5)
    if 'spikepursuit.pickle' in os.listdir(savedir):
        print('already done.. skipped')
        

        spikepursuit = pickle.load(open(os.path.join(savedir, 'spikepursuit.pickle'), 'rb'))
        if 'dFF' not in spikepursuit['estimates'].keys():
            if movie['movie_frame_num']>500:
                print('rerunning only spikepursuit')
                #time.sleep(1000)
                #%%
                run_denoised_caiman_pipeline(movie,fr,loaddir,savedir,usematlabroi,onlyspikepursuit = True)
    else:
        roidir = savedir[:savedir.find('denoised_volpy')] + 'Spikepursuit' + savedir[savedir.find('denoised_volpy')+len('denoised_volpy'):]
        try:
            files = os.listdir(roidir)
        except:
            files= []
        if usematlabroi  and 'ROIs.mat' not in files:
            print('no matlab ROIs found')
        else:
            #print('waiting')
            #time.sleep(1000)
            if movie['movie_frame_num']>500:
                print('starting')
                time.sleep(3)
                run_denoised_caiman_pipeline(movie,fr,loaddir,savedir,usematlabroi)
