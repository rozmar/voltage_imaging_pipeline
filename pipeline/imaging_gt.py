import datajoint as dj
import pipeline.lab as lab#, ccf
import pipeline.experiment as experiment
import pipeline.ephys_patch as ephys_patch
import pipeline.ephysanal as ephysanal
import pipeline.imaging as imaging
from pipeline.pipeline_tools import get_schema_name
schema = dj.schema(get_schema_name('imaging'),locals())

import numpy as np
import scipy

@schema
class ROIEphysCorrelation(dj.Imported): 
# ROI (Region of interest - e.g. cells)
    definition = """
    -> imaging.ROI
    -> ephys_patch.Sweep
    ---
    time_lag                        : float #ms   
    corr_coeff                      : float #-1 - 1
    """

@schema
class ROIAPWave(dj.Imported): 
# ROI (Region of interest - e.g. cells)
    definition = """ # this is the optical AP waveform relative to the real AP peak
    -> imaging.ROI
    -> ephysanal.ActionPotential
    ---
    apwave_time                     : longblob
    apwave_dff                      : longblob
    apwave_snratio                  : float
    apwave_peak_amplitude           : float
    apwave_noise                    : float
    """

@schema
class GroundTruthROI(dj.Computed):
    definition = """ # this is the optical AP waveform relative to the real AP peak
    -> ephys_patch.Cell
    -> imaging.ROI
    --- 
    ephys_matched_ap_times                  : longblob # in seconds, from the start of the session
    ophys_matched_ap_times                  : longblob # in seconds, from the start of the session
    ephys_unmatched_ap_times                : longblob # in seconds, from the start of the session
    ophys_unmatched_ap_times                : longblob # in seconds, from the start of the session
    """
    def make(self, key):
        
        #key = {'subject_id': 454597, 'session': 1, 'cell_number': 0, 'motion_correction_method': 'Matlab', 'roi_type': 'SpikePursuit', 'roi_number': 1}
        #key = {'subject_id': 454597, 'session': 1, 'cell_number': 1, 'movie_number': 0, 'motion_correction_method': 'Matlab', 'roi_type': 'SpikePursuit', 'roi_number': 1}
        if len(ROIEphysCorrelation&key)>0:#  and key['roi_type'] == 'SpikePursuit' #only spikepursuit for now..
            key_to_compare = key.copy()
            del key_to_compare['roi_number']
            #print(key)
            if np.max((ROIEphysCorrelation&key).fetch('roi_number')) == np.min((ROIEphysCorrelation&key_to_compare).fetch('roi_number')):#np.max(np.abs((ROIEphysCorrelation&key).fetch('corr_coeff'))) == np.max(np.abs((ROIEphysCorrelation&key_to_compare).fetch('corr_coeff'))):
                print('this is it')
                cellstarttime = (ephys_patch.Cell()&key).fetch1('cell_recording_start')
                sessionstarttime = (experiment.Session()&key).fetch1('session_time')
                aptimes = np.asarray((ephysanal.ActionPotential()&key).fetch('ap_max_time'),float)+(cellstarttime-sessionstarttime).total_seconds()
                sweep_start_times,sweep_end_times = (ephys_patch.Sweep()&key).fetch('sweep_start_time','sweep_end_time')
                sweep_start_times = np.asarray(sweep_start_times,float)+(cellstarttime-sessionstarttime).total_seconds()
                sweep_end_times = np.asarray(sweep_end_times,float)+(cellstarttime-sessionstarttime).total_seconds()
                frame_timess,roi_spike_indicess=(imaging.MovieFrameTimes()*imaging.Movie()*imaging.ROI()&key).fetch('frame_times','roi_spike_indices')
                movie_start_times=list()
                movie_end_times = list()
                roi_ap_times = list()
                for frame_times,roi_spike_indices in zip(frame_timess,roi_spike_indicess):
                    movie_start_times.append(frame_times[0])
                    movie_end_times.append(frame_times[-1])
                    roi_ap_times.append(frame_times[roi_spike_indices])
                movie_start_times = np.sort(movie_start_times)
                movie_end_times = np.sort(movie_end_times)
                roi_ap_times=np.sort(np.concatenate(roi_ap_times))
                #%
                ##delete spikes in optical traces where there was no ephys recording
                for start_t,end_t in zip(np.concatenate([sweep_start_times,[np.inf]]),np.concatenate([[0],sweep_end_times])):
                    idxtodel = np.where((roi_ap_times>end_t) & (roi_ap_times<start_t))[0]
                    if len(idxtodel)>0:
                        roi_ap_times = np.delete(roi_ap_times,idxtodel)
                ##delete spikes in ephys traces where there was no imaging
                for start_t,end_t in zip(np.concatenate([movie_start_times,[np.inf]]),np.concatenate([[0],movie_end_times])):
                    idxtodel = np.where((aptimes>end_t) & (aptimes<start_t))[0]
                    if len(idxtodel)>0:
                        #print(idxtodel)
                        aptimes = np.delete(aptimes,idxtodel)
                        #%
                D = np.zeros([len(aptimes),len(roi_ap_times)])
                for idx,apt in enumerate(aptimes):
                    D[idx,:]=(roi_ap_times-apt)*1000
                D_test = np.abs(D)    
                D_test[D_test>15]=1000
                D_test[D<-1]=1000
                X = scipy.optimize.linear_sum_assignment(D_test)    
                #%
                cost = D_test[X[0],X[1]]
                unmatched = np.where(cost == 1000)[0]
                X0_final = np.delete(X[0],unmatched)
                X1_final = np.delete(X[1],unmatched)
                ephys_ap_times = aptimes[X0_final]
                ophys_ap_times = roi_ap_times[X1_final]
                false_positive_time_imaging = list()
                for roi_ap_time in roi_ap_times:
                    if roi_ap_time not in ophys_ap_times:
                        false_positive_time_imaging.append(roi_ap_time)
                false_negative_time_ephys = list()
                for aptime in aptimes:
                    if aptime not in ephys_ap_times:
                        false_negative_time_ephys.append(aptime)
                
                key['ephys_matched_ap_times'] = ephys_ap_times
                key['ophys_matched_ap_times'] = ophys_ap_times
                key['ephys_unmatched_ap_times'] = false_negative_time_ephys
                key['ophys_unmatched_ap_times'] = false_positive_time_imaging
                #print(imaging.ROI()&key)
                #print([len(aptimes),'vs',len(roi_ap_times)])
                self.insert1(key,skip_duplicates=True)

            #else:
                
                #print('this is not it')
            
        
    
    