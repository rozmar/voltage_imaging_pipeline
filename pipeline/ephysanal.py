import numpy as np
import matplotlib.pyplot as plt
import datajoint as dj
import pandas as pd
import scipy.signal as signal
import scipy.ndimage as ndimage
from pipeline import pipeline_tools, lab, experiment, behavioranal, ephys_patch
#dj.conn()
#%%
schema = dj.schema(pipeline_tools.get_schema_name('ephys-anal'),locals())

#%%

@schema
class ActionPotential(dj.Computed):
    definition = """
    -> ephys_patch.Sweep
    ap_num : smallint unsigned # action potential number in sweep
    ---
    ap_max_index=null : int unsigned # index of AP max on sweep
    ap_max_time=null : decimal(8,4) # time of the AP max relative to recording start
    """
    def make(self, key):
        
        #%%
        #key = {'subject_id': 454263, 'session': 1, 'cell_number': 1, 'sweep_number': 62}
        #print(key)
        keynow = key.copy()
        if len(ActionPotential()&keynow) == 0:
            pd_sweep = pd.DataFrame((ephys_patch.Sweep()&key)*(ephys_patch.SweepResponse()&key)*(ephys_patch.SweepStimulus()&key)*(ephys_patch.SweepMetadata()&key))
            if len(pd_sweep)>0:
                trace = pd_sweep['response_trace'].values[0]
                sr = pd_sweep['sample_rate'][0]
                si = 1/sr
                sigma = .00005
                trace_f = ndimage.gaussian_filter(trace,sigma/si)
                d_trace_f = np.diff(trace_f)/si
                peaks = d_trace_f > 40
                peaks = ndimage.morphology.binary_dilation(peaks,np.ones(int(round(.002/si))))
        
                spikemaxidxes = list()
                while np.any(peaks):
                    #%%
                    spikestart = np.argmax(peaks)
                    spikeend = np.argmin(peaks[spikestart:])+spikestart
                    if spikestart == spikeend:
                        if sum(peaks[spikestart:]) == len(peaks[spikestart:]):
                            spikeend = len(trace)
                    try:
                        sipeidx = np.argmax(trace[spikestart:spikeend])+spikestart
                    except:
                        print(key)
                        sipeidx = np.argmax(trace[spikestart:spikeend])+spikestart
                    spikemaxidxes.append(sipeidx)
                    peaks[spikestart:spikeend] = False
                    #%%
                if len(spikemaxidxes)>0:
                    spikemaxtimes = spikemaxidxes/sr + float(pd_sweep['sweep_start_time'].values[0])
                    spikenumbers = np.arange(len(spikemaxidxes))+1
                    keylist = list()
                    for spikenumber,spikemaxidx,spikemaxtime in zip(spikenumbers,spikemaxidxes,spikemaxtimes):
                        keynow =key.copy()
                        keynow['ap_num'] = spikenumber
                        keynow['ap_max_index'] = spikemaxidx
                        keynow['ap_max_time'] = spikemaxtime
                        keylist.append(keynow)
                        #%%
                    self.insert(keylist,skip_duplicates=True)
       
@schema
class SquarePulse(dj.Computed):
    definition = """
    -> ephys_patch.Sweep
    square_pulse_num : smallint unsigned # action potential number in sweep
    ---
    square_pulse_start_idx: int unsigned # index of sq pulse start
    square_pulse_end_idx: int unsigned # index of sq pulse end
    square_pulse_start_time: decimal(8,4) # time of the sq pulse start relative to recording start
    square_pulse_length: decimal(8,4) # length of the square pulse in seconds
    square_pulse_amplitude: float #amplitude of square pulse
    """
    def make(self, key):
        
        pd_sweep = pd.DataFrame((ephys_patch.Sweep()&key)*(ephys_patch.SweepStimulus()&key)*(ephys_patch.SweepMetadata()&key))
        if len(pd_sweep)>0:
            
            stim = pd_sweep['stimulus_trace'].values[0]
            sr = pd_sweep['sample_rate'].values[0]
            sweepstart = pd_sweep['sweep_start_time'].values[0]
            dstim = np.diff(stim)
            square_pulse_num = -1
            while sum(dstim!=0)>0:
                square_pulse_num += 1
                stimstart = np.argmax(dstim!=0)
                amplitude = dstim[stimstart]
                dstim[stimstart] = 0
                stimend = np.argmax(dstim!=0)
                dstim[stimend] = 0
                stimstart += 1
                stimend += 1
                key['square_pulse_num'] = square_pulse_num
                key['square_pulse_start_idx'] = stimstart
                key['square_pulse_end_idx'] = stimend
                key['square_pulse_start_time'] = stimstart/sr + float(sweepstart)
                key['square_pulse_length'] = (stimend-stimstart)/sr
                key['square_pulse_amplitude'] = amplitude
                self.insert1(key,skip_duplicates=True)
            
@schema
class SquarePulseSeriesResistance(dj.Computed):
    definition = """
    -> SquarePulse
    ---
    series_resistance_squarepulse: decimal(8,2) # series resistance in MOhms 
    """    
    def make(self, key):
        time_back = .0002
        time_capacitance = .0001
        time_forward = .0002
        df_squarepulse = pd.DataFrame((SquarePulse()&key)*ephys_patch.Sweep()*ephys_patch.SweepResponse()*ephys_patch.SweepMetadata())
        stimamplitude = df_squarepulse['square_pulse_amplitude'].values[0]
        if np.abs(stimamplitude)>=40*10**-12:
            trace = df_squarepulse['response_trace'].values[0]
            start_idx = df_squarepulse['square_pulse_start_idx'][0]
            end_idx = df_squarepulse['square_pulse_end_idx'][0]
            sr = df_squarepulse['sample_rate'][0]
            step_back = int(np.round(time_back*sr))
            step_capacitance = int(np.round(time_capacitance*sr))
            step_forward = int(np.round(time_forward*sr))
            
            
            v0_start = np.mean(trace[start_idx-step_back:start_idx])
            vrs_start = np.mean(trace[start_idx+step_capacitance:start_idx+step_capacitance+step_forward])
            v0_end = np.mean(trace[end_idx-step_back:end_idx])
            vrs_end = np.mean(trace[end_idx+step_capacitance:end_idx+step_capacitance+step_forward])
            
            dv_start = vrs_start-v0_start
            RS_start = dv_start/stimamplitude 
            dv_end = vrs_end-v0_end
            RS_end = dv_end/stimamplitude*-1
            
            RS = np.round(np.mean([RS_start,RS_end])/1000000,2)
            key['series_resistance_squarepulse'] = RS
            self.insert1(key,skip_duplicates=True)

@schema
class SweepSeriesResistance(dj.Computed):
    definition = """
    -> ephys_patch.Sweep
    ---
    series_resistance_residual = null: decimal(8,2) # residual series resistance after bridge balance in MOhms 
    series_resistance_bridged = null: decimal(8,2) # bridged series resistance in MOhms 
    series_resistance = null: decimal(8,2) # total series resistance in MOhms 
    """    
    def make(self, key):
        #%%
        #key = {'subject_id':454263,'session':1,'cell_number':0,'sweep_number':0}
        if len((SquarePulseSeriesResistance()&key).fetch('series_resistance_squarepulse'))>0:
            if (ephys_patch.SweepMetadata()&key).fetch('bridgebalenable')[0] == 1:
                bridgeR = (ephys_patch.SweepMetadata()&key).fetch('bridgebalresist')[0]/10**6
            else:
                bridgeR = 0
            try:
                rs_residual = float(np.mean((SquarePulseSeriesResistance()&key).fetch('series_resistance_squarepulse')))
            except:
                print(key)
                rs_residual = float(np.mean((SquarePulseSeriesResistance()&key).fetch('series_resistance_squarepulse')))
            key['series_resistance_residual'] = rs_residual
            key['series_resistance_bridged'] = bridgeR
            key['series_resistance'] = rs_residual + bridgeR
        self.insert1(key,skip_duplicates=True)
        #%%
        
@schema
class SweepFrameTimes(dj.Computed):
    definition = """
    -> ephys_patch.Sweep
    ---
    frame_idx : longblob # index of positive square pulses
    frame_sweep_time : longblob  # time of exposure relative to sweep start in seconds
    frame_time : longblob  # time of exposure relative to recording start in seconds
    """    
    def make(self, key):
        #%%
        #key = {'subject_id':456462,'session':1,'cell_number':3,'sweep_number':24}
        exposure = (ephys_patch.SweepImagingExposure()&key).fetch('imaging_exposure_trace')
        if len(exposure)>0:
            si = 1/(ephys_patch.SweepMetadata()&key).fetch('sample_rate')[0]
            sweeptime = float((ephys_patch.Sweep()&key).fetch('sweep_start_time')[0])
            exposure = np.diff(exposure[0])
            peaks = signal.find_peaks(exposure)
            peaks_idx = peaks[0]
            key['frame_idx']= peaks_idx 
            key['frame_sweep_time']= peaks_idx*si
            key['frame_time']= peaks_idx*si + sweeptime
            self.insert1(key,skip_duplicates=True)
    
    
    
    
# =============================================================================
#     step = 100
#     key = {
#            'subject_id' : 453476,
#            'session':29,
#            'cell_number':1,
#            'sweep_number':3,
#            'square_pulse':0}
#     sqstart = trace[start_idx-step:start_idx+step]
#     sqend = trace[end_idx-step:end_idx+step]
#     time = np.arange(-step,step)/sr*1000
#     fig=plt.figure()
#     ax_v=fig.add_axes([0,0,.8,.8])
#     ax_v.plot(time,sqstart)    
#     ax_v.plot([time_back/2*-1*1000,(time_forward/2+time_capacitance)*1000],[v0_start,vrs_start],'o')    
#     ax_v.plot(time,sqend)    
#     ax_v.plot([time_back/2*-1*1000,(time_forward/2+time_capacitance)*1000],[v0_end,vrs_end],'o')    
# =============================================================================

