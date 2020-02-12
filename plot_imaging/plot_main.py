import numpy as np
from pipeline import pipeline_tools
from pipeline import lab, experiment, ephys_patch, ephysanal, imaging, imaging_gt
import matplotlib.pyplot as plt
def plot_precision_recall(key_cell,binwidth =  30,frbinwidth = 0.01,firing_rate_window = 3):
    fr_kernel = np.ones(int(firing_rate_window/frbinwidth))/(firing_rate_window/frbinwidth)
    first_movie_start_time = np.min(np.asarray(((imaging.Movie()*imaging_gt.GroundTruthROI())&key_cell).fetch('movie_start_time'),float))
    ephys_matched_ap_times,ephys_unmatched_ap_times,ophys_matched_ap_times,ophys_unmatched_ap_times = (imaging_gt.GroundTruthROI()&key_cell).fetch('ephys_matched_ap_times','ephys_unmatched_ap_times','ophys_matched_ap_times','ophys_unmatched_ap_times')
    #%
    ephys_matched_ap_times = np.concatenate(ephys_matched_ap_times) - first_movie_start_time 
    ephys_unmatched_ap_times = np.concatenate(ephys_unmatched_ap_times) - first_movie_start_time 
    all_ephys_ap_times = np.concatenate([ephys_matched_ap_times,ephys_unmatched_ap_times])
    ophys_matched_ap_times = np.concatenate(ophys_matched_ap_times) - first_movie_start_time 
    ophys_unmatched_ap_times = np.concatenate(ophys_unmatched_ap_times) - first_movie_start_time 
    all_ophys_ap_times = np.concatenate([ophys_matched_ap_times,ophys_unmatched_ap_times])
    all_times = np.concatenate([ephys_matched_ap_times,ephys_unmatched_ap_times,ophys_matched_ap_times,ophys_unmatched_ap_times])
    maxtime = np.max(all_times)
    #%
    fr_bincenters = np.arange(frbinwidth/2,maxtime+frbinwidth,frbinwidth)
    fr_binedges = np.concatenate([fr_bincenters-frbinwidth/2,[fr_bincenters[-1]+frbinwidth/2]])
    fr_e = np.histogram(all_ephys_ap_times,fr_binedges)[0]/frbinwidth
    fr_e = np.convolve(fr_e, fr_kernel,'same')
    fr_o = np.histogram(all_ophys_ap_times,fr_binedges)[0]/frbinwidth
    fr_o = np.convolve(fr_o, fr_kernel,'same')
    #%
    bincenters = np.arange(binwidth/2,maxtime+binwidth,binwidth)
    binedges = np.concatenate([bincenters-binwidth/2,[bincenters[-1]+binwidth/2]])
    ephys_matched_ap_num_binned,tmp = np.histogram(ephys_matched_ap_times,binedges)
    ephys_unmatched_ap_num_binned,tmp = np.histogram(ephys_unmatched_ap_times,binedges)
    ophys_matched_ap_num_binned,tmp = np.histogram(ophys_matched_ap_times,binedges)
    ophys_unmatched_ap_num_binned,tmp = np.histogram(ophys_unmatched_ap_times,binedges)
    precision_binned = ophys_matched_ap_num_binned/(ophys_matched_ap_num_binned+ophys_unmatched_ap_num_binned)
    recall_binned = ephys_matched_ap_num_binned/(ephys_matched_ap_num_binned+ephys_unmatched_ap_num_binned)
    f1_binned = 2*precision_binned*recall_binned/(precision_binned+recall_binned)
    #%
    fig=plt.figure()
    ax_rates = fig.add_axes([0,1.4,2,.3])
    ax_spikes = fig.add_axes([0,1,2,.3])
    ax = fig.add_axes([0,0,2,.8])
    ax_latency = fig.add_axes([0,-1,2,.8])
    ax_latency_hist = fig.add_axes([.5,-2,1,.8])
    
    ax.plot(bincenters,precision_binned,'go-',label = 'precision')    
    ax.plot(bincenters,recall_binned,'ro-',label = 'recall')    
    ax.plot(bincenters,f1_binned,'bo-',label = 'f1 score')    
    ax.set_xlim([binedges[0],binedges[-1]])
    ax.legend()
    ax_spikes.plot(ephys_unmatched_ap_times,np.zeros(len(ephys_unmatched_ap_times)),'r|', ms = 10)
    ax_spikes.plot(all_ephys_ap_times,np.zeros(len(all_ephys_ap_times))+.33,'k|', ms = 10)
    ax_spikes.plot(ophys_unmatched_ap_times,np.ones(len(ophys_unmatched_ap_times)),'r|',ms = 10)
    ax_spikes.plot(all_ophys_ap_times,np.ones(len(all_ophys_ap_times))-.33,'g|', ms = 10)
    ax_spikes.set_yticks([0,.33,.67,1])
    ax_spikes.set_yticklabels(['false negative','ephys','ophys','false positive'])
    ax_spikes.set_ylim([-.2,1.2])
    ax_spikes.set_xlim([binedges[0],binedges[-1]])
    ax_rates.plot(fr_bincenters,fr_e,'k-',label = 'ephys')
    ax_rates.plot(fr_bincenters,fr_o,'g-',label = 'ophys')
    ax_rates.legend()
    ax_rates.set_xlim([binedges[0],binedges[-1]])
    ax_rates.set_ylabel('Firing rate (Hz)')
    ax_rates.set_title('subject_id: %d, cell number: %s' %(key_cell['subject_id'],key_cell['cell_number']))
    ax_latency.plot(ephys_matched_ap_times,1000*(ophys_matched_ap_times-ephys_matched_ap_times),'ko')
    ax_latency.set_ylabel('ephys-ophys spike latency (ms)')
    ax_latency.set_ylim([0,10])
    ax_latency.set_xlabel('time from first movie start (s)')
    
    ax_latency_hist.hist(1000*(ophys_matched_ap_times-ephys_matched_ap_times),np.arange(-5,15,.1))
    ax_latency_hist.set_xlabel('ephys-ophys spike latency (ms)')
    ax_latency_hist.set_ylabel('matched ap count')
    plot_ephys_ophys_trace(key_cell,ephys_matched_ap_times[0],trace_window = 1)
    
def plot_ephys_ophys_trace(key_cell,time_to_plot=None,trace_window = 1):
    #%%
    fig=plt.figure()
    ax_ophys = fig.add_axes([0,0,2,.8])
    ax_ephys = fig.add_axes([0,-1,2,.8])
    
    session_time, cell_recording_start = (experiment.Session()*ephys_patch.Cell()&key_cell).fetch1('session_time','cell_recording_start')
    first_movie_start_time =  np.min(np.asarray(((imaging.Movie()*imaging_gt.GroundTruthROI())&key_cell).fetch('movie_start_time'),float))
    if not time_to_plot:
        time_to_plot = trace_window/2#ephys_matched_ap_times[0]
    session_time_to_plot = time_to_plot+first_movie_start_time  + session_time.total_seconds()# time relative to session start
    cell_time_to_plot= session_time_to_plot + -cell_recording_start.total_seconds() # time relative to recording start
    #%%
    sweep_start_times,sweep_end_times,sweep_nums = (ephys_patch.Sweep()&key_cell).fetch('sweep_start_time','sweep_end_time','sweep_number')
    needed_start_time = cell_time_to_plot - trace_window/2
    needed_end_time = cell_time_to_plot + trace_window/2
    #%%
    sweep_nums = sweep_nums[((sweep_start_times > needed_start_time) & (sweep_start_times < needed_end_time)) |
                            ((sweep_end_times > needed_start_time) & (sweep_end_times < needed_end_time)) | 
                            ((sweep_end_times > needed_end_time) & (sweep_start_times < needed_start_time)) ]
    #%%
    ephys_traces = list()
    ephys_trace_times = list()
    ephys_sweep_start_times = list()
    for sweep_num in sweep_nums:
        sweep = ephys_patch.Sweep()&key_cell&'sweep_number = %d' % sweep_num
        trace,sr= (ephys_patch.SweepMetadata()*ephys_patch.SweepResponse()&sweep).fetch1('response_trace','sample_rate')
        sweep_start_time  = float(sweep.fetch('sweep_start_time')) 
        trace_time = np.arange(len(trace))/sr+ sweep_start_time  
        
        trace_idx = (cell_time_to_plot-trace_window/2 < trace_time) & (cell_time_to_plot+trace_window/2 > trace_time)
                
        ax_ephys.plot(trace_time[trace_idx]-first_movie_start_time,trace[trace_idx],'k-')
        
        ephys_traces.append(trace)
        ephys_trace_times.append(trace_time)
        ephys_sweep_start_times.append(sweep_start_time)

    ephysdata = {'ephys_traces':ephys_traces,'ephys_trace_times':ephys_trace_times}
            #%%
    movie_nums, movie_start_times,movie_frame_rates,movie_frame_nums = ((imaging.Movie()*imaging_gt.GroundTruthROI())&key_cell).fetch('movie_number','movie_start_time','movie_frame_rate','movie_frame_num')
    movie_start_times = np.asarray(movie_start_times, float)
    movie_end_times = np.asarray(movie_start_times, float)+np.asarray(movie_frame_nums, float)/np.asarray(movie_frame_rates, float)
    needed_start_time = session_time_to_plot - trace_window/2
    needed_end_time = session_time_to_plot + trace_window/2
    movie_nums = movie_nums[((movie_start_times >= needed_start_time) & (movie_start_times <= needed_end_time)) |
                            ((movie_end_times >= needed_start_time) & (movie_end_times <= needed_end_time)) | 
                            ((movie_end_times >= needed_end_time) & (movie_start_times <= needed_start_time)) ]
    dffs=list()
    frametimes = list()
    for movie_num in movie_nums:
    #movie_num = movie_nums[(session_time_to_plot>movie_start_times)&(session_time_to_plot<movie_end_times)][0]
        key_movie = key_cell.copy()
        key_movie['movie_number'] = movie_num

        dff = ((imaging.ROI()*imaging_gt.GroundTruthROI())&key_movie).fetch1('roi_dff')*-1
        frame_times = ((imaging.MovieFrameTimes()*imaging_gt.GroundTruthROI())&key_movie).fetch1('frame_times') + (session_time-cell_recording_start).total_seconds() #modified to cell time
        frame_idx = (cell_time_to_plot-trace_window/2 < frame_times) & (cell_time_to_plot+trace_window/2 > frame_times)
        ax_ophys.plot(frame_times[frame_idx]-first_movie_start_time,dff[frame_idx],'g-')
        dffs.append(dff)
        frametimes.append(frame_times)
    ophysdata = {'ophys_traces':dffs,'ophys_trace_times':frametimes}
    
    ax_ophys.set_xlim([cell_time_to_plot-trace_window/2-first_movie_start_time,cell_time_to_plot+trace_window/2-first_movie_start_time])
    ax_ophys.set_ylabel('df/f')
    
    ax_ephys.set_xlim([cell_time_to_plot-trace_window/2-first_movie_start_time,cell_time_to_plot+trace_window/2-first_movie_start_time])
    ax_ephys.set_ylabel('Vm (mV))')
    ax_ephys.set_xlabel('time from first movie start (s)')
    outdict = {'ephys':ephysdata,'ophys':ophysdata}
    #%%
    return outdict 