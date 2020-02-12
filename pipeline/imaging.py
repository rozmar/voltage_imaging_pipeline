import datajoint as dj
import pipeline.lab as lab#, ccf
import pipeline.experiment as experiment
import pipeline.ephys_patch as ephys_patch
import pipeline.ephysanal as ephysanal
from pipeline.pipeline_tools import get_schema_name
schema = dj.schema(get_schema_name('imaging'),locals())


@schema
class Movie(dj.Imported):
    definition = """
    -> experiment.Session
    movie_number                : smallint
    ---
    movie_name                  : varchar(200)          # movie name
    movie_x_size                : double                # (pixels)
    movie_y_size                : double                # (pixels)
    movie_frame_rate            : double                # (Hz)             
    movie_frame_num             : int                   # number of frames        
    movie_start_time            : decimal(10, 4)        # (s) from session start # it should be 10,6??? maybe end_time would also be useful
    movie_pixel_size            : decimal(5,2)          # in microns
    """ 
    
@schema
class MovieFrameTimes(dj.Imported):
    definition = """
    -> Movie
    ---
    frame_times                : longblob              # timing of each frame relative to Session start
    """
    
@schema
class MovieFile(dj.Imported): #MovieFile
    definition = """
    -> Movie 
    movie_file_number         : smallint
    ---
    movie_file_repository     : varchar(200)          # name of the repository where the data are
    movie_file_directory      : varchar(200)          # location of the files  
    movie_file_name           : varchar(100)          # file name
    movie_file_start_frame    : int                   # first frame of this file that belongs to the movie
    movie_file_end_frame      : int                   # last frame of this file that belongs to the movie
    """

@schema
class MotionCorrectionMethod(dj.Lookup): 
    definition = """
    #
    motion_correction_method  :  varchar(30)
    """
    contents = zip(['Matlab','VolPy','Suite2P'])

@schema
class RegisteredMovie(dj.Imported): #MovieFile
    definition = """
    -> Movie
    -> MotionCorrectionMethod
    ---
    registered_movie_mean_image : longblob
    """

@schema
class MotionCorrection(dj.Imported): 
    definition = """
    -> RegisteredMovie
    motion_correction_id    : smallint             # id of motion correction in case of multiple motion corrections
    ---
    motion_corr_description     : varchar(300)         #description of the motion correction
    motion_corr_vectors         : longblob             # registration vectors   #motion_corr_parameters      : longblob              # probably a dict?  ##motion_corr_metrics         : longblob              # ??
    """


@schema
class ROIType(dj.Lookup): 
    definition = """
    #
    roi_type  :  varchar(30)
    """
    contents = zip(['SpikePursuit','SpikePursuit_dexpF0','Suite2P','VolPy','VolPy_dexpF0'])

@schema
class ROI(dj.Imported): 
# ROI (Region of interest - e.g. cells)
    definition = """
    -> RegisteredMovie
    -> ROIType    
    roi_number                      : int           # roi number (restarts for every registered movie)
    ---
    roi_dff                         : longblob      # spikepursuit
    roi_f0                          : longblob      # spikepursuit
    roi_spike_indices               : longblob      # spikepursuit 
    roi_centroid_x                  : double        # ROI centroid  x, pixels
    roi_centroid_y                  : double        # ROI centroid  y, pixels
    roi_mask                        : longblob      # pixel mask 
    """

#-----------------------GROUND TRUTH RELATED STUFF    
# =============================================================================
# @schema
# class ROIEphysCorrelation(dj.Imported): 
# # ROI (Region of interest - e.g. cells)
#     definition = """
#     -> ROI
#     -> ephys_patch.Sweep
#     ---
#     time_lag                        : float #ms   
#     corr_coeff                      : float #-1 - 1
#     """
#     
# @schema
# class ROIAPWave(dj.Imported): 
# # ROI (Region of interest - e.g. cells)
#     definition = """ # this is the optical AP waveform relative to the real AP peak
#     -> ROI
#     -> ephysanal.ActionPotential
#     ---
#     apwave_time                     : longblob
#     apwave_dff                      : longblob
#     apwave_snratio                  : float
#     """
# =============================================================================
#-----------------------GROUND TRUTH RELATED STUFF    