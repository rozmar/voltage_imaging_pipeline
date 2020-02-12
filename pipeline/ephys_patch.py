import datajoint as dj

# =============================================================================
# import numpy as np
# 
import pipeline.lab as lab#, ccf
import pipeline.experiment as experiment#, ccf
from pipeline.pipeline_tools import get_schema_name
#from . import get_schema_name
# 
schema = dj.schema(get_schema_name('ephys_patch'),locals())
#schema = dj.schema('rozmar_foraging_experiment',locals())
# =============================================================================
#schema = dj.schema('rozmar_tutorial', locals())


@schema
class CellType(dj.Lookup):
    definition = """
    #
    cell_type  :  varchar(100)
    ---
    cell_type_description :  varchar(4000)
    """
    contents = [
        ('pyr', 'putative pyramidal'),
        ('int', 'putative interneuron'),
        ('glia', 'astrocyte or oligodendrocyte precursor like cell'),
        ('unidentified', 'can''t tell based on electrophysiological recording')
    ]


@schema
class Cell(dj.Imported):
    definition = """
    -> experiment.Session
    cell_number: smallint
    ---
    -> CellType
    depth: smallint # microns from the surface of the brain
    cell_recording_start: time #time at the start of the first sweep
    """

@schema
class CellNotes(dj.Imported): #TO DO: fill in metadata
    definition = """
    -> Cell
    ---
    notes: varchar(1000) # notes during the recording
    """

@schema
class Sweep(dj.Imported):
    definition = """
    -> Cell
    sweep_number: smallint
    ---
    sweep_start_time: decimal(12, 6)   # (s) from recording start
    sweep_end_time: decimal(12, 6)   # (s) from recording start
    protocol_name: varchar(100)
    protocol_sweep_number: smallint
    """

@schema
class ClampParams(dj.Imported):
    definition = """
    #
    bridgebalenable  :  tinyint  # 0 or 1
    bridgebalresist  :  float # in Ohms
    fastcompcap : float # im Farads
    fastcomptau : float # in seconds
    holding : float # in amps/volts
    holdingenable : tinyint  # 0 or 1
    leaksubenable : tinyint  # 0 or 1
    leaksubresist : float
    neutralizationcap : float
    neutralizationenable : tinyint  # 0 or 1
    outputzeroamplitude : float
    outputzeroenable : tinyint  # 0 or 1
    pipetteoffset : float
    primarysignalhpf : float
    primarysignallpf : float
    rscompbandwidth : float
    rscompcorrection : float
    rscompenable : tinyint  # 0 or 1
    slowcompcap : float
    slowcomptau : float
    wholecellcompcap : float
    wholecellcompenable : tinyint  # 0 or 1
    wholecellcompresist : float
    """

@schema
class RecordingMode(dj.Lookup):
    definition = """
    #
    recording_mode  :  varchar(100)
    """
    contents = zip(['current clamp', 'voltage clamp'])
    

@schema
class SweepMetadata(dj.Imported): #TO DO: fill in metadata
    definition = """
    -> Sweep
    ---
    -> RecordingMode
    -> ClampParams
    sample_rate: int
    """



@schema
class SweepResponse(dj.Imported): #TO DO: fill in metadata
    definition = """
    -> Sweep
    ---
    response_trace  : longblob #
    response_units: varchar(5)
    """

@schema
class SweepStimulus(dj.Imported): #TO DO: fill in metadata
    definition = """
    -> Sweep
    ---
    stimulus_trace  : longblob #
    stimulus_units: varchar(5)
    """
    
@schema
class SweepTemperature(dj.Imported): #TO DO: fill in metadata
    definition = """
    -> Sweep
    ---
    temperature_trace  : longblob #
    temperature_units: varchar(5)
    """
@schema
class SweepImagingExposure(dj.Imported): #TO DO: fill in metadata
    definition = """
    -> Sweep
    ---
    imaging_exposure_trace  : longblob #
    """