
import json
project = 'voltage imaging'
with open('dj_local_conf.json') as json_file:
    variables = json.load(json_file)
variables['project'] = project
with open('dj_local_conf.json', 'w') as outfile:
    json.dump(variables, outfile, indent=2, sort_keys=True)
#import datapipeline_metadata
#import datapipeline_behavior
#import datapipeline_elphys
import datapipeline_imaging
homefolder = '/nrs/svoboda/rozsam'
#%%
# =============================================================================
# datapipeline_metadata.populatemetadata()
# #datapipeline_behavior.populatebehavior()
# #datapipeline_behavior.populatemytables()
# datapipeline_elphys.populateelphys()
# datapipeline_elphys.populatemytables()
# 
# =============================================================================
#datapipeline_imaging.upload_movie_metadata()
#datapipeline_imaging.calculate_exposition_times()

#datapipeline_imaging.save_spikepursuit_pipeline()
#datapipeline_imaging.save_volpy_pipeline(roitype = 'VolPy',motion_corr = 'VolPy')
#datapipeline_imaging.save_volpy_pipeline(roitype = 'VolPy_denoised',motion_corr = 'VolPy2x')
#datapipeline_imaging.save_volpy_pipeline(roitype = 'VolPy_raw',motion_corr = 'VolPy')
#datapipeline_imaging.save_volpy_pipeline(roitype = 'VolPy_denoised_raw',motion_corr = 'VolPy2x')
datapipeline_imaging.upload_gt_correlations_apwaves(cores = 8)
datapipeline_imaging.populatemytables_gt(cores = 8)
print('done')