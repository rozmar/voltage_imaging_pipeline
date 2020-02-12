
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
#datapipeline_imaging.save_volpy_pipeline()

#datapipeline_imaging.upload_gt_correlations_apwaves()
datapipeline_imaging.populatemytables_gt(cores = 1)
print('done')