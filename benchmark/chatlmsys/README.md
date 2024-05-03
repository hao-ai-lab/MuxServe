# End-to-End Evaluation For Real Scenarios

* `basic setting:`
```
4 x 8 A100
55-55.5 days, 8091 requests
16 models, 4x7b, 6x13b, 3x30b, 3x65b
maxrate from 8 to 32 (req/s)
avg rate from 2 to 11 (req/s)
```

* `file structure:`
```
---
 - merged: The tpt data folder of chatlmsys
 - plot.py: plot the tpt and slo
 - yamls: basic yaml file for placement gen

 - chatlmsys_translation.py: translate the dataset
 - cfg_gen.py: generate the placement for muxserve
 - muxserve2spatial.py: translate the muxserve config into spatial config
 - muxserve2temporal.py: translate the muxserve config into temporal config
 - cmd_gen.py: generate the running command for muxserve,spatial and temporal
 - profile.sh: script for run
 - merge.py: merge the file into `merged` folder
```
