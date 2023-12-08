# annotation_nerf_pytorch
The content of this repository is a detailed chinese annotation of nerf_pytorch, which can be repeated for reference    
**Source code:**&emsp;[nerf_pytorch](https://github.com/yenchenlin/nerf-pytorch)
# Note
* In the part of loading data, because it involves the actual needs of the project, this repository only implements the **llff** data format.  
* The detailed location of the configuration file path is on line 623 in run_nerf, which can be adjusted as needed.  
* The code defaults to using CPU. To change it to CUDA, adjust the comments in run_nerf at lines 18 and 927.  
# The meaning of each comment type in the code 
* TODO: Wait for completion  
* SECTION: The code in the same section is closely related   
* LINK: The code at the same link is understood together, where LINK:END depends on LINK:START  
* dir_: Small directory, placed under each section or function  
* FUNCTION: Functions in the same section are closely related and work towards the same goal  
* EXPLAIN: Interpretation of function  

* Ps: It is recommended to set the above annotation types to different colors in advance in pycharm for easy understanding
