# nerf_pytorch_detailed_annotation
仓库内容为nerf_pytorch的详细注释，可供复现参考  
在加载数据部分，由于涉及到项目的实际需求，仓库仅对llff数据格式进行了实现，其余数据格式未作处理    
附nerf_pytorch源码位置：https://github.com/yenchenlin/nerf-pytorch
# 运行方法
不需要命令行操作，pycharm中直接运行即可  
配置文件路径的详细位置在run_nerf中第623行，可自行调整
## 设备更改：
代码默认使用cpu，如需更改为cuda，将run_nerf中第18行与第927行注释调整即可
# 代码中各注释类型的含义  
TODO: 待完成  
SECTION: 模块，同一个模块中的代码关联紧密  
LINK: 链接，相同链接处的代码搭配理解，其中LINK:END依赖LINK:START  
dir_: 小型目录，置于各模块或函数下  
FUNCTION: 函数模块，同一个模块中的函数关联紧密  
EXPLAIN: 函数的解释  

Ps: 建议在pycharm中将上述各注释类型提前设置成不同颜色，方便理解
