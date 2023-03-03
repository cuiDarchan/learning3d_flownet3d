# 测试文件使用说明
<br/>

## 1. 通过setup.py 安装库文件
setup.py文件中提供pytorch新版本（默认）和Pytorch老版本
```
cd utils/lib/
python setup.py install
```
## 2. 下载数据
将下载好的数据放在本工程的`data目录`下，然后修改`data_utils/dataloaders.py`414行的数据绝对路径，如
```
root = os.path.abspath('/home/cui/workspace/deepLearning/learning3d/data/')
```
<br/>

## 3. 运行测试脚本
### 3.1 参数修改
预训练模型地址（--pretrained），适配自己工程内部，如`/home/cui/workspace/deepLearning/learning3d/pretrained/exp_flownet/models/model.best.t7`
### 3.2 官方测试脚本
```
python examples/test_flownet.py
```
### 3.3 自己添加一些功能的测试脚本
脚本内部提供各种显示方式，如场景流箭头，添加场景流后的预测点云显示。
```
python test/test_flownet.py
```