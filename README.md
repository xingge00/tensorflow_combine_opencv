# tensorflow_combine_opencv
环境搭建
opencv4.0.0
python3.7
detection_objectAPI https://github.com/tensorflow/models/tree/master/research/object_detection
在tensorflow/models/research/目录下执行cmd命令 
protoc object_detection/protocs/*.proto --python_out=.
生成可使用的py文件


训练数据来源
视频数据
http://www.robots.ox.ac.uk/ActiveVision/Research/Projects/2009bbenfold_headpose/Datasets/TownCentreXVID.avi
视频数据的标记文件
http://www.robots.ox.ac.uk/ActiveVision/Research/Projects/2009bbenfold_headpose/Datasets/TownCentre-groundtruth.top

将数据转化为VOC2012的格式

使用object_detection/dataset_tools/create_pascal_tf_record.py脚本生成训练和验证的tfrecord
使用的ssd_mobilenet_v2_coco模型进行迁移学习
使用object_detection/export_inference_graph.py训练导出pb模型
使用opencv的tf_text_graph_ssd生成模型的pbtxt文件(生成时出现了一个断言错误，使用Asserterror_resolve.py改一下模型格式)
有了pb模型和pbtxt就可以脱离tensorflow环境在opencv的dnn模块中使用tensorflow模型了。