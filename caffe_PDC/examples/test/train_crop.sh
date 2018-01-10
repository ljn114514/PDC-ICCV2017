GLOG_logtostderr=0 GLOG_log_dir=./log ../../build/tools/caffe train --solver=facial_point_crop_solver.prototxt --gpu=1 --weights=./model/init_v2.0.caffemodel
#--weights=./model/facial_point_iter_200000.caffemodel
#--snapshot=/home/lvjiangjing/Landmark/caffe_68/examples/VGG/model/facial_point_iter_20000.solverstate

#
