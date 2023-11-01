# SAM_TensorRT
This is the code to implement Segment Anything (SAM) using TensorRT(C++).


We use MobileSAM as SAM in order to be suitable for lightweight deployment of applications.
## Export engine model


## Export encoder
```
/usr/local/TensorRT-8.5.3.1/bin/trtexec --onnx=./models/mobile_sam_encoder.onnx --saveEngine=./models/mobile_sam_encoder.engine --memPoolSize=workspace:10240
```

## Export decoder
```
/usr/local/TensorRT-8.5.3.1/bin/trtexec --onnx=./models/mobile_sam_decoder.onnx --minShapes=point_coords:1x1x2,point_labels:1x1 --optShapes=point_coords:1x1x2,point_labels:1x1 --maxShapes=point_coords:1x10x2,point_labels:1x10 --saveEngine=./models/mobile_sam_decoder.engine
```

## Run
```
mkdir build
cd build
cmake ..
make -j8
./sam
```

