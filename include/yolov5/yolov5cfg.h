#pragma once
#include "logging.h"
namespace yolov5 {
	#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32
	#define DEVICE 0  // GPU id
	#define NMS_THRESH 0.4
	#define CONF_THRESH 0.5
	#define BATCH_SIZE 1

	static constexpr int CHECK_COUNT = 3;
	static constexpr float IGNORE_THRESH = 0.1f;
	struct YoloKernel
	{
		int width;
		int height;
		float anchors[CHECK_COUNT * 2];
	};

	// stuff we know about the network and the input/output blobs
	static const int INPUT_H = 640;		// yolov5's input height and width must be divisible by 32.
	static const int INPUT_W = 640;
	static const int CLASS_NUM = 80;
	static constexpr int MAX_OUTPUT_BBOX_COUNT = 1000;

	static constexpr int LOCATIONS = 4;
	struct alignas(float) Detection {
		//center_x center_y w h
		float bbox[LOCATIONS];
		float conf;  // bbox_conf * cls_conf
		float class_id;
	};

	static const int OUTPUT_SIZE = MAX_OUTPUT_BBOX_COUNT * sizeof(Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
	static const char* INPUT_BLOB_NAME = "data";
	static const char* OUTPUT_BLOB_NAME = "prob";
	static Logger gLogger;

	// p3Ç°µÄ
	#define NET s  // s m l x
	#define NETSTRUCT(str) createEngine_##str
	#define CREATENET(net) NETSTRUCT(net)
	#define STR1(x) #x
	#define STR2(x) STR1(x)
}