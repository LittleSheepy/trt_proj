#pragma once
// stuff we know about the network and the input/output blobs
#include "logging.h"
namespace yolov4
{
	// yololayer
	static constexpr int CHECK_COUNT = 3;
	static constexpr float IGNORE_THRESH = 0.1f;
	static constexpr int MAX_OUTPUT_BBOX_COUNT = 1000;
	static constexpr int CLASS_NUM = 80;
	static constexpr int INPUT_H = 608;
	static constexpr int INPUT_W = 608;

	struct YoloKernel
	{
		int width;
		int height;
		float anchors[CHECK_COUNT * 2];
	};

	static constexpr YoloKernel yolo1 = {
		INPUT_W / 8,
		INPUT_H / 8,
		{12,16, 19,36, 40,28}
	};
	static constexpr YoloKernel yolo2 = {
		INPUT_W / 16,
		INPUT_H / 16,
		{36,75, 76,55, 72,146}
	};
	static constexpr YoloKernel yolo3 = {
		INPUT_W / 32,
		INPUT_H / 32,
		{142,110, 192,243, 459,401}
	};

	static constexpr int LOCATIONS = 4;
	struct alignas(float) Detection {
		//x y w h
		float bbox[LOCATIONS];
		float det_confidence;
		float class_id;
		float class_confidence;
	};


	#define USE_FP16  // comment out this if want to use FP32
	#define DEVICE 0  // GPU id
	#define NMS_THRESH 0.4
	#define BBOX_CONF_THRESH 0.5
	#define BATCH_SIZE 1
	//static const int INPUT_H = 608;
	//static const int INPUT_W = 608;
	static const int DETECTION_SIZE = sizeof(Detection) / sizeof(float);
	static const int OUTPUT_SIZE = MAX_OUTPUT_BBOX_COUNT * DETECTION_SIZE + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
	static const char* INPUT_BLOB_NAME = "data";
	static const char* OUTPUT_BLOB_NAME = "prob";
	static Logger gLogger;
}