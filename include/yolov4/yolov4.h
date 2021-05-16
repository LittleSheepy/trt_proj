#ifndef YOLOV4_H
#define YOLOV4_H
//#include "common.hpp"
#include "NvInferRuntime.h"
#include "NvInfer.h"
#include "logging.h"

#include <fstream>
#include <map>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
//#include "yololayer.h"
#include "yolov4cfg.h"
using namespace nvinfer1;
using namespace yolov4;
namespace ObjDet {
	class YOLOV4
	{
	public:
		YOLOV4();
		~YOLOV4();
		void Init();
		ILayer* convBnMish(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int p, int linx);
		ILayer* convBnLeaky(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int p, int linx);
		ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt);
		int LoadEngine();
		void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream);
		void doInference(IExecutionContext& context, float* input, float* output, int batchSize);
		std::vector<yolov4::Detection> predict(cv::Mat& img);
	public:
		IRuntime*			m_runtime = nullptr;					// 
		ICudaEngine*		m_engine = nullptr;						// 
		IExecutionContext*	m_context = nullptr;					// 
		cudaStream_t		m_stream = nullptr;						// 
		std::string			m_engine_name = "";
		void*				buffers[2];								//

		float				input[BATCH_SIZE * 3 * INPUT_H * INPUT_W];		// 图片数据
		float				output[BATCH_SIZE * OUTPUT_SIZE];				// 输出数据
		int					inputIndex;
		int					outputIndex;
	};
}
#endif // YOLOV5_H
