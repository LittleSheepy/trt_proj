/*******************************************************************
** 文件名:	AIServer.cpp
** 版  权:	(C) littlesheepy 2020 - All Rights Reserved
** 创建人:	littlesheepy
** 日  期:	12/02/2020
** 版  本:	1.0
** 描  述:
** 应  用:
**************************** 修改记录 ******************************
** 修改人:
** 日  期:
** 描  述:
********************************************************************/
#include <chrono>
#include "cuda_runtime_api.h"
#include "NvInfer.h"
#include "yolov5.h"
#include "common.h"
#include "utils.h"
#include "yolov5cfg.h"


using namespace nvinfer1;
//using namespace Yolo;
using namespace yolov5;

#include "yolov5layer.h"
REGISTER_TENSORRT_PLUGIN(YoloPluginCreator);
//extern std::string IMG_NAME;
namespace ObjDet {
	YOLOV5::YOLOV5()
	{

	}
	YOLOV5::~YOLOV5()
	{
		// Release stream and buffers
		cudaStreamDestroy(this->m_stream);
		CHECK(cudaFree(this->buffers[inputIndex]));
		CHECK(cudaFree(this->buffers[outputIndex]));
		// Destroy the engine
		m_context->destroy();
		m_engine->destroy();
		m_runtime->destroy();
	}

	void YOLOV5::Init(std::string& engine_name)
	{
		cudaSetDevice(DEVICE);
		m_engine_name = engine_name;
		// cfg
		//Config cfg("cfg.txt");
	}


	ICudaEngine* YOLOV5::build_engine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, float& gd, float& gw, std::string& wts_name) {
		INetworkDefinition* network = builder->createNetworkV2(0U);

		// Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
		ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
		assert(data);

		std::map<std::string, Weights> weightMap = loadWeights(wts_name);

		/* ------ yolov5 backbone------ */
		auto focus0 = focus(network, weightMap, *data, 3, get_width(64, gw), 3, "model.0");
		auto conv1 = convBlock(network, weightMap, *focus0->getOutput(0), get_width(128, gw), 3, 2, 1, "model.1");
		auto bottleneck_CSP2 = C3(network, weightMap, *conv1->getOutput(0), get_width(128, gw), get_width(128, gw), get_depth(3, gd), true, 1, 0.5, "model.2");
		auto conv3 = convBlock(network, weightMap, *bottleneck_CSP2->getOutput(0), get_width(256, gw), 3, 2, 1, "model.3");
		auto bottleneck_csp4 = C3(network, weightMap, *conv3->getOutput(0), get_width(256, gw), get_width(256, gw), get_depth(9, gd), true, 1, 0.5, "model.4");
		auto conv5 = convBlock(network, weightMap, *bottleneck_csp4->getOutput(0), get_width(512, gw), 3, 2, 1, "model.5");
		auto bottleneck_csp6 = C3(network, weightMap, *conv5->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(9, gd), true, 1, 0.5, "model.6");
		auto conv7 = convBlock(network, weightMap, *bottleneck_csp6->getOutput(0), get_width(1024, gw), 3, 2, 1, "model.7");
		auto spp8 = SPP(network, weightMap, *conv7->getOutput(0), get_width(1024, gw), get_width(1024, gw), 5, 9, 13, "model.8");

		/* ------ yolov5 head ------ */
		auto bottleneck_csp9 = C3(network, weightMap, *spp8->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.9");
		auto conv10 = convBlock(network, weightMap, *bottleneck_csp9->getOutput(0), get_width(512, gw), 1, 1, 1, "model.10");

		auto upsample11 = network->addResize(*conv10->getOutput(0));
		assert(upsample11);
		upsample11->setResizeMode(ResizeMode::kNEAREST);
		upsample11->setOutputDimensions(bottleneck_csp6->getOutput(0)->getDimensions());

		ITensor* inputTensors12[] = { upsample11->getOutput(0), bottleneck_csp6->getOutput(0) };
		auto cat12 = network->addConcatenation(inputTensors12, 2);
		auto bottleneck_csp13 = C3(network, weightMap, *cat12->getOutput(0), get_width(1024, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.13");
		auto conv14 = convBlock(network, weightMap, *bottleneck_csp13->getOutput(0), get_width(256, gw), 1, 1, 1, "model.14");

		auto upsample15 = network->addResize(*conv14->getOutput(0));
		assert(upsample15);
		upsample15->setResizeMode(ResizeMode::kNEAREST);
		upsample15->setOutputDimensions(bottleneck_csp4->getOutput(0)->getDimensions());

		ITensor* inputTensors16[] = { upsample15->getOutput(0), bottleneck_csp4->getOutput(0) };
		auto cat16 = network->addConcatenation(inputTensors16, 2);

		auto bottleneck_csp17 = C3(network, weightMap, *cat16->getOutput(0), get_width(512, gw), get_width(256, gw), get_depth(3, gd), false, 1, 0.5, "model.17");

		/* ------ detect ------ */
		IConvolutionLayer* det0 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), 3 * (CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.0.weight"], weightMap["model.24.m.0.bias"]);
		auto conv18 = convBlock(network, weightMap, *bottleneck_csp17->getOutput(0), get_width(256, gw), 3, 2, 1, "model.18");
		ITensor* inputTensors19[] = { conv18->getOutput(0), conv14->getOutput(0) };
		auto cat19 = network->addConcatenation(inputTensors19, 2);
		auto bottleneck_csp20 = C3(network, weightMap, *cat19->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.20");
		IConvolutionLayer* det1 = network->addConvolutionNd(*bottleneck_csp20->getOutput(0), 3 * (yolov5::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.1.weight"], weightMap["model.24.m.1.bias"]);
		auto conv21 = convBlock(network, weightMap, *bottleneck_csp20->getOutput(0), get_width(512, gw), 3, 2, 1, "model.21");
		ITensor* inputTensors22[] = { conv21->getOutput(0), conv10->getOutput(0) };
		auto cat22 = network->addConcatenation(inputTensors22, 2);
		auto bottleneck_csp23 = C3(network, weightMap, *cat22->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.23");
		IConvolutionLayer* det2 = network->addConvolutionNd(*bottleneck_csp23->getOutput(0), 3 * (yolov5::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.2.weight"], weightMap["model.24.m.2.bias"]);

		auto yolo = addYoLoLayer(network, weightMap, "model.24", std::vector<IConvolutionLayer*>{det0, det1, det2});
		yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
		network->markOutput(*yolo->getOutput(0));

		// Build engine
		builder->setMaxBatchSize(maxBatchSize);
		config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#if defined(USE_FP16)
		config->setFlag(BuilderFlag::kFP16);
#elif defined(USE_INT8)
		std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
		assert(builder->platformHasFastInt8());
		config->setFlag(BuilderFlag::kINT8);
		Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(1, INPUT_W, INPUT_H, "./coco_calib/", "int8calib.table", INPUT_BLOB_NAME);
		config->setInt8Calibrator(calibrator);
#endif

		std::cout << "Building engine, please wait for a while..." << std::endl;
		ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
		std::cout << "Build engine successfully!" << std::endl;

		// Don't need the network any more
		network->destroy();

		// Release host memory
		for (auto& mem : weightMap)
		{
			free((void*)(mem.second.values));
		}

		return engine;
	}

	ICudaEngine* YOLOV5::build_engine_p6(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, float& gd, float& gw, std::string& wts_name) {
		INetworkDefinition* network = builder->createNetworkV2(0U);

		// Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
		ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
		assert(data);

		std::map<std::string, Weights> weightMap = loadWeights(wts_name);

		/* ------ yolov5 backbone------ */
		auto focus0 = focus(network, weightMap, *data, 3, get_width(64, gw), 3, "model.0");
		auto conv1 = convBlock(network, weightMap, *focus0->getOutput(0), get_width(128, gw), 3, 2, 1, "model.1");
		auto c3_2 = C3(network, weightMap, *conv1->getOutput(0), get_width(128, gw), get_width(128, gw), get_depth(3, gd), true, 1, 0.5, "model.2");
		auto conv3 = convBlock(network, weightMap, *c3_2->getOutput(0), get_width(256, gw), 3, 2, 1, "model.3");
		auto c3_4 = C3(network, weightMap, *conv3->getOutput(0), get_width(256, gw), get_width(256, gw), get_depth(9, gd), true, 1, 0.5, "model.4");
		auto conv5 = convBlock(network, weightMap, *c3_4->getOutput(0), get_width(512, gw), 3, 2, 1, "model.5");
		auto c3_6 = C3(network, weightMap, *conv5->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(9, gd), true, 1, 0.5, "model.6");
		auto conv7 = convBlock(network, weightMap, *c3_6->getOutput(0), get_width(768, gw), 3, 2, 1, "model.7");
		auto c3_8 = C3(network, weightMap, *conv7->getOutput(0), get_width(768, gw), get_width(768, gw), get_depth(3, gd), true, 1, 0.5, "model.8");
		auto conv9 = convBlock(network, weightMap, *c3_8->getOutput(0), get_width(1024, gw), 3, 2, 1, "model.9");
		auto spp10 = SPP(network, weightMap, *conv9->getOutput(0), get_width(1024, gw), get_width(1024, gw), 3, 5, 7, "model.10");
		auto c3_11 = C3(network, weightMap, *spp10->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.11");

		/* ------ yolov5 head ------ */
		auto conv12 = convBlock(network, weightMap, *c3_11->getOutput(0), get_width(768, gw), 1, 1, 1, "model.12");
		auto upsample13 = network->addResize(*conv12->getOutput(0));
		assert(upsample13);
		upsample13->setResizeMode(ResizeMode::kNEAREST);
		upsample13->setOutputDimensions(c3_8->getOutput(0)->getDimensions());
		ITensor* inputTensors14[] = { upsample13->getOutput(0), c3_8->getOutput(0) };
		auto cat14 = network->addConcatenation(inputTensors14, 2);
		auto c3_15 = C3(network, weightMap, *cat14->getOutput(0), get_width(1536, gw), get_width(768, gw), get_depth(3, gd), false, 1, 0.5, "model.15");

		auto conv16 = convBlock(network, weightMap, *c3_15->getOutput(0), get_width(512, gw), 1, 1, 1, "model.16");
		auto upsample17 = network->addResize(*conv16->getOutput(0));
		assert(upsample17);
		upsample17->setResizeMode(ResizeMode::kNEAREST);
		upsample17->setOutputDimensions(c3_6->getOutput(0)->getDimensions());
		ITensor* inputTensors18[] = { upsample17->getOutput(0), c3_6->getOutput(0) };
		auto cat18 = network->addConcatenation(inputTensors18, 2);
		auto c3_19 = C3(network, weightMap, *cat18->getOutput(0), get_width(1024, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.19");

		auto conv20 = convBlock(network, weightMap, *c3_19->getOutput(0), get_width(256, gw), 1, 1, 1, "model.20");
		auto upsample21 = network->addResize(*conv20->getOutput(0));
		assert(upsample21);
		upsample21->setResizeMode(ResizeMode::kNEAREST);
		upsample21->setOutputDimensions(c3_4->getOutput(0)->getDimensions());
		ITensor* inputTensors21[] = { upsample21->getOutput(0), c3_4->getOutput(0) };
		auto cat22 = network->addConcatenation(inputTensors21, 2);
		auto c3_23 = C3(network, weightMap, *cat22->getOutput(0), get_width(512, gw), get_width(256, gw), get_depth(3, gd), false, 1, 0.5, "model.23");

		auto conv24 = convBlock(network, weightMap, *c3_23->getOutput(0), get_width(256, gw), 3, 2, 1, "model.24");
		ITensor* inputTensors25[] = { conv24->getOutput(0), conv20->getOutput(0) };
		auto cat25 = network->addConcatenation(inputTensors25, 2);
		auto c3_26 = C3(network, weightMap, *cat25->getOutput(0), get_width(1024, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.26");

		auto conv27 = convBlock(network, weightMap, *c3_26->getOutput(0), get_width(512, gw), 3, 2, 1, "model.27");
		ITensor* inputTensors28[] = { conv27->getOutput(0), conv16->getOutput(0) };
		auto cat28 = network->addConcatenation(inputTensors28, 2);
		auto c3_29 = C3(network, weightMap, *cat28->getOutput(0), get_width(1536, gw), get_width(768, gw), get_depth(3, gd), false, 1, 0.5, "model.29");

		auto conv30 = convBlock(network, weightMap, *c3_29->getOutput(0), get_width(768, gw), 3, 2, 1, "model.30");
		ITensor* inputTensors31[] = { conv30->getOutput(0), conv12->getOutput(0) };
		auto cat31 = network->addConcatenation(inputTensors31, 2);
		auto c3_32 = C3(network, weightMap, *cat31->getOutput(0), get_width(2048, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.32");

		/* ------ detect ------ */
		IConvolutionLayer* det0 = network->addConvolutionNd(*c3_23->getOutput(0), 3 * (yolov5::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.33.m.0.weight"], weightMap["model.33.m.0.bias"]);
		IConvolutionLayer* det1 = network->addConvolutionNd(*c3_26->getOutput(0), 3 * (yolov5::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.33.m.1.weight"], weightMap["model.33.m.1.bias"]);
		IConvolutionLayer* det2 = network->addConvolutionNd(*c3_29->getOutput(0), 3 * (yolov5::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.33.m.2.weight"], weightMap["model.33.m.2.bias"]);
		IConvolutionLayer* det3 = network->addConvolutionNd(*c3_32->getOutput(0), 3 * (yolov5::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.33.m.3.weight"], weightMap["model.33.m.3.bias"]);

		auto yolo = addYoLoLayer(network, weightMap, "model.33", std::vector<IConvolutionLayer*>{det0, det1, det2, det3});
		yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
		network->markOutput(*yolo->getOutput(0));

		// Build engine
		builder->setMaxBatchSize(maxBatchSize);
		config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#if defined(USE_FP16)
		config->setFlag(BuilderFlag::kFP16);
#elif defined(USE_INT8)
		std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
		assert(builder->platformHasFastInt8());
		config->setFlag(BuilderFlag::kINT8);
		Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(1, INPUT_W, INPUT_H, "./coco_calib/", "int8calib.table", INPUT_BLOB_NAME);
		config->setInt8Calibrator(calibrator);
#endif

		std::cout << "Building engine, please wait for a while..." << std::endl;
		ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
		std::cout << "Build engine successfully!" << std::endl;

		// Don't need the network any more
		network->destroy();

		// Release host memory
		for (auto& mem : weightMap)
		{
			free((void*)(mem.second.values));
		}

		return engine;
	}

	int YOLOV5::LoadEngine() {
		// create a model using the API directly and serialize it to a stream
		char *trtModelStream{ nullptr };
		size_t size{ 0 };
		std::string engine_name = STR2(NET);
		engine_name = "../yolov5" + engine_name + ".engine";
		std::ifstream file(engine_name, std::ios::binary);
		if (file.good()) {
			file.seekg(0, file.end);
			size = file.tellg();
			file.seekg(0, file.beg);
			trtModelStream = new char[size];
			assert(trtModelStream);
			file.read(trtModelStream, size);
			file.close();
		}
		m_runtime = createInferRuntime(gLogger);
		assert(m_runtime != nullptr);
		m_engine = m_runtime->deserializeCudaEngine(trtModelStream, size);
		assert(m_engine != nullptr);
		this->m_context = m_engine->createExecutionContext();
		assert(this->m_context != nullptr);
		delete[] trtModelStream;
		assert(m_engine->getNbBindings() == 2);
		// In order to bind the buffers, we need to know the names of the input and output tensors.
		// Note that indices are guaranteed to be less than IEngine::getNbBindings()
		inputIndex = m_engine->getBindingIndex(INPUT_BLOB_NAME);
		outputIndex = m_engine->getBindingIndex(OUTPUT_BLOB_NAME);
		assert(inputIndex == 0);
		assert(outputIndex == 1);
		// Create GPU buffers on device
		buffers[2];
		CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
		CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
		// Create stream
		CHECK(cudaStreamCreate(&m_stream));
	}
	void YOLOV5::APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream, bool& is_p6, float& gd, float& gw, std::string& wts_name) {
		// Create builder
		IBuilder* builder = createInferBuilder(gLogger);
		IBuilderConfig* config = builder->createBuilderConfig();

		// Create model to populate the network, then set the outputs and create an engine
		ICudaEngine *engine = nullptr;
		if (is_p6) {
			engine = build_engine_p6(maxBatchSize, builder, config, DataType::kFLOAT, gd, gw, wts_name);
		}
		else {
			engine = build_engine(maxBatchSize, builder, config, DataType::kFLOAT, gd, gw, wts_name);
		}
		assert(engine != nullptr);

		// Serialize the engine
		(*modelStream) = engine->serialize();

		// Close everything down
		engine->destroy();
		builder->destroy();
		config->destroy();
	}
    void YOLOV5::doInference() {
        // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
        CHECK(cudaMemcpyAsync(buffers[0], this->input, BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, m_stream));
        this->m_context->enqueue(BATCH_SIZE, this->buffers, m_stream, nullptr);
        CHECK(cudaMemcpyAsync(this->output, this->buffers[1], BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, m_stream));
        cudaStreamSynchronize(m_stream);
    }
	int YOLOV5::serialize(std::string& wts_name, std::string& engine_name, bool& is_p6, float& gd, float& gw) {
		if (wts_name.empty()) { return -1; }
		IHostMemory* modelStream{ nullptr };
		APIToModel(BATCH_SIZE, &modelStream, is_p6, gd, gw, wts_name);
		assert(modelStream != nullptr);
		std::ofstream p(engine_name, std::ios::binary);
		if (!p) {
			std::cerr << "could not open plan output file" << std::endl;
			return -1;
		}
		p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
		modelStream->destroy();
		return 0;
	}


    int cmpr(yolov5::Detection& x1, yolov5::Detection& x2){
        return x1.bbox[0] < x2.bbox[0];
    }
	std::vector<yolov5::Detection> YOLOV5::predict(cv::Mat& img) {
		float* data = this->input;
		cv::Mat pr_img = preprocess_img(img, INPUT_W, INPUT_H); // letterbox BGR to RGB
		int i = 0;
		for (int row = 0; row < INPUT_H; ++row) {
			uchar* uc_pixel = pr_img.data + row * pr_img.step;
			for (int col = 0; col < INPUT_W; ++col) {
				data[i] = (float)uc_pixel[2] / 255.0;
				data[i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
				data[i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
				uc_pixel += 3;
				++i;
			}
		}
		// Run inference
		auto start = std::chrono::system_clock::now();
		doInference();
		auto end = std::chrono::system_clock::now();
		std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
		std::vector<std::vector<yolov5::Detection>> batch_res(1);
		for (int b = 0; b < 1; b++) {
			auto& res = batch_res[b];
			nms_all(res, &this->output[0], CONF_THRESH, NMS_THRESH);
		}
        sort(batch_res[0].begin(), batch_res[0].end(), cmpr);
		return batch_res[0];
	}

    std::vector<std::string> YOLOV5::predict_str(cv::Mat& img) {
        auto res = predict(img);
        std::vector<std::string> rec_result_list;
        std::cout << "size:" <<res.size() << std::endl;
        std::string str = "";
		std::string label_list[] = { "0","1","2", "3","4", "5","6", "7","8", "9",
							 "A", "b","C", "d","E", "F","J","h","P","t",
							 "L", "U", "u","o", "_" };
		cv::Mat img_clone;
		img_clone = img.clone();
        for (size_t j = 0; j < res.size(); j++) {
            cv::Rect r = get_rect(img_clone, res[j].bbox);
            cv::rectangle(img_clone, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
			//cv::putText(img_clone, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
			cv::putText(img_clone, std::string(label_list[(int)res[j].class_id]), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
			if (j > 0){
                if ( (res[j].bbox[0] - res[j-1].bbox[0]) > 1.5 * res[j].bbox[2]){
                    str.append(std::string(" "));
                }
            }
            str.append(std::string(label_list[(int)res[j].class_id]));
        }
		//cv::imwrite("../images/" + IMG_NAME, img_clone);
        std::cout << "str:"<<str<<std::endl;
        rec_result_list.push_back(str);

		return rec_result_list;
    }
}
