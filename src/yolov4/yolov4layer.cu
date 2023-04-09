#include <assert.h>
#include <device_launch_parameters.h>
#include "yolov4layer.h"
#include "utils.h"
#include "yolov4cfg.h"
#include <cuda_runtime_api.h>

#include "cuda_runtime.h"
namespace yolov4{
	__device__ float Logist(float data) 
	{ 
		return 1.0f / (1.0f + expf(-data)); 
	}
	
	/* 
	input:
	output:
	noElements: 5776
	yoloWidth:  76 
	yoloHeight: 76
	anchors:
	classes:
	outputElem:
	*/
    __global__ void CalDetection(const float *input, float *output,int noElements, 
            int yoloWidth,int yoloHeight,const float anchors[CHECK_COUNT*2],int classes,int outputElem) {

        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx >= noElements) return;

        int total_grid = yoloWidth * yoloHeight;  // 5776 = 76*76
        int bnIdx = idx / total_grid;
        idx = idx - total_grid*bnIdx;
        int info_len_i = 5 + classes;		// 85 = 5 + 80
        const float* curInput = input + bnIdx * (info_len_i * total_grid * CHECK_COUNT);	// (85 * 5776 * 3)

        for (int k = 0; k < 3; ++k) {
            int class_id = 0;
            float max_cls_prob = 0.0;
            for (int i = 5; i < info_len_i; ++i) {
                float p = Logist(curInput[idx + k * info_len_i * total_grid + i * total_grid]);
                if (p > max_cls_prob) {
                    max_cls_prob = p;
                    class_id = i - 5;
                }
            }
            float box_prob = Logist(curInput[idx + k * info_len_i * total_grid + 4 * total_grid]);
            if (max_cls_prob < IGNORE_THRESH || box_prob < IGNORE_THRESH) continue;

            float *res_count = output + bnIdx*outputElem;
            int count = (int)atomicAdd(res_count, 1);
            if (count >= MAX_OUTPUT_BBOX_COUNT) return;
            char* data = (char * )res_count + sizeof(float) + count*sizeof(Detection);
            Detection* det =  (Detection*)(data);

            int row = idx / yoloWidth;
            int col = idx % yoloWidth;

            //Location
            det->bbox[0] = (col + Logist(curInput[idx + k * info_len_i * total_grid + 0 * total_grid])) * INPUT_W / yoloWidth;
            det->bbox[1] = (row + Logist(curInput[idx + k * info_len_i * total_grid + 1 * total_grid])) * INPUT_H / yoloHeight;
            det->bbox[2] = exp(curInput[idx + k * info_len_i * total_grid + 2 * total_grid]) * anchors[2*k];
            det->bbox[3] = exp(curInput[idx + k * info_len_i * total_grid + 3 * total_grid]) * anchors[2*k + 1];
            det->det_confidence = box_prob;
            det->class_id = class_id;
            det->class_confidence = max_cls_prob;
        }
    }
}
using namespace yolov4;

namespace nvinfer1
{
    YoloV4LayerPlugin::YoloV4LayerPlugin()
    {
        mClassCount = CLASS_NUM;
        mYoloKernel.clear();
        mYoloKernel.push_back(yolo1);
        mYoloKernel.push_back(yolo2);
        mYoloKernel.push_back(yolo3);

        mKernelCount = mYoloKernel.size();

        CUDA_CHECK(cudaMallocHost(&mAnchor, mKernelCount * sizeof(void*)));
        size_t AnchorLen = sizeof(float)* CHECK_COUNT*2;
        for(int ii = 0; ii < mKernelCount; ii ++)
        {
            CUDA_CHECK(cudaMalloc(&mAnchor[ii],AnchorLen));
            const auto& yolo = mYoloKernel[ii];
            CUDA_CHECK(cudaMemcpy(mAnchor[ii], yolo.anchors, AnchorLen, cudaMemcpyHostToDevice));
        }
    }
    // create the plugin at runtime from a byte stream
    YoloV4LayerPlugin::YoloV4LayerPlugin(const void* data, size_t length)
    {
        using namespace Tn;
        const char *d = reinterpret_cast<const char *>(data), *a = d;
        read(d, mClassCount);
        read(d, mThreadCount);
        read(d, mKernelCount);
        mYoloKernel.resize(mKernelCount);
        auto kernelSize = mKernelCount*sizeof(YoloKernel);
        memcpy(mYoloKernel.data(),d,kernelSize);
        d += kernelSize;

        CUDA_CHECK(cudaMallocHost(&mAnchor, mKernelCount * sizeof(void*)));
        size_t AnchorLen = sizeof(float)* CHECK_COUNT*2;
        for(int ii = 0; ii < mKernelCount; ii ++)
        {
            CUDA_CHECK(cudaMalloc(&mAnchor[ii],AnchorLen));
            const auto& yolo = mYoloKernel[ii];
            CUDA_CHECK(cudaMemcpy(mAnchor[ii], yolo.anchors, AnchorLen, cudaMemcpyHostToDevice));
        }

        assert(d == a + length);
    }
    
    YoloV4LayerPlugin::~YoloV4LayerPlugin()
    {
    }

    const char* YoloV4LayerPlugin::getPluginType() const
    {
        return "YoloV4Layer_TRT";
    }

    const char* YoloV4LayerPlugin::getPluginVersion() const
    {
        return "1";
    }
	// 该层返回输出的张量个数
    int YoloV4LayerPlugin::getNbOutputs() const
    {
        return 1;
    }
    
    Dims YoloV4LayerPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
    {
        //output the result to channel
        int totalsize = MAX_OUTPUT_BBOX_COUNT * sizeof(Detection) / sizeof(float);

        return Dims3(totalsize + 1, 1, 1);
    }

    int YoloV4LayerPlugin::initialize()
    { 
        return 0;
    }

    void YoloV4LayerPlugin::forwardGpu(const float *const * inputs, float* output, cudaStream_t stream, int batchSize) {

        int outputElem = 1 + MAX_OUTPUT_BBOX_COUNT * sizeof(Detection) / sizeof(float); // 7001 = 1+ 1000*7

        for(int idx = 0 ; idx < batchSize; ++idx) {
            CUDA_CHECK(cudaMemset(output + idx*outputElem, 0, sizeof(float)));
        }
        int numElem = 0;
        for (unsigned int i = 0;i< mYoloKernel.size();++i)
        {
            const auto& yolo = mYoloKernel[i];
            numElem = yolo.width*yolo.height*batchSize; // 5776=76*76  1444=38*38
            if (numElem < mThreadCount)
                mThreadCount = numElem;
            CalDetection <<< (yolo.width*yolo.height*batchSize + mThreadCount - 1) / mThreadCount, mThreadCount>>>
                (inputs[i],output, numElem, yolo.width, yolo.height, (float *)mAnchor[i], mClassCount ,outputElem);
        }

    }

    int YoloV4LayerPlugin::enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream)
    {
        //assert(batchSize == 1);
        //GPU
        //CUDA_CHECK(cudaStreamSynchronize(stream));
        forwardGpu((const float *const *)inputs, (float*)outputs[0], stream, batchSize);
        return 0;
    }
	
    size_t YoloV4LayerPlugin::getSerializationSize() const
    {  
        return sizeof(mClassCount) + sizeof(mThreadCount) + sizeof(mKernelCount)  + sizeof(yolov4::YoloKernel) * mYoloKernel.size();
    }
	
    void YoloV4LayerPlugin::serialize(void* buffer) const
    {
        using namespace Tn;
        char* d = static_cast<char*>(buffer), *a = d;
        write(d, mClassCount);
        write(d, mThreadCount);
        write(d, mKernelCount);
        auto kernelSize = mKernelCount*sizeof(YoloKernel);
        memcpy(d,mYoloKernel.data(),kernelSize);
        d += kernelSize;

        assert(d == a + getSerializationSize());
    }
	
    void YoloV4LayerPlugin::destroy()
    {
        delete this;
    }

    // Clone the plugin
    IPluginV2IOExt* YoloV4LayerPlugin::clone() const
    {
        YoloV4LayerPlugin *p = new YoloV4LayerPlugin();
        p->setPluginNamespace(mPluginNamespace);
        return p;
    }

    // Set plugin namespace
    void YoloV4LayerPlugin::setPluginNamespace(const char* pluginNamespace)
    {
        mPluginNamespace = pluginNamespace;
    }

    const char* YoloV4LayerPlugin::getPluginNamespace() const
    {
        return mPluginNamespace;
    }
	// ***IPluginV2Ext
    // Return the DataType of the plugin output at the requested index
    DataType YoloV4LayerPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
    {
        return DataType::kFLOAT;
    }

    // Return true if output tensor is broadcast across a batch.
    bool YoloV4LayerPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
    {
        return false;
    }

    // Return true if plugin can use input that is broadcast across batch without replication.
    bool YoloV4LayerPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
    {
        return false;
    }

    // Attach the plugin object to an execution context and grant the plugin the access to some context resource.
    void YoloV4LayerPlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
    {
    }

    // Detach the plugin object from its execution context.
    void YoloV4LayerPlugin::detachFromContext() {}

	// IPluginV2IOExt
	void YoloV4LayerPlugin::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput)
    {
    }
	
	// TensorRT调用此方法以判断pos索引的输入/输出是否支持inOut[pos].format和inOut[pos].type指定的格式/数据类型。
	bool YoloV4LayerPlugin::supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const
	{
        return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
    }
	
	/////////////////////YoloPluginCreator////////////////////////
    PluginFieldCollection YoloPluginCreator::mFC{};
    std::vector<PluginField> YoloPluginCreator::mPluginAttributes;

    YoloPluginCreator::YoloPluginCreator()
    {
        mPluginAttributes.clear();

        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    const char* YoloPluginCreator::getPluginName() const
    {
            return "YoloV4Layer_TRT";
    }

    const char* YoloPluginCreator::getPluginVersion() const
    {
            return "1";
    }

    const PluginFieldCollection* YoloPluginCreator::getFieldNames()
    {
            return &mFC;
    }

    IPluginV2IOExt* YoloPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
    {
        YoloV4LayerPlugin* obj = new YoloV4LayerPlugin();
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

    IPluginV2IOExt* YoloPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
    {
        // This object will be deleted when the network is destroyed, which will
        // call MishPlugin::destroy()
        YoloV4LayerPlugin* obj = new YoloV4LayerPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
	void YoloPluginCreator::setPluginNamespace(const char* libNamespace)
	{
		mNamespace = libNamespace;
	}

}
