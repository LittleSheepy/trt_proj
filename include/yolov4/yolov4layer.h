#ifndef _YOLO_LAYER_H
#define _YOLO_LAYER_H

#include <iostream>
#include <vector>
#include "NvInfer.h"
#include "yolov4cfg.h"


namespace nvinfer1
{
    class YoloV4LayerPlugin: public IPluginV2IOExt
    {
        public:
			// IPluginV2
            explicit YoloV4LayerPlugin();
			YoloV4LayerPlugin(const void* data, size_t length);
            ~YoloV4LayerPlugin();

			const char* getPluginType() const override;
			const char* getPluginVersion() const override;
			int getNbOutputs() const override;// 该层返回输出的张量个数
			Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override; // 返回输出的张量维度
            int initialize() override;
            virtual void terminate() override {};
            virtual size_t getWorkspaceSize(int maxBatchSize) const override { return 0;}
            virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override;
            virtual size_t getSerializationSize() const override;
            virtual void serialize(void* buffer) const override;
			void destroy() override;
			IPluginV2IOExt* clone() const override;
			void setPluginNamespace(const char* pluginNamespace) override;
			const char* getPluginNamespace() const override;

			// IPluginV2Ext
			DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;
			bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override;
			bool canBroadcastInputAcrossBatch(int inputIndex) const override;
			void attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) override;
			void detachFromContext() override;

			// IPluginV2IOExt
			void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) override;
			// TensorRT调用此方法以判断pos索引的输入/输出是否支持inOut[pos].format和inOut[pos].type指定的格式/数据类型。
			bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const override;
        private:
            void forwardGpu(const float *const * inputs,float * output, cudaStream_t stream,int batchSize = 1);
            int mClassCount;
            int mKernelCount;
            std::vector<yolov4::YoloKernel> mYoloKernel;
            int mThreadCount = 256;
            void** mAnchor;
            const char* mPluginNamespace;
    };

    class YoloPluginCreator : public IPluginCreator
    {
        public:
            YoloPluginCreator();

            ~YoloPluginCreator() override = default;

            const char* getPluginName() const override;

            const char* getPluginVersion() const override;

            const PluginFieldCollection* getFieldNames() override;

            IPluginV2IOExt* createPlugin(const char* name, const PluginFieldCollection* fc) override;

            IPluginV2IOExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

			void setPluginNamespace(const char* libNamespace) override;

            const char* getPluginNamespace() const override
            {
                return mNamespace.c_str();
            }

        private:
            std::string mNamespace;
            static PluginFieldCollection mFC;
            static std::vector<PluginField> mPluginAttributes;
    };
    REGISTER_TENSORRT_PLUGIN(YoloPluginCreator);
};

#endif 
