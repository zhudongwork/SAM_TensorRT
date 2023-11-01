#include "samEncoder.h"
#include <iostream>

SamEncoder::SamEncoder(const std::string& engine_file)
    : m_engine_file(engine_file)
{
    cudaSetDevice(0);
    cudaStreamCreate(&m_stream);
    LoadModel();
}

SamEncoder::~SamEncoder()
{
    cudaStreamSynchronize(m_stream);
    cudaStreamDestroy(m_stream);
    if (d_input_ptr != nullptr)
        cudaFree(d_input_ptr);
    if (d_output_ptr != nullptr)
        cudaFree(d_output_ptr);
}


bool SamEncoder::LoadModel()
{
    if (IsExists(m_engine_file))
        return LoadTRTModel();
    else
        return false;
}

bool SamEncoder::LoadTRTModel()
{
    std::ifstream fgie(m_engine_file, std::ios_base::in | std::ios_base::binary);
    if (!fgie)
        return false;

    std::stringstream buffer;
    buffer << fgie.rdbuf();

    std::string stream_model(buffer.str());

    deserializeCudaEngine(stream_model.data(), stream_model.size());

    return true;
}

bool SamEncoder::deserializeCudaEngine(const void* blob, std::size_t size)
{
    m_runtime = nvinfer1::createInferRuntime(m_loger);
    assert(m_runtime != nullptr);

    bool didInitPlugins = initLibNvInferPlugins(nullptr, "");
    m_engine = m_runtime->deserializeCudaEngine(blob, size, nullptr);
    assert(m_engine != nullptr);

    m_context = m_engine->createExecutionContext();
    assert(m_context != nullptr);

    mallocInputOutput();

    return true;
}

bool SamEncoder::mallocInputOutput()
{
    m_buffers.clear();

    int nb_bind = m_engine->getNbBindings();

    nvinfer1::Dims input_dim = m_engine->getBindingDimensions(0); // -1*3*1024*1024
    nvinfer1::Dims output_dim = m_engine->getBindingDimensions(1); // -1*256x64x64


    // Create GPU buffers on device
    cudaMalloc((void**)&d_input_ptr, m_max_batchsize * // 1
        input_dim.d[1] *
        input_dim.d[2] *
        input_dim.d[3] * sizeof(float));

    cudaMalloc((void**)&d_output_ptr, m_max_batchsize *
        output_dim.d[1] * 
        output_dim.d[2] * 
        output_dim.d[3] * sizeof(float));

    m_buffers.emplace_back(d_input_ptr);
    m_buffers.emplace_back(d_output_ptr);

    return true;
}



const cv::Mat SamEncoder::transform(const cv::Mat &imageBGR)
{
    cv::Mat img;
    cv::resize(imageBGR, img,
                cv::Size(1024, 1024));

    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    img.convertTo(img, CV_32FC3);

    const cv::Scalar m_mean = cv::Scalar(123.675, 116.28, 103.53);
	const cv::Scalar m_std = cv::Scalar(58.395, 57.12, 57.375);
    const cv::Mat mean = cv::Mat(img.size(), CV_32FC3, m_mean);
    img = img - mean;
    const cv::Mat std_mat = cv::Mat(img.size(), CV_32FC3, m_std);
    img = img / std_mat;
  
    return std::move(img);
}

void SamEncoder::preprocess(const cv::Mat &image)
{
    cv::Mat img = image.clone();
    img = transform(img);

    std::vector<cv::Mat> channels;
	cv::split(img, channels);

	int offset = 0;
	for(const auto &channel : channels)
	{   
		cudaMemcpy(d_input_ptr + offset, channel.data, 
					channel.total() * sizeof(float), cudaMemcpyHostToDevice);
		offset += channel.total();
	}

}

const std::vector<float> SamEncoder::getFeature(const cv::Mat &img)
{

    const cv::Mat mattmp = img.clone();
    
    preprocess(mattmp);

    Forward();
  
    auto res = postProcess();

    return std::move(res);
}


void SamEncoder::Forward()
{
    assert(m_engine != nullptr);

    m_context->enqueueV2(m_buffers.data(), m_stream, nullptr);

    cudaStreamSynchronize(m_stream);
}

const std::vector<float> SamEncoder::postProcess()
{
    std::vector<float> result(output_size);
    cudaMemcpyAsync(result.data(), d_output_ptr, m_batchsize *
        output_size * sizeof(float), cudaMemcpyDeviceToHost, m_stream);

    cudaStreamSynchronize(m_stream);

    return std::move(result);
}


