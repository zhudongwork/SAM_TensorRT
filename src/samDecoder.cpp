#include "samDecoder.h"
#include <iostream>

SamDecoder::SamDecoder(const std::string& encoder_engine_file, const std::string& decoder_engine_file)
    : m_encoder_engine_file(encoder_engine_file),
    m_decoder_engine_file(decoder_engine_file),
    sam_encoder(encoder_engine_file)
{
    cudaSetDevice(0);
    cudaStreamCreate(&m_stream);
    LoadModel();
}

SamDecoder::~SamDecoder()
{
    cudaStreamSynchronize(m_stream);
    cudaStreamDestroy(m_stream);
    if (d_input_image_embeddings_ptr != nullptr)
        cudaFree(d_input_image_embeddings_ptr);
    if (d_input_point_coords_ptr != nullptr)
        cudaFree(d_input_point_coords_ptr);
    if (d_input_point_labels_ptr != nullptr)
        cudaFree(d_input_point_labels_ptr);
    if (d_input_mask_input_ptr != nullptr)
        cudaFree(d_input_mask_input_ptr);
    if (d_input_has_mask_input_ptr != nullptr)
        cudaFree(d_input_has_mask_input_ptr);
    
    
    if (d_output_low_res_masks_ptr != nullptr)
        cudaFree(d_output_low_res_masks_ptr);
    if (d_output_iou_predictions_ptr != nullptr)
        cudaFree(d_output_iou_predictions_ptr);
    if (d_output_masks_ptr != nullptr)
        cudaFree(d_output_masks_ptr);

    if (h_output_masks_ptr != nullptr)
        cudaFree(h_output_masks_ptr);
}


bool SamDecoder::LoadModel()
{
    if (IsExists(m_decoder_engine_file))
        return LoadTRTModel();
    else
        return false;
}


bool SamDecoder::LoadTRTModel()
{
    std::ifstream fgie(m_decoder_engine_file, std::ios_base::in | std::ios_base::binary);
    if (!fgie)
        return false;

    std::stringstream buffer;
    buffer << fgie.rdbuf();

    std::string stream_model(buffer.str());

    deserializeCudaEngine(stream_model.data(), stream_model.size());

    return true;
}

bool SamDecoder::deserializeCudaEngine(const void* blob, std::size_t size)
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

bool SamDecoder::mallocInputOutput()
{
    m_buffers.clear();

    // Create GPU buffers on device
    cudaMalloc((void**)&d_input_image_embeddings_ptr, m_max_batchsize * (256*64*64) * sizeof(float));
    cudaMalloc((void**)&d_input_point_coords_ptr, m_max_batchsize * (1*2) * sizeof(float));
    cudaMalloc((void**)&d_input_point_labels_ptr, m_max_batchsize * 1 * sizeof(float));
    cudaMalloc((void**)&d_input_mask_input_ptr, m_max_batchsize * (1*256*256) * sizeof(float));
    cudaMalloc((void**)&d_input_has_mask_input_ptr, m_max_batchsize * 1 * sizeof(float));
    


    cudaMalloc((void**)&d_output_low_res_masks_ptr, m_max_batchsize *(4*256*256)* sizeof(float));
    cudaMalloc((void**)&d_output_iou_predictions_ptr, m_max_batchsize *(4)* sizeof(float));
    cudaMalloc((void**)&d_output_masks_ptr, m_max_batchsize *(4*1024*1024)* sizeof(float));

    h_output_masks_ptr = (float*)malloc(m_max_batchsize * (4*1024*1024)* sizeof(float));

    m_buffers.emplace_back(d_input_image_embeddings_ptr);
    m_buffers.emplace_back(d_input_point_coords_ptr);
    m_buffers.emplace_back(d_input_point_labels_ptr);
    m_buffers.emplace_back(d_input_mask_input_ptr);
    m_buffers.emplace_back(d_input_has_mask_input_ptr);

    m_buffers.emplace_back(d_output_low_res_masks_ptr);
    m_buffers.emplace_back(d_output_iou_predictions_ptr);
    m_buffers.emplace_back(d_output_masks_ptr);

    return true;
}



void SamDecoder::preprocess(const std::vector<float> &embedding, const float x, const float y)
{

    cudaMemcpyAsync(d_input_image_embeddings_ptr, (float*)embedding.data(), 
					embedding.size() * sizeof(float), cudaMemcpyHostToDevice, m_stream);
    
    const float point_coord[] = {x, y};
    cudaMemcpyAsync(d_input_point_coords_ptr, point_coord, 
					2 * sizeof(float), cudaMemcpyHostToDevice, m_stream);

    const float point_labels[] = {1.0f};
    cudaMemcpyAsync(d_input_point_labels_ptr, point_labels, 
					1 * sizeof(float), cudaMemcpyHostToDevice, m_stream);
    
    const std::vector<float> mask(256*256, 0.0f);
    cudaMemcpyAsync(d_input_mask_input_ptr, mask.data(), 
					256*256 * sizeof(float), cudaMemcpyHostToDevice, m_stream);
    
    const float has_mask_input[] = {0.0f};
    cudaMemcpyAsync(d_input_has_mask_input_ptr, has_mask_input, 
					1 * sizeof(float), cudaMemcpyHostToDevice, m_stream);
    
    cudaStreamSynchronize(m_stream);

}

const cv::Mat SamDecoder::getMask(const cv::Mat &image, const float x, const float y)
{
 
    const std::vector<float> embedding = sam_encoder.getFeature(image);

    preprocess(embedding, x, y);

    Forward();

    auto res = postProcess();

    return std::move(res);
}


void SamDecoder::Forward()
{
    assert(m_engine != nullptr);
    // 固定动态输入
    m_context->setBindingDimensions(1, nvinfer1::Dims3(1, 1, 2)); //point_coords: 1x1x2
    m_context->setBindingDimensions(2, nvinfer1::Dims2(1, 1));    //point_labels: 1x1

    m_context->enqueueV2(m_buffers.data(), m_stream, nullptr);

    cudaStreamSynchronize(m_stream);
}

const cv::Mat SamDecoder::postProcess()
{

    cudaMemcpyAsync(h_output_masks_ptr, d_output_masks_ptr, m_batchsize *
        (4*1024*1024) * sizeof(float), cudaMemcpyDeviceToHost, m_stream);
    cudaStreamSynchronize(m_stream);

    int index = 0;
    auto outputMaskSam = cv::Mat(1024, 1024, CV_8UC1);
    for (int i = 0; i < outputMaskSam.rows; ++i) {
      for (int j = 0; j < outputMaskSam.cols; ++j) {
        auto val = h_output_masks_ptr[i * outputMaskSam.cols + j + index*1024*1024] ;
        outputMaskSam.at<uchar>(i, j) = val > 0 ? 255 : 0;
      }
    }

    return std::move(outputMaskSam);
}


