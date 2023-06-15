#include <iostream>
#include <opencv2/opencv.hpp>

#define CHANNELS 3
#define BLOCK_SIZE 16

__global__ void colorToGreyscaleConversion(unsigned char* Pout, unsigned char* Pin, int width, int height) {
    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    int Row = threadIdx.y + blockIdx.y * blockDim.y;

    if (Col < width && Row < height) {
        int greyOffset = Row * width + Col;
        int rgbOffset = greyOffset * CHANNELS;

        unsigned char r = Pin[rgbOffset];
        unsigned char g = Pin[rgbOffset + 1];
        unsigned char b = Pin[rgbOffset + 2];

        Pout[greyOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
    }
}

int main() {
    cv::Mat image = cv::imread("input.jpg", cv::IMREAD_COLOR);
    int width = image.cols;
    int height = image.rows;

    unsigned char* h_in = image.data;
    unsigned char* h_out = new unsigned char[width * height];
    unsigned char* d_in;
    unsigned char* d_out;

    cudaMalloc((void**)&d_in, width * height * CHANNELS * sizeof(unsigned char));
    cudaMalloc((void**)&d_out, width * height * sizeof(unsigned char));

    cudaMemcpy(d_in, h_in, width * height * CHANNELS * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    colorToGreyscaleConversion<<<gridDim, blockDim>>>(d_out, d_in, width, height);

    cudaMemcpy(h_out, d_out, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cv::Mat output(height, width, CV_8UC1, h_out);
    cv::imwrite("output.jpg", output);

    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_out;

    return 0;
}