#include <iostream>
#include <opencv2/opencv.hpp>

#define BLUR_SIZE 5
#define BLOCK_SIZE 16

__global__ void blurKernel(unsigned char* in, unsigned char* out, int w, int h) {
    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    int Row = threadIdx.y + blockIdx.y * blockDim.y;

    if (Col < w && Row < h) {
        int pixVal = 0;
        int pixels = 0;

        for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow) {
            for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol) {
                int curRow = Row + blurRow;
                int curCol = Col + blurCol;

                if (curRow > -1 && curRow < h && curCol > -1 && curCol < w) {
                    pixVal += in[curRow * w + curCol];
                    pixels++;
                }
            }
        }

        out[Row * w + Col] = static_cast<unsigned char>(pixVal / pixels);
    }
}

int main() {
    cv::Mat image = cv::imread("input.jpg", cv::IMREAD_GRAYSCALE);
    int width = image.cols;
    int height = image.rows;

    unsigned char* h_in = image.data;
    unsigned char* h_out = new unsigned char[width * height];
    unsigned char* d_in;
    unsigned char* d_out;

    cudaMalloc((void**)&d_in, width * height * sizeof(unsigned char));
    cudaMalloc((void**)&d_out, width * height * sizeof(unsigned char));

    cudaMemcpy(d_in, h_in, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    blurKernel<<<gridDim, blockDim>>>(d_in, d_out, width, height);

    cudaMemcpy(h_out, d_out, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cv::Mat output(height, width, CV_8UC1, h_out);
    cv::imwrite("output.jpg", output);

    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_out;

    return 0;
}