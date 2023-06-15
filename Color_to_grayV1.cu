#include <stdio.h>
#include <stdlib.h>
#include <jpeglib.h>

#define CHANNELS 3

__global__ void colorToGreyscaleConversion(unsigned char *Pout, unsigned char *Pin, int width, int height) {
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

unsigned char *read_JPEG_file(const char *filename, int *width, int *height) {
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE *infile;
    JSAMPARRAY buffer;
    int row_stride;

    if ((infile = fopen(filename, "rb")) == NULL) {
        fprintf(stderr, "Error opening jpeg file %s\n", filename);
        return NULL;
    }

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);

    jpeg_stdio_src(&cinfo, infile);
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);

    *width = cinfo.output_width;
    *height = cinfo.output_height;
    int channels = cinfo.num_components;
    int row_size = *width * channels;

    unsigned char *image = (unsigned char *)malloc(row_size * (*height));
    buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE, row_size, 1);

    int y = 0;
    while (cinfo.output_scanline < cinfo.output_height) {
        jpeg_read_scanlines(&cinfo, buffer, 1);
        for (int x = 0; x < row_size; x++) {
            image[y * row_size + x] = buffer[0][x];
        }
        y++;
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);

    return image;
}

void write_JPEG_file(const char *filename, unsigned char *image, int width, int height) {
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE *outfile;
    JSAMPROW row_pointer[1];
    int row_stride;

    if ((outfile = fopen(filename, "wb")) == NULL) {
        fprintf(stderr, "Error opening output jpeg file %s\n", filename);
        return;
    }

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, outfile);

    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = 1;
    cinfo.in_color_space = JCS_GRAYSCALE;

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 75, TRUE);

    jpeg_start_compress(&cinfo, TRUE);

    row_stride = width;

    while (cinfo.next_scanline < cinfo.image_height) {
        row_pointer[0] = &image[cinfo.next_scanline * row_stride];
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }

    jpeg_finish_compress(&cinfo);
    fclose(outfile);
    jpeg_destroy_compress(&cinfo);
}




int main() {
    const char *input_file = "input.jpg";
    const char *output_file = "output.jpg";

    int width, height;
    unsigned char *image = read_JPEG_file(input_file, &width, &height);

    if (image == NULL) {
        return 1;
    }

    // Define las dimensiones de la cuadrícula de bloques y los bloques de hilos
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);

    // Crea arreglos en el dispositivo CUDA para los datos de entrada y salida
    unsigned char *Pout_gpu, *Pin_gpu;
    cudaMalloc((void **)&Pout_gpu, width * height);
    cudaMalloc((void **)&Pin_gpu, width * height * CHANNELS);
    cudaMemcpy(Pin_gpu, image, width * height * CHANNELS, cudaMemcpyHostToDevice);

    // Ejecuta el kernel CUDA
    colorToGreyscaleConversion<<<grid_size, block_size>>>(Pout_gpu, Pin_gpu, width, height);

    // Copia el resultado del dispositivo al host
    unsigned char *output_image = (unsigned char *)malloc(width * height);
    cudaMemcpy(output_image, Pout_gpu, width * height, cudaMemcpyDeviceToHost);

    // Guarda la imagen resultante en un archivo de salida
    write_JPEG_file(output_file, output_image, width, height);

    // Libera la memoria en el dispositivo CUDA y en el host
    cudaFree(Pout_gpu);
    cudaFree(Pin_gpu);
    free(image);
    free(output_image);

    return 0;
}