# Image Filters Implementation using CUDA
CUDA-based GPU Image Filters: Efficiently apply color-to-grayscale conversion and blur filters to images using parallel computing. Accelerate image processing with CUDA, C++, and OpenCV.

## Features

- Color-to-Grayscale Conversion: The code `color_to_gray.cu` demonstrates how to convert a color image to grayscale using CUDA parallelization.

- Image Blur Filter: The code `blur_image.cu` showcases the application of a blur filter to an image using CUDA for accelerated processing.

## Getting Started

To run the code in Colab, follow these steps:

1. Install the necessary dependencies by running the following commands in a code cell (this is not necessary):
   ``` 
   !pip install opencv-python
   !pip install pycuda
   ```

3. Upload the input image file (`input.jpg`) to the Colab runtime environment.

4. Open the desired code file (`color_to_gray.cu` or `blur_image.cu`) in Colab.

5. Compile and run the code by executing the following commands in separate code cells:
- For color-to-grayscale conversion:
  ```
  !nvcc -o ejecutable color_to_gray.cu `pkg-config --cflags --libs opencv4`
  !./ejecutable
  ```

- For image blur:
  ```
  !nvcc -o ejecutable2 blur_image.cu `pkg-config --cflags --libs opencv4`
  !./ejecutable2
  ```

5. The output image (`output.jpg`) will be generated and can be downloaded from Colab.

## Dependencies

- Colab environment (GPU runtime)
  
## License

This project is licensed under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## References

If you are interested in learning more about GPU programming and CUDA, you can refer to the following book:

- Kirk, D.B., Wen-mei W. Hwu. "Programming Massively Parallel Processors: A Hands-on Approach, Third Edition". Morgan Kaufmann, 2016.

The code provided in this repository was inspired by examples from this book.

