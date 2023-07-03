
#include "stdio.h"
#include"iostream"
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

// sobel 边缘检测算法
// kernel: x方向， y方向梯度计算，在将梯度绝对值求平均
// k00 k01 k02          m00 m01 m02 m03 m04             r00 r01 r02 r03 r04
// k10 k11 k12   @      m10 m11 m12 m13 m14             r10 r11 r12 r13 r14
// k20 k21 k22          m20 m21 m22 m23 m24    =》      r20 r21 r22 r23 r24
//                      m30 m31 m32 m33 m34             r30 r31 r32 r33 r34
//                      m40 m41 m42 m43 m44             r40 r41 r42 r43 r44
// Gx   1 0 -1  Gy    1  2  1
//      2 0 -2        0  0  0
//      1 0 -1        -1 -2 -1
// Gx_r11 = (m00 + 2* m10 + m20) - (m02+2*m12 + m22)
// Gy_r11 = (m00 + 2* m01 + m02) - (m20 + 2* m21 + m22)
// ret = (abs(Gx) + abs(Gy)) / 2

__global__ void soble_gpu(unsigned char *in ,unsigned char* out, int height, int width){
    // 1. xy方向的索引
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    // 梯度计算用到的值索引，中间为0，所以没有使用x4
    // 核函数的变量，存储在SM有限的register内，能省则省
    unsigned char x0, x1, x2, x3, x5, x6, x7, x8;
    signed int gx, gy;
    // 结果的外层忽略不计算
    if(x>0 && x<width-1 && y>0 && y<height-1){
        // 由结果的索引，反推回核函数位置的idx, 并取计算用到的值
        x0 = in[(y-1)*width + x-1];
        x1 = in[(y-1)*width + x];
        x2 = in[(y-1)*width + x+1];
        x3 = in[y*width + x-1];
        // x4
        x5 = in[y*width + x+1];
        x6 = in[(y+1)*width + x-1];
        x7 = in[(y+1)*width + x];
        x8 = in[(y+1)*width + x+1];

        gx = (x0 + 2*x3 +x6) - (x2 + 2*x5 + x8);
        gy = (x0 + 2*x1 +x2) - (x6 + 2*x7 + x8);

        out[y*width+x] = (abs(gx) + abs(gy)) / 2;
    }

}

int main(int argc, char** argv){
    cout << "hello" <<endl;
    Mat img = imread("Lenna.jpg",0); // 灰度图

    unsigned int height = img.rows;
    unsigned int width = img.cols;
    unsigned int memsize = height * width;

    Mat gaussianimg;
    GaussianBlur(img,gaussianimg,Size(3,3),1,1);

    // 定义存放在cpu的内存
    Mat soble = Mat::zeros(Size(width,height),CV_8UC1);

    // 定义GPU内存
    unsigned char * in =0;
    unsigned char * out=0;
    cudaMalloc((void**)&in, memsize);
    cudaMalloc((void**)&out, memsize);

    // copy to device
    cudaMemcpy(in, gaussianimg.data, memsize, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32,32);
    dim3 blocksPerGrid(int((height+threadsPerBlock.y-1)/threadsPerBlock.y), int((width + threadsPerBlock.x -1)/ threadsPerBlock.x));
    soble_gpu<<<blocksPerGrid, threadsPerBlock>>>(in, out, height, width);
    cudaDeviceSynchronize();

    cudaMemcpy(soble.data, out, memsize, cudaMemcpyDeviceToHost);
    
    
    
    imwrite("sobel_save.jpg", soble);

    cudaFree(in);
    cudaFree(out);

    return 0;
}
