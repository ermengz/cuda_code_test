#include "stdio.h"

void print_from_cpu()
{

    printf("hello world wrom cpu\n");
}

// thread -> block -> grid
// SM(stream mutil-processor) 流多处理器
__global__ void print_from_gpu_dim1()
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    printf("hello from gpu (%d,%d)\n", bid,tid);
}

/*
    二维Grid 二维Block
    // 一个grid, block为2维，thread为2维
    关键点1： 一个grid中，有若干个blocks， blocks的数量由变量GridDim.x\y\z表示，block在Grid中的坐标由blockIdx.x\y\z表示
    关键点2： 一个block中，有若干个thread，thread的数量表示为blockDim.x\y\z表示，thread在block中的坐标由threadIdx.x\y\z表示

    例如： （x,y,z）在坐标的大小为(Dx,Dy,Dz)， Z的维度最高，Y次之，Z维度最低。
        则，高维度转化为低纬度的坐标变换方式为：id = z* Dx*Dy + y* Dx, + x
*/
__global__ void print_from_gpu_dim2()
{
    // grid中，block的坐标表示为
    int bidx =  blockIdx.x;
    int bidy =  blockIdx.y;

    // block中，thread的坐标表示为：
    int tidx =threadIdx.x;
    int tidy =threadIdx.y;

    
    // 二维Grid 二维Block
    // 1. block 的索引
    int blockid = blockIdx.x + blockIdx.y*gridDim.x;
    // 2. 线程的全局id
    int gtid = (blockid * blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    printf("hello from gpu block_id=(%d,%d), thread_id=(%d,%d), gtid=(%d)\n", bidx,bidy,tidx,tidy,gtid);
}

/*
    三维grid、三维block、三维thread
    同样的方式：1. 先索引grid索引、block索引、线程索引

    思想：将一个三维立方体，展平的思维， 降维展平
*/ 
__global__ void print_from_gpu_dim3()
{
    // 
    // int gid = gridDim

     // grid中，block的坐标表示为
    int bidx =  blockIdx.x;
    int bidy =  blockIdx.y;
    int bidz =  blockIdx.z;

    // block中，thread的坐标表示为：
    int tidx =threadIdx.x;
    int tidy =threadIdx.y;
    int tidz =threadIdx.z;

    // x,y,z 3维索引转成1维索引 block索引
    // blockIdx.x 1维偏置 + [(blockIdx.y * gridDim.x) 2维转1维] + [blockIdx.z * (gridDim.x*gridDim.y) 3维转2维];
    int block_id = blockIdx.x + (blockIdx.y * gridDim.x) + blockIdx.z * (gridDim.x*gridDim.y);
    // thread索引
    int gtid = block_id * blockDim.x * blockDim.y * blockDim.z // 索引个数
            + threadIdx.z * blockDim.x * blockDim.y // 3维转2维
            + threadIdx.y * blockDim.x              // 2维转1维
            + threadIdx.x ;                         // 1维偏置

    printf("%d\n",gtid);
    // printf(" gtid=(%d)\n",gtid);
    // printf("hello world from gpu, grid=(%d,%d), block=(%d,%d), thread=(%d,%d), gtid=(%d)\n");
}

int main(int argc,char** argv)
{
    // print_from_cpu();

    // // 一维 
    // printf("grid dim=1, block dim=1 \n");
    // constexpr int block_size = 4;
    // constexpr int grid_size = 2;
    // print_from_gpu_dim1<<<grid_size, block_size>>>();

    if(0)
    {
        // 二维
        printf("block dim=2\n");
        dim3 grid_size_d2(2,2);
        // g00 g01
        // g10 g11
        dim3 block_size_d2(3,3);
        // t00 t01 t02
        // t10 t11 t12
        // t20 t21 t22
        print_from_gpu_dim2<<<grid_size_d2, block_size_d2>>>();
    }
    

    // 三维
    {
        printf("3 dim\n");
        dim3 gridsize(2,2,2);
        dim3 blocksize(3,3,3);
        print_from_gpu_dim3<<<gridsize,blocksize>>>();
    }



    cudaDeviceSynchronize();

    return 0;
}