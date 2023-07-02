
// 方阵乘法

#include <stdio.h>

void maxtrix_mutil_cpu(const int*x, const int*y, int*z, const int N){

    for(int r = 0;r<N;r++)
    {
        for(int c = 0;c<N;c++){
            int temp=0;
            for(int step = 0;step<N; step++){
                temp += x[r*N+step] * y[step*N+c];
            }
            z[r*N+c] = temp;

        }
    }
}

__global__ void maxtrix_mutil_gpu(int*x, int*y, int*z, const int N)
{
    // x的索引
    int r = blockIdx.y * blockDim.y + threadIdx.y ;
    int c = blockIdx.x * blockDim.x + threadIdx.x ; 
    if (r<N && c<N){
        int temp=0;
        for(int step=0; step<N; step++){
            temp += x[r*N+step] * y[step*N+c];
        }
        z[r*N+c] = temp;
    }
}

int main(int argc,char** argv){

    // 随机生产矩阵
    constexpr int matrix_size = 32;
    constexpr int memsize = sizeof(int)*matrix_size*matrix_size;

    // cuda分配主机的锁页内存。
    int * hx =0;
    int * hy = 0;
    int * hz = 0;
    int * hzz = 0;
    cudaMallocHost((void**)&hx, memsize);
    cudaMallocHost((void**)&hy, memsize);
    cudaMallocHost((void**)&hz, memsize);
    cudaMallocHost((void**)&hzz, memsize);

    // 内存设置值
    for(int r = 0; r < matrix_size; r++){
        for(int c = 0; c < matrix_size; c++){
            hx[r*matrix_size +c] = 1;//int(rand() *1024);
        }
    }
    for(int r = 0; r < matrix_size; r++){
        for(int c = 0; c < matrix_size; c++){
            hy[r*matrix_size +c] =2;//int(rand() *1024);
        }
    }

    maxtrix_mutil_cpu(hx,hy,hz, matrix_size);

    // alloc gpu memory
    int* dx=0;
    int *dy=0;
    int *dz=0;
    cudaMalloc((void**)&dx, memsize);
    cudaMalloc((void**)&dy, memsize);
    cudaMalloc((void**)&dz, memsize);

    // memcopy to gpu
    cudaMemcpy(dx,hx,memsize,cudaMemcpyHostToDevice);
    cudaMemcpy(dy,hy,memsize,cudaMemcpyHostToDevice);

    // kernel function
    constexpr unsigned int blocks = 16;
    unsigned int grids = (matrix_size + blocks -1) / blocks;
    dim3 blocksize(blocks,blocks);
    dim3 gridsize(grids,grids);
    maxtrix_mutil_gpu<<<gridsize,blocksize>>>(dx,dy,dz,matrix_size);
    cudaDeviceSynchronize();
    cudaMemcpy(hzz, dz, memsize,cudaMemcpyDeviceToHost);

    // compare result between gpu and cpu
    bool error = false;
    for(int r = 0; r < matrix_size; r++){
        for(int c = 0; c < matrix_size; c++){
            if(fabs(hz[r*matrix_size +c] - hzz[r*matrix_size +c]) > (1.0e-10)){
                if(r<10 && c<10)printf("%d,%d,%d,%d\n",r,c,hz[r*matrix_size +c],hzz[r*matrix_size +c]);
                error=true;
            }
        }
    }

    printf("result: %s\n",error?"WRONG":"RIGHT");
    cudaFreeHost(hx);
    cudaFreeHost(hy);
    cudaFreeHost(hz);
    cudaFreeHost(hzz);

    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dz);

    return  0;
}