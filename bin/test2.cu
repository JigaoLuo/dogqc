#include <cstdint>
#include <cstdio>

constexpr int HT_SIZE = 1024;

__global__ void kernel() {
    __shared__ int aht2[HT_SIZE];
    volatile __shared__ int HT_FULL_FLAG; HT_FULL_FLAG = 0;
    __syncthreads();

    int key = threadIdx.x;
    while (key < 512) {
        int hash_value = key % 53;
        if (aht2[hash_value] == 0) atomicAdd((int *)&HT_FULL_FLAG, 1);
int pred1; __match_all_sync(0xffffffff, HT_FULL_FLAG, &pred1); //TODO: SO with a easy example
if (pred1 == 0) printf("%d\n", pred1);
//printf("%d : %d \n", threadIdx.x, HT_FULL_FLAG);
        if (HT_FULL_FLAG != 0) {
            __syncthreads();  ////
            if (threadIdx.x == 0) printf("================\n");
            atomicExch((int*)&HT_FULL_FLAG, 0);
            __syncthreads();  ////
        }
        key += (blockDim.x);
    }
}

int main() {
    kernel<<<1, 64>>>();
    cudaDeviceSynchronize();
    return 0;
}