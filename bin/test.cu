#include <cstdint>
#include <cstdio>

constexpr uint64_t HASH_EMPTY = 0xffffffffffffffff;

/*
 A lock that ensures that a section is only executed once.
 E.g. assigning the key to a ht entry
 */
struct OnceLock_SM {

    static const unsigned LOCK_FRESH   = 0;
    static const unsigned LOCK_WORKING = 1;
    static const unsigned LOCK_DONE    = 2;

    volatile unsigned lock;

    __device__ void init() {
        lock = LOCK_FRESH;
    }

    __device__ bool enter() {
        unsigned lockState = atomicCAS ( (unsigned*) &lock, LOCK_FRESH, LOCK_WORKING );
        return lockState == LOCK_FRESH;
    }

    __device__ void done() {
        __threadfence_block();
        lock = LOCK_DONE;
        __threadfence_block();
    }

    __device__ void wait() {
        while ( lock != LOCK_DONE );
    }
};

template <typename T>
struct agg_ht_sm {
    OnceLock_SM lock;
    uint64_t hash;
    T payload;
};


template <typename T>
__device__ int hashAggregateGetBucket ( agg_ht_sm<T>* ht, int32_t ht_size, uint64_t grouphash, int& numLookups, T* payl ) {
    constexpr static int N_PROBE_LIMIT = 50;
    int location=-1;
    bool done=false;
    while ( !done ) {
        if ( numLookups >= N_PROBE_LIMIT ) {
            return -1;  /// flag for hash table being full
        }

        location = ( grouphash + numLookups ) % ht_size;
        agg_ht_sm<T>& entry = ht [ location ];
        numLookups++;
        if ( entry.lock.enter() ) {
            entry.payload = *payl;
            entry.hash = grouphash;
            entry.lock.done();
        }
        entry.lock.wait();
        done = (entry.hash == grouphash);
    }
    return location;
}







constexpr int HT_SIZE = 1024;

__global__ void kernel() {
    __shared__ agg_ht_sm<int> aht2[HT_SIZE];
    volatile __shared__ int HT_FULL_FLAG; HT_FULL_FLAG = 0;
    {
        int ht_index;
        unsigned loopVar = threadIdx.x;
        unsigned step = blockDim.x;
        while(loopVar < HT_SIZE) {
            ht_index = loopVar;
            aht2[ht_index].lock.init();
            aht2[ht_index].hash = HASH_EMPTY;
            loopVar += step;
        }
    }

    __syncthreads();

    int key = threadIdx.x;
    while (key < 512) {
        int hash_value = key % 53;

        int bucket = 0;
        int bucketFound = 0;
        int numLookups = 0;
        while(!(bucketFound)) {
            bucket = hashAggregateGetBucket ( aht2, HT_SIZE, hash_value, numLookups, &(key));
            if (bucket != -1) {
                int probepayl = aht2[bucket].payload;
                bucketFound = 1;
                bucketFound &= ((key == probepayl));
            } else {
                atomicAdd((int *)&HT_FULL_FLAG, 1); /// Flag not 0 anymore.
                break;
            }
        }

printf("%d : %d \n", threadIdx.x, HT_FULL_FLAG);
        if (HT_FULL_FLAG != 0) {
            __syncthreads();  ////
            if (threadIdx.x == 0) printf("================\n");
            {
                int ht_index;
                unsigned loopVar = threadIdx.x;
                unsigned step = blockDim.x;
                while(loopVar < HT_SIZE) {
                    ht_index = loopVar;
                    aht2[ht_index].lock.init();
                    aht2[ht_index].hash = HASH_EMPTY;
                    loopVar += step;
                }
            }
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