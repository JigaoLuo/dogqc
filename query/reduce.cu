#include <stdio.h>
#include <string>
#include <cassert>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void warpReduce() {
    int laneId = threadIdx.x & 0x1f;
    // Seed starting value as inverse lane ID
    int value = (31 - laneId) % 3;

    unsigned int group_mask = __match_any_sync(0xffffffff, value);
    int elected_lane = __ffs(group_mask) - 1;
    int value_agg = 0;
    unsigned laneid; asm volatile ("mov.u32 %0, %laneid;" : "=r"(laneid));
//    printf("Thread %d: ", threadIdx.x);
    for (int offset=16; offset>=1; offset/=2) {
        int sync_value = __shfl_down_sync(0xffffffff, value, offset);
        unsigned source_lane = (laneid + offset) % 32;
        unsigned int source_lane_group_mask = __shfl_sync(0xffffffff, group_mask, source_lane);
        if ( source_lane_group_mask == group_mask ) {
            value_agg += sync_value + value;
            if (threadIdx.x == 0) printf(" + %d + %d + \n", value, sync_value);
        } else {
            value_agg = value;
        }
    }

    // "value" now contains the sum across all threads
    printf("Thread %d final value = %d, value_agg = %d\n", threadIdx.x, value, value_agg);
}

__global__ void warpReduceCG() {
    int laneId = threadIdx.x & 0x1f;
    // Seed starting value as inverse lane ID
    int value = (31 - laneId) % 3;
    int value_agg = value;

    cg::thread_block cta = cg::this_thread_block();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
    unsigned int group_mask = __match_any_sync(0xffffffff, value);
    cg::coalesced_group g = cg::coalesced_threads();
    unsigned int group_mask_cg = g.match_any(value);
    assert (group_mask == group_mask_cg);
//    cg::coalesced_group subtile = cg::labeled_partition(tile32, group_mask);
    cg::coalesced_group subtile = cg::labeled_partition(g, group_mask);

    for (int i = subtile.size() / 2; i > 0; i /= 2) {
        int value_shfl = subtile.shfl_down(value, i);
        if (threadIdx.x == 0) printf(" + %d + %d + \n", value, value_shfl);
        value_agg += value_shfl;
    }


    // "value" now contains the sum across all threads
    printf("Thread %d final value = %d, value_agg = %d\n", threadIdx.x, value, value_agg);
}

int main() {
    warpReduce<<< 1, 32 >>>();
    cudaDeviceSynchronize();

    warpReduceCG<<< 1, 32 >>>();
    cudaDeviceSynchronize();


    return 0;
}