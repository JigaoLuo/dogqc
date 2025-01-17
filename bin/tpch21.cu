#include <list>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <ctime>
#include <limits.h>
#include <float.h>
#include "../dogqc/include/csv.h"
#include "../dogqc/include/util.h"
#include "../dogqc/include/mappedmalloc.h"
#include "../dogqc/include/util.cuh"
#include "../dogqc/include/hashing.cuh"
struct jpayl15 {
    int att2_lorderke;
    int att4_lsuppkey;
};
struct jpayl14 {
    int att18_lorderke;
    int att20_lsuppkey;
};
struct jpayl7 {
    int att34_nnationk;
};
struct jpayl10 {
    int att38_ssuppkey;
    str_t att39_sname;
};
struct jpayl13 {
    str_t att39_sname;
    int att45_lorderke;
    int att47_lsuppkey;
};
struct apayl16 {
    str_t att39_sname;
};

__global__ void krnl_lineitem1(
    int* iatt2_lorderke, int* iatt4_lsuppkey, multi_ht* jht15, jpayl15* jht15_payload) {
    int att2_lorderke;
    int att4_lsuppkey;

    int tid_lineitem1 = 0;
    unsigned loopVar = ((blockIdx.x * blockDim.x) + threadIdx.x);
    unsigned step = (blockDim.x * gridDim.x);
    unsigned flushPipeline = 0;
    int active = 0;
    while(!(flushPipeline)) {
        tid_lineitem1 = loopVar;
        active = (loopVar < 6001215);
        // flush pipeline if no new elements
        flushPipeline = !(__ballot_sync(ALL_LANES,active));
        if(active) {
            att2_lorderke = iatt2_lorderke[tid_lineitem1];
            att4_lsuppkey = iatt4_lsuppkey[tid_lineitem1];
        }
        // -------- hash join build (opId: 15) --------
        if(active) {
            uint64_t hash15 = 0;
            if(active) {
                hash15 = 0;
                if(active) {
                    hash15 = hash ( (hash15 + ((uint64_t)att2_lorderke)));
                }
            }
            hashCountMulti ( jht15, 3000607, hash15);
        }
        loopVar += step;
    }

}

__global__ void krnl_lineitem1_ins(
    int* iatt2_lorderke, int* iatt4_lsuppkey, multi_ht* jht15, jpayl15* jht15_payload, int* offs15) {
    int att2_lorderke;
    int att4_lsuppkey;

    int tid_lineitem1 = 0;
    unsigned loopVar = ((blockIdx.x * blockDim.x) + threadIdx.x);
    unsigned step = (blockDim.x * gridDim.x);
    unsigned flushPipeline = 0;
    int active = 0;
    while(!(flushPipeline)) {
        tid_lineitem1 = loopVar;
        active = (loopVar < 6001215);
        // flush pipeline if no new elements
        flushPipeline = !(__ballot_sync(ALL_LANES,active));
        if(active) {
            att2_lorderke = iatt2_lorderke[tid_lineitem1];
            att4_lsuppkey = iatt4_lsuppkey[tid_lineitem1];
        }
        // -------- hash join build (opId: 15) --------
        if(active) {
            uint64_t hash15 = 0;
            if(active) {
                hash15 = 0;
                if(active) {
                    hash15 = hash ( (hash15 + ((uint64_t)att2_lorderke)));
                }
            }
            jpayl15 payl;
            payl.att2_lorderke = att2_lorderke;
            payl.att4_lsuppkey = att4_lsuppkey;
            hashInsertMulti ( jht15, jht15_payload, offs15, 3000607, hash15, &(payl));
        }
        loopVar += step;
    }

}

__global__ void krnl_lineitem22(
    int* iatt18_lorderke, int* iatt20_lsuppkey, unsigned* iatt29_lcommitd, unsigned* iatt30_lreceipt, multi_ht* jht14, jpayl14* jht14_payload) {
    int att18_lorderke;
    int att20_lsuppkey;
    unsigned att29_lcommitd;
    unsigned att30_lreceipt;

    int tid_lineitem2 = 0;
    unsigned loopVar = ((blockIdx.x * blockDim.x) + threadIdx.x);
    unsigned step = (blockDim.x * gridDim.x);
    unsigned flushPipeline = 0;
    int active = 0;
    while(!(flushPipeline)) {
        tid_lineitem2 = loopVar;
        active = (loopVar < 6001215);
        // flush pipeline if no new elements
        flushPipeline = !(__ballot_sync(ALL_LANES,active));
        if(active) {
            att18_lorderke = iatt18_lorderke[tid_lineitem2];
            att20_lsuppkey = iatt20_lsuppkey[tid_lineitem2];
            att29_lcommitd = iatt29_lcommitd[tid_lineitem2];
            att30_lreceipt = iatt30_lreceipt[tid_lineitem2];
        }
        // -------- selection (opId: 3) --------
        if(active) {
            active = (att30_lreceipt > att29_lcommitd);
        }
        // -------- hash join build (opId: 14) --------
        if(active) {
            uint64_t hash14 = 0;
            if(active) {
                hash14 = 0;
                if(active) {
                    hash14 = hash ( (hash14 + ((uint64_t)att18_lorderke)));
                }
            }
            hashCountMulti ( jht14, 3000607, hash14);
        }
        loopVar += step;
    }

}

__global__ void krnl_lineitem22_ins(
    int* iatt18_lorderke, int* iatt20_lsuppkey, unsigned* iatt29_lcommitd, unsigned* iatt30_lreceipt, multi_ht* jht14, jpayl14* jht14_payload, int* offs14) {
    int att18_lorderke;
    int att20_lsuppkey;
    unsigned att29_lcommitd;
    unsigned att30_lreceipt;

    int tid_lineitem2 = 0;
    unsigned loopVar = ((blockIdx.x * blockDim.x) + threadIdx.x);
    unsigned step = (blockDim.x * gridDim.x);
    unsigned flushPipeline = 0;
    int active = 0;
    while(!(flushPipeline)) {
        tid_lineitem2 = loopVar;
        active = (loopVar < 6001215);
        // flush pipeline if no new elements
        flushPipeline = !(__ballot_sync(ALL_LANES,active));
        if(active) {
            att18_lorderke = iatt18_lorderke[tid_lineitem2];
            att20_lsuppkey = iatt20_lsuppkey[tid_lineitem2];
            att29_lcommitd = iatt29_lcommitd[tid_lineitem2];
            att30_lreceipt = iatt30_lreceipt[tid_lineitem2];
        }
        // -------- selection (opId: 3) --------
        if(active) {
            active = (att30_lreceipt > att29_lcommitd);
        }
        // -------- hash join build (opId: 14) --------
        if(active) {
            uint64_t hash14 = 0;
            if(active) {
                hash14 = 0;
                if(active) {
                    hash14 = hash ( (hash14 + ((uint64_t)att18_lorderke)));
                }
            }
            jpayl14 payl;
            payl.att18_lorderke = att18_lorderke;
            payl.att20_lsuppkey = att20_lsuppkey;
            hashInsertMulti ( jht14, jht14_payload, offs14, 3000607, hash14, &(payl));
        }
        loopVar += step;
    }

}

__global__ void krnl_nation4(
    int* iatt34_nnationk, size_t* iatt35_nname_offset, char* iatt35_nname_char, unique_ht<jpayl7>* jht7) {
    int att34_nnationk;
    str_t att35_nname;
    str_t c1 = stringConstant ( "SAUDI ARABIA", 12);

    int tid_nation1 = 0;
    unsigned loopVar = ((blockIdx.x * blockDim.x) + threadIdx.x);
    unsigned step = (blockDim.x * gridDim.x);
    unsigned flushPipeline = 0;
    int active = 0;
    while(!(flushPipeline)) {
        tid_nation1 = loopVar;
        active = (loopVar < 25);
        // flush pipeline if no new elements
        flushPipeline = !(__ballot_sync(ALL_LANES,active));
        if(active) {
            att34_nnationk = iatt34_nnationk[tid_nation1];
            att35_nname = stringScan ( iatt35_nname_offset, iatt35_nname_char, tid_nation1);
        }
        // -------- selection (opId: 5) --------
        if(active) {
            active = stringEquals ( att35_nname, c1);
        }
        // -------- hash join build (opId: 7) --------
        if(active) {
            jpayl7 payl7;
            payl7.att34_nnationk = att34_nnationk;
            uint64_t hash7;
            hash7 = 0;
            if(active) {
                hash7 = hash ( (hash7 + ((uint64_t)att34_nnationk)));
            }
            hashBuildUnique ( jht7, 50, hash7, &(payl7));
        }
        loopVar += step;
    }

}

__global__ void krnl_supplier6(
    int* iatt38_ssuppkey, size_t* iatt39_sname_offset, char* iatt39_sname_char, int* iatt41_snationk, unique_ht<jpayl7>* jht7, unique_ht<jpayl10>* jht10) {
    int att38_ssuppkey;
    str_t att39_sname;
    int att41_snationk;
    int att34_nnationk;

    int tid_supplier1 = 0;
    unsigned loopVar = ((blockIdx.x * blockDim.x) + threadIdx.x);
    unsigned step = (blockDim.x * gridDim.x);
    unsigned flushPipeline = 0;
    int active = 0;
    while(!(flushPipeline)) {
        tid_supplier1 = loopVar;
        active = (loopVar < 10000);
        // flush pipeline if no new elements
        flushPipeline = !(__ballot_sync(ALL_LANES,active));
        if(active) {
            att38_ssuppkey = iatt38_ssuppkey[tid_supplier1];
            att39_sname = stringScan ( iatt39_sname_offset, iatt39_sname_char, tid_supplier1);
            att41_snationk = iatt41_snationk[tid_supplier1];
        }
        // -------- hash join probe (opId: 7) --------
        uint64_t hash7 = 0;
        if(active) {
            hash7 = 0;
            if(active) {
                hash7 = hash ( (hash7 + ((uint64_t)att41_snationk)));
            }
        }
        jpayl7* probepayl7;
        int numLookups7 = 0;
        if(active) {
            active = hashProbeUnique ( jht7, 50, hash7, numLookups7, &(probepayl7));
        }
        int bucketFound7 = 0;
        int probeActive7 = active;
        while((probeActive7 && !(bucketFound7))) {
            jpayl7 jprobepayl7 = *(probepayl7);
            att34_nnationk = jprobepayl7.att34_nnationk;
            bucketFound7 = 1;
            bucketFound7 &= ((att34_nnationk == att41_snationk));
            if(!(bucketFound7)) {
                probeActive7 = hashProbeUnique ( jht7, 50, hash7, numLookups7, &(probepayl7));
            }
        }
        active = bucketFound7;
        // -------- hash join build (opId: 10) --------
        if(active) {
            jpayl10 payl10;
            payl10.att38_ssuppkey = att38_ssuppkey;
            payl10.att39_sname = att39_sname;
            uint64_t hash10;
            hash10 = 0;
            if(active) {
                hash10 = hash ( (hash10 + ((uint64_t)att38_ssuppkey)));
            }
            hashBuildUnique ( jht10, 20000, hash10, &(payl10));
        }
        loopVar += step;
    }

}

__global__ void krnl_lineitem38(
    int* iatt45_lorderke, int* iatt47_lsuppkey, unsigned* iatt56_lcommitd, unsigned* iatt57_lreceipt, unique_ht<jpayl10>* jht10, multi_ht* jht13, jpayl13* jht13_payload) {
    int att45_lorderke;
    int att47_lsuppkey;
    unsigned att56_lcommitd;
    unsigned att57_lreceipt;
    int att38_ssuppkey;
    str_t att39_sname;

    int tid_lineitem3 = 0;
    unsigned loopVar = ((blockIdx.x * blockDim.x) + threadIdx.x);
    unsigned step = (blockDim.x * gridDim.x);
    unsigned flushPipeline = 0;
    int active = 0;
    while(!(flushPipeline)) {
        tid_lineitem3 = loopVar;
        active = (loopVar < 6001215);
        // flush pipeline if no new elements
        flushPipeline = !(__ballot_sync(ALL_LANES,active));
        if(active) {
            att45_lorderke = iatt45_lorderke[tid_lineitem3];
            att47_lsuppkey = iatt47_lsuppkey[tid_lineitem3];
            att56_lcommitd = iatt56_lcommitd[tid_lineitem3];
            att57_lreceipt = iatt57_lreceipt[tid_lineitem3];
        }
        // -------- selection (opId: 9) --------
        if(active) {
            active = (att57_lreceipt > att56_lcommitd);
        }
        // -------- hash join probe (opId: 10) --------
        uint64_t hash10 = 0;
        if(active) {
            hash10 = 0;
            if(active) {
                hash10 = hash ( (hash10 + ((uint64_t)att47_lsuppkey)));
            }
        }
        jpayl10* probepayl10;
        int numLookups10 = 0;
        if(active) {
            active = hashProbeUnique ( jht10, 20000, hash10, numLookups10, &(probepayl10));
        }
        int bucketFound10 = 0;
        int probeActive10 = active;
        while((probeActive10 && !(bucketFound10))) {
            jpayl10 jprobepayl10 = *(probepayl10);
            att38_ssuppkey = jprobepayl10.att38_ssuppkey;
            att39_sname = jprobepayl10.att39_sname;
            bucketFound10 = 1;
            bucketFound10 &= ((att38_ssuppkey == att47_lsuppkey));
            if(!(bucketFound10)) {
                probeActive10 = hashProbeUnique ( jht10, 20000, hash10, numLookups10, &(probepayl10));
            }
        }
        active = bucketFound10;
        // -------- hash join build (opId: 13) --------
        if(active) {
            uint64_t hash13 = 0;
            if(active) {
                hash13 = 0;
                if(active) {
                    hash13 = hash ( (hash13 + ((uint64_t)att45_lorderke)));
                }
            }
            hashCountMulti ( jht13, 240048, hash13);
        }
        loopVar += step;
    }

}

__global__ void krnl_lineitem38_ins(
    int* iatt45_lorderke, int* iatt47_lsuppkey, unsigned* iatt56_lcommitd, unsigned* iatt57_lreceipt, unique_ht<jpayl10>* jht10, multi_ht* jht13, jpayl13* jht13_payload, int* offs13) {
    int att45_lorderke;
    int att47_lsuppkey;
    unsigned att56_lcommitd;
    unsigned att57_lreceipt;
    int att38_ssuppkey;
    str_t att39_sname;

    int tid_lineitem3 = 0;
    unsigned loopVar = ((blockIdx.x * blockDim.x) + threadIdx.x);
    unsigned step = (blockDim.x * gridDim.x);
    unsigned flushPipeline = 0;
    int active = 0;
    while(!(flushPipeline)) {
        tid_lineitem3 = loopVar;
        active = (loopVar < 6001215);
        // flush pipeline if no new elements
        flushPipeline = !(__ballot_sync(ALL_LANES,active));
        if(active) {
            att45_lorderke = iatt45_lorderke[tid_lineitem3];
            att47_lsuppkey = iatt47_lsuppkey[tid_lineitem3];
            att56_lcommitd = iatt56_lcommitd[tid_lineitem3];
            att57_lreceipt = iatt57_lreceipt[tid_lineitem3];
        }
        // -------- selection (opId: 9) --------
        if(active) {
            active = (att57_lreceipt > att56_lcommitd);
        }
        // -------- hash join probe (opId: 10) --------
        uint64_t hash10 = 0;
        if(active) {
            hash10 = 0;
            if(active) {
                hash10 = hash ( (hash10 + ((uint64_t)att47_lsuppkey)));
            }
        }
        jpayl10* probepayl10;
        int numLookups10 = 0;
        if(active) {
            active = hashProbeUnique ( jht10, 20000, hash10, numLookups10, &(probepayl10));
        }
        int bucketFound10 = 0;
        int probeActive10 = active;
        while((probeActive10 && !(bucketFound10))) {
            jpayl10 jprobepayl10 = *(probepayl10);
            att38_ssuppkey = jprobepayl10.att38_ssuppkey;
            att39_sname = jprobepayl10.att39_sname;
            bucketFound10 = 1;
            bucketFound10 &= ((att38_ssuppkey == att47_lsuppkey));
            if(!(bucketFound10)) {
                probeActive10 = hashProbeUnique ( jht10, 20000, hash10, numLookups10, &(probepayl10));
            }
        }
        active = bucketFound10;
        // -------- hash join build (opId: 13) --------
        if(active) {
            uint64_t hash13 = 0;
            if(active) {
                hash13 = 0;
                if(active) {
                    hash13 = hash ( (hash13 + ((uint64_t)att45_lorderke)));
                }
            }
            jpayl13 payl;
            payl.att39_sname = att39_sname;
            payl.att45_lorderke = att45_lorderke;
            payl.att47_lsuppkey = att47_lsuppkey;
            hashInsertMulti ( jht13, jht13_payload, offs13, 240048, hash13, &(payl));
        }
        loopVar += step;
    }

}

__global__ void krnl_orders11(
    int* iatt61_oorderke, char* iatt63_oorderst, multi_ht* jht13, jpayl13* jht13_payload, multi_ht* jht14, jpayl14* jht14_payload, multi_ht* jht15, jpayl15* jht15_payload, agg_ht<apayl16>* aht16, int* agg1) {
    int att61_oorderke;
    char att63_oorderst;
    unsigned warplane = (threadIdx.x % 32);
    str_t att39_sname;
    int att45_lorderke;
    int att47_lsuppkey;
    int att18_lorderke;
    int att20_lsuppkey;
    int att2_lorderke;
    int att4_lsuppkey;

    int tid_orders1 = 0;
    unsigned loopVar = ((blockIdx.x * blockDim.x) + threadIdx.x);
    unsigned step = (blockDim.x * gridDim.x);
    unsigned flushPipeline = 0;
    int active = 0;
    while(!(flushPipeline)) {
        tid_orders1 = loopVar;
        active = (loopVar < 1500000);
        // flush pipeline if no new elements
        flushPipeline = !(__ballot_sync(ALL_LANES,active));
        if(active) {
            att61_oorderke = iatt61_oorderke[tid_orders1];
            att63_oorderst = iatt63_oorderst[tid_orders1];
        }
        // -------- selection (opId: 12) --------
        if(active) {
            active = (att63_oorderst == 'F');
        }
        // -------- hash join probe (opId: 13) --------
        // -------- multiprobe multi broadcast (opId: 13) --------
        int matchEnd13 = 0;
        int matchEndBuf13 = 0;
        int matchOffset13 = 0;
        int matchOffsetBuf13 = 0;
        int probeActive13 = active;
        int att61_oorderke_bcbuf13;
        uint64_t hash13 = 0;
        if(probeActive13) {
            hash13 = 0;
            if(active) {
                hash13 = hash ( (hash13 + ((uint64_t)att61_oorderke)));
            }
            probeActive13 = hashProbeMulti ( jht13, 240048, hash13, matchOffsetBuf13, matchEndBuf13);
        }
        unsigned activeProbes13 = __ballot_sync(ALL_LANES,probeActive13);
        int num13 = 0;
        num13 = (matchEndBuf13 - matchOffsetBuf13);
        unsigned wideProbes13 = __ballot_sync(ALL_LANES,(num13 >= 32));
        att61_oorderke_bcbuf13 = att61_oorderke;
        while((activeProbes13 > 0)) {
            unsigned tupleLane;
            unsigned broadcastLane;
            int numFilled = 0;
            int num = 0;
            while(((numFilled < 32) && activeProbes13)) {
                if((wideProbes13 > 0)) {
                    tupleLane = (__ffs(wideProbes13) - 1);
                    wideProbes13 -= (1 << tupleLane);
                }
                else {
                    tupleLane = (__ffs(activeProbes13) - 1);
                }
                num = __shfl_sync(ALL_LANES,num13,tupleLane);
                if((numFilled && ((numFilled + num) > 32))) {
                    break;
                }
                if((warplane >= numFilled)) {
                    broadcastLane = tupleLane;
                    matchOffset13 = (warplane - numFilled);
                }
                numFilled += num;
                activeProbes13 -= (1 << tupleLane);
            }
            matchOffset13 += __shfl_sync(ALL_LANES,matchOffsetBuf13,broadcastLane);
            matchEnd13 = __shfl_sync(ALL_LANES,matchEndBuf13,broadcastLane);
            att61_oorderke = __shfl_sync(ALL_LANES,att61_oorderke_bcbuf13,broadcastLane);
            probeActive13 = (matchOffset13 < matchEnd13);
            while(__any_sync(ALL_LANES,probeActive13)) {
                active = probeActive13;
                active = 0;
                jpayl13 payl;
                if(probeActive13) {
                    payl = jht13_payload[matchOffset13];
                    att39_sname = payl.att39_sname;
                    att45_lorderke = payl.att45_lorderke;
                    att47_lsuppkey = payl.att47_lsuppkey;
                    active = 1;
                    active &= ((att45_lorderke == att61_oorderke));
                    matchOffset13 += 32;
                    probeActive13 &= ((matchOffset13 < matchEnd13));
                }
                // -------- hash join probe (opId: 14) --------
                int matchEnd14 = 0;
                int matchOffset14 = 0;
                int matchStep14 = 1;
                int filterMatch14 = 0;
                int probeActive14 = active;
                uint64_t hash14 = 0;
                if(probeActive14) {
                    hash14 = 0;
                    if(active) {
                        hash14 = hash ( (hash14 + ((uint64_t)att45_lorderke)));
                    }
                    probeActive14 = hashProbeMulti ( jht14, 3000607, hash14, matchOffset14, matchEnd14);
                }
                while(probeActive14) {
                    jpayl14 payl;
                    payl = jht14_payload[matchOffset14];
                    att18_lorderke = payl.att18_lorderke;
                    att20_lsuppkey = payl.att20_lsuppkey;
                    filterMatch14 = 1;
                    filterMatch14 &= ((att18_lorderke == att45_lorderke));
                    filterMatch14 &= (!((att20_lsuppkey == att47_lsuppkey)));
                    matchOffset14 += matchStep14;
                    probeActive14 &= (!(filterMatch14));
                    probeActive14 &= ((matchOffset14 < matchEnd14));
                }
                active &= (!(filterMatch14));
                // -------- hash join probe (opId: 15) --------
                int matchEnd15 = 0;
                int matchOffset15 = 0;
                int matchStep15 = 1;
                int filterMatch15 = 0;
                int probeActive15 = active;
                uint64_t hash15 = 0;
                if(probeActive15) {
                    hash15 = 0;
                    if(active) {
                        hash15 = hash ( (hash15 + ((uint64_t)att45_lorderke)));
                    }
                    probeActive15 = hashProbeMulti ( jht15, 3000607, hash15, matchOffset15, matchEnd15);
                }
                while(probeActive15) {
                    jpayl15 payl;
                    payl = jht15_payload[matchOffset15];
                    att2_lorderke = payl.att2_lorderke;
                    att4_lsuppkey = payl.att4_lsuppkey;
                    filterMatch15 = 1;
                    filterMatch15 &= ((att2_lorderke == att45_lorderke));
                    filterMatch15 &= (!((att4_lsuppkey == att47_lsuppkey)));
                    matchOffset15 += matchStep15;
                    probeActive15 &= (!(filterMatch15));
                    probeActive15 &= ((matchOffset15 < matchEnd15));
                }
                active &= (filterMatch15);
                // -------- aggregation (opId: 16) --------
                int bucket = 0;
                if(active) {
                    uint64_t hash16 = 0;
                    hash16 = 0;
                    hash16 = hash ( (hash16 + stringHash ( att39_sname)));
                    apayl16 payl;
                    payl.att39_sname = att39_sname;
                    int bucketFound = 0;
                    int numLookups = 0;
                    while(!(bucketFound)) {
                        bucket = hashAggregateGetBucket ( aht16, 240048, hash16, numLookups, &(payl));
                        apayl16 probepayl = aht16[bucket].payload;
                        bucketFound = 1;
                        bucketFound &= (stringEquals ( payl.att39_sname, probepayl.att39_sname));
                    }
                }
                if(active) {
                    atomicAdd(&(agg1[bucket]), ((int)1));
                }
            }
        }
        loopVar += step;
    }

}

__global__ void krnl_aggregation16(
    agg_ht<apayl16>* aht16, int* agg1, int* nout_result, str_offs* oatt39_sname_offset, char* iatt39_sname_char, int* oatt1_numwait) {
    str_t att39_sname;
    int att1_numwait;
    unsigned warplane = (threadIdx.x % 32);
    unsigned prefixlanes = (0xffffffff >> (32 - warplane));

    int tid_aggregation16 = 0;
    unsigned loopVar = ((blockIdx.x * blockDim.x) + threadIdx.x);
    unsigned step = (blockDim.x * gridDim.x);
    unsigned flushPipeline = 0;
    int active = 0;
    while(!(flushPipeline)) {
        tid_aggregation16 = loopVar;
        active = (loopVar < 240048);
        // flush pipeline if no new elements
        flushPipeline = !(__ballot_sync(ALL_LANES,active));
        if(active) {
        }
        // -------- scan aggregation ht (opId: 16) --------
        if(active) {
            active &= ((aht16[tid_aggregation16].lock.lock == OnceLock::LOCK_DONE));
        }
        if(active) {
            apayl16 payl = aht16[tid_aggregation16].payload;
            att39_sname = payl.att39_sname;
        }
        if(active) {
            att1_numwait = agg1[tid_aggregation16];
        }
        // -------- materialize (opId: 17) --------
        int wp;
        int writeMask;
        int numProj;
        writeMask = __ballot_sync(ALL_LANES,active);
        numProj = __popc(writeMask);
        if((warplane == 0)) {
            wp = atomicAdd(nout_result, numProj);
        }
        wp = __shfl_sync(ALL_LANES,wp,0);
        wp = (wp + __popc((writeMask & prefixlanes)));
        if(active) {
            oatt39_sname_offset[wp] = toStringOffset ( iatt39_sname_char, att39_sname);
            oatt1_numwait[wp] = att1_numwait;
        }
        loopVar += step;
    }

}

int main() {
    int* iatt2_lorderke;
    iatt2_lorderke = ( int*) map_memory_file ( "mmdb/lineitem_l_orderkey" );
    int* iatt4_lsuppkey;
    iatt4_lsuppkey = ( int*) map_memory_file ( "mmdb/lineitem_l_suppkey" );
    int* iatt18_lorderke;
    iatt18_lorderke = ( int*) map_memory_file ( "mmdb/lineitem_l_orderkey" );
    int* iatt20_lsuppkey;
    iatt20_lsuppkey = ( int*) map_memory_file ( "mmdb/lineitem_l_suppkey" );
    unsigned* iatt29_lcommitd;
    iatt29_lcommitd = ( unsigned*) map_memory_file ( "mmdb/lineitem_l_commitdate" );
    unsigned* iatt30_lreceipt;
    iatt30_lreceipt = ( unsigned*) map_memory_file ( "mmdb/lineitem_l_receiptdate" );
    int* iatt34_nnationk;
    iatt34_nnationk = ( int*) map_memory_file ( "mmdb/nation_n_nationkey" );
    size_t* iatt35_nname_offset;
    iatt35_nname_offset = ( size_t*) map_memory_file ( "mmdb/nation_n_name_offset" );
    char* iatt35_nname_char;
    iatt35_nname_char = ( char*) map_memory_file ( "mmdb/nation_n_name_char" );
    int* iatt38_ssuppkey;
    iatt38_ssuppkey = ( int*) map_memory_file ( "mmdb/supplier_s_suppkey" );
    size_t* iatt39_sname_offset;
    iatt39_sname_offset = ( size_t*) map_memory_file ( "mmdb/supplier_s_name_offset" );
    char* iatt39_sname_char;
    iatt39_sname_char = ( char*) map_memory_file ( "mmdb/supplier_s_name_char" );
    int* iatt41_snationk;
    iatt41_snationk = ( int*) map_memory_file ( "mmdb/supplier_s_nationkey" );
    int* iatt45_lorderke;
    iatt45_lorderke = ( int*) map_memory_file ( "mmdb/lineitem_l_orderkey" );
    int* iatt47_lsuppkey;
    iatt47_lsuppkey = ( int*) map_memory_file ( "mmdb/lineitem_l_suppkey" );
    unsigned* iatt56_lcommitd;
    iatt56_lcommitd = ( unsigned*) map_memory_file ( "mmdb/lineitem_l_commitdate" );
    unsigned* iatt57_lreceipt;
    iatt57_lreceipt = ( unsigned*) map_memory_file ( "mmdb/lineitem_l_receiptdate" );
    int* iatt61_oorderke;
    iatt61_oorderke = ( int*) map_memory_file ( "mmdb/orders_o_orderkey" );
    char* iatt63_oorderst;
    iatt63_oorderst = ( char*) map_memory_file ( "mmdb/orders_o_orderstatus" );

    int nout_result;
    std::vector < str_offs > oatt39_sname_offset(120024);
    std::vector < int > oatt1_numwait(120024);

    // wake up gpu
    cudaDeviceSynchronize();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in wake up gpu! " << cudaGetErrorString( err ) << std::endl;
            ERROR("wake up gpu")
        }
    }

    int* d_iatt2_lorderke;
    cudaMalloc((void**) &d_iatt2_lorderke, 6001215* sizeof(int) );
    int* d_iatt4_lsuppkey;
    cudaMalloc((void**) &d_iatt4_lsuppkey, 6001215* sizeof(int) );
    int* d_iatt18_lorderke;
    d_iatt18_lorderke = d_iatt2_lorderke;
    int* d_iatt20_lsuppkey;
    d_iatt20_lsuppkey = d_iatt4_lsuppkey;
    unsigned* d_iatt29_lcommitd;
    cudaMalloc((void**) &d_iatt29_lcommitd, 6001215* sizeof(unsigned) );
    unsigned* d_iatt30_lreceipt;
    cudaMalloc((void**) &d_iatt30_lreceipt, 6001215* sizeof(unsigned) );
    int* d_iatt34_nnationk;
    cudaMalloc((void**) &d_iatt34_nnationk, 25* sizeof(int) );
    size_t* d_iatt35_nname_offset;
    cudaMalloc((void**) &d_iatt35_nname_offset, (25 + 1)* sizeof(size_t) );
    char* d_iatt35_nname_char;
    cudaMalloc((void**) &d_iatt35_nname_char, 186* sizeof(char) );
    int* d_iatt38_ssuppkey;
    cudaMalloc((void**) &d_iatt38_ssuppkey, 10000* sizeof(int) );
    size_t* d_iatt39_sname_offset;
    cudaMalloc((void**) &d_iatt39_sname_offset, (10000 + 1)* sizeof(size_t) );
    char* d_iatt39_sname_char;
    cudaMalloc((void**) &d_iatt39_sname_char, 180009* sizeof(char) );
    int* d_iatt41_snationk;
    cudaMalloc((void**) &d_iatt41_snationk, 10000* sizeof(int) );
    int* d_iatt45_lorderke;
    d_iatt45_lorderke = d_iatt2_lorderke;
    int* d_iatt47_lsuppkey;
    d_iatt47_lsuppkey = d_iatt4_lsuppkey;
    unsigned* d_iatt56_lcommitd;
    d_iatt56_lcommitd = d_iatt29_lcommitd;
    unsigned* d_iatt57_lreceipt;
    d_iatt57_lreceipt = d_iatt30_lreceipt;
    int* d_iatt61_oorderke;
    cudaMalloc((void**) &d_iatt61_oorderke, 1500000* sizeof(int) );
    char* d_iatt63_oorderst;
    cudaMalloc((void**) &d_iatt63_oorderst, 1500000* sizeof(char) );
    int* d_nout_result;
    cudaMalloc((void**) &d_nout_result, 1* sizeof(int) );
    str_offs* d_oatt39_sname_offset;
    cudaMalloc((void**) &d_oatt39_sname_offset, 120024* sizeof(str_offs) );
    int* d_oatt1_numwait;
    cudaMalloc((void**) &d_oatt1_numwait, 120024* sizeof(int) );
    cudaDeviceSynchronize();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in cuda malloc! " << cudaGetErrorString( err ) << std::endl;
            ERROR("cuda malloc")
        }
    }


    // show memory usage of GPU
    {   size_t free_byte ;
        size_t total_byte ;
        cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;
        if ( cudaSuccess != cuda_status ) {
            printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
            exit(1);
        }
        double free_db = (double)free_byte ;
        double total_db = (double)total_byte ;
        double used_db = total_db - free_db ;
        fprintf(stderr, "Memory %.1f / %.1f GB\n",
                used_db/(1024*1024*1024), total_db/(1024*1024*1024) );
        fflush(stdout);
    }

    multi_ht* d_jht15;
    cudaMalloc((void**) &d_jht15, 3000607* sizeof(multi_ht) );
    jpayl15* d_jht15_payload;
    cudaMalloc((void**) &d_jht15_payload, 12002430* sizeof(jpayl15) );
    {
        int gridsize=920;
        int blocksize=128;
        initMultiHT<<<gridsize, blocksize>>>(d_jht15, 3000607);
    }
    int* d_offs15;
    cudaMalloc((void**) &d_offs15, 1* sizeof(int) );
    {
        int gridsize=920;
        int blocksize=128;
        initArray<<<gridsize, blocksize>>>(d_offs15, 0, 1);
    }
    multi_ht* d_jht14;
    cudaMalloc((void**) &d_jht14, 3000607* sizeof(multi_ht) );
    jpayl14* d_jht14_payload;
    cudaMalloc((void**) &d_jht14_payload, 6001214* sizeof(jpayl14) );
    {
        int gridsize=920;
        int blocksize=128;
        initMultiHT<<<gridsize, blocksize>>>(d_jht14, 3000607);
    }
    int* d_offs14;
    cudaMalloc((void**) &d_offs14, 1* sizeof(int) );
    {
        int gridsize=920;
        int blocksize=128;
        initArray<<<gridsize, blocksize>>>(d_offs14, 0, 1);
    }
    unique_ht<jpayl7>* d_jht7;
    cudaMalloc((void**) &d_jht7, 50* sizeof(unique_ht<jpayl7>) );
    {
        int gridsize=920;
        int blocksize=128;
        initUniqueHT<<<gridsize, blocksize>>>(d_jht7, 50);
    }
    unique_ht<jpayl10>* d_jht10;
    cudaMalloc((void**) &d_jht10, 20000* sizeof(unique_ht<jpayl10>) );
    {
        int gridsize=920;
        int blocksize=128;
        initUniqueHT<<<gridsize, blocksize>>>(d_jht10, 20000);
    }
    multi_ht* d_jht13;
    cudaMalloc((void**) &d_jht13, 240048* sizeof(multi_ht) );
    jpayl13* d_jht13_payload;
    cudaMalloc((void**) &d_jht13_payload, 240048* sizeof(jpayl13) );
    {
        int gridsize=920;
        int blocksize=128;
        initMultiHT<<<gridsize, blocksize>>>(d_jht13, 240048);
    }
    int* d_offs13;
    cudaMalloc((void**) &d_offs13, 1* sizeof(int) );
    {
        int gridsize=920;
        int blocksize=128;
        initArray<<<gridsize, blocksize>>>(d_offs13, 0, 1);
    }
    agg_ht<apayl16>* d_aht16;
    cudaMalloc((void**) &d_aht16, 240048* sizeof(agg_ht<apayl16>) );
    {
        int gridsize=920;
        int blocksize=128;
        initAggHT<<<gridsize, blocksize>>>(d_aht16, 240048);
    }
    int* d_agg1;
    cudaMalloc((void**) &d_agg1, 240048* sizeof(int) );
    {
        int gridsize=920;
        int blocksize=128;
        initArray<<<gridsize, blocksize>>>(d_agg1, 0, 240048);
    }
    {
        int gridsize=920;
        int blocksize=128;
        initArray<<<gridsize, blocksize>>>(d_nout_result, 0, 1);
    }
    cudaDeviceSynchronize();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in cuda mallocHT! " << cudaGetErrorString( err ) << std::endl;
            ERROR("cuda mallocHT")
        }
    }


    // show memory usage of GPU
    {   size_t free_byte ;
        size_t total_byte ;
        cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;
        if ( cudaSuccess != cuda_status ) {
            printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
            exit(1);
        }
        double free_db = (double)free_byte ;
        double total_db = (double)total_byte ;
        double used_db = total_db - free_db ;
        fprintf(stderr, "Memory %.1f / %.1f GB\n",
                used_db/(1024*1024*1024), total_db/(1024*1024*1024) );
        fflush(stdout);
    }

    cudaMemcpy( d_iatt2_lorderke, iatt2_lorderke, 6001215 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt4_lsuppkey, iatt4_lsuppkey, 6001215 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt29_lcommitd, iatt29_lcommitd, 6001215 * sizeof(unsigned), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt30_lreceipt, iatt30_lreceipt, 6001215 * sizeof(unsigned), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt34_nnationk, iatt34_nnationk, 25 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt35_nname_offset, iatt35_nname_offset, (25 + 1) * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt35_nname_char, iatt35_nname_char, 186 * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt38_ssuppkey, iatt38_ssuppkey, 10000 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt39_sname_offset, iatt39_sname_offset, (10000 + 1) * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt39_sname_char, iatt39_sname_char, 180009 * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt41_snationk, iatt41_snationk, 10000 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt61_oorderke, iatt61_oorderke, 1500000 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( d_iatt63_oorderst, iatt63_oorderst, 1500000 * sizeof(char), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in cuda memcpy in! " << cudaGetErrorString( err ) << std::endl;
            ERROR("cuda memcpy in")
        }
    }

    std::clock_t start_totalKernelTime178 = std::clock();
    std::clock_t start_krnl_lineitem1179 = std::clock();
    {
        int gridsize=920;
        int blocksize=128;
        krnl_lineitem1<<<gridsize, blocksize>>>(d_iatt2_lorderke, d_iatt4_lsuppkey, d_jht15, d_jht15_payload);
    }
    cudaDeviceSynchronize();
    std::clock_t stop_krnl_lineitem1179 = std::clock();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in krnl_lineitem1! " << cudaGetErrorString( err ) << std::endl;
            ERROR("krnl_lineitem1")
        }
    }

    std::clock_t start_scanMultiHT180 = std::clock();
    {
        int gridsize=920;
        int blocksize=128;
        scanMultiHT<<<gridsize, blocksize>>>(d_jht15, 3000607, d_offs15);
    }
    cudaDeviceSynchronize();
    std::clock_t stop_scanMultiHT180 = std::clock();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in scanMultiHT! " << cudaGetErrorString( err ) << std::endl;
            ERROR("scanMultiHT")
        }
    }

    std::clock_t start_krnl_lineitem1_ins181 = std::clock();
    {
        int gridsize=920;
        int blocksize=128;
        krnl_lineitem1_ins<<<gridsize, blocksize>>>(d_iatt2_lorderke, d_iatt4_lsuppkey, d_jht15, d_jht15_payload, d_offs15);
    }
    cudaDeviceSynchronize();
    std::clock_t stop_krnl_lineitem1_ins181 = std::clock();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in krnl_lineitem1_ins! " << cudaGetErrorString( err ) << std::endl;
            ERROR("krnl_lineitem1_ins")
        }
    }

    std::clock_t start_krnl_lineitem22182 = std::clock();
    {
        int gridsize=920;
        int blocksize=128;
        krnl_lineitem22<<<gridsize, blocksize>>>(d_iatt18_lorderke, d_iatt20_lsuppkey, d_iatt29_lcommitd, d_iatt30_lreceipt, d_jht14, d_jht14_payload);
    }
    cudaDeviceSynchronize();
    std::clock_t stop_krnl_lineitem22182 = std::clock();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in krnl_lineitem22! " << cudaGetErrorString( err ) << std::endl;
            ERROR("krnl_lineitem22")
        }
    }

    std::clock_t start_scanMultiHT183 = std::clock();
    {
        int gridsize=920;
        int blocksize=128;
        scanMultiHT<<<gridsize, blocksize>>>(d_jht14, 3000607, d_offs14);
    }
    cudaDeviceSynchronize();
    std::clock_t stop_scanMultiHT183 = std::clock();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in scanMultiHT! " << cudaGetErrorString( err ) << std::endl;
            ERROR("scanMultiHT")
        }
    }

    std::clock_t start_krnl_lineitem22_ins184 = std::clock();
    {
        int gridsize=920;
        int blocksize=128;
        krnl_lineitem22_ins<<<gridsize, blocksize>>>(d_iatt18_lorderke, d_iatt20_lsuppkey, d_iatt29_lcommitd, d_iatt30_lreceipt, d_jht14, d_jht14_payload, d_offs14);
    }
    cudaDeviceSynchronize();
    std::clock_t stop_krnl_lineitem22_ins184 = std::clock();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in krnl_lineitem22_ins! " << cudaGetErrorString( err ) << std::endl;
            ERROR("krnl_lineitem22_ins")
        }
    }

    std::clock_t start_krnl_nation4185 = std::clock();
    {
        int gridsize=920;
        int blocksize=128;
        krnl_nation4<<<gridsize, blocksize>>>(d_iatt34_nnationk, d_iatt35_nname_offset, d_iatt35_nname_char, d_jht7);
    }
    cudaDeviceSynchronize();
    std::clock_t stop_krnl_nation4185 = std::clock();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in krnl_nation4! " << cudaGetErrorString( err ) << std::endl;
            ERROR("krnl_nation4")
        }
    }

    std::clock_t start_krnl_supplier6186 = std::clock();
    {
        int gridsize=920;
        int blocksize=128;
        krnl_supplier6<<<gridsize, blocksize>>>(d_iatt38_ssuppkey, d_iatt39_sname_offset, d_iatt39_sname_char, d_iatt41_snationk, d_jht7, d_jht10);
    }
    cudaDeviceSynchronize();
    std::clock_t stop_krnl_supplier6186 = std::clock();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in krnl_supplier6! " << cudaGetErrorString( err ) << std::endl;
            ERROR("krnl_supplier6")
        }
    }

    std::clock_t start_krnl_lineitem38187 = std::clock();
    {
        int gridsize=920;
        int blocksize=128;
        krnl_lineitem38<<<gridsize, blocksize>>>(d_iatt45_lorderke, d_iatt47_lsuppkey, d_iatt56_lcommitd, d_iatt57_lreceipt, d_jht10, d_jht13, d_jht13_payload);
    }
    cudaDeviceSynchronize();
    std::clock_t stop_krnl_lineitem38187 = std::clock();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in krnl_lineitem38! " << cudaGetErrorString( err ) << std::endl;
            ERROR("krnl_lineitem38")
        }
    }

    std::clock_t start_scanMultiHT188 = std::clock();
    {
        int gridsize=920;
        int blocksize=128;
        scanMultiHT<<<gridsize, blocksize>>>(d_jht13, 240048, d_offs13);
    }
    cudaDeviceSynchronize();
    std::clock_t stop_scanMultiHT188 = std::clock();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in scanMultiHT! " << cudaGetErrorString( err ) << std::endl;
            ERROR("scanMultiHT")
        }
    }

    std::clock_t start_krnl_lineitem38_ins189 = std::clock();
    {
        int gridsize=920;
        int blocksize=128;
        krnl_lineitem38_ins<<<gridsize, blocksize>>>(d_iatt45_lorderke, d_iatt47_lsuppkey, d_iatt56_lcommitd, d_iatt57_lreceipt, d_jht10, d_jht13, d_jht13_payload, d_offs13);
    }
    cudaDeviceSynchronize();
    std::clock_t stop_krnl_lineitem38_ins189 = std::clock();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in krnl_lineitem38_ins! " << cudaGetErrorString( err ) << std::endl;
            ERROR("krnl_lineitem38_ins")
        }
    }

    std::clock_t start_krnl_orders11190 = std::clock();
    {
        int gridsize=920;
        int blocksize=128;
        krnl_orders11<<<gridsize, blocksize>>>(d_iatt61_oorderke, d_iatt63_oorderst, d_jht13, d_jht13_payload, d_jht14, d_jht14_payload, d_jht15, d_jht15_payload, d_aht16, d_agg1);
    }
    cudaDeviceSynchronize();
    std::clock_t stop_krnl_orders11190 = std::clock();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in krnl_orders11! " << cudaGetErrorString( err ) << std::endl;
            ERROR("krnl_orders11")
        }
    }

    std::clock_t start_krnl_aggregation16191 = std::clock();
    {
        int gridsize=920;
        int blocksize=128;
        krnl_aggregation16<<<gridsize, blocksize>>>(d_aht16, d_agg1, d_nout_result, d_oatt39_sname_offset, d_iatt39_sname_char, d_oatt1_numwait);
    }
    cudaDeviceSynchronize();
    std::clock_t stop_krnl_aggregation16191 = std::clock();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in krnl_aggregation16! " << cudaGetErrorString( err ) << std::endl;
            ERROR("krnl_aggregation16")
        }
    }

    std::clock_t stop_totalKernelTime178 = std::clock();
    cudaMemcpy( &nout_result, d_nout_result, 1 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy( oatt39_sname_offset.data(), d_oatt39_sname_offset, 120024 * sizeof(str_offs), cudaMemcpyDeviceToHost);
    cudaMemcpy( oatt1_numwait.data(), d_oatt1_numwait, 120024 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in cuda memcpy out! " << cudaGetErrorString( err ) << std::endl;
            ERROR("cuda memcpy out")
        }
    }

    cudaFree( d_iatt2_lorderke);
    cudaFree( d_iatt4_lsuppkey);
    cudaFree( d_jht15);
    cudaFree( d_jht15_payload);
    cudaFree( d_offs15);
    cudaFree( d_iatt29_lcommitd);
    cudaFree( d_iatt30_lreceipt);
    cudaFree( d_jht14);
    cudaFree( d_jht14_payload);
    cudaFree( d_offs14);
    cudaFree( d_iatt34_nnationk);
    cudaFree( d_iatt35_nname_offset);
    cudaFree( d_iatt35_nname_char);
    cudaFree( d_jht7);
    cudaFree( d_iatt38_ssuppkey);
    cudaFree( d_iatt39_sname_offset);
    cudaFree( d_iatt39_sname_char);
    cudaFree( d_iatt41_snationk);
    cudaFree( d_jht10);
    cudaFree( d_jht13);
    cudaFree( d_jht13_payload);
    cudaFree( d_offs13);
    cudaFree( d_iatt61_oorderke);
    cudaFree( d_iatt63_oorderst);
    cudaFree( d_aht16);
    cudaFree( d_agg1);
    cudaFree( d_nout_result);
    cudaFree( d_oatt39_sname_offset);
    cudaFree( d_oatt1_numwait);
    cudaDeviceSynchronize();
    {
        cudaError err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cerr << "Cuda Error in cuda free! " << cudaGetErrorString( err ) << std::endl;
            ERROR("cuda free")
        }
    }

    std::clock_t start_finish192 = std::clock();
    printf("\nResult: %i tuples\n", nout_result);
    if((nout_result > 120024)) {
        ERROR("Index out of range. Output size larger than allocated with expected result number.")
    }
    for ( int pv = 0; ((pv < 10) && (pv < nout_result)); pv += 1) {
        printf("s_name: ");
        stringPrint ( iatt39_sname_char, oatt39_sname_offset[pv]);
        printf("  ");
        printf("numwait: ");
        printf("%8i", oatt1_numwait[pv]);
        printf("  ");
        printf("\n");
    }
    if((nout_result > 10)) {
        printf("[...]\n");
    }
    printf("\n");
    std::clock_t stop_finish192 = std::clock();

    printf("<timing>\n");
    printf ( "%32s: %6.1f ms\n", "krnl_lineitem1", (stop_krnl_lineitem1179 - start_krnl_lineitem1179) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "scanMultiHT", (stop_scanMultiHT180 - start_scanMultiHT180) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "krnl_lineitem1_ins", (stop_krnl_lineitem1_ins181 - start_krnl_lineitem1_ins181) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "krnl_lineitem22", (stop_krnl_lineitem22182 - start_krnl_lineitem22182) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "scanMultiHT", (stop_scanMultiHT183 - start_scanMultiHT183) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "krnl_lineitem22_ins", (stop_krnl_lineitem22_ins184 - start_krnl_lineitem22_ins184) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "krnl_nation4", (stop_krnl_nation4185 - start_krnl_nation4185) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "krnl_supplier6", (stop_krnl_supplier6186 - start_krnl_supplier6186) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "krnl_lineitem38", (stop_krnl_lineitem38187 - start_krnl_lineitem38187) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "scanMultiHT", (stop_scanMultiHT188 - start_scanMultiHT188) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "krnl_lineitem38_ins", (stop_krnl_lineitem38_ins189 - start_krnl_lineitem38_ins189) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "krnl_orders11", (stop_krnl_orders11190 - start_krnl_orders11190) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "krnl_aggregation16", (stop_krnl_aggregation16191 - start_krnl_aggregation16191) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "finish", (stop_finish192 - start_finish192) / (double) (CLOCKS_PER_SEC / 1000) );
    printf ( "%32s: %6.1f ms\n", "totalKernelTime", (stop_totalKernelTime178 - start_totalKernelTime178) / (double) (CLOCKS_PER_SEC / 1000) );
    printf("</timing>\n");
}
