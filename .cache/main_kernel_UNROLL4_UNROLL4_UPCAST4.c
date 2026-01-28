// opts: Opt(op=OptOps.UNROLL, axis=0, arg=4), Opt(op=OptOps.UNROLL, axis=0, arg=4), Opt(op=OptOps.UPCAST, axis=0, arg=4)
// source: .cache/beam_latest_2048_candidates_debug.log
// kernel: test
typedef float float4 __attribute__((aligned(16),ext_vector_type(4)));
void test(float* restrict data0_256, float* restrict data1_4194304, int core_id) {
  float acc0[4];
  for (int Lidx2 = 0; Lidx2 < 64; Lidx2++) {
    *(acc0+0) = 0.0f;
    *(acc0+1) = 0.0f;
    *(acc0+2) = 0.0f;
    *(acc0+3) = 0.0f;
    for (int Ridx0 = 0; Ridx0 < 1024; Ridx0++) {
      int alu4 = ((Ridx0<<4)+(Lidx2<<16));
      float4 val0 = (*((float4*)((data1_4194304+(alu4+4)))));
      float4 val1 = (*((float4*)((data1_4194304+(alu4+8)))));
      float4 val2 = (*((float4*)((data1_4194304+(alu4+12)))));
      float4 val3 = (*((float4*)((data1_4194304+(alu4+16384)))));
      float4 val4 = (*((float4*)((data1_4194304+(alu4+16388)))));
      float4 val5 = (*((float4*)((data1_4194304+(alu4+16392)))));
      float4 val6 = (*((float4*)((data1_4194304+(alu4+16396)))));
      float4 val7 = (*((float4*)((data1_4194304+(alu4+32768)))));
      float4 val8 = (*((float4*)((data1_4194304+(alu4+32772)))));
      float4 val9 = (*((float4*)((data1_4194304+(alu4+32776)))));
      float4 val10 = (*((float4*)((data1_4194304+(alu4+32780)))));
      float4 val11 = (*((float4*)((data1_4194304+(alu4+49152)))));
      float4 val12 = (*((float4*)((data1_4194304+(alu4+49156)))));
      float4 val13 = (*((float4*)((data1_4194304+(alu4+49160)))));
      float4 val14 = (*((float4*)((data1_4194304+(alu4+49164)))));
      float4 val15 = (*((float4*)((data1_4194304+alu4))));
      *(acc0+0) = ((*(acc0+0))+val15[0]+val15[1]+val15[2]+val15[3]+val0[0]+val0[1]+val0[2]+val0[3]+val1[0]+val1[1]+val1[2]+val1[3]+val2[0]+val2[1]+val2[2]+val2[3]);
      *(acc0+1) = ((*(acc0+1))+val3[0]+val3[1]+val3[2]+val3[3]+val4[0]+val4[1]+val4[2]+val4[3]+val5[0]+val5[1]+val5[2]+val5[3]+val6[0]+val6[1]+val6[2]+val6[3]);
      *(acc0+2) = ((*(acc0+2))+val7[0]+val7[1]+val7[2]+val7[3]+val8[0]+val8[1]+val8[2]+val8[3]+val9[0]+val9[1]+val9[2]+val9[3]+val10[0]+val10[1]+val10[2]+val10[3]);
      *(acc0+3) = ((*(acc0+3))+val11[0]+val11[1]+val11[2]+val11[3]+val12[0]+val12[1]+val12[2]+val12[3]+val13[0]+val13[1]+val13[2]+val13[3]+val14[0]+val14[1]+val14[2]+val14[3]);
    }
    *((float4*)((data0_256+(Lidx2<<2)))) = (float4){(*(acc0+0)),(*(acc0+1)),(*(acc0+2)),(*(acc0+3))};
  }
}
