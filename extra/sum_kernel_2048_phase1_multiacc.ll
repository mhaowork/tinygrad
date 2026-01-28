define void @r_32_8_4096_4(float* noalias align 32 %data0, float* noalias align 32 %data1, i32 %core_id) #0
{
  %reg_0 = alloca [1 x <4 x float>]
  %reg_1 = alloca [1 x <4 x float>]
  %reg_2 = alloca [1 x <4 x float>]
  %reg_3 = alloca [1 x <4 x float>]
  %v0 = add i32 %core_id, 0
  %v1_z = insertelement <1 x float> poison, float 0.0, i32 0
  %v1 = shufflevector <1 x float> %v1_z, <1 x float> poison, <4 x i32> zeroinitializer
  %v2 = mul i32 %v0, 32
  %v3 = mul i32 %v0, 524288
  br label %loop_entry_2
loop_entry_2:
  br label %loop_latch_2
loop_latch_2:
  %v4 = phi i32 [ 0, %loop_entry_2 ], [ %v4phi, %loop_footer_2 ]
  %v4phi = add i32 %v4, 1
  %v4cmp = icmp ult i32 %v4, 32
  br i1 %v4cmp, label %loop_body_2, label %loop_exit_2
loop_body_2:
  %v5 = getelementptr inbounds <4 x float>, <4 x float>* %reg_0, i32 0
  %v6 = getelementptr inbounds <4 x float>, <4 x float>* %reg_1, i32 0
  %v7 = getelementptr inbounds <4 x float>, <4 x float>* %reg_2, i32 0
  %v8 = getelementptr inbounds <4 x float>, <4 x float>* %reg_3, i32 0
  %v9 = mul i32 %v4, 16384
  %v10 = add nsw i32 %v3, %v9
  store <4 x float> %v1, <4 x float>* %v5
  store <4 x float> %v1, <4 x float>* %v6
  store <4 x float> %v1, <4 x float>* %v7
  store <4 x float> %v1, <4 x float>* %v8
  br label %loop_entry_0
loop_entry_0:
  br label %loop_latch_0
loop_latch_0:
  %v15 = phi i32 [ 0, %loop_entry_0 ], [ %v15phi, %loop_footer_0 ]
  %v15phi = add i32 %v15, 1
  %v15cmp = icmp ult i32 %v15, 1024
  br i1 %v15cmp, label %loop_body_0, label %loop_exit_0
loop_body_0:
  %v16 = getelementptr inbounds <4 x float>, <4 x float>* %reg_0, i32 0
  %v17 = load <4 x float>, <4 x float>* %v16
  %v18 = getelementptr inbounds <4 x float>, <4 x float>* %reg_1, i32 0
  %v19 = load <4 x float>, <4 x float>* %v18
  %v20 = getelementptr inbounds <4 x float>, <4 x float>* %reg_2, i32 0
  %v21 = load <4 x float>, <4 x float>* %v20
  %v22 = getelementptr inbounds <4 x float>, <4 x float>* %reg_3, i32 0
  %v23 = load <4 x float>, <4 x float>* %v22
  %v24 = mul i32 %v15, 16
  %v25 = add nsw i32 %v10, %v24
  %v26 = add nsw i32 %v25, 4
  %v27 = getelementptr inbounds float, float* %data1, i32 %v26
  %v28 = load <4 x float>, <4 x float>* %v27
  %v29 = add nsw i32 %v25, 8
  %v30 = getelementptr inbounds float, float* %data1, i32 %v29
  %v31 = load <4 x float>, <4 x float>* %v30
  %v32 = add nsw i32 %v25, 12
  %v33 = getelementptr inbounds float, float* %data1, i32 %v32
  %v34 = load <4 x float>, <4 x float>* %v33
  %v35 = getelementptr inbounds float, float* %data1, i32 %v25
  %v36 = load <4 x float>, <4 x float>* %v35
  %v37 = fadd nsz arcp contract afn <4 x float> %v17, %v36
  %v38 = fadd nsz arcp contract afn <4 x float> %v19, %v28
  %v39 = fadd nsz arcp contract afn <4 x float> %v21, %v31
  %v40 = fadd nsz arcp contract afn <4 x float> %v23, %v34
  store <4 x float> %v37, <4 x float>* %v16
  store <4 x float> %v38, <4 x float>* %v18
  store <4 x float> %v39, <4 x float>* %v20
  store <4 x float> %v40, <4 x float>* %v22
  br label %loop_footer_0
loop_footer_0:
  br label %loop_latch_0
loop_exit_0:
  %v46 = getelementptr inbounds <4 x float>, <4 x float>* %reg_0, i32 0
  %v47 = load <4 x float>, <4 x float>* %v46
  %v48 = getelementptr inbounds <4 x float>, <4 x float>* %reg_1, i32 0
  %v49 = load <4 x float>, <4 x float>* %v48
  %v50 = getelementptr inbounds <4 x float>, <4 x float>* %reg_2, i32 0
  %v51 = load <4 x float>, <4 x float>* %v50
  %v52 = getelementptr inbounds <4 x float>, <4 x float>* %reg_3, i32 0
  %v53 = load <4 x float>, <4 x float>* %v52
  %v54 = extractelement <4 x float> %v47, i32 0
  %v55 = extractelement <4 x float> %v49, i32 0
  %v56 = extractelement <4 x float> %v51, i32 0
  %v57 = extractelement <4 x float> %v53, i32 0
  %v58 = extractelement <4 x float> %v47, i32 1
  %v59 = extractelement <4 x float> %v49, i32 1
  %v60 = extractelement <4 x float> %v51, i32 1
  %v61 = extractelement <4 x float> %v53, i32 1
  %v62 = extractelement <4 x float> %v47, i32 2
  %v63 = extractelement <4 x float> %v49, i32 2
  %v64 = extractelement <4 x float> %v51, i32 2
  %v65 = extractelement <4 x float> %v53, i32 2
  %v66 = extractelement <4 x float> %v47, i32 3
  %v67 = extractelement <4 x float> %v49, i32 3
  %v68 = extractelement <4 x float> %v51, i32 3
  %v69 = extractelement <4 x float> %v53, i32 3
  %v70 = add nsw i32 %v2, %v4
  %v71 = getelementptr inbounds float, float* %data0, i32 %v70
  %v72 = fadd nsz arcp contract afn float %v54, %v55
  %v73 = fadd nsz arcp contract afn float %v58, %v59
  %v74 = fadd nsz arcp contract afn float %v62, %v63
  %v75 = fadd nsz arcp contract afn float %v66, %v67
  %v76 = fadd nsz arcp contract afn float %v72, %v56
  %v77 = fadd nsz arcp contract afn float %v73, %v60
  %v78 = fadd nsz arcp contract afn float %v74, %v64
  %v79 = fadd nsz arcp contract afn float %v75, %v68
  %v80 = fadd nsz arcp contract afn float %v76, %v57
  %v81 = fadd nsz arcp contract afn float %v77, %v61
  %v82 = fadd nsz arcp contract afn float %v78, %v65
  %v83 = fadd nsz arcp contract afn float %v79, %v69
  %v84 = fadd nsz arcp contract afn float %v80, %v81
  %v85 = fadd nsz arcp contract afn float %v84, %v82
  %v86 = fadd nsz arcp contract afn float %v85, %v83
  store float %v86, float* %v71
  br label %loop_footer_2
loop_footer_2:
  br label %loop_latch_2
loop_exit_2:
  ret void
}
attributes #0 = { alwaysinline nounwind "no-builtins" "no-trapping-math"="true" }