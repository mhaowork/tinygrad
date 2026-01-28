; ---- r[90m_[0m[97m64[0m[90m_[0m[94m4[0m[90m_[0m[31m1024[0m[90m_[0m[35m4[0m[90m_[0m[35m4[0m[90m[0m gs=[4, 1, 1] ----
define void @r_64_4_1024_4_4(float* noalias align 32 %data0, float* noalias align 32 %data1, i32 %core_id) #0
{
  %reg_0 = alloca [16 x float]
  %v0 = add i32 %core_id, 0
  %v1 = getelementptr inbounds float, float* %reg_0, i32 0
  %v2 = getelementptr inbounds float, float* %reg_0, i32 1
  %v3 = getelementptr inbounds float, float* %reg_0, i32 2
  %v4 = getelementptr inbounds float, float* %reg_0, i32 3
  %v5 = getelementptr inbounds float, float* %reg_0, i32 4
  %v6 = getelementptr inbounds float, float* %reg_0, i32 5
  %v7 = getelementptr inbounds float, float* %reg_0, i32 6
  %v8 = getelementptr inbounds float, float* %reg_0, i32 7
  %v9 = getelementptr inbounds float, float* %reg_0, i32 8
  %v10 = getelementptr inbounds float, float* %reg_0, i32 9
  %v11 = getelementptr inbounds float, float* %reg_0, i32 10
  %v12 = getelementptr inbounds float, float* %reg_0, i32 11
  %v13 = getelementptr inbounds float, float* %reg_0, i32 12
  %v14 = getelementptr inbounds float, float* %reg_0, i32 13
  %v15 = getelementptr inbounds float, float* %reg_0, i32 14
  %v16 = getelementptr inbounds float, float* %reg_0, i32 15
  %v17 = mul i32 %v0, 64
  %v18 = mul i32 %v0, 1048576
  br label %loop_entry_2
loop_entry_2:
  br label %loop_latch_2
loop_latch_2:
  %v19 = phi i32 [ 0, %loop_entry_2 ], [ %v19phi, %loop_footer_2 ]
  %v19phi = add i32 %v19, 1
  %v19cmp = icmp ult i32 %v19, 64
  br i1 %v19cmp, label %loop_body_2, label %loop_exit_2
loop_body_2:
  %v20 = getelementptr inbounds float, float* %reg_0, i32 0
  %v21 = getelementptr inbounds float, float* %reg_0, i32 1
  %v22 = getelementptr inbounds float, float* %reg_0, i32 2
  %v23 = getelementptr inbounds float, float* %reg_0, i32 3
  %v24 = getelementptr inbounds float, float* %reg_0, i32 4
  %v25 = getelementptr inbounds float, float* %reg_0, i32 5
  %v26 = getelementptr inbounds float, float* %reg_0, i32 6
  %v27 = getelementptr inbounds float, float* %reg_0, i32 7
  %v28 = getelementptr inbounds float, float* %reg_0, i32 8
  %v29 = getelementptr inbounds float, float* %reg_0, i32 9
  %v30 = getelementptr inbounds float, float* %reg_0, i32 10
  %v31 = getelementptr inbounds float, float* %reg_0, i32 11
  %v32 = getelementptr inbounds float, float* %reg_0, i32 12
  %v33 = getelementptr inbounds float, float* %reg_0, i32 13
  %v34 = getelementptr inbounds float, float* %reg_0, i32 14
  %v35 = getelementptr inbounds float, float* %reg_0, i32 15
  %v36 = mul i32 %v19, 16384
  %v37 = add nsw i32 %v18, %v36
  store float 0.0, float* %v20
  store float 0.0, float* %v21
  store float 0.0, float* %v22
  store float 0.0, float* %v23
  store float 0.0, float* %v24
  store float 0.0, float* %v25
  store float 0.0, float* %v26
  store float 0.0, float* %v27
  store float 0.0, float* %v28
  store float 0.0, float* %v29
  store float 0.0, float* %v30
  store float 0.0, float* %v31
  store float 0.0, float* %v32
  store float 0.0, float* %v33
  store float 0.0, float* %v34
  store float 0.0, float* %v35
  br label %loop_entry_0
loop_entry_0:
  br label %loop_latch_0
loop_latch_0:
  %v54 = phi i32 [ 0, %loop_entry_0 ], [ %v54phi, %loop_footer_0 ]
  %v54phi = add i32 %v54, 1
  %v54cmp = icmp ult i32 %v54, 1024
  br i1 %v54cmp, label %loop_body_0, label %loop_exit_0
loop_body_0:
  %v55 = getelementptr inbounds float, float* %reg_0, i32 0
  %v56 = load float, float* %v55
  %v57 = getelementptr inbounds float, float* %reg_0, i32 1
  %v58 = load float, float* %v57
  %v59 = getelementptr inbounds float, float* %reg_0, i32 2
  %v60 = load float, float* %v59
  %v61 = getelementptr inbounds float, float* %reg_0, i32 3
  %v62 = load float, float* %v61
  %v63 = getelementptr inbounds float, float* %reg_0, i32 4
  %v64 = load float, float* %v63
  %v65 = getelementptr inbounds float, float* %reg_0, i32 5
  %v66 = load float, float* %v65
  %v67 = getelementptr inbounds float, float* %reg_0, i32 6
  %v68 = load float, float* %v67
  %v69 = getelementptr inbounds float, float* %reg_0, i32 7
  %v70 = load float, float* %v69
  %v71 = getelementptr inbounds float, float* %reg_0, i32 8
  %v72 = load float, float* %v71
  %v73 = getelementptr inbounds float, float* %reg_0, i32 9
  %v74 = load float, float* %v73
  %v75 = getelementptr inbounds float, float* %reg_0, i32 10
  %v76 = load float, float* %v75
  %v77 = getelementptr inbounds float, float* %reg_0, i32 11
  %v78 = load float, float* %v77
  %v79 = getelementptr inbounds float, float* %reg_0, i32 12
  %v80 = load float, float* %v79
  %v81 = getelementptr inbounds float, float* %reg_0, i32 13
  %v82 = load float, float* %v81
  %v83 = getelementptr inbounds float, float* %reg_0, i32 14
  %v84 = load float, float* %v83
  %v85 = getelementptr inbounds float, float* %reg_0, i32 15
  %v86 = load float, float* %v85
  %v87 = mul i32 %v54, 16
  %v88 = add nsw i32 %v37, %v87
  %v89 = add nsw i32 %v88, 4
  %v90 = getelementptr inbounds float, float* %data1, i32 %v89
  %v91 = load <4 x float>, <4 x float>* %v90
  %v92 = add nsw i32 %v88, 8
  %v93 = getelementptr inbounds float, float* %data1, i32 %v92
  %v94 = load <4 x float>, <4 x float>* %v93
  %v95 = add nsw i32 %v88, 12
  %v96 = getelementptr inbounds float, float* %data1, i32 %v95
  %v97 = load <4 x float>, <4 x float>* %v96
  %v98 = getelementptr inbounds float, float* %data1, i32 %v88
  %v99 = load <4 x float>, <4 x float>* %v98
  %v100 = extractelement <4 x float> %v91, i32 0
  %v101 = extractelement <4 x float> %v94, i32 0
  %v102 = extractelement <4 x float> %v97, i32 0
  %v103 = extractelement <4 x float> %v99, i32 0
  %v104 = extractelement <4 x float> %v91, i32 1
  %v105 = extractelement <4 x float> %v94, i32 1
  %v106 = extractelement <4 x float> %v97, i32 1
  %v107 = extractelement <4 x float> %v99, i32 1
  %v108 = extractelement <4 x float> %v91, i32 2
  %v109 = extractelement <4 x float> %v94, i32 2
  %v110 = extractelement <4 x float> %v97, i32 2
  %v111 = extractelement <4 x float> %v99, i32 2
  %v112 = extractelement <4 x float> %v91, i32 3
  %v113 = extractelement <4 x float> %v94, i32 3
  %v114 = extractelement <4 x float> %v97, i32 3
  %v115 = extractelement <4 x float> %v99, i32 3
  %v116 = fadd nsz arcp contract afn float %v56, %v103
  %v117 = fadd nsz arcp contract afn float %v58, %v107
  %v118 = fadd nsz arcp contract afn float %v60, %v111
  %v119 = fadd nsz arcp contract afn float %v62, %v115
  %v120 = fadd nsz arcp contract afn float %v64, %v100
  %v121 = fadd nsz arcp contract afn float %v66, %v104
  %v122 = fadd nsz arcp contract afn float %v68, %v108
  %v123 = fadd nsz arcp contract afn float %v70, %v112
  %v124 = fadd nsz arcp contract afn float %v72, %v101
  %v125 = fadd nsz arcp contract afn float %v74, %v105
  %v126 = fadd nsz arcp contract afn float %v76, %v109
  %v127 = fadd nsz arcp contract afn float %v78, %v113
  %v128 = fadd nsz arcp contract afn float %v80, %v102
  %v129 = fadd nsz arcp contract afn float %v82, %v106
  %v130 = fadd nsz arcp contract afn float %v84, %v110
  %v131 = fadd nsz arcp contract afn float %v86, %v114
  store float %v116, float* %v1
  store float %v117, float* %v2
  store float %v118, float* %v3
  store float %v119, float* %v4
  store float %v120, float* %v5
  store float %v121, float* %v6
  store float %v122, float* %v7
  store float %v123, float* %v8
  store float %v124, float* %v9
  store float %v125, float* %v10
  store float %v126, float* %v11
  store float %v127, float* %v12
  store float %v128, float* %v13
  store float %v129, float* %v14
  store float %v130, float* %v15
  store float %v131, float* %v16
  br label %loop_footer_0
loop_footer_0:
  br label %loop_latch_0
loop_exit_0:
  %v149 = getelementptr inbounds float, float* %reg_0, i32 0
  %v150 = load float, float* %v149
  %v151 = getelementptr inbounds float, float* %reg_0, i32 1
  %v152 = load float, float* %v151
  %v153 = getelementptr inbounds float, float* %reg_0, i32 2
  %v154 = load float, float* %v153
  %v155 = getelementptr inbounds float, float* %reg_0, i32 3
  %v156 = load float, float* %v155
  %v157 = getelementptr inbounds float, float* %reg_0, i32 4
  %v158 = load float, float* %v157
  %v159 = getelementptr inbounds float, float* %reg_0, i32 5
  %v160 = load float, float* %v159
  %v161 = getelementptr inbounds float, float* %reg_0, i32 6
  %v162 = load float, float* %v161
  %v163 = getelementptr inbounds float, float* %reg_0, i32 7
  %v164 = load float, float* %v163
  %v165 = getelementptr inbounds float, float* %reg_0, i32 8
  %v166 = load float, float* %v165
  %v167 = getelementptr inbounds float, float* %reg_0, i32 9
  %v168 = load float, float* %v167
  %v169 = getelementptr inbounds float, float* %reg_0, i32 10
  %v170 = load float, float* %v169
  %v171 = getelementptr inbounds float, float* %reg_0, i32 11
  %v172 = load float, float* %v171
  %v173 = getelementptr inbounds float, float* %reg_0, i32 12
  %v174 = load float, float* %v173
  %v175 = getelementptr inbounds float, float* %reg_0, i32 13
  %v176 = load float, float* %v175
  %v177 = getelementptr inbounds float, float* %reg_0, i32 14
  %v178 = load float, float* %v177
  %v179 = getelementptr inbounds float, float* %reg_0, i32 15
  %v180 = load float, float* %v179
  %v181 = add nsw i32 %v17, %v19
  %v182 = getelementptr inbounds float, float* %data0, i32 %v181
  %v183 = fadd nsz arcp contract afn float %v150, %v152
  %v184 = fadd nsz arcp contract afn float %v183, %v154
  %v185 = fadd nsz arcp contract afn float %v184, %v156
  %v186 = fadd nsz arcp contract afn float %v185, %v158
  %v187 = fadd nsz arcp contract afn float %v186, %v160
  %v188 = fadd nsz arcp contract afn float %v187, %v162
  %v189 = fadd nsz arcp contract afn float %v188, %v164
  %v190 = fadd nsz arcp contract afn float %v189, %v166
  %v191 = fadd nsz arcp contract afn float %v190, %v168
  %v192 = fadd nsz arcp contract afn float %v191, %v170
  %v193 = fadd nsz arcp contract afn float %v192, %v172
  %v194 = fadd nsz arcp contract afn float %v193, %v174
  %v195 = fadd nsz arcp contract afn float %v194, %v176
  %v196 = fadd nsz arcp contract afn float %v195, %v178
  %v197 = fadd nsz arcp contract afn float %v196, %v180
  store float %v197, float* %v182
  br label %loop_footer_2
loop_footer_2:
  br label %loop_latch_2
loop_exit_2:
  ret void
}
attributes #0 = { alwaysinline nounwind "no-builtins" "no-trapping-math"="true" }

; ---- r[90m_[0m[31m4[0m[90m_[0m[35m4[0m[90m_[0m[35m4[0m[90m_[0m[35m4[0m[90m[0m gs=[1, 1, 1] ----
define void @r_4_4_4_4(float* noalias align 32 %data0, float* noalias align 32 %data1, i32 %core_id) #0
{
  %reg_0 = alloca [64 x float]
  %v0 = getelementptr inbounds float, float* %reg_0, i32 0
  %v1 = getelementptr inbounds float, float* %reg_0, i32 1
  %v2 = getelementptr inbounds float, float* %reg_0, i32 2
  %v3 = getelementptr inbounds float, float* %reg_0, i32 3
  %v4 = getelementptr inbounds float, float* %reg_0, i32 4
  %v5 = getelementptr inbounds float, float* %reg_0, i32 5
  %v6 = getelementptr inbounds float, float* %reg_0, i32 6
  %v7 = getelementptr inbounds float, float* %reg_0, i32 7
  %v8 = getelementptr inbounds float, float* %reg_0, i32 8
  %v9 = getelementptr inbounds float, float* %reg_0, i32 9
  %v10 = getelementptr inbounds float, float* %reg_0, i32 10
  %v11 = getelementptr inbounds float, float* %reg_0, i32 11
  %v12 = getelementptr inbounds float, float* %reg_0, i32 12
  %v13 = getelementptr inbounds float, float* %reg_0, i32 13
  %v14 = getelementptr inbounds float, float* %reg_0, i32 14
  %v15 = getelementptr inbounds float, float* %reg_0, i32 15
  %v16 = getelementptr inbounds float, float* %reg_0, i32 16
  %v17 = getelementptr inbounds float, float* %reg_0, i32 17
  %v18 = getelementptr inbounds float, float* %reg_0, i32 18
  %v19 = getelementptr inbounds float, float* %reg_0, i32 19
  %v20 = getelementptr inbounds float, float* %reg_0, i32 20
  %v21 = getelementptr inbounds float, float* %reg_0, i32 21
  %v22 = getelementptr inbounds float, float* %reg_0, i32 22
  %v23 = getelementptr inbounds float, float* %reg_0, i32 23
  %v24 = getelementptr inbounds float, float* %reg_0, i32 24
  %v25 = getelementptr inbounds float, float* %reg_0, i32 25
  %v26 = getelementptr inbounds float, float* %reg_0, i32 26
  %v27 = getelementptr inbounds float, float* %reg_0, i32 27
  %v28 = getelementptr inbounds float, float* %reg_0, i32 28
  %v29 = getelementptr inbounds float, float* %reg_0, i32 29
  %v30 = getelementptr inbounds float, float* %reg_0, i32 30
  %v31 = getelementptr inbounds float, float* %reg_0, i32 31
  %v32 = getelementptr inbounds float, float* %reg_0, i32 32
  %v33 = getelementptr inbounds float, float* %reg_0, i32 33
  %v34 = getelementptr inbounds float, float* %reg_0, i32 34
  %v35 = getelementptr inbounds float, float* %reg_0, i32 35
  %v36 = getelementptr inbounds float, float* %reg_0, i32 36
  %v37 = getelementptr inbounds float, float* %reg_0, i32 37
  %v38 = getelementptr inbounds float, float* %reg_0, i32 38
  %v39 = getelementptr inbounds float, float* %reg_0, i32 39
  %v40 = getelementptr inbounds float, float* %reg_0, i32 40
  %v41 = getelementptr inbounds float, float* %reg_0, i32 41
  %v42 = getelementptr inbounds float, float* %reg_0, i32 42
  %v43 = getelementptr inbounds float, float* %reg_0, i32 43
  %v44 = getelementptr inbounds float, float* %reg_0, i32 44
  %v45 = getelementptr inbounds float, float* %reg_0, i32 45
  %v46 = getelementptr inbounds float, float* %reg_0, i32 46
  %v47 = getelementptr inbounds float, float* %reg_0, i32 47
  %v48 = getelementptr inbounds float, float* %reg_0, i32 48
  %v49 = getelementptr inbounds float, float* %reg_0, i32 49
  %v50 = getelementptr inbounds float, float* %reg_0, i32 50
  %v51 = getelementptr inbounds float, float* %reg_0, i32 51
  %v52 = getelementptr inbounds float, float* %reg_0, i32 52
  %v53 = getelementptr inbounds float, float* %reg_0, i32 53
  %v54 = getelementptr inbounds float, float* %reg_0, i32 54
  %v55 = getelementptr inbounds float, float* %reg_0, i32 55
  %v56 = getelementptr inbounds float, float* %reg_0, i32 56
  %v57 = getelementptr inbounds float, float* %reg_0, i32 57
  %v58 = getelementptr inbounds float, float* %reg_0, i32 58
  %v59 = getelementptr inbounds float, float* %reg_0, i32 59
  %v60 = getelementptr inbounds float, float* %reg_0, i32 60
  %v61 = getelementptr inbounds float, float* %reg_0, i32 61
  %v62 = getelementptr inbounds float, float* %reg_0, i32 62
  %v63 = getelementptr inbounds float, float* %reg_0, i32 63
  store float 0.0, float* %v0
  store float 0.0, float* %v1
  store float 0.0, float* %v2
  store float 0.0, float* %v3
  store float 0.0, float* %v4
  store float 0.0, float* %v5
  store float 0.0, float* %v6
  store float 0.0, float* %v7
  store float 0.0, float* %v8
  store float 0.0, float* %v9
  store float 0.0, float* %v10
  store float 0.0, float* %v11
  store float 0.0, float* %v12
  store float 0.0, float* %v13
  store float 0.0, float* %v14
  store float 0.0, float* %v15
  store float 0.0, float* %v16
  store float 0.0, float* %v17
  store float 0.0, float* %v18
  store float 0.0, float* %v19
  store float 0.0, float* %v20
  store float 0.0, float* %v21
  store float 0.0, float* %v22
  store float 0.0, float* %v23
  store float 0.0, float* %v24
  store float 0.0, float* %v25
  store float 0.0, float* %v26
  store float 0.0, float* %v27
  store float 0.0, float* %v28
  store float 0.0, float* %v29
  store float 0.0, float* %v30
  store float 0.0, float* %v31
  store float 0.0, float* %v32
  store float 0.0, float* %v33
  store float 0.0, float* %v34
  store float 0.0, float* %v35
  store float 0.0, float* %v36
  store float 0.0, float* %v37
  store float 0.0, float* %v38
  store float 0.0, float* %v39
  store float 0.0, float* %v40
  store float 0.0, float* %v41
  store float 0.0, float* %v42
  store float 0.0, float* %v43
  store float 0.0, float* %v44
  store float 0.0, float* %v45
  store float 0.0, float* %v46
  store float 0.0, float* %v47
  store float 0.0, float* %v48
  store float 0.0, float* %v49
  store float 0.0, float* %v50
  store float 0.0, float* %v51
  store float 0.0, float* %v52
  store float 0.0, float* %v53
  store float 0.0, float* %v54
  store float 0.0, float* %v55
  store float 0.0, float* %v56
  store float 0.0, float* %v57
  store float 0.0, float* %v58
  store float 0.0, float* %v59
  store float 0.0, float* %v60
  store float 0.0, float* %v61
  store float 0.0, float* %v62
  store float 0.0, float* %v63
  br label %loop_entry_0
loop_entry_0:
  br label %loop_latch_0
loop_latch_0:
  %v128 = phi i32 [ 0, %loop_entry_0 ], [ %v128phi, %loop_footer_0 ]
  %v128phi = add i32 %v128, 1
  %v128cmp = icmp ult i32 %v128, 4
  br i1 %v128cmp, label %loop_body_0, label %loop_exit_0
loop_body_0:
  %v129 = getelementptr inbounds float, float* %reg_0, i32 0
  %v130 = load float, float* %v129
  %v131 = getelementptr inbounds float, float* %reg_0, i32 1
  %v132 = load float, float* %v131
  %v133 = getelementptr inbounds float, float* %reg_0, i32 2
  %v134 = load float, float* %v133
  %v135 = getelementptr inbounds float, float* %reg_0, i32 3
  %v136 = load float, float* %v135
  %v137 = getelementptr inbounds float, float* %reg_0, i32 4
  %v138 = load float, float* %v137
  %v139 = getelementptr inbounds float, float* %reg_0, i32 5
  %v140 = load float, float* %v139
  %v141 = getelementptr inbounds float, float* %reg_0, i32 6
  %v142 = load float, float* %v141
  %v143 = getelementptr inbounds float, float* %reg_0, i32 7
  %v144 = load float, float* %v143
  %v145 = getelementptr inbounds float, float* %reg_0, i32 8
  %v146 = load float, float* %v145
  %v147 = getelementptr inbounds float, float* %reg_0, i32 9
  %v148 = load float, float* %v147
  %v149 = getelementptr inbounds float, float* %reg_0, i32 10
  %v150 = load float, float* %v149
  %v151 = getelementptr inbounds float, float* %reg_0, i32 11
  %v152 = load float, float* %v151
  %v153 = getelementptr inbounds float, float* %reg_0, i32 12
  %v154 = load float, float* %v153
  %v155 = getelementptr inbounds float, float* %reg_0, i32 13
  %v156 = load float, float* %v155
  %v157 = getelementptr inbounds float, float* %reg_0, i32 14
  %v158 = load float, float* %v157
  %v159 = getelementptr inbounds float, float* %reg_0, i32 15
  %v160 = load float, float* %v159
  %v161 = getelementptr inbounds float, float* %reg_0, i32 16
  %v162 = load float, float* %v161
  %v163 = getelementptr inbounds float, float* %reg_0, i32 17
  %v164 = load float, float* %v163
  %v165 = getelementptr inbounds float, float* %reg_0, i32 18
  %v166 = load float, float* %v165
  %v167 = getelementptr inbounds float, float* %reg_0, i32 19
  %v168 = load float, float* %v167
  %v169 = getelementptr inbounds float, float* %reg_0, i32 20
  %v170 = load float, float* %v169
  %v171 = getelementptr inbounds float, float* %reg_0, i32 21
  %v172 = load float, float* %v171
  %v173 = getelementptr inbounds float, float* %reg_0, i32 22
  %v174 = load float, float* %v173
  %v175 = getelementptr inbounds float, float* %reg_0, i32 23
  %v176 = load float, float* %v175
  %v177 = getelementptr inbounds float, float* %reg_0, i32 24
  %v178 = load float, float* %v177
  %v179 = getelementptr inbounds float, float* %reg_0, i32 25
  %v180 = load float, float* %v179
  %v181 = getelementptr inbounds float, float* %reg_0, i32 26
  %v182 = load float, float* %v181
  %v183 = getelementptr inbounds float, float* %reg_0, i32 27
  %v184 = load float, float* %v183
  %v185 = getelementptr inbounds float, float* %reg_0, i32 28
  %v186 = load float, float* %v185
  %v187 = getelementptr inbounds float, float* %reg_0, i32 29
  %v188 = load float, float* %v187
  %v189 = getelementptr inbounds float, float* %reg_0, i32 30
  %v190 = load float, float* %v189
  %v191 = getelementptr inbounds float, float* %reg_0, i32 31
  %v192 = load float, float* %v191
  %v193 = getelementptr inbounds float, float* %reg_0, i32 32
  %v194 = load float, float* %v193
  %v195 = getelementptr inbounds float, float* %reg_0, i32 33
  %v196 = load float, float* %v195
  %v197 = getelementptr inbounds float, float* %reg_0, i32 34
  %v198 = load float, float* %v197
  %v199 = getelementptr inbounds float, float* %reg_0, i32 35
  %v200 = load float, float* %v199
  %v201 = getelementptr inbounds float, float* %reg_0, i32 36
  %v202 = load float, float* %v201
  %v203 = getelementptr inbounds float, float* %reg_0, i32 37
  %v204 = load float, float* %v203
  %v205 = getelementptr inbounds float, float* %reg_0, i32 38
  %v206 = load float, float* %v205
  %v207 = getelementptr inbounds float, float* %reg_0, i32 39
  %v208 = load float, float* %v207
  %v209 = getelementptr inbounds float, float* %reg_0, i32 40
  %v210 = load float, float* %v209
  %v211 = getelementptr inbounds float, float* %reg_0, i32 41
  %v212 = load float, float* %v211
  %v213 = getelementptr inbounds float, float* %reg_0, i32 42
  %v214 = load float, float* %v213
  %v215 = getelementptr inbounds float, float* %reg_0, i32 43
  %v216 = load float, float* %v215
  %v217 = getelementptr inbounds float, float* %reg_0, i32 44
  %v218 = load float, float* %v217
  %v219 = getelementptr inbounds float, float* %reg_0, i32 45
  %v220 = load float, float* %v219
  %v221 = getelementptr inbounds float, float* %reg_0, i32 46
  %v222 = load float, float* %v221
  %v223 = getelementptr inbounds float, float* %reg_0, i32 47
  %v224 = load float, float* %v223
  %v225 = getelementptr inbounds float, float* %reg_0, i32 48
  %v226 = load float, float* %v225
  %v227 = getelementptr inbounds float, float* %reg_0, i32 49
  %v228 = load float, float* %v227
  %v229 = getelementptr inbounds float, float* %reg_0, i32 50
  %v230 = load float, float* %v229
  %v231 = getelementptr inbounds float, float* %reg_0, i32 51
  %v232 = load float, float* %v231
  %v233 = getelementptr inbounds float, float* %reg_0, i32 52
  %v234 = load float, float* %v233
  %v235 = getelementptr inbounds float, float* %reg_0, i32 53
  %v236 = load float, float* %v235
  %v237 = getelementptr inbounds float, float* %reg_0, i32 54
  %v238 = load float, float* %v237
  %v239 = getelementptr inbounds float, float* %reg_0, i32 55
  %v240 = load float, float* %v239
  %v241 = getelementptr inbounds float, float* %reg_0, i32 56
  %v242 = load float, float* %v241
  %v243 = getelementptr inbounds float, float* %reg_0, i32 57
  %v244 = load float, float* %v243
  %v245 = getelementptr inbounds float, float* %reg_0, i32 58
  %v246 = load float, float* %v245
  %v247 = getelementptr inbounds float, float* %reg_0, i32 59
  %v248 = load float, float* %v247
  %v249 = getelementptr inbounds float, float* %reg_0, i32 60
  %v250 = load float, float* %v249
  %v251 = getelementptr inbounds float, float* %reg_0, i32 61
  %v252 = load float, float* %v251
  %v253 = getelementptr inbounds float, float* %reg_0, i32 62
  %v254 = load float, float* %v253
  %v255 = getelementptr inbounds float, float* %reg_0, i32 63
  %v256 = load float, float* %v255
  %v257 = mul i32 %v128, 64
  %v258 = add nsw i32 %v257, 4
  %v259 = getelementptr inbounds float, float* %data1, i32 %v258
  %v260 = load <4 x float>, <4 x float>* %v259
  %v261 = add nsw i32 %v257, 8
  %v262 = getelementptr inbounds float, float* %data1, i32 %v261
  %v263 = load <4 x float>, <4 x float>* %v262
  %v264 = add nsw i32 %v257, 12
  %v265 = getelementptr inbounds float, float* %data1, i32 %v264
  %v266 = load <4 x float>, <4 x float>* %v265
  %v267 = add nsw i32 %v257, 16
  %v268 = getelementptr inbounds float, float* %data1, i32 %v267
  %v269 = load <4 x float>, <4 x float>* %v268
  %v270 = add nsw i32 %v257, 20
  %v271 = getelementptr inbounds float, float* %data1, i32 %v270
  %v272 = load <4 x float>, <4 x float>* %v271
  %v273 = add nsw i32 %v257, 24
  %v274 = getelementptr inbounds float, float* %data1, i32 %v273
  %v275 = load <4 x float>, <4 x float>* %v274
  %v276 = add nsw i32 %v257, 28
  %v277 = getelementptr inbounds float, float* %data1, i32 %v276
  %v278 = load <4 x float>, <4 x float>* %v277
  %v279 = add nsw i32 %v257, 32
  %v280 = getelementptr inbounds float, float* %data1, i32 %v279
  %v281 = load <4 x float>, <4 x float>* %v280
  %v282 = add nsw i32 %v257, 36
  %v283 = getelementptr inbounds float, float* %data1, i32 %v282
  %v284 = load <4 x float>, <4 x float>* %v283
  %v285 = add nsw i32 %v257, 40
  %v286 = getelementptr inbounds float, float* %data1, i32 %v285
  %v287 = load <4 x float>, <4 x float>* %v286
  %v288 = add nsw i32 %v257, 44
  %v289 = getelementptr inbounds float, float* %data1, i32 %v288
  %v290 = load <4 x float>, <4 x float>* %v289
  %v291 = add nsw i32 %v257, 48
  %v292 = getelementptr inbounds float, float* %data1, i32 %v291
  %v293 = load <4 x float>, <4 x float>* %v292
  %v294 = add nsw i32 %v257, 52
  %v295 = getelementptr inbounds float, float* %data1, i32 %v294
  %v296 = load <4 x float>, <4 x float>* %v295
  %v297 = add nsw i32 %v257, 56
  %v298 = getelementptr inbounds float, float* %data1, i32 %v297
  %v299 = load <4 x float>, <4 x float>* %v298
  %v300 = add nsw i32 %v257, 60
  %v301 = getelementptr inbounds float, float* %data1, i32 %v300
  %v302 = load <4 x float>, <4 x float>* %v301
  %v303 = getelementptr inbounds float, float* %data1, i32 %v257
  %v304 = load <4 x float>, <4 x float>* %v303
  %v305 = extractelement <4 x float> %v260, i32 0
  %v306 = extractelement <4 x float> %v263, i32 0
  %v307 = extractelement <4 x float> %v266, i32 0
  %v308 = extractelement <4 x float> %v269, i32 0
  %v309 = extractelement <4 x float> %v272, i32 0
  %v310 = extractelement <4 x float> %v275, i32 0
  %v311 = extractelement <4 x float> %v278, i32 0
  %v312 = extractelement <4 x float> %v281, i32 0
  %v313 = extractelement <4 x float> %v284, i32 0
  %v314 = extractelement <4 x float> %v287, i32 0
  %v315 = extractelement <4 x float> %v290, i32 0
  %v316 = extractelement <4 x float> %v293, i32 0
  %v317 = extractelement <4 x float> %v296, i32 0
  %v318 = extractelement <4 x float> %v299, i32 0
  %v319 = extractelement <4 x float> %v302, i32 0
  %v320 = extractelement <4 x float> %v304, i32 0
  %v321 = extractelement <4 x float> %v260, i32 1
  %v322 = extractelement <4 x float> %v263, i32 1
  %v323 = extractelement <4 x float> %v266, i32 1
  %v324 = extractelement <4 x float> %v269, i32 1
  %v325 = extractelement <4 x float> %v272, i32 1
  %v326 = extractelement <4 x float> %v275, i32 1
  %v327 = extractelement <4 x float> %v278, i32 1
  %v328 = extractelement <4 x float> %v281, i32 1
  %v329 = extractelement <4 x float> %v284, i32 1
  %v330 = extractelement <4 x float> %v287, i32 1
  %v331 = extractelement <4 x float> %v290, i32 1
  %v332 = extractelement <4 x float> %v293, i32 1
  %v333 = extractelement <4 x float> %v296, i32 1
  %v334 = extractelement <4 x float> %v299, i32 1
  %v335 = extractelement <4 x float> %v302, i32 1
  %v336 = extractelement <4 x float> %v304, i32 1
  %v337 = extractelement <4 x float> %v260, i32 2
  %v338 = extractelement <4 x float> %v263, i32 2
  %v339 = extractelement <4 x float> %v266, i32 2
  %v340 = extractelement <4 x float> %v269, i32 2
  %v341 = extractelement <4 x float> %v272, i32 2
  %v342 = extractelement <4 x float> %v275, i32 2
  %v343 = extractelement <4 x float> %v278, i32 2
  %v344 = extractelement <4 x float> %v281, i32 2
  %v345 = extractelement <4 x float> %v284, i32 2
  %v346 = extractelement <4 x float> %v287, i32 2
  %v347 = extractelement <4 x float> %v290, i32 2
  %v348 = extractelement <4 x float> %v293, i32 2
  %v349 = extractelement <4 x float> %v296, i32 2
  %v350 = extractelement <4 x float> %v299, i32 2
  %v351 = extractelement <4 x float> %v302, i32 2
  %v352 = extractelement <4 x float> %v304, i32 2
  %v353 = extractelement <4 x float> %v260, i32 3
  %v354 = extractelement <4 x float> %v263, i32 3
  %v355 = extractelement <4 x float> %v266, i32 3
  %v356 = extractelement <4 x float> %v269, i32 3
  %v357 = extractelement <4 x float> %v272, i32 3
  %v358 = extractelement <4 x float> %v275, i32 3
  %v359 = extractelement <4 x float> %v278, i32 3
  %v360 = extractelement <4 x float> %v281, i32 3
  %v361 = extractelement <4 x float> %v284, i32 3
  %v362 = extractelement <4 x float> %v287, i32 3
  %v363 = extractelement <4 x float> %v290, i32 3
  %v364 = extractelement <4 x float> %v293, i32 3
  %v365 = extractelement <4 x float> %v296, i32 3
  %v366 = extractelement <4 x float> %v299, i32 3
  %v367 = extractelement <4 x float> %v302, i32 3
  %v368 = extractelement <4 x float> %v304, i32 3
  %v369 = fadd nsz arcp contract afn float %v130, %v320
  %v370 = fadd nsz arcp contract afn float %v132, %v336
  %v371 = fadd nsz arcp contract afn float %v134, %v352
  %v372 = fadd nsz arcp contract afn float %v136, %v368
  %v373 = fadd nsz arcp contract afn float %v138, %v305
  %v374 = fadd nsz arcp contract afn float %v140, %v321
  %v375 = fadd nsz arcp contract afn float %v142, %v337
  %v376 = fadd nsz arcp contract afn float %v144, %v353
  %v377 = fadd nsz arcp contract afn float %v146, %v306
  %v378 = fadd nsz arcp contract afn float %v148, %v322
  %v379 = fadd nsz arcp contract afn float %v150, %v338
  %v380 = fadd nsz arcp contract afn float %v152, %v354
  %v381 = fadd nsz arcp contract afn float %v154, %v307
  %v382 = fadd nsz arcp contract afn float %v156, %v323
  %v383 = fadd nsz arcp contract afn float %v158, %v339
  %v384 = fadd nsz arcp contract afn float %v160, %v355
  %v385 = fadd nsz arcp contract afn float %v162, %v308
  %v386 = fadd nsz arcp contract afn float %v164, %v324
  %v387 = fadd nsz arcp contract afn float %v166, %v340
  %v388 = fadd nsz arcp contract afn float %v168, %v356
  %v389 = fadd nsz arcp contract afn float %v170, %v309
  %v390 = fadd nsz arcp contract afn float %v172, %v325
  %v391 = fadd nsz arcp contract afn float %v174, %v341
  %v392 = fadd nsz arcp contract afn float %v176, %v357
  %v393 = fadd nsz arcp contract afn float %v178, %v310
  %v394 = fadd nsz arcp contract afn float %v180, %v326
  %v395 = fadd nsz arcp contract afn float %v182, %v342
  %v396 = fadd nsz arcp contract afn float %v184, %v358
  %v397 = fadd nsz arcp contract afn float %v186, %v311
  %v398 = fadd nsz arcp contract afn float %v188, %v327
  %v399 = fadd nsz arcp contract afn float %v190, %v343
  %v400 = fadd nsz arcp contract afn float %v192, %v359
  %v401 = fadd nsz arcp contract afn float %v194, %v312
  %v402 = fadd nsz arcp contract afn float %v196, %v328
  %v403 = fadd nsz arcp contract afn float %v198, %v344
  %v404 = fadd nsz arcp contract afn float %v200, %v360
  %v405 = fadd nsz arcp contract afn float %v202, %v313
  %v406 = fadd nsz arcp contract afn float %v204, %v329
  %v407 = fadd nsz arcp contract afn float %v206, %v345
  %v408 = fadd nsz arcp contract afn float %v208, %v361
  %v409 = fadd nsz arcp contract afn float %v210, %v314
  %v410 = fadd nsz arcp contract afn float %v212, %v330
  %v411 = fadd nsz arcp contract afn float %v214, %v346
  %v412 = fadd nsz arcp contract afn float %v216, %v362
  %v413 = fadd nsz arcp contract afn float %v218, %v315
  %v414 = fadd nsz arcp contract afn float %v220, %v331
  %v415 = fadd nsz arcp contract afn float %v222, %v347
  %v416 = fadd nsz arcp contract afn float %v224, %v363
  %v417 = fadd nsz arcp contract afn float %v226, %v316
  %v418 = fadd nsz arcp contract afn float %v228, %v332
  %v419 = fadd nsz arcp contract afn float %v230, %v348
  %v420 = fadd nsz arcp contract afn float %v232, %v364
  %v421 = fadd nsz arcp contract afn float %v234, %v317
  %v422 = fadd nsz arcp contract afn float %v236, %v333
  %v423 = fadd nsz arcp contract afn float %v238, %v349
  %v424 = fadd nsz arcp contract afn float %v240, %v365
  %v425 = fadd nsz arcp contract afn float %v242, %v318
  %v426 = fadd nsz arcp contract afn float %v244, %v334
  %v427 = fadd nsz arcp contract afn float %v246, %v350
  %v428 = fadd nsz arcp contract afn float %v248, %v366
  %v429 = fadd nsz arcp contract afn float %v250, %v319
  %v430 = fadd nsz arcp contract afn float %v252, %v335
  %v431 = fadd nsz arcp contract afn float %v254, %v351
  %v432 = fadd nsz arcp contract afn float %v256, %v367
  store float %v369, float* %v0
  store float %v370, float* %v1
  store float %v371, float* %v2
  store float %v372, float* %v3
  store float %v373, float* %v4
  store float %v374, float* %v5
  store float %v375, float* %v6
  store float %v376, float* %v7
  store float %v377, float* %v8
  store float %v378, float* %v9
  store float %v379, float* %v10
  store float %v380, float* %v11
  store float %v381, float* %v12
  store float %v382, float* %v13
  store float %v383, float* %v14
  store float %v384, float* %v15
  store float %v385, float* %v16
  store float %v386, float* %v17
  store float %v387, float* %v18
  store float %v388, float* %v19
  store float %v389, float* %v20
  store float %v390, float* %v21
  store float %v391, float* %v22
  store float %v392, float* %v23
  store float %v393, float* %v24
  store float %v394, float* %v25
  store float %v395, float* %v26
  store float %v396, float* %v27
  store float %v397, float* %v28
  store float %v398, float* %v29
  store float %v399, float* %v30
  store float %v400, float* %v31
  store float %v401, float* %v32
  store float %v402, float* %v33
  store float %v403, float* %v34
  store float %v404, float* %v35
  store float %v405, float* %v36
  store float %v406, float* %v37
  store float %v407, float* %v38
  store float %v408, float* %v39
  store float %v409, float* %v40
  store float %v410, float* %v41
  store float %v411, float* %v42
  store float %v412, float* %v43
  store float %v413, float* %v44
  store float %v414, float* %v45
  store float %v415, float* %v46
  store float %v416, float* %v47
  store float %v417, float* %v48
  store float %v418, float* %v49
  store float %v419, float* %v50
  store float %v420, float* %v51
  store float %v421, float* %v52
  store float %v422, float* %v53
  store float %v423, float* %v54
  store float %v424, float* %v55
  store float %v425, float* %v56
  store float %v426, float* %v57
  store float %v427, float* %v58
  store float %v428, float* %v59
  store float %v429, float* %v60
  store float %v430, float* %v61
  store float %v431, float* %v62
  store float %v432, float* %v63
  br label %loop_footer_0
loop_footer_0:
  br label %loop_latch_0
loop_exit_0:
  %v498 = getelementptr inbounds float, float* %reg_0, i32 0
  %v499 = load float, float* %v498
  %v500 = getelementptr inbounds float, float* %reg_0, i32 1
  %v501 = load float, float* %v500
  %v502 = getelementptr inbounds float, float* %reg_0, i32 2
  %v503 = load float, float* %v502
  %v504 = getelementptr inbounds float, float* %reg_0, i32 3
  %v505 = load float, float* %v504
  %v506 = getelementptr inbounds float, float* %reg_0, i32 4
  %v507 = load float, float* %v506
  %v508 = getelementptr inbounds float, float* %reg_0, i32 5
  %v509 = load float, float* %v508
  %v510 = getelementptr inbounds float, float* %reg_0, i32 6
  %v511 = load float, float* %v510
  %v512 = getelementptr inbounds float, float* %reg_0, i32 7
  %v513 = load float, float* %v512
  %v514 = getelementptr inbounds float, float* %reg_0, i32 8
  %v515 = load float, float* %v514
  %v516 = getelementptr inbounds float, float* %reg_0, i32 9
  %v517 = load float, float* %v516
  %v518 = getelementptr inbounds float, float* %reg_0, i32 10
  %v519 = load float, float* %v518
  %v520 = getelementptr inbounds float, float* %reg_0, i32 11
  %v521 = load float, float* %v520
  %v522 = getelementptr inbounds float, float* %reg_0, i32 12
  %v523 = load float, float* %v522
  %v524 = getelementptr inbounds float, float* %reg_0, i32 13
  %v525 = load float, float* %v524
  %v526 = getelementptr inbounds float, float* %reg_0, i32 14
  %v527 = load float, float* %v526
  %v528 = getelementptr inbounds float, float* %reg_0, i32 15
  %v529 = load float, float* %v528
  %v530 = getelementptr inbounds float, float* %reg_0, i32 16
  %v531 = load float, float* %v530
  %v532 = getelementptr inbounds float, float* %reg_0, i32 17
  %v533 = load float, float* %v532
  %v534 = getelementptr inbounds float, float* %reg_0, i32 18
  %v535 = load float, float* %v534
  %v536 = getelementptr inbounds float, float* %reg_0, i32 19
  %v537 = load float, float* %v536
  %v538 = getelementptr inbounds float, float* %reg_0, i32 20
  %v539 = load float, float* %v538
  %v540 = getelementptr inbounds float, float* %reg_0, i32 21
  %v541 = load float, float* %v540
  %v542 = getelementptr inbounds float, float* %reg_0, i32 22
  %v543 = load float, float* %v542
  %v544 = getelementptr inbounds float, float* %reg_0, i32 23
  %v545 = load float, float* %v544
  %v546 = getelementptr inbounds float, float* %reg_0, i32 24
  %v547 = load float, float* %v546
  %v548 = getelementptr inbounds float, float* %reg_0, i32 25
  %v549 = load float, float* %v548
  %v550 = getelementptr inbounds float, float* %reg_0, i32 26
  %v551 = load float, float* %v550
  %v552 = getelementptr inbounds float, float* %reg_0, i32 27
  %v553 = load float, float* %v552
  %v554 = getelementptr inbounds float, float* %reg_0, i32 28
  %v555 = load float, float* %v554
  %v556 = getelementptr inbounds float, float* %reg_0, i32 29
  %v557 = load float, float* %v556
  %v558 = getelementptr inbounds float, float* %reg_0, i32 30
  %v559 = load float, float* %v558
  %v560 = getelementptr inbounds float, float* %reg_0, i32 31
  %v561 = load float, float* %v560
  %v562 = getelementptr inbounds float, float* %reg_0, i32 32
  %v563 = load float, float* %v562
  %v564 = getelementptr inbounds float, float* %reg_0, i32 33
  %v565 = load float, float* %v564
  %v566 = getelementptr inbounds float, float* %reg_0, i32 34
  %v567 = load float, float* %v566
  %v568 = getelementptr inbounds float, float* %reg_0, i32 35
  %v569 = load float, float* %v568
  %v570 = getelementptr inbounds float, float* %reg_0, i32 36
  %v571 = load float, float* %v570
  %v572 = getelementptr inbounds float, float* %reg_0, i32 37
  %v573 = load float, float* %v572
  %v574 = getelementptr inbounds float, float* %reg_0, i32 38
  %v575 = load float, float* %v574
  %v576 = getelementptr inbounds float, float* %reg_0, i32 39
  %v577 = load float, float* %v576
  %v578 = getelementptr inbounds float, float* %reg_0, i32 40
  %v579 = load float, float* %v578
  %v580 = getelementptr inbounds float, float* %reg_0, i32 41
  %v581 = load float, float* %v580
  %v582 = getelementptr inbounds float, float* %reg_0, i32 42
  %v583 = load float, float* %v582
  %v584 = getelementptr inbounds float, float* %reg_0, i32 43
  %v585 = load float, float* %v584
  %v586 = getelementptr inbounds float, float* %reg_0, i32 44
  %v587 = load float, float* %v586
  %v588 = getelementptr inbounds float, float* %reg_0, i32 45
  %v589 = load float, float* %v588
  %v590 = getelementptr inbounds float, float* %reg_0, i32 46
  %v591 = load float, float* %v590
  %v592 = getelementptr inbounds float, float* %reg_0, i32 47
  %v593 = load float, float* %v592
  %v594 = getelementptr inbounds float, float* %reg_0, i32 48
  %v595 = load float, float* %v594
  %v596 = getelementptr inbounds float, float* %reg_0, i32 49
  %v597 = load float, float* %v596
  %v598 = getelementptr inbounds float, float* %reg_0, i32 50
  %v599 = load float, float* %v598
  %v600 = getelementptr inbounds float, float* %reg_0, i32 51
  %v601 = load float, float* %v600
  %v602 = getelementptr inbounds float, float* %reg_0, i32 52
  %v603 = load float, float* %v602
  %v604 = getelementptr inbounds float, float* %reg_0, i32 53
  %v605 = load float, float* %v604
  %v606 = getelementptr inbounds float, float* %reg_0, i32 54
  %v607 = load float, float* %v606
  %v608 = getelementptr inbounds float, float* %reg_0, i32 55
  %v609 = load float, float* %v608
  %v610 = getelementptr inbounds float, float* %reg_0, i32 56
  %v611 = load float, float* %v610
  %v612 = getelementptr inbounds float, float* %reg_0, i32 57
  %v613 = load float, float* %v612
  %v614 = getelementptr inbounds float, float* %reg_0, i32 58
  %v615 = load float, float* %v614
  %v616 = getelementptr inbounds float, float* %reg_0, i32 59
  %v617 = load float, float* %v616
  %v618 = getelementptr inbounds float, float* %reg_0, i32 60
  %v619 = load float, float* %v618
  %v620 = getelementptr inbounds float, float* %reg_0, i32 61
  %v621 = load float, float* %v620
  %v622 = getelementptr inbounds float, float* %reg_0, i32 62
  %v623 = load float, float* %v622
  %v624 = getelementptr inbounds float, float* %reg_0, i32 63
  %v625 = load float, float* %v624
  %v626 = getelementptr inbounds float, float* %data0, i32 0
  %v627 = fadd nsz arcp contract afn float %v499, %v501
  %v628 = fadd nsz arcp contract afn float %v627, %v503
  %v629 = fadd nsz arcp contract afn float %v628, %v505
  %v630 = fadd nsz arcp contract afn float %v629, %v507
  %v631 = fadd nsz arcp contract afn float %v630, %v509
  %v632 = fadd nsz arcp contract afn float %v631, %v511
  %v633 = fadd nsz arcp contract afn float %v632, %v513
  %v634 = fadd nsz arcp contract afn float %v633, %v515
  %v635 = fadd nsz arcp contract afn float %v634, %v517
  %v636 = fadd nsz arcp contract afn float %v635, %v519
  %v637 = fadd nsz arcp contract afn float %v636, %v521
  %v638 = fadd nsz arcp contract afn float %v637, %v523
  %v639 = fadd nsz arcp contract afn float %v638, %v525
  %v640 = fadd nsz arcp contract afn float %v639, %v527
  %v641 = fadd nsz arcp contract afn float %v640, %v529
  %v642 = fadd nsz arcp contract afn float %v641, %v531
  %v643 = fadd nsz arcp contract afn float %v642, %v533
  %v644 = fadd nsz arcp contract afn float %v643, %v535
  %v645 = fadd nsz arcp contract afn float %v644, %v537
  %v646 = fadd nsz arcp contract afn float %v645, %v539
  %v647 = fadd nsz arcp contract afn float %v646, %v541
  %v648 = fadd nsz arcp contract afn float %v647, %v543
  %v649 = fadd nsz arcp contract afn float %v648, %v545
  %v650 = fadd nsz arcp contract afn float %v649, %v547
  %v651 = fadd nsz arcp contract afn float %v650, %v549
  %v652 = fadd nsz arcp contract afn float %v651, %v551
  %v653 = fadd nsz arcp contract afn float %v652, %v553
  %v654 = fadd nsz arcp contract afn float %v653, %v555
  %v655 = fadd nsz arcp contract afn float %v654, %v557
  %v656 = fadd nsz arcp contract afn float %v655, %v559
  %v657 = fadd nsz arcp contract afn float %v656, %v561
  %v658 = fadd nsz arcp contract afn float %v657, %v563
  %v659 = fadd nsz arcp contract afn float %v658, %v565
  %v660 = fadd nsz arcp contract afn float %v659, %v567
  %v661 = fadd nsz arcp contract afn float %v660, %v569
  %v662 = fadd nsz arcp contract afn float %v661, %v571
  %v663 = fadd nsz arcp contract afn float %v662, %v573
  %v664 = fadd nsz arcp contract afn float %v663, %v575
  %v665 = fadd nsz arcp contract afn float %v664, %v577
  %v666 = fadd nsz arcp contract afn float %v665, %v579
  %v667 = fadd nsz arcp contract afn float %v666, %v581
  %v668 = fadd nsz arcp contract afn float %v667, %v583
  %v669 = fadd nsz arcp contract afn float %v668, %v585
  %v670 = fadd nsz arcp contract afn float %v669, %v587
  %v671 = fadd nsz arcp contract afn float %v670, %v589
  %v672 = fadd nsz arcp contract afn float %v671, %v591
  %v673 = fadd nsz arcp contract afn float %v672, %v593
  %v674 = fadd nsz arcp contract afn float %v673, %v595
  %v675 = fadd nsz arcp contract afn float %v674, %v597
  %v676 = fadd nsz arcp contract afn float %v675, %v599
  %v677 = fadd nsz arcp contract afn float %v676, %v601
  %v678 = fadd nsz arcp contract afn float %v677, %v603
  %v679 = fadd nsz arcp contract afn float %v678, %v605
  %v680 = fadd nsz arcp contract afn float %v679, %v607
  %v681 = fadd nsz arcp contract afn float %v680, %v609
  %v682 = fadd nsz arcp contract afn float %v681, %v611
  %v683 = fadd nsz arcp contract afn float %v682, %v613
  %v684 = fadd nsz arcp contract afn float %v683, %v615
  %v685 = fadd nsz arcp contract afn float %v684, %v617
  %v686 = fadd nsz arcp contract afn float %v685, %v619
  %v687 = fadd nsz arcp contract afn float %v686, %v621
  %v688 = fadd nsz arcp contract afn float %v687, %v623
  %v689 = fadd nsz arcp contract afn float %v688, %v625
  store float %v689, float* %v626
  ret void
}
attributes #0 = { alwaysinline nounwind "no-builtins" "no-trapping-math"="true" }