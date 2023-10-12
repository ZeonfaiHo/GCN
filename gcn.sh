#!/bin/bash
#SBATCH -J gcn
#SBATCH -p ty_xd
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=dcu:1

#load env
module switch compiler/dtk/22.10.1 compiler/dtk/23.04

# echo ${SLURM_JOB_NAME}
# submitFile=${SLURM_JOB_NAME:10} # 分割字符串，获取第11个字符以后的子字符串

#读取提交文件
# cp ~/${submitFile}.zip ./
# unzip -o ${submitFile}.zip -d ${submitFile}
# cd ${submitFile}

#CUDA code to HIP code
#hipify-perl ${submitFile}.cu > gcn_verify.cpp
hipify-perl gcn_verify.cu > gcn_verify.cpp
#to compile HIP code
hipcc gcn_verify.cpp -o gcn_verify

#第一行输出值在10e-6之内
#To check the correctness, the error should be less than 10e-6 compared with the results on GPU
stde=10^-6

#运行baseline, 第一行输出为结果 ，第二行为时间
hipify-perl gcn.cu > gcn_baseline.cpp
hipcc gcn_baseline.cpp -o gcn_baseline
#保存结果
output_var=($(./gcn_baseline 64 16 graph/1024_example_graph.txt embedding/1024.bin weight/W_64_16.bin))
len=${#output_var[@]}
num_baseline=${output_var[len-2]}
time1=${output_var[len-1]}
# echo $num_baseline
# echo $time1

#运行比赛代码
## for correctness
output_var=($(./gcn_verify 64 16 graph/1024_example_graph.txt embedding/1024.bin weight/W_64_16.bin))
len=${#output_var[@]}
num_verify=${output_var[len-2]}
time2=${output_var[len-1]}
# echo $num_verify
# echo $time2
if [ "$num_verify" == "$num_baseline" ]; then
   echo $num_baseline
   echo "Correctness OK!!!"
else
   echo "Wrong !!!"
   time2=-100
fi

#to evaluate performance, using average consumed time of 10-200 timestep (10 times)
echo "Total time for scoring: $time2" > ./rank.txt




