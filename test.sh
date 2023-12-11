nvcc gcn.cu -o gcn arch=sm_89

#运行比赛代码
## for correctness

#./gcn 128 16 graph/web-stanford.txt embedding/web-stanford_F0_128.bin weight/web-stanford_F0_128_F1_16.bin
output_stanford=($(./gcn 128 16 ./graph/web-stanford.txt ./embedding/web-stanford_F0_128.bin ./weight/web-stanford_F0_128_F1_16.bin))
len_stanford=${#output_stanford[@]}
verify_stanford=${output_stanford[len_stanford-2]}
time_stanford=${output_stanford[len_stanford-1]}

echo "verify stanford"
echo $verify_stanford
echo $time_stanford

#./gcn 128 16 graph/com-dblp.txt embedding/dblp_F0_128.bin weight/dblp_F0_128_F1_16.bin
output_dblp=($(./gcn 128 16 ./graph/com-dblp.txt ./embedding/dblp_F0_128.bin ./weight/dblp_F0_128_F1_16.bin))
len_dblp=${#output_dblp[@]}
verify_dblp=${output_dblp[len_dblp-2]}
time_dblp=${output_dblp[len_dblp-1]}

echo "verify dblp"
echo $verify_dblp
echo $time_dblp


#./gcn 128 16 graph/ak_2010.txt embedding/ak_2010_F0_128.bin weight/ak_2010_F0_128_F1_16.bin
output_ak=($(./gcn 128 16 ./graph/ak_2010.txt ./embedding/ak_2010_F0_128.bin ./weight/ak_2010_F0_128_F1_16.bin))
len_ak=${#output_ak[@]}
verify_ak=${output_ak[len_ak-2]}
time_ak=${output_ak[len_ak-1]}

echo "verify ak"
echo $verify_ak
echo $time_ak

time_all=`echo "$time_stanford + $time_dblp + $time_ak"|bc`

# # echo $num_verify
# # echo $time2
if [ $verify_stanford == "True" ] && [ $verify_dblp == "True" ] && [ $verify_ak == "True" ]; then
   echo "Correctness OK!!!"
   echo "all time = " $time_all "ms" 
else
   echo "Wrong !!!"
   echo "verify stanford = " $verify_stanford
   echo "verify dblp = "$verify_dblp
   echo "verify ak  = " $verify_ak
   time_all=100000000
fi

#to evaluate performance, using average consumed time (5 times)
echo "Total time for scoring: $time_all" 