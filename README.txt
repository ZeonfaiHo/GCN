代码说明：gcn.cu 为比赛团队上传代码 slurm.sh为自评测脚本代码

提交材料：
	1、程序源代码。
		格式及命名：
			提交gcu.cu文件
		内容：
			只允许修改gcn.cu中的gcn()函数
	2、技术报告文档：
		报告内容包括但不限于基本算法介绍、设计思路和方法、算法优化、详细算法设计与实现、实验结果与分析、程序代码模块说明、详细程序代码编译说明、详细代码运行使用说明等。

	
测试方式：
	编译：
	hipify-perl gcn.cu > gcn_baseline.cpp
	hipcc gcn_baseline.cpp -o gcn_baseline
	执行：
	可执行程序需接收5个参数，分别为：
		输入顶点特征长度F0，第一层顶点特征长度F1，图结构文件名，输入顶点特征矩阵文件名，第一层权重矩阵文件名
		./gcn 128 16 graph/web-stanford_nodes_281903_edges_1992636_core_71.txt embedding/web-stanford_F0_128.bin weight/web-stanford_F0_128_F1_16.bin
        		./gcn 128 16 graph/com-dblp_nodes_317080_edges_1049866_core_113.txt embedding/dblp_F0_128.bin weight/dblp_F0_128_F1_16.bin
       		./gcn 128 16 graph/ak_2010.txt embedding/ak_2010_F0_128.bin weight/ak_2010_F0_128_F1_16.bin
	
	具体参考gcn.cpp

	输入的文件名均包含相对路径

	图结构文件为文本文件，第一行两个整数分别为图顶点数量（v_num）和边数量，之后每一行为一条边，格式为“源顶点id 目的顶点id”，顶点id从0开始
	图结构文件中包含自环（即有边“i i”），包含反向边（即同时有边“i j”和边“j i”）

	输入顶点特征矩阵文件为二进制文件，包含v_num*F0个float32，大小为v_num*F0*4字节
	权重矩阵文件为二进制文件，包含F0*F1个float32，大小为F0*F1*4字节
	
	只统计核函数和softmax()的计算时间

	具体参考gcn.cu 中 GCN()函数

评分方式：
	对于结果正确的队伍，将结合其性能计算排名。