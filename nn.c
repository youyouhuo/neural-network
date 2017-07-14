#include"he.c"

// #include”total.c”
	// 配置网络的结构：输入为35维  第一层为30  第二层为30 输入层为26
	// int dataAmount[4]={35,30,30,26};


int dataAmount[4]={13,10,10,3};

int flag=0; // bigger than 0 is to trainning

	//配置网络训练的步长alpha=0.01

// 1

int totalTrainTimes=30000;
double alpha=0.01;
double sigmoid(double input){
	double result=0.0;
	result=exp(-input);
	result=1.0/(1+result);
	return result;
}
	//计算两个数相乘
double times(double first,double second){
	return first*second;
}
	//计算sigmoid函数的导数
double derivateSigmoid(double input){
	double result=sigmoid(input);
	return result*(1-result);
}
	//求解反向传播中F_dot矩阵
double **getFdot(double (*actaFun)(double),double **input,int row,int col){
	double **result=getMemory(row,row);
	for(int i=0;i<row;++i){
		result[i][i]=actaFun(input[i][col]);
	}
	return result;
}
	//计算矩阵与数的操作
double timesOrDevMatrix(double(*func)(double,double),double **input,double number,int row,int col){
	for(int i=0;i<row;++i){
		for(int j=0;j<col;++j){
			input[i][j]=func(number,input[i][j]);
		}
	}
}
	//随机初始化矩阵 取值为0到1之间
double **randMatrix(int row,int col){
	double **result=getMemory(row,col);
	for(int i=0;i<row;i++){
		for(int j=0;j<col;++j){
			result[i][j] = rand() / (double)(RAND_MAX);
		}
	}
	return result;
}
	//使用随机梯度下降进行网络参数的训练
double trainNN(double **w1,double **w2,double **w3,double **b1,double **b2,double **b3,double **a0,double**t,int trainCol){
	double totalError=0.0;
	//计算 w1*a0
	double **n1=getTimes(w1,a0,dataAmount[1],dataAmount[0],trainCol);  
	//计算得到 w1*a0+b1
	getAddOrMinus(add,n1,b1,dataAmount[1],trainCol);
	//计算反向传播中使用的F_dot(n1)
	double **fdot_n1=getFdot(derivateSigmoid,n1,dataAmount[1],trainCol);
	//计算 sigmoid(w1*a0+b1)
	activateMatrix(sigmoid,n1,dataAmount[1],trainCol);
	// a1=n1;
	//计算a1的转置
	double **a1_transfer=getTransfer(n1,dataAmount[1],trainCol);
	//计算w2*a1
	double **n2=getTimes(w2,n1,dataAmount[2],dataAmount[1],trainCol);
	freeMemory(n1,dataAmount[1],trainCol);
	//计算w2*a1+b2
	getAddOrMinus(add,n2,b2,dataAmount[2],trainCol);
	//计算反向传播中使用的F_dot(n2)
	double **fdot_n2=getFdot(derivateSigmoid,n2,dataAmount[2],trainCol);
	//计算 sigmoid(w*a1+b2)
	activateMatrix(sigmoid,n2,dataAmount[2],trainCol);
	//a2=n2
	//计算a2的转置
	double **a2_transfer=getTransfer(n2,dataAmount[2],trainCol);
	//计算 w3*a2
	double **n3=getTimes(w3,n2,dataAmount[3],dataAmount[2],trainCol);
	freeMemory(n2,dataAmount[2],trainCol);
	//计算 w3*a2+b3
	getAddOrMinus(add,n3,b3,dataAmount[3],trainCol);
	//计算反向传播中使用的F_dot(n3)
	double **fdot_n3=getFdot(derivateSigmoid,n3,dataAmount[3],trainCol);
	//计算 sigmoid(w3*a2+b3)
	activateMatrix(sigmoid,n3,dataAmount[3],trainCol);
	//a3=n3
	//计算t-a3
	for(int i=0;i<dataAmount[3];++i){
		for(int j=0;j<trainCol;++j){
			n3[i][j]=t[i][j]-n3[i][j];
			totalError+=fabs(n3[i][j]);
		}
	}
	//计算-2*F_dot(n3)
	timesOrDevMatrix(times,fdot_n3,-2.0,dataAmount[3],dataAmount[3]);
	//计算s3=-2*F_dot(n3)*（t-a3)
	double **s3=getTimes(fdot_n3,n3,dataAmount[3],dataAmount[3],trainCol);
	freeMemory(n3,dataAmount[3],trainCol);
	freeMemory(fdot_n3,dataAmount[3],dataAmount[3]);
	//计算w3的转置
	double **w3_transfer=getTransfer(w3,dataAmount[3],dataAmount[2]);
	//计算F_dot(n2)*w3_transfer
	double **fdot_n2_times_w3_transfer=getTimes(fdot_n2,w3_transfer,dataAmount[2],dataAmount[2],dataAmount[3]);
	//计算s2=F_dot(n2)*w3_transfer*s3
	double**s2=getTimes(fdot_n2_times_w3_transfer,s3,dataAmount[2],dataAmount[3],trainCol);
	freeMemory(w3_transfer,dataAmount[2],dataAmount[3]);
	freeMemory(fdot_n2,dataAmount[2],dataAmount[2]);
	freeMemory(fdot_n2_times_w3_transfer,dataAmount[2],dataAmount[3]);
	//计算w2的转置
	double **w2_transfer=getTransfer(w2,dataAmount[2],dataAmount[1]);
	//计算F_dot(n1)*w2_transfer
	double **fdot_n1_times_w2_transfer=getTimes(fdot_n1,w2_transfer,dataAmount[1],dataAmount[1],dataAmount[2]);
	//计算s1=F_dot(n1)*w2_transfer*s2
	double **s1=getTimes(fdot_n1_times_w2_transfer,s2,dataAmount[1],dataAmount[2],trainCol);
	freeMemory(w2_transfer,dataAmount[1],dataAmount[2]);
	freeMemory(fdot_n1,dataAmount[1],dataAmount[1]);
	freeMemory(fdot_n1_times_w2_transfer,dataAmount[1],dataAmount[2]);
	//计算a0的转置
	double **a0_transfer=getTransfer(a0,dataAmount[0],trainCol);
	//计算alpha*s
	timesOrDevMatrix(times,s1,alpha,dataAmount[1],trainCol);
	timesOrDevMatrix(times,s2,alpha,dataAmount[2],trainCol);
	timesOrDevMatrix(times,s3,alpha,dataAmount[3],trainCol);
	//计算s1*a0_transfer
	double **s1_times_a0_transfer=getTimes(s1,a0_transfer,dataAmount[1],trainCol,dataAmount[0]);
	//计算w1=w1-alpha*s1*a0_transfer
	getAddOrMinus(minus,w1,s1_times_a0_transfer,dataAmount[1],dataAmount[0]);
	//计算b1=b1-alpha*s1
	getAddOrMinus(minus,b1,s1,dataAmount[1],trainCol);
	freeMemory(s1_times_a0_transfer,dataAmount[1],dataAmount[0]);
	freeMemory(s1,dataAmount[1],trainCol);
	freeMemory(a0_transfer,trainCol,dataAmount[0]);
	//计算s2*a1_transfer
	double **s2_times_a1_transfer=getTimes(s2,a1_transfer,dataAmount[2],trainCol,dataAmount[1]);
	//计算w2=w2-alpha*s2*a1_transfer
	getAddOrMinus(minus,w2,s2_times_a1_transfer,dataAmount[2],dataAmount[1]);
	//计算b2=b2-alpha*s2
	getAddOrMinus(minus,b2,s2,dataAmount[2],trainCol);
	freeMemory(s2_times_a1_transfer,dataAmount[2],dataAmount[1]);
	freeMemory(s2,dataAmount[2],trainCol);
	freeMemory(a1_transfer,trainCol,dataAmount[1]);
	//计算s3*a2_transfer
	double **s3_times_a2_transfer=getTimes(s3,a2_transfer,dataAmount[3],trainCol,dataAmount[2]);
	//计算w3=w3-alpha*s3*a1_transfer
	getAddOrMinus(minus,w3,s3_times_a2_transfer,dataAmount[3],dataAmount[2]);
	//计算b3=b3-alpha*s3
	getAddOrMinus(minus,b3,s3,dataAmount[3],trainCol);
	freeMemory(s3_times_a2_transfer,dataAmount[3],dataAmount[2]);
	freeMemory(s3,dataAmount[3],trainCol);
	freeMemory(a2_transfer,trainCol,dataAmount[2]);
	return totalError;
}
double **getBitArray(int number){
	int totalBit=(int)((log(number)/log(2))+0.5);
	double **result=getMemory(totalBit,number);
	for(int i=0;i<number;++i){
		int temp=i+1;
		for(int j=0;j<totalBit;++j){
			result[j][i]=temp%2;
			temp=temp/2;
			if(temp==0) break;
		}
	}
	return result;
}
	//得到训练数据的t矩阵
double **getResultDataArray(int number){
	double **result=getMemory(26,26);
	for(int i=0;i<26;++i){
		result[i][i]=1.0;
	}
	return result;
}
	//计算网络对输入p的输出结果
double **calculateOneTest(double **w1,double **w2,double **w3,double **b1,double **b2,double **b3,double **p,int trainCol){
	double **n1=getTimes(w1,p,dataAmount[1],dataAmount[0],trainCol);  
	getAddOrMinus(add,n1,b1,dataAmount[1],trainCol);
	activateMatrix(sigmoid,n1,dataAmount[1],trainCol);
	double **n2=getTimes(w2,n1,dataAmount[2],dataAmount[1],trainCol);
	getAddOrMinus(add,n2,b2,dataAmount[2],trainCol);
	activateMatrix(sigmoid,n2,dataAmount[2],trainCol);
	double **n3=getTimes(w3,n2,dataAmount[3],dataAmount[2],trainCol);
	getAddOrMinus(add,n3,b3,dataAmount[3],trainCol);
	activateMatrix(sigmoid,n3,dataAmount[3],trainCol);
	freeMemory(n1,dataAmount[1],trainCol);
	freeMemory(n2,dataAmount[2],trainCol);
	return n3;
}
void train(double **input,int rowTotal,int colTotal,double **result,int resultRow,int resultCol){
		// double **input=loadData("s.txt",rowTotal,colTotal);
	int trainCol=1;
	int trainResultRow=colTotal; 
	//随机初始化权值矩阵
	double **w1=randMatrix(dataAmount[1],dataAmount[0]);
	double **w2=randMatrix(dataAmount[2],dataAmount[1]);
	double **w3=randMatrix(dataAmount[3],dataAmount[2]);
	double **b1=randMatrix(dataAmount[1],trainCol);
	double **b2=randMatrix(dataAmount[2],trainCol);
	double **b3=randMatrix(dataAmount[3],trainCol);
	//获取训练数据中的t
		// double **result=getResultDataArray(colTotal);
	double totalError=1000.0;
	//迭代
	for(int timeNow=0;timeNow<totalTrainTimes;++timeNow){
		double tempError=0.0;
		for(int i=0;i<colTotal;++i){
			double **p=getOneCol(input,rowTotal,colTotal,i);
			double **t=getOneCol(result,resultRow,resultCol,i);
			tempError+=trainNN(w1,w2,w3,b1,b2,b3,p,t,trainCol);
			freeMemory(p,rowTotal,1);
			freeMemory(t,resultRow,1);
		}
	//求取训练到目前为止，最小的误差
		if(tempError<totalError) totalError=tempError;
		printf("times :%d  temp error %lf  minmus error :%lf\n",timeNow,tempError,totalError);
			// if(tempError<1) break;
	
	//将训练结束的网络参数保存到文件中
	saveData("weight11-35-20-10.txt",w1,dataAmount[1],dataAmount[0]);
	saveData("weight21-35-20-10.txt",w2,dataAmount[2],dataAmount[1]);
	saveData("weight31-35-20-10.txt",w3,dataAmount[3],dataAmount[2]);
	saveData("b11-35-20-10.txt",b1,dataAmount[1],trainCol);
	saveData("b21-35-20-10.txt",b2,dataAmount[2],trainCol);
	saveData("b31-35-20-10.txt",b3,dataAmount[3],trainCol);

	}
	// freeMemory(input,rowTotal,colTotal);
	freeMemory(w1,dataAmount[1],dataAmount[0]);
	freeMemory(w2,dataAmount[2],dataAmount[1]);
	freeMemory(w3,dataAmount[3],dataAmount[2]);
	freeMemory(b1,dataAmount[1],trainCol);
	freeMemory(b2,dataAmount[2],trainCol);
	freeMemory(b3,dataAmount[3],trainCol);
	// freeMemory(result,resultRow,resultCol);
}
	//进行数据的测试
void dataTest(double **input,int rowTotal,int colTotal,double **result,int resultRow,int resultCol){
	//读取测试的数据
		// double **input=loadData("s.txt",rowTotal,colTotal);
	int trainCol=1;
	//读取训练完成后的网络参数
	double **w1=loadData("weight11-35-20-10.txt",dataAmount[1],dataAmount[0]);
	double **w2=loadData("weight21-35-20-10.txt",dataAmount[2],dataAmount[1]);
	double **w3=loadData("weight31-35-20-10.txt",dataAmount[3],dataAmount[2]);
	double **b1=loadData("b11-35-20-10.txt",dataAmount[1],trainCol);
	double **b2=loadData("b21-35-20-10.txt",dataAmount[2],trainCol);
	double **b3=loadData("b31-35-20-10.txt",dataAmount[3],trainCol);
	//读取测试数据对应的t
		// double **result=getResultDataArray(colTotal);
	//保存计算的总误差，||t-a||
	double totalError=0.0;
	//保存分类的总误差
	double class_error=0.0;
	printf("the net is : %d %d %d %d\n",dataAmount[0],dataAmount[1],dataAmount[2],dataAmount[3]);

	for(int class=0;class<colTotal;++class){
		double tempError=0.0;
		double **p=getOneCol(input,rowTotal,colTotal,class);
		double **t=getOneCol(result,resultRow,resultCol,class);
		double **n3=calculateOneTest(w1,w2,w3,b1,b2,b3,p,trainCol);



	//判断当前训练数据所属的类别
		double max=n3[0][0];
		int Classtype=0;
		double resu=t[0][0];
		int resuClasstype=0;
		for(int i=0;i<dataAmount[3];++i){
			if(n3[i][0]>max) {
				max=n3[i][0];
				Classtype=i;
			}
			if(t[i][0]>max) {
				resu=t[i][0];
				resuClasstype=i;
			}
		}

		printf("result is:%d   realResult is:%d\n",resuClasstype+1,Classtype+1);
		// printf("%d \n",Classtype+1);

	// //当网络得到的类别不等于实际类别时，分类总误差+1
	// 	if(class!=Classtype)
	// 		class_error+=1;
	//计算当前测试样本的总计算误差
		for(int i1=0;i1<dataAmount[3];++i1){
			for(int j=0;j<trainCol;++j){
				n3[i1][j]=t[i1][j]-n3[i1][j];
				tempError+=fabs(n3[i1][j]);

				if(t[i1][j]>0&&(Classtype!=i1)){
					class_error+=1;
				}
			}
		}
		totalError+=tempError;
		freeMemory(n3,dataAmount[3],trainCol);
	// printf("temp error :%lf\n",tempError);
		freeMemory(p,rowTotal,1);
		freeMemory(t,resultRow,1);
	// getchar();
	}
	printf("total error :%lf\n", totalError);
	printf("total class_error :%lf\n",class_error);

	freeMemory(w1,dataAmount[1],dataAmount[0]);
	freeMemory(w2,dataAmount[2],dataAmount[1]);
	freeMemory(w3,dataAmount[3],dataAmount[2]);
	freeMemory(b1,dataAmount[1],trainCol);
	freeMemory(b2,dataAmount[2],trainCol);
	freeMemory(b3,dataAmount[3],trainCol);

}

//load init data from file
void initData(char *filePath,double **input,int rowTotal,int colTotal,double **result,int resultRow,int resultCol){

	double **inputTemp=loadData(filePath,colTotal,rowTotal+1);



		// double **result=getMemory(resultRow,resultCol);



	for(int i=0;i<resultCol;++i){

		int temp=(int)inputTemp[i][0]-1;

		result[temp][i]=1;

	}


		// double **input=getMemory(rowTotal,colTotal);



	for(int i=0;i<colTotal;++i){
		for(int j=1;j<rowTotal+1;++j){
			input[j-1][i]=inputTemp[i][j];
		}
	}

	freeMemory(inputTemp,colTotal,rowTotal+1);


}

void using_NN(void){
	srand( (unsigned)time( NULL ) );


	int resultRow=3;
	int resultCol=160;


	int rowTotal=13;
	int colTotal=160;

	double **result=getMemory(resultRow,resultCol);



	double **input=getMemory(rowTotal,colTotal);


	initData("data1.txt",input, rowTotal, colTotal,result, resultRow, resultCol);

// printArray(result,resultRow,resultCol);

	int TestResultRow=3;
	int TestResultCol=18;


	int TestRowTotal=13;
	int TestColTotal=18;

	double **Testresult=getMemory(TestResultRow,TestResultCol);



	double **TestInput=getMemory(TestRowTotal,TestColTotal);


	initData("test1.txt",TestInput, TestRowTotal, TestColTotal,Testresult, TestResultRow, TestResultCol);

// printArray(Testresult, TestResultRow, TestResultCol);


if(flag>0)
	train(input, rowTotal, colTotal,result, resultRow, resultCol);
	dataTest(TestInput, TestRowTotal, TestColTotal,Testresult, TestResultRow, TestResultCol);

	freeMemory(input, rowTotal, colTotal);
	freeMemory(result, resultRow, resultCol);

	freeMemory(TestInput, TestRowTotal, TestColTotal);
	freeMemory(Testresult, TestResultRow, TestResultCol);

}