	#include<stdio.h>
	#include<malloc.h>
	#include <math.h>
	#include <stdlib.h>
	#include <time.h>
	#include <unistd.h>
	double **getMemory(int row,int col){ //申请row*col的空间
		double **result=(double**)malloc(row*sizeof(double*));
		int i=0;
		for(i=0;i<row;++i){
			result[i]=(double*)malloc(col*sizeof(double));
		}
		int j=0;
		for(i=0;i<row;++i){
			for(j=0;j<col;++j)
			{
				result[i][j]=0.0;
			}
		}
		return result;
	}
	void freeMemory(double **input,int row,int col){ //释放row*col的空间
		for(int i=0;i<row;++i){
			free(input[i]);
		}
		free(input);
	}

	int getRandom(int max){ //产生随机数
		int number=rand()%max;
		return number;
	}
	
	void printArray(double **outData,int row,int col){ //输出矩阵
		int i=0; int j=0;
		for( i=0;i<row;++i){
			for(j=0;j<col;j++){
				printf("%lf ",outData[i][j]);
			}
			printf("\n");
		} }

	//矩阵乘法
		double** getTimes(double **leftOne,double **right,int leftOneRow,int leftOneCol,int rightOneCol){
			int i=0; int j=0; int k=0;
			double ** result=getMemory(leftOneRow,rightOneCol);
			for(i=0;i<leftOneRow;i++){
				for(j=0;j<rightOneCol;j++){
					for(k=0;k<leftOneCol;k++){
						result[i][j]+=leftOne[i][k]*right[k][j];
					}
				}
			}
			return result;
		}

	double **getTransfer(double **input,int row,int col){ 	//矩阵转置
		double **result=getMemory(col,row);
		for(int i=0;i<row;++i){
			for(int j=0;j<col;++j){
				result[j][i]=input[i][j];
			}
		}
		return result;
	}
	//代数余子式
	double **getCofactor(double **input,int row,int col,int rowNumber,int colNumber){
		if(row<=0) return NULL;
		double **result=getMemory(row-1,col-1);
		int i0=0; int j0=0;
		for(int i=0;i<row;++i){
			if(i==rowNumber) continue;
			j0=0;
			for(int j=0;j<col;++j){
				if(j==colNumber) continue;
				result[i0][j0]=input[i][j];
				j0++;
			}
			i0++;
		}
		return result;
	}
	double getDeterminant(double **input,int row,int col){ // 行列式
		if(row==2) return input[0][0]*input[1][1]-input[0][1]*input[1][0];
		if(row==1) return input[0][0];
		double sum=0.0;
		for(int i=0;i<col;++i){
			double **tempInput=getCofactor(input,row,col,0,i);
			double tempSum=input[0][i]*getDeterminant(tempInput,row-1,col-1);
			if(i%2==0) sum+=tempSum;
			else sum+=-tempSum;
			freeMemory(tempInput,row-1,col-1);
		}
		return sum;
	}

	double **getINverse(double **input,int row,int col){ 	//矩阵的逆矩阵
		double **result=getMemory(row,row);
		double Determinant=getDeterminant(input,row,row);
		for(int i=0;i<row;++i){
			for(int j=0;j<row;++j){
				double **tempCofactor=getCofactor(input,row,row,i,j);
				if((i+j)%2==0){
					result[j][i]=getDeterminant(tempCofactor,row-1,row-1)/Determinant;
				}else result[j][i]=-getDeterminant(tempCofactor,row-1,row-1)/Determinant;
				freeMemory(tempCofactor,row-1,row-1);
			}
		}
		return result;
	}

	//两个矩阵的对位操作
	void getAddOrMinus(double (*fun)(double,double),double** firstOne,double **secondOne,int row,int col){
		for(int i=0;i<row;++i){
			for(int j=0;j<col;++j){
				firstOne[i][j]=fun(firstOne[i][j],secondOne[i][j]);
			}}}

	double hardlim(double input){ //硬极限函数
		if(input<0.0) return -1;
		else return 1;
	}

		double add(double first,double second){ //两个数相加
			return first+second;
		}

	double minus(double first,double second){ //两个数相减
		return first-second;
	}
	//将对矩阵中的每一个数调用fun函数
	void activateMatrix(double(*fun)(double),double** input,int row,int col){
		for(int i=0;i<row;++i){
			for(int j=0;j<col;++j){
				input[i][j]=fun(input[i][j]);
			}
		}
	}

	//从指定文件路径中加载数据
	double **loadData(char *filePath,int row,int col){
		if(filePath==NULL) return NULL;
		double **result=getMemory(row, col);
		FILE *fp=fopen(filePath,"r");
		if(fp==NULL) return result; 
		for(int i=0;i<row;++i){
			for(int j=0;j<col;++j){
				fscanf(fp,"%lf",&result[i][j]);
			}
		}
		fclose(fp);
		return result;
	}
	//数据矩阵对应的图形
	void printChart(double **input,int row,int col){
		for(int i=0;i<row;++i){
			if(i%5==0) printf("\n");
			if(input[i][0]>0.0)
				printf("*");
			else printf(" ");
		}
		printf("\n");
	}
	//将数据保存到文件中
	void saveData(char *filePath,double **result,int row,int col){
		if(filePath==NULL) return;

		FILE *fp=fopen(filePath,"w");
		if(fp==NULL) return ; 
		for(int i=0;i<row;++i){
			for(int j=0;j<col;++j){
				fprintf(fp,"%lf ",result[i][j]);
			}
			fprintf(fp, "\n");
		}
		fclose(fp);
	}
	//获取矩阵中的某一行
	double **getOneRow(double **input,int row,int col,int rowNumber){
		double **result=getMemory(col,1);
		for(int i=0;i<col;++i){
			result[i][0]=input[rowNumber][i];
		}
		return result;
	}
	//获取矩阵中的某一列
	double **getOneCol(double **input,int row,int col,int colNumber){
		double **result=getMemory(row,1);
		for(int i=0;i<row;++i){
			result[i][0]=input[i][colNumber];
		}
		return result;
	}


	void zeroOneData(double **input,int row,int col,int indexCol,double *min1,double *max1,int minisCalculate){

		double min=*min1;
		double max=*max1;

		if(minisCalculate<1){

			// Means=0.0;

			// Variance=0.0;
			min=input[0][indexCol];
			max=input[0][indexCol];

			for(int i=1;i<row;++i){

				if(input[i][indexCol]<=min){
					min=input[i][indexCol];

				} 
				if(input[i][indexCol]>=max){
					max=input[i][indexCol];

				} 
			}

		}


		for(int i=0;i<row;++i){

			input[i][indexCol]=(input[i][indexCol]-min)/(max-min);
		}


		*min1=min;
		*max1=max;

	}
	


	void getMeanAndVariance(double **input,int row,int col,int indexCol,double **meansAndVariance,int meanRow){

		double Means=0.0;
		double Variance=0.0;
		for(int i=0;i<row;++i){

				Means+=input[i][indexCol];

			}
			Means=Means/(row);


			for(int i=0;i<row;++i){

				Variance+=(input[i][indexCol]-Means)*(input[i][indexCol]-Means);
			}

			Variance=sqrt(Variance/(row));

			meansAndVariance[meanRow][0]=Means;
			meansAndVariance[meanRow][1]=Variance;

	}

	void normalData(double **input,int row,int col,int indexCol,double *Means1,double *Variance1,int meanisCalculate){

		double Means=*Means1;
		double Variance=*Variance1;

		if(meanisCalculate<1){

			// Means=0.0;

			// Variance=0.0;

			for(int i=0;i<row;++i){

				Means+=input[i][indexCol];

			}
			Means=Means/(row);


			for(int i=0;i<row;++i){

				Variance+=(input[i][indexCol]-Means)*(input[i][indexCol]-Means);
			}

			Variance=sqrt(Variance/(row));



		}



		for(int i=0;i<row;++i){

			input[i][indexCol]=(input[i][indexCol]-Means)/Variance;
		}


		*Means1=Means;
		*Variance1=Variance;

	}