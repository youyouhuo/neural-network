#include "nn.c"

double** KNN(double **input,int inputRow,int inputCol,double **test,int testRow,int testCol);

int main(void){

using_NN();

	return 0;
}


int main1(void){


int inputRow=160;
int inputCol=14;
int testRow=18;
int testCol=13;

double **input=loadData("data.txt",inputRow,inputCol);

double **test=loadData("test.txt",testRow,testCol+1);

double **testNow=getMemory(testRow,testCol);

for(int i=0;i<testRow;++i){

	for(int j=1;j<testCol+1;++j){
		testNow[i][j-1]=test[i][j];
	}
}


double **Means_Variance=getMemory(inputCol,2);

for(int i=1;i<inputCol;i++){

double means=0.0;
double variance=0.0;

// normalData( input, inputRow, inputCol, i, &means, &variance,0);
zeroOneData( input, inputRow, inputCol, i, &means, &variance,0);

printf("after calculate %lf %lf \n",means,variance);

Means_Variance[i][0]=means;
Means_Variance[i][1]=variance;

}


double **menas_and_variance_afterNormal=getMemory(inputCol,2);
for(int i=1;i<inputCol;i++){
// getMeanAndVariance(input,inputRow, inputCol, i,menas_and_variance_afterNormal,i);

}


printArray(menas_and_variance_afterNormal,inputCol,2);

for(int i=0;i<testCol;++i){

// normalData( testNow, testRow, testCol, i, &Means_Variance[i+1][0], &Means_Variance[i+1][1],1);
zeroOneData( testNow, testRow, testCol, i, &Means_Variance[i+1][0], &Means_Variance[i+1][1],1);
// normalData( test, testRow, testCol+1, i+1, &Means_Variance[i+1][0], &Means_Variance[i+1][1],1);
}


// saveData("test1.txt",test,testRow,testCol+1);
// saveData("data1.txt",input, inputRow, inputCol);







double **trainResult=KNN(input, inputRow, inputCol,testNow, testRow, testCol);

double totalError=0.0;

for(int i=0;i<testRow;++i){

if(fabs(test[i][0]-trainResult[i][0])>0.5) totalError=totalError+1;
printf("result is:%lf   realResult is:%lf\n",test[i][0],trainResult[i][0]);


}

printf("total error is %lf\n",totalError/(testRow+0.0));


freeMemory(Means_Variance,inputCol,2);

freeMemory(menas_and_variance_afterNormal,inputCol,2);

freeMemory(test,testRow,testCol+1);

freeMemory(input,inputRow,inputCol);

freeMemory(testNow,testRow,testCol);
freeMemory(trainResult,testRow,1);



	return 0;
}

double** KNN(double **input,int inputRow,int inputCol,double **test,int testRow,int testCol){

	double **result=getMemory(testRow,1);

	for(int i=0;i<testRow;++i){

		int maxIndex=-1;
		double maxSum=-1.0;

		for(int indexRow=0;indexRow<inputRow;++indexRow){

			double totalSum=0.0;

			for(int indexCol=1;indexCol<inputCol;++indexCol){

				totalSum+=(input[indexRow][indexCol]-test[i][indexCol-1])*(input[indexRow][indexCol]-test[i][indexCol-1]);


			}

			if(maxIndex==-1){

				maxIndex=indexRow;
				maxSum=totalSum;

			}else if((maxIndex>=0)&&(maxSum>=totalSum)){

				maxIndex=indexRow;
				maxSum=totalSum;

			}

		}

		printf("%d %lf %lf\n",maxIndex,input[maxIndex][0],maxSum);

		result[i][0]=input[maxIndex][0];

	}
	return result;
}