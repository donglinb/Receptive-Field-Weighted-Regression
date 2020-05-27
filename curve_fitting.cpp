/* Coded by Bai Donglin, 2018.5.10. */

#include<iostream>
#include<vector>
#include<random>
#include<time.h>

#include<Eigen/Dense>
#include<matio.h>

#include"ReceptiveField.h"

using namespace std;
using namespace Eigen;

typedef struct TrainDataType
{
	Vector2d point;
	double realvalue;
	double noisyvalue;
}TrainDataType;

double FunctionEval(double x, double y)
{
	double f1 = exp(-10 * x * x);
	double f2 = exp(-50 * y * y);
	double f3 = 1.25 * exp(-5 * (x * x + y * y));
	double f = f1 > f2 ? (f1 > f3 ? f1 : f3) : (f2 > f3 ? f2 : f3);
	return f;
}

int SavetoMat(std::vector<TrainDataType> traindata, string filename = "traindata.mat")
{
	double* X = new double[2 * traindata.size()];
	double* realvalue = new double[traindata.size()];
	double* noisyvalue = new double[traindata.size()];
	for (int k = 0; k < traindata.size(); k++)
	{
		X[2 * k] = traindata.at(k).point(0);
		X[2 * k + 1] = traindata.at(k).point(1);
		realvalue[k] = traindata.at(k).realvalue;
		noisyvalue[k] = traindata.at(k).noisyvalue;
	}
	mat_t* mat = 0;
	matvar_t* matvar = 0;
	size_t dims1[2] = { 2,traindata.size() };
	size_t dims2[2] = { traindata.size(),1 };
	mat = Mat_CreateVer(filename.c_str(), NULL, MAT_FT_MAT5);
	if (mat)
	{
		matvar = Mat_VarCreate("X", MAT_C_DOUBLE, MAT_T_DOUBLE, 2, dims1, X, 0);
		Mat_VarWrite(mat, matvar, matio_compression::MAT_COMPRESSION_NONE);
		Mat_VarFree(matvar);
		matvar = Mat_VarCreate("realvalue", MAT_C_DOUBLE, MAT_T_DOUBLE, 2, dims2, realvalue, 0);
		Mat_VarWrite(mat, matvar, matio_compression::MAT_COMPRESSION_NONE);
		Mat_VarFree(matvar);
		matvar = Mat_VarCreate("noisyvalue", MAT_C_DOUBLE, MAT_T_DOUBLE, 2, dims2, noisyvalue, 0);
		Mat_VarWrite(mat, matvar, matio_compression::MAT_COMPRESSION_NONE);
		Mat_VarFree(matvar);
		Mat_Close(mat);
		cout << "Write Successfully." << endl;
	}
	else
	{
		cout << "Open File Failed." << endl;
		return 1;
	}

	delete[] X;
	delete[] realvalue;
	delete[] noisyvalue;

	return 0;
}

int ReadFromMat(std::vector<TrainDataType>& traindata, string filename = "traindata.mat")
{
	double* X = 0, * realvalue = 0, * noisyvalue = 0;
	int size = 0;
	mat_t* mat = 0;
	matvar_t* matvar = 0;

	//mat = Mat_Open("traindata.mat", MAT_ACC_RDONLY);
	mat = Mat_Open(filename.c_str(), MAT_ACC_RDONLY);
	if (mat)
	{
		matvar = Mat_VarRead(mat, "X");
		if (matvar)
		{
			X = new double[matvar->nbytes / matvar->data_size];
			memcpy(X, matvar->data, matvar->nbytes);
			size = matvar->nbytes / matvar->data_size / 2;
		}
		else
		{
			cout << "Read Variable X Failed." << endl;
			return 1;
		}

		matvar = Mat_VarRead(mat, "noisyvalue");
		if (matvar)
		{
			noisyvalue = new double[matvar->nbytes / matvar->data_size];
			memcpy(noisyvalue, matvar->data, matvar->nbytes);
		}
		else
		{
			matvar = Mat_VarRead(mat, "Y");
			if (matvar)
			{
				noisyvalue = new double[matvar->nbytes / matvar->data_size];
				memcpy(noisyvalue, matvar->data, matvar->nbytes);
			}
			else
			{
				cout << "Read Variable Y Failed." << endl;
				return 1;
			}
		}

		matvar = Mat_VarRead(mat, "realvalue");
		if (matvar)
		{
			realvalue = new double[matvar->nbytes / matvar->data_size];
			memcpy(realvalue, matvar->data, matvar->nbytes);
		}
		else
		{
			realvalue = new double[size];
			memcpy(realvalue, noisyvalue, size * sizeof(double));
		}

		traindata.clear();
		for (int k = 0; k < size; k++)
		{
			TrainDataType data;
			data.point(0) = X[2 * k];
			data.point(1) = X[2 * k + 1];
			data.noisyvalue = noisyvalue[k];
			data.realvalue = realvalue[k];
			traindata.push_back(data);
		}

		cout << "Read From Mat " + filename + " Succeed. " << size << " samples." << endl;
	}
	else
	{
		cout << "Open File Failed." << endl;
		return 1;
	}
	return 0;
}

//  fitting a 2-d function using receptive field weighted regression.
int main()
{
	//generate the train data
	std::vector<TrainDataType> traindata;
	std::default_random_engine e;
	std::normal_distribution<double> n(0, 0.01);
	for (double i = -1; i <= 1; i += 0.02)
	{
		for (double j = -1; j <= 1; j += 0.02)
		{
			TrainDataType data;
			data.point << i, j;
			data.realvalue = FunctionEval(i, j);
			data.noisyvalue = data.realvalue + n(e);
			traindata.push_back(data);
		}
	}
	cout << "number of samples:" << traindata.size() << endl;
	SavetoMat(traindata);

	//convert train data to Matrix
	MatrixXd X(traindata.size(), 2);
	VectorXd Y(traindata.size());
	for (int k = 0; k < traindata.size(); k++)
	{
		X.row(k) = traindata.at(k).point.transpose();
		Y(k) = traindata.at(k).noisyvalue;
	}

	//create receptive field object
	ReceptiveField::ReceptiveField rtf(0.1, 0.9);

	//learn for 10 epochs
	time_t t1, t2;
	t1 = time(NULL);
	//rtf.LearnReceptiveFieldErr(traindata.size(), X, Y, 0.5, 10, 0.1);
	rtf.LearnReceptiveField(traindata.size(), X, Y, 10, 0.1);
	t2 = time(NULL);
	cout << "number of learned receptive fields on 10 epoch:" << rtf.getnumReceptiveFields() << endl;
	cout << "consuming time: " << t2 - t1 << " seconds" << endl;

	//predict on the original samples
	VectorXd out = rtf.PredictOndataSet(X);
	cout << "Testing Error: MSE=" << ReceptiveField::ReceptiveField::errorMSE(out, Y) << endl;

	//save the results
	mat_t* mat = Mat_CreateVer("predict.mat", NULL, MAT_FT_MAT5);
	size_t dims[2] = { out.rows(),1 };
	matvar_t* matvar = Mat_VarCreate("predict", MAT_C_DOUBLE, MAT_T_DOUBLE, 1, dims, out.data(), 0);
	Mat_VarWrite(mat, matvar, MAT_COMPRESSION_NONE);
	Mat_VarFree(matvar);
	Mat_Close(mat);
	cout << "The predictions are successfully saved." << endl;

	return 0;
}
