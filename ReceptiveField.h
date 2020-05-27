/* Coded by Bai Donglin, 2018.5.10. */

#pragma once
#ifndef RECEPTIVEFIELD_H

#include<Eigen/Dense>
#include<vector>

namespace ReceptiveField
{
	using namespace Eigen;
	class ReceptiveField
	{
	public:
		ReceptiveField();
		ReceptiveField(double w_gen, double w_prun, double w_a = 0.001, double e_a = 0.001);
		double LinearPredict(Vector2d x, Vector2d c, Vector3d Beta);
		double GaussianKernel(Vector2d x, Vector2d c, Matrix2d D);
		Matrix2d GaussianKernelDiffM(Vector2d x, Vector2d c, Matrix2d M);
		Matrix2d GaussianKernelDiffD(Vector2d x, Vector2d c, Matrix2d D);
		int UpdateParams(Vector3d &Beta, Matrix3d &P, Vector2d c, Matrix2d D, Vector2d x, double y, double lambda=1);
		double CostFunc(Vector2d x, double y);
		std::vector<Matrix2d> CostFuncDiff(Vector2d x, double y, double G=1);
		int UpdateM(Vector2d x, double y, double alpha = 1, double G = 1);
		int UpdateParamsAll(Vector2d x, double y, double lambda=1);
		int CreateReceptiveField(Vector2d x, double y);
		int RemoveReceptiveField(int k);
		int getnumReceptiveFields();
		double getWeightSum(Vector2d x);
		int LearnReceptiveFieldStep(Vector2d x, double y, double alpha = 1, double G = 1, double lambda = 1);
		int LearnReceptiveFieldStepErr(Vector2d x, double y, double Threrr, double alpha = 1, double G = 1, double lambda = 1, int nIter=10);
		int LearnReceptiveField(int numSamples, MatrixXd X, VectorXd Y, int numEpoch = 1, double alpha = 1, double G = 1, double lambda = 1);
		int LearnReceptiveFieldErr(int numSamples, MatrixXd X, VectorXd Y, double Threrr, int numEpoch = 1, double alpha = 1, double G = 1, double lambda = 1);
		double PredictOndata(Vector2d x);
		VectorXd PredictOndataSet(MatrixXd X);
		static double PredictFusionData(Vector2d x1, Vector2d x2, ReceptiveField &rtf1, ReceptiveField &rtf2);
		static VectorXd PredictFusionDataSet(MatrixXd X1, MatrixXd X2, ReceptiveField &rtf1, ReceptiveField &rtf2);
		static double errorMSE(VectorXd predictvalue, VectorXd realvalue);
	private:
		Matrix2d M0 = Matrix2d::Identity() * 5;
		Matrix3d P0 = Matrix3d::Identity() * 1 / (0.001*0.001);
		double w_gen = 0.1;
		double w_prun = 0.9;
		double w_a = 0.001;
		double e_a = 0.001;
		double G = 100;
		
		typedef struct ReceptiveFieldDataType
		{
			Vector2d c;
			Matrix2d D;
			Matrix2d M;
			Vector3d Beta;
			Matrix3d P;
		}ReceptiveFieldDataType;
		std::vector < ReceptiveFieldDataType> ReceptiveFieldData;
		int numReceptiveFields=0;
	};
}  //  namespace ReceptiveField

#endif  //  RECEPTIVEFIELD_H