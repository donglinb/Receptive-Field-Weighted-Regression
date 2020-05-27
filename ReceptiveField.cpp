/* Coded by Bai Donglin, 2018.5.10. */

#include"ReceptiveField.h"
#include<iostream>

namespace ReceptiveField
{
	ReceptiveField::ReceptiveField()
	{
		M0 = Matrix2d::Identity() * 5;
		P0 = Matrix3d::Identity() * 1 / (0.001*0.001);
		w_gen = 0.1;
		w_prun = 0.9;
		w_a = 0.001;
		e_a = 0.001;
		G = 100;
		
		numReceptiveFields = 0;
	}
	ReceptiveField::ReceptiveField(double w_gen, double w_prun, double w_a, double e_a)
	{
		M0 = Matrix2d::Identity() * 5;
		P0 = Matrix3d::Identity() * 1 / (0.001*0.001);
		this->w_gen = w_gen;
		this->w_prun = w_prun;
		this->w_a = w_a;
		this->e_a = e_a;
		G = 100;

		numReceptiveFields = 0;
	}
	double ReceptiveField::LinearPredict(Vector2d x, Vector2d c, Vector3d Beta)
	{
		Vector3d x1;
		x1 << (x - c), 1;
		double y = x1.transpose()*Beta;
		return y;
	}
	double ReceptiveField::GaussianKernel(Vector2d x, Vector2d c, Matrix2d D)
	{
		Vector2d x_c = x - c;
		double d=-0.5*x_c.transpose()*D*x_c;
		double w = exp(d);
		return w;
	}
	Matrix2d ReceptiveField::GaussianKernelDiffM(Vector2d x, Vector2d c, Matrix2d M)
	{
		Vector2d x_c = x - c;
		double w = exp(-0.5*x_c.transpose()*M.transpose()*M*x_c);
		Matrix2d Diff = -w*M*x_c*x_c.transpose();
		return Diff;
	}
	Matrix2d ReceptiveField::GaussianKernelDiffD(Vector2d x, Vector2d c, Matrix2d D)
	{
		Matrix2d M = D.llt().matrixL().transpose();
		Vector2d x_c = x - c;
		double w = exp(-0.5*x_c.transpose()*D*x_c);
		Matrix2d Diff = -w*M*x_c*x_c.transpose();
		return Diff;
	}
	int ReceptiveField::UpdateParams(Vector3d &Beta, Matrix3d &P, Vector2d c, Matrix2d D, Vector2d x, double y, double lambda)
	{
		Vector3d x1;
		x1 << (x - c), 1;
		double e_cv = y - Beta.transpose()*x1;
		double w = GaussianKernel(x, c, D);
		Matrix3d P1 = (P - (P*x1*x1.transpose()*P) / (lambda / w + x1.transpose()*P*x1)) / lambda;
		Vector3d Beta1 = Beta + w*P1*x1*e_cv;
		Beta = Beta1;
		P = P1;
		return 0;
	}
	double ReceptiveField::CostFunc(Vector2d x, double y)
	{
		std::vector<double> weights;
		std::vector<double> predicts;
		double sumWeights=0;
		for (int k = 0; k < ReceptiveFieldData.size(); k++)
		{
			double w = GaussianKernel(x, ReceptiveFieldData.at(k).c, ReceptiveFieldData.at(k).D);
			weights.push_back(w);
			sumWeights += w;
			double yp = LinearPredict(x, ReceptiveFieldData.at(k).c, ReceptiveFieldData.at(k).Beta);
			predicts.push_back(yp);
		}
		double yp = 0;
		for (int k = 0; k < ReceptiveFieldData.size(); k++)
		{
			yp += weights.at(k)*predicts.at(k);
		}
		yp /= sumWeights;
		double J1 = 0.5*(y - yp)*(y - yp);
		double J2 = 0;
		for (int k = 0; k < ReceptiveFieldData.size(); k++)
		{
			J2 += weights.at(k)*(y - predicts.at(k))*(y - predicts.at(k));
		}
		J2 /= sumWeights;
		double J = J1 + J2;
		return J;
	}
	std::vector<Matrix2d> ReceptiveField::CostFuncDiff(Vector2d x, double y, double G)
	{
		std::vector<Matrix2d> Diffs;
		std::vector<double> weights;
		std::vector<double> predicts;
		double sumWeights = 0;
		for (int k = 0; k < ReceptiveFieldData.size(); k++)
		{
			double w = GaussianKernel(x, ReceptiveFieldData.at(k).c, ReceptiveFieldData.at(k).D);
			weights.push_back(w);
			sumWeights += w;
			double yp = LinearPredict(x, ReceptiveFieldData.at(k).c, ReceptiveFieldData.at(k).Beta);
			predicts.push_back(yp);
		}
		double yp = 0;
		for (int k = 0; k < ReceptiveFieldData.size(); k++)
		{
			yp += weights.at(k)*predicts.at(k);
		}
		yp /= sumWeights;

		double J1 = 0, J2 = 0, J3 = 0;
		J2 = sumWeights;
		for (int k = 0; k < ReceptiveFieldData.size(); k++)
		{
			J1 += weights.at(k)*predicts.at(k);
			J3 += weights.at(k)*(y - predicts.at(k))*(y - predicts.at(k));
		}
		for (int k = 0; k < ReceptiveFieldData.size(); k++)
		{
			Matrix2d DiffM = GaussianKernelDiffM(x, ReceptiveFieldData.at(k).c, ReceptiveFieldData.at(k).M);
			Matrix2d Diff = (yp - y)*(predicts.at(k) - yp) / J2*DiffM + ((y - predicts.at(k))*(y - predicts.at(k)) - J3 / J2) / J2*DiffM;
			Diffs.push_back(Diff*G);
		}
		return Diffs;
	}
	int ReceptiveField::UpdateM(Vector2d x, double y, double alpha, double G)
	{
		std::vector<Matrix2d> Diffs = CostFuncDiff(x, y, G);
		for (int k = 0; k < ReceptiveFieldData.size(); k++)
		{
			ReceptiveFieldData.at(k).M = ReceptiveFieldData.at(k).M - alpha*Diffs.at(k);
			ReceptiveFieldData.at(k).D = ReceptiveFieldData.at(k).M.transpose()*ReceptiveFieldData.at(k).M;
		}
		return 0;
	}
	int ReceptiveField::UpdateParamsAll(Vector2d x, double y, double lambda)
	{
		for (int k = 0; k < ReceptiveFieldData.size(); k++)
		{
			UpdateParams(ReceptiveFieldData.at(k).Beta, ReceptiveFieldData.at(k).P, ReceptiveFieldData.at(k).c, ReceptiveFieldData.at(k).D, x, y, lambda);
		}
		return 0;
	}
	int ReceptiveField::CreateReceptiveField(Vector2d x, double y)
	{
		ReceptiveFieldDataType data;
		data.c = x;
		data.M = M0;
		data.D = M0.transpose()*M0;
		data.P = P0;

		Vector3d x1;
		x1 << x, 1;
		//data.Beta = (x1.transpose()*x1).inverse()*x1.transpose()*y;
		data.Beta << VectorXd::Ones(x1.rows() - 1), (y - x1.head(x1.rows() - 1).sum()) / x1(x1.rows()-1);

		ReceptiveFieldData.push_back(data);
		numReceptiveFields += 1;
		return 0;
	}
	int ReceptiveField::RemoveReceptiveField(int k)
	{
		ReceptiveFieldData.erase(ReceptiveFieldData.begin()+k);
		numReceptiveFields -= 1;
		return 0;
	}
	int ReceptiveField::getnumReceptiveFields()
	{
		return numReceptiveFields;
	}
	double ReceptiveField::getWeightSum(Vector2d x)
	{
		double WeightSum = 0;
		for (int k = 0; k < ReceptiveFieldData.size(); k++)
		{
			WeightSum += GaussianKernel(x, ReceptiveFieldData.at(k).c, ReceptiveFieldData.at(k).D);
		}
		return WeightSum;
	}
	int ReceptiveField::LearnReceptiveFieldStep(Vector2d x, double y, double alpha, double G, double lambda)
	{
		if (ReceptiveFieldData.size() == 0)
		{
			CreateReceptiveField(x, y);
		}
		else
		{
			std::vector<double> weights;
			std::vector<int> prunCount;
			bool gen = true;
			for (int k = 0; k < ReceptiveFieldData.size(); k++)
			{
				double w = GaussianKernel(x, ReceptiveFieldData.at(k).c, ReceptiveFieldData.at(k).D);
				weights.push_back(w);
				if (w >= w_gen)
				{
					gen = false;
				}
				if (w >= w_prun)
				{
					prunCount.push_back(k);
				}
			}
			if (gen == true)
			{
				CreateReceptiveField(x, y);
			}
			else if (prunCount.size() >= 2)
			{
				int index = prunCount.at(0);
				double maxdet = ReceptiveFieldData.at(index).D.determinant();
				for (int i = 1; i < prunCount.size(); i++)
				{
					if (ReceptiveFieldData.at(prunCount.at(i)).D.determinant()>maxdet)
					{
						index = prunCount.at(i);
					}
				}
				RemoveReceptiveField(index);
			}
			UpdateM(x, y, alpha, G);
			UpdateParamsAll(x, y, lambda);
		}
		return 0;
	}
	int ReceptiveField::LearnReceptiveFieldStepErr(Vector2d x, double y, double Threrr, double alpha, double G, double lambda, int nIter)
	{
		if (ReceptiveFieldData.size() == 0)
		{
			CreateReceptiveField(x, y);
		}
		else
		{
			for (int k = 0; k < nIter; k++)
			{
				UpdateM(x, y, alpha, G);
				UpdateParamsAll(x, y, lambda);
			}
			double yp = PredictOndata(x);
			double err = abs(y - yp);
			if (err > Threrr)
			{
				CreateReceptiveField(x, y);
				UpdateM(x, y, alpha, G);
				UpdateParamsAll(x, y, lambda);
			}
			else
			{
				std::vector<int> prunCount;
				for (int k = 0; k < ReceptiveFieldData.size(); k++)
				{
					double w = GaussianKernel(x, ReceptiveFieldData.at(k).c, ReceptiveFieldData.at(k).D);
					if (w >= w_prun)
					{
						prunCount.push_back(k);
					}
				}
				if (prunCount.size() >= 2)
				{
					int index = prunCount.at(0);
					double maxdet = ReceptiveFieldData.at(index).D.determinant();
					for (int i = 1; i < prunCount.size(); i++)
					{
						if (ReceptiveFieldData.at(prunCount.at(i)).D.determinant()>maxdet)
						{
							index = prunCount.at(i);
						}
					}
					RemoveReceptiveField(index);
					UpdateM(x, y, alpha, G);
					UpdateParamsAll(x, y, lambda);
				}
			}
		}
		return 0;
	}
	int ReceptiveField::LearnReceptiveField(int numSamples, MatrixXd X, VectorXd Y, int numEpoch, double alpha, double G, double lambda)
	{
		if (X.rows() != numSamples || X.cols() != 2)
		{
			return -1;
		}
		for (int epoch = 0; epoch < numEpoch; epoch++)
		{
			for (int k = 0; k < numSamples; k++)
			{
				Vector2d x = X.row(k).transpose();
				double y = Y(k);
				LearnReceptiveFieldStep(x, y, alpha, G, lambda);
			}
			std::cout << epoch + 1 << "/" << numEpoch << "epoch Finished, MSE=" << errorMSE(PredictOndataSet(X),Y) << std::endl;
			std::cout << numReceptiveFields << " Receptive Fields are Created." << std::endl;
		}
		return 0;
	}
	int ReceptiveField::LearnReceptiveFieldErr(int numSamples, MatrixXd X, VectorXd Y, double Threrr, int numEpoch, double alpha, double G, double lambda)
	{
		if (X.rows() != numSamples || X.cols() != 2)
		{
			return -1;
		}
		for (int epoch = 0; epoch < numEpoch; epoch++)
		{
			for (int k = 0; k < numSamples; k++)
			{
				Vector2d x = X.row(k).transpose();
				double y = Y(k);
				LearnReceptiveFieldStepErr(x, y, Threrr, alpha, G, lambda);
			}
			std::cout << epoch + 1 << "/" << numEpoch << "epoch Finished, MSE=" << errorMSE(PredictOndataSet(X), Y) << std::endl;
			std::cout << numReceptiveFields << " Receptive Fields are Created." << std::endl;
		}
		return 0;
	}
	double ReceptiveField::PredictOndata(Vector2d x)
	{
		double sumWeights=0;
		double y=0;
		for (int k = 0; k < ReceptiveFieldData.size(); k++)
		{
			double w = GaussianKernel(x, ReceptiveFieldData.at(k).c, ReceptiveFieldData.at(k).D);
			double yp = LinearPredict(x, ReceptiveFieldData.at(k).c, ReceptiveFieldData.at(k).Beta);
			y += w*yp;
			sumWeights += w;
		}
		y /= sumWeights;
		return y;
	}
	VectorXd ReceptiveField::PredictOndataSet(MatrixXd X)
	{
		if (X.cols() != 2)
		{
			return Vector2d();
		}

		int numSamples = X.rows();
		VectorXd y(numSamples);
		for (int k = 0; k < numSamples; k++)
		{
			y(k) = PredictOndata(X.row(k).transpose());
		}
		return y;
	}
	double ReceptiveField::PredictFusionData(Vector2d x1, Vector2d x2, ReceptiveField &rtf1, ReceptiveField &rtf2)
	{
		double y1 = 0, y2 = 0, y = 0;
		double w1 = 0, w2 = 0;
		y1 = rtf1.PredictOndata(x1);
		y2 = rtf2.PredictOndata(x2);
		w1 = rtf1.getWeightSum(x1);
		w2 = rtf2.getWeightSum(x2);
		y = (y1*w1 + y2 * w2) / (w1 + w2);
		return y;
	}
	VectorXd ReceptiveField::PredictFusionDataSet(MatrixXd X1, MatrixXd X2, ReceptiveField &rtf1, ReceptiveField &rtf2)
	{
		if (X1.cols() != 2)
		{
			return Vector2d();
		}
		if (X2.cols() != 2)
		{
			return Vector2d();
		}
		if (X1.rows() != X2.rows())
		{
			return Vector2d();
		}

		int numSamples = X1.rows();
		VectorXd y(numSamples);
		for (int k = 0; k < numSamples; k++)
		{
			y(k) = PredictFusionData(X1.row(k).transpose(), X2.row(k).transpose(), rtf1, rtf2);
		}
		return y;
	}
	double ReceptiveField::errorMSE(VectorXd predictvalue, VectorXd realvalue)
	{
		if (predictvalue.rows() != realvalue.rows())
		{
			return -1;
		}
		int nSample = predictvalue.rows();
		double error = 0;
		for (int k = 0; k < nSample; k++)
		{
			error += (predictvalue(k) - realvalue(k))*(predictvalue(k) - realvalue(k));
		}
		error /= nSample;
		error = sqrt(error);
		return error;
	}
}