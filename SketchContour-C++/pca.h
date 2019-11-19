#pragma once
//#include <flann/flann.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <ANN/ANN.h>


using namespace std;

void compute_pca(cv::Mat & gridx, cv::Mat & gridy, std::vector<cv::Point2d> & norms,
	std::vector<std::vector<cv::Point2d>> & strokes, 
	std::vector<std::vector<cv::Point2i>> & positions, 
	double radius,
	cv::Mat & pca)
{
	int total_num = 0;
	for (size_t i = 0; i < strokes.size(); i++)
	{
		total_num += int(strokes[i].size());
	}

	ANNpointArray dataset = annAllocPts(total_num, 2);

	total_num = 0;
	for (size_t i = 0; i < strokes.size(); i++)
	{
		for (size_t j = 0; j < strokes[i].size(); j++)
		{
			dataset[total_num][0] = strokes[i][j].x;
			dataset[total_num][1] = strokes[i][j].y;
			total_num++;
		}
	}


 	ANNkd_tree * kd = new ANNkd_tree(dataset, total_num, 2);
	
	
	cv::Mat count(gridx.size(), CV_32S, cv::Scalar(0));
	
	for (size_t i = 0; i < positions.size(); i++)
	{
		for (size_t j = 0; j < positions[i].size(); j++)
		{
			count.at<int>(positions[i][j].x, positions[i][j].y) += 1;
		}
	}

	pca = cv::Mat(gridx.size(), CV_64FC2, cv::Scalar(0, 0));

	int M = gridx.rows;
	int N = gridx.cols;
	
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			if (count.at<int>(i, j) >= 1)
			{
				ANNpoint pt = new ANNcoord[2];
				pt[0] = gridx.at<double>(i, j);
				pt[1] = gridy.at<double>(i, j);

				int k = kd->annkFRSearch(pt, radius * radius, 0);

				ANNidxArray nnIdx = new ANNidx[k];
				ANNdistArray nnDist = new ANNdist[k];
	
				kd->annkFRSearch(pt, radius * radius, k, nnIdx, nnDist);

				cv::Mat local_points(k, 2, CV_64F);
				for (int pp = 0; pp < k; pp++)
				{
					local_points.at<double>(pp, 0) = dataset[nnIdx[pp]][0];
					local_points.at<double>(pp, 1) = dataset[nnIdx[pp]][1];
				}

				cv::Mat cov, Mu;
				
				cv::calcCovarMatrix(local_points, cov, Mu, cv::COVAR_NORMAL | cv::COVAR_ROWS);
				cov = cov / (k - 1);

				cv::Mat eigval, eigvec;
				cv::eigen(cov, eigval, eigvec);

				int maxeigidx = 0;
				if (eigval.at<double>(1, 0) > eigval.at<double>(0, 0))
				{
					maxeigidx = 1;
				}

				double xxx = eigvec.at<double>(maxeigidx, 0);
				double yyy = eigvec.at<double>(maxeigidx, 1);

				pca.at<cv::Vec2d>(i, j) = cv::Vec2d(xxx, yyy);

				cv::Point2d norm = norms[i];
				cv::Point2d tang(-norm.y, norm.x);
				double flag = pca.at<cv::Vec2d>(i, j)[0] * tang.x + pca.at<cv::Vec2d>(i, j)[1] * tang.y;
				if (flag < 0)
				{
					pca.at<cv::Vec2d>(i, j)[0] = -pca.at<cv::Vec2d>(i, j)[0];
					pca.at<cv::Vec2d>(i, j)[1] = -pca.at<cv::Vec2d>(i, j)[1];
				}

				delete[] pt;
				delete[] nnIdx;
				delete[] nnDist;
			}
		}
	}

	annDeallocPts(dataset);
	delete kd;
	annClose();
}