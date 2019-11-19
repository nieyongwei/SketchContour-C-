#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include "gpb.h"

using namespace std;

void compute_grid_points(const std::vector<cv::Point2d> & samples,
	const std::vector<cv::Point2d> & norms,
	double r,
	int N,
	cv::Mat & gridx,
	cv::Mat & gridy)
{
	double ds = r / N;
	int M = int(samples.size());
	
	std::vector<double> di;
	
	for (int i = -N; i < N + 1; i++)
	{
		di.push_back(i*ds);
	}

	gridx = cv::Mat(M, 2 * N + 1, CV_64F);
	gridy = cv::Mat(M, 2 * N + 1, CV_64F);

	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < 2 * N + 1; j++)
		{
			cv::Point2d pt = samples[i] + di[j] * norms[i];
			gridx.at<double>(i, j) = pt.x;
			gridy.at<double>(i, j) = pt.y;
		}
	}
}

void dp(cv::Mat & gridx, cv::Mat & gridy, int sss, int ttt, double * gpb, int height, int width,
	int cmax, double alpha, double beta,
	double & mean_energy,
	std::vector<cv::Point2d> & stroke,
	std::vector<cv::Point2i> & position)
{
	int M = ttt - sss;
	int N = int(gridx.cols / 2);

	cv::Mat dpM = cv::Mat(M, 2 * N + 1, CV_64F, cv::Scalar(0)), dpP = cv::Mat(M, 2 * N + 1, CV_32S, cv::Scalar(0));

	// Row first:average gradeint of 8 directions
	for (int j = 0; j < 2 * N + 1; j++)
	{
		double grad = 0.0;
		for (int k = 0; k < 8; k++)
		{
			double angle = 3.14159265359 / 8.0 * (k + 0.01);
			double g = get_gPb(gpb, height, width, 
				cv::Point2d(gridx.at<double>(0+sss,j),gridy.at<double>(0+sss,j)), 
				cv::Point2d(cos(angle), sin(angle)));
			grad += g;
		}
		dpM.at<double>(0, j) = grad / 8.0;
	}

	// Second row: gradient only
	for (int j = 0; j < 2 * N + 1; j++)
	{
		double max_value = -1e10;
		int max_index = -1;
		cv::Point2d curp(gridx.at<double>(sss + 1, j), gridy.at<double>(sss + 1, j));
		for (int k = -cmax; k <= cmax; k++)
		{
			int kk = j + k;
			if (kk >= 0 && kk < 2 * N + 1)
			{
				cv::Point2d prep(gridx.at<double>(sss + 0, kk), gridy.at<double>(sss + 0, kk));
				cv::Point2d v = curp - prep;
				double curg = get_gPb(gpb, height, width, curp, v);
				curg += dpM.at<double>(0, kk);
				if (curg > max_value)
				{
					max_value = curg;
					max_index = kk;
				}
			}
		}
		dpM.at<double>(1, j) = max_value;
		dpP.at<int>(1, j) = max_index;
	}

	// Remaining rows: gradient + direction
	for (int i = 2; i < M; i++)
	{
		for (int j = 0; j < 2 * N + 1; j++)
		{
			double max_value = -1e10;
			int max_index = -1;
			cv::Point2d curp(gridx.at<double>(sss + i, j), gridy.at<double>(sss + i, j));
			for (int k = -cmax; k <= cmax; k++)
			{
				int kk = j + k;
				if (kk >= 0 && kk < 2 * N + 1)
				{
					cv::Point2d prep(gridx.at<double>(sss + i - 1, kk), gridy.at<double>(sss + i - 1, kk));
					int kkk = dpP.at<int>(i - 1, kk);
					cv::Point2d preprep(gridx.at<double>(sss + i - 2, kkk), gridy.at<double>(sss + i - 2, kkk));
					cv::Point2d v = curp - prep;
					double curg = get_gPb(gpb, height, width, curp, v);
					cv::Point2d v2 = prep - preprep;
					double lv = sqrt(v.x*v.x + v.y*v.y);
					double lv2 = sqrt(v2.x*v2.x + v2.y*v2.y);
					v /= lv;
					v2 /= lv2;

					double cur_curv = v.x * v2.x + v.y * v2.y;
					double val = dpM.at<double>(i - 1, kk) + alpha * curg + beta * cur_curv;
					if (val > max_value)
					{
						max_value = val;
						max_index = kk;
					}
				}
			}
			dpM.at<double>(i, j) = max_value;
			dpP.at<int>(i, j) = max_index;
		}
	}

	double emax = -1e10;
	int eindex = -1;

	for (int j = 0; j < 2 * N + 1; j++)
	{
		if (dpM.at<double>(M - 1, j) > emax)
		{
			emax = dpM.at<double>(M - 1, j);
			eindex = j;
		}
	}

	mean_energy = emax / M;

	stroke.push_back(cv::Point2d(gridx.at<double>(sss + M - 1, eindex), gridy.at<double>(sss + M - 1, eindex)));
	position.push_back(cv::Point2i(sss + M - 1, eindex));

	for (int i = M - 1; i > 0; i--)
	{
		int ind = dpP.at<int>(i, eindex);
		stroke.push_back(cv::Point2d(gridx.at<double>(sss + i - 1, ind), gridy.at<double>(sss + i - 1, ind)));
		position.push_back(cv::Point2i(sss + i - 1, ind));
		eindex = ind;
	}
}