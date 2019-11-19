#pragma once

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "pca.h"

using namespace std;

void compute_stroke_start_and_end(int M, std::vector<cv::Point2i> & strokes)
{
	int l[9] = { 12,15,18,21,24,30,39,51,66 };
	int space[9];
	int lnum = 9;
	
	double overlap_rate = 1.0 / 3;
	for (int i = 0; i < lnum; i++)
	{
		space[i] = int(l[i] * overlap_rate);
	}

	strokes.clear();

	for (int i = 0; i < lnum; i++)
	{
		std::vector<int> start;
		for (int j = 0; j < M - l[i]; j += space[i])
		{
			start.push_back(j);
		}
		for (int j = 0; j < int(start.size()); j++)
		{
			strokes.push_back(cv::Point2i(start[j], l[i]));
		}
		if (M >= l[i])
		{
			strokes.push_back(cv::Point2i(M - l[i], l[i]));
		}
	}
	strokes.push_back(cv::Point2i(0, M));
	for (size_t i = 0; i < strokes.size(); i++)
	{
		strokes[i].y = strokes[i].x + strokes[i].y;
	}
}

void global_integrate_dp(cv::Mat & gridx, cv::Mat & gridy,
	std::vector<cv::Point2d> & norms,
	std::vector<std::vector<cv::Point2d>> & strokes,
	std::vector<std::vector<cv::Point2i>> & positions,
	std::vector<double> mean_energies,
	int cmax,
	double radius,
	std::vector<cv::Point2d> & contour)
{
	cv::Mat pca;
	compute_pca(gridx, gridy, norms, strokes, positions, radius, pca);
	int M = gridx.rows;
	int N = gridx.cols;

	cv::Mat count(gridx.size(), CV_32S, cv::Scalar(0));
	cv::Mat energy(gridx.size(), CV_64F, cv::Scalar(0));

	for (size_t i = 0; i < positions.size(); i++)
	{
		double mean_energy = mean_energies[i];
		for (size_t j = 0; j < positions[i].size(); j++)
		{
			count.at<int>(positions[i][j].x, positions[i][j].y) += 1;
			energy.at<double>(positions[i][j].x, positions[i][j].y) += mean_energy;
		}
	}

	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			if (count.at<int>(i, j) != 0)
			{
				energy.at<double>(i, j) /= count.at<int>(i, j);
			}
		}
	}

	std::vector<std::vector<std::vector<int>>> edge;
	edge.resize(M);
	for (int i = 0; i < M; i++)
	{
		edge[i].resize(N);
		for (int j = 0; j < N; j++)
		{
			edge[i][j].resize(2 * cmax + 1, 0);
		}
	}

	for (size_t i = 0; i < positions.size(); i++)
	{
		for (int j = 0; j < int(positions[i].size()) - 1; j++)
		{
			cv::Point2i cpos = positions[i][j];
			cv::Point2i ppos = positions[i][j + 1];
			edge[cpos.x][cpos.y][ppos.y - cpos.y + cmax] = 1;
		}
	}

	cv::Mat dpM(gridx.size(), CV_64F, cv::Scalar(0));
	cv::Mat dpI(gridx.size(), CV_32S, cv::Scalar(0));

	for (int i = 1; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			double maxc = -1e10;
			int maxj = -1;
			cv::Point2d curp(gridx.at<double>(i, j), gridy.at<double>(i, j));
			for (int k = -cmax; k <= cmax; k++)
			{
				if (edge[i][j][k + cmax] == 1)
				{
					cv::Vec2d pca_direction = pca.at<cv::Vec2d>(i - 1, j + k);
					cv::Point2d prep(gridx.at<double>(i - 1, j + k), gridy.at<double>(i - 1, j + k));
					cv::Point2d v = curp - prep;
					double lv = sqrt(v.x*v.x + v.y*v.y);
					v /= lv;
					double c = pca_direction[0] * v.x + pca_direction[1] * v.y + dpM.at<double>(i - 1, j + k);
					if (c > maxc)
					{
						maxc = c;
						maxj = j + k;
					}
				}
			}
			if (maxj != -1)
			{
				dpM.at<double>(i, j) = maxc;
				dpI.at<int>(i, j) = maxj;
			}
		}
	}
	
	contour.clear();
	double maxc = -1e10;
	int maxj = -1;

	for (int j = 0; j < N; j++)
	{
		if (dpM.at<double>(M - 1, j) > maxc)
		{
			maxc = dpM.at<double>(M - 1, j);
			maxj = j;
		}
	}

	contour.push_back(cv::Point2d(gridx.at<double>(M - 1, maxj), gridy.at<double>(M - 1, maxj)));

	for (int i = M - 1; i > 0; i--)
	{
		int ind = dpI.at<int>(i, maxj);
		contour.push_back(cv::Point2d(gridx.at<double>(i - 1, ind), gridy.at<double>(i - 1, ind)));
		maxj = ind;
	}
}


void test_multiscale()
{
	std::vector<cv::Point2i> strokes;
	compute_stroke_start_and_end(100, strokes);
	
	for (size_t i = 0; i < strokes.size(); i++)
	{
		cout << strokes[i] << endl;
	}
	std::cout << strokes.size() << endl;
}