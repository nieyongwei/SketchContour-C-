#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include "CRSpline.h"
#include "dp.h"
#include "multiscale.h"

using namespace std;

void run_UI(char * name,
	double * gpb,
	const std::vector<cv::Point2d> & pts,
	int width, int height,
	std::vector<cv::Point2d> & contour, double r = 10)
{
	double space = 2;
	int N = 100;
	double alpha = 1;
	double beta = 3;
	int cmax = 3;
	double radius = 30;

	std::vector<cv::Point2d> sams, norms;
	crspline_multiple_inputs_sample(pts, space, sams, norms);

	cv::Mat gridx, gridy;
	compute_grid_points(sams, norms, r, N, gridx, gridy);

	std::vector<cv::Point2i> start_end;
	compute_stroke_start_and_end(gridx.rows, start_end);

	std::vector<std::vector<cv::Point2d>> strokes;
	std::vector<std::vector<cv::Point2i>> positions;
	std::vector<double> mean_energies;

	for (size_t i = 0; i < start_end.size(); i++)
	{
		int sss = start_end[i].x;
		int ttt = start_end[i].y;
		std::vector<cv::Point2d> stroke;
		std::vector<cv::Point2i> position;
		double mean_energy;
		dp(gridx, gridy, sss, ttt, gpb, height, width, cmax, alpha, beta, mean_energy, stroke, position);
		strokes.push_back(stroke);
		positions.push_back(position);
		mean_energies.push_back(mean_energy);
	}

	/*cv::Mat src = cv::imread("C:/Users/nyw/Desktop/BSR/BSDS500/data/images/test/a.jpg");
	for (size_t i = 0; i < strokes.size(); i++)
	{
		for (int j = 0; j < int(strokes[i].size())-1; j++)
		{
			cv::line(src, strokes[i][j], strokes[i][j + 1], cv::Scalar(0, 255, 0));
		}
	}
	cv::imwrite("C:/Users/nyw/Desktop/strokes.png", src);*/

	contour.clear();
	global_integrate_dp(gridx, gridy, norms, strokes, positions, mean_energies, cmax, radius, contour);
}

