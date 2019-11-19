#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include "dp.h"

using namespace std;

cv::Point2d crspline_point(const cv::Point2d & p0, 
	const cv::Point2d & p1, 
	const cv::Point2d & p2, 
	const cv::Point2d & p3, 
	double t)
{
	double t3 = t * t * t;
	double t2 = t * t;
	
	double w0 = -0.5 * t3 + t2 - 0.5 * t;
	double w1 = 1.5 * t3 - 2.5 * t2 + 1.0;
	double w2 = -1.5 * t3 + 2 * t2 + 0.5 * t;
	double w3 = 0.5 * t3 - 0.5 * t2;
	
	cv::Point2d ans = p0 * w0 + p1 * w1 + p2 * w2 + p3 * w3;
	return ans;
}

// Length from p1 to p2
double crspline_length(cv::Point2d p0, cv::Point2d p1, cv::Point2d p2, cv::Point2d p3)
{
	int num = 1000;
	double length = 0.0;
	
	cv::Point2d a = p1;
	
	for (int i = 1; i < num + 1; i++)
	{
		double t = double(i) / double(num);
		cv::Point2d b = crspline_point(p0, p1, p2, p3, t);
		double x = b.x - a.x;
		double y = b.y - a.y;
		length += sqrt(x*x + y*y);
		a = b;
	}

	return length;
}

void linear_sample(cv::Point2d p0, cv::Point2d p1, double space, std::vector<cv::Point2d> & sams,
	std::vector<cv::Point2d> & norms)
{
	double x = p0.x - p1.x;
	double y = p0.y - p1.y;
	
	double l = sqrt(x*x + y*y);

	cv::Point2d v = p1 - p0;
	
	double vvx = v.x / l;
	double vvy = v.y / l;
	
	cv::Point2d norm(vvy, -vvx);

	/*if (l < space)
	{
		sams.push_back(p0);
		norms.push_back(norm);
		sams.push_back(p1);
		norms.push_back(norm);
		return;

	}*/
	assert(l > space);

	int n = int(l / space);
	double s = l / double(n);

	sams.resize(n + 1);
	norms.resize(n + 1);

	
	for (int i = 0; i < n + 1; i++)
	{
		double t = double(i) / double(n);
		cv::Point2d pt = p0 * (1 - t) + p1 *t;
		sams[i] = pt;
		norms[i] = norm;
	}
}

void crspline_sample(cv::Point2d p0, cv::Point2d p1, cv::Point2d p2, cv::Point2d p3,
	double space,
	std::vector<cv::Point2d> & sams,
	std::vector<cv::Point2d> & norms)
{
	int num = 1000;
	double length = 0.0;
	
	std::vector<cv::Point2d> allp(num + 1);
	std::vector<double> dist(num + 1);
	
	cv::Point2d a = p1;
	
	for (int i = 0; i < num + 1; i++)
	{
		double t = double(i) / num;
		cv::Point2d b = crspline_point(p0, p1, p2, p3, t);
		allp[i] = b;
		double x = b.x - a.x;
		double y = b.y - a.y;
		double d = sqrt(x*x + y*y);
		length += d;
		dist[i] = length;
		a = b;
	}

	assert(length >= space);

	int n = (int)(length / space);
	double s = length / n;

	sams.resize(n + 1);
	std::vector<int> sam_index(n + 1);

	sams[0] = p1;
	sam_index[0] = 0;
	sams[n] = p2;
	sam_index[n] = num;

	int j = 1;
	for (int i = 1; i < num; i++)
	{
		if (dist[i] >= j*s)
		{
			sams[j] = allp[i];
			sam_index[j] = i;
			j += 1;
		}
	}

	norms.resize(n + 1);
	for (int i = 0; i < n + 1; i++)
	{
		j = sam_index[i];
		cv::Point2d cur = allp[j];
		cv::Point2d pre = cur;
		cv::Point2d post = cur;

		if (j > 0)
		{
			pre = allp[j - 1];
		}
		if (j < num - 1)
		{
			post = allp[j + 1];
		}

		cv::Point2d v1 = cur - pre;
		cv::Point2d v2 = post - cur;

		cv::Point2d v = v1 + v2;
		double lv = sqrt(v.x * v.x + v.y * v.y);
		v = v / lv;

		norms[i] = cv::Point2d(v.y, -v.x);
	}
}

void crspline_multiple_inputs_sample(const std::vector<cv::Point2d> & pts, double space,
	std::vector<cv::Point2d> & sams,
	std::vector<cv::Point2d> & norms)
{
	int m = int(pts.size());
	assert(m >= 4);

	std::vector<std::vector<cv::Point2d>> alls, alln;
	std::vector<cv::Point2d> aa, bb, cc, dd, ss, nn;

	linear_sample(pts[0], pts[1], space,aa,bb);
	aa.pop_back();
	bb.pop_back();
	alls.push_back(aa);
	alln.push_back(bb);

	for (int i = 0; i < m - 3; i++)
	{
		cv::Point2d p0 = pts[i];
		cv::Point2d p1 = pts[i+1];
		cv::Point2d p2 = pts[i+2];
		cv::Point2d p3 = pts[i+3];

		crspline_sample(p0, p1, p2, p3, space, ss, nn);
		ss.pop_back();
		nn.pop_back();
		alls.push_back(ss);
		alln.push_back(nn);
	}

	linear_sample(pts[m - 2], pts[m - 1], space, cc, dd);
	alls.push_back(cc);
	alln.push_back(dd);

	sams.clear();
	norms.clear();

	for (size_t i = 0; i < alls.size(); i++)
	{
		for (size_t j = 0; j < alls[i].size(); j++)
		{
			sams.push_back(alls[i][j]);
			norms.push_back(alln[i][j]);
		}
	}
}

void test_crspline()
{
	cv::Point2d p0(100, 100);
	cv::Point2d p1(120, 400);
	cv::Point2d p2(350, 200);
	cv::Point2d p3(400, 100);
	cv::Point2d p4(450, 200);
	cv::Point2d p5(480, 300);
	
	std::vector<cv::Point2d> pts, sams, norms;
	pts.push_back(p0);
	pts.push_back(p1);
	pts.push_back(p2);
	pts.push_back(p3);
	pts.push_back(p4);
	pts.push_back(p5);

	crspline_multiple_inputs_sample(pts, 5, sams, norms);

	cout << sams.size() << endl;
	for (size_t i = 0; i < sams.size(); i++)
	{
		cout << sams[i] << endl;
	}

	cout << norms.size() << endl;
	for (size_t i = 0; i < norms.size(); i++)
	{
		cout << norms[i] << endl;
	}

	cv::Mat gridx, gridy;
	compute_grid_points(sams, norms, 50, 20, gridx, gridy);
	
	for (int i = 0; i < gridx.rows; i++)
	{
		for (int j = 0; j < gridx.cols; j++)
		{
			cout << gridy.at<double>(i, j) << ",";
		}
		cout << endl;
	}
}