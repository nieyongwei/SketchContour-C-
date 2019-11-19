#pragma once
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace std;

double * load_gPb(string filename)
{
	ifstream fin(filename.c_str(), ios::binary);

	istream::pos_type curpos = fin.tellg();
	fin.seekg(0, ios_base::end);
	istream::pos_type sz = fin.tellg();
	fin.seekg(curpos);
	sz = sz - curpos;
	int num = (int)(sz / sizeof(double));
	double * gpb = new double[num];
	
	for (int i = 0; i < num; i++)
	{
		double ele;
		fin.read((char*)&ele, sizeof(double));
		gpb[i] = ele;
	}
	
	return gpb;
}

double get_gpb_at(double * gpb, int cols, int r, int c, int d)
{
	int index = r * cols * 8 + c * 8 + d;
	return gpb[index];
}

int tangent_to_index(cv::Point2d tang)
{
	int arr[32] = { 4,3,3,2,2,1,1,0,0,7,7,6,6,5,5,4,4,3,3,2,2,1,1,0,0,7,7,6,6,5,5,4 };
	double angle = atan2(tang.y, tang.x);
	if (angle < 0)
	{
		angle += 2 * 3.14159265359;
	}

	int n = (int)(angle / (3.14159265359 / 16));
	if (n >= 32)
	{
		n = 31;
	}

	return arr[n];
}

double get_gPb(double * gpb, int rows, int cols, cv::Point2d pt, cv::Point2d tang)
{
	if (pt.x < 0 || pt.x >= cols - 1|| pt.y < 0 || pt.y >= rows - 1)
	{
		return -1e5;
	}

	int index = tangent_to_index(tang);
	int x = int(pt.x);
	int y = int(pt.y);

	double t = pt.x - x;
	double s = pt.y - y;

	double a = get_gpb_at(gpb, cols, y, x, index);
	double b = get_gpb_at(gpb, cols, y, x + 1, index);
	double c = get_gpb_at(gpb, cols, y + 1, x + 1, index);
	double d = get_gpb_at(gpb, cols, y + 1, x, index);

	double e = a * (1 - t) + b * t;
	double f = d * (1 - t) + c * t;

	double g = e * (1 - s) + f * s;

	return g;
}

void test_gpb()
{
	//double * gpb;
	//gpb = load_gPb("C:/Users/nyw/Desktop/BSR/BSDS500/data/images/gpb_py/79073.bin");
	//int width = 321;
	//cout << get_gpb_at(gpb, width, 0, 0, 0) << endl;
	//cout << get_gpb_at(gpb, width, 30, 50, 6) << endl;


	///*cv::Point2d tang(0, 4), pt(176, 108);
	//double g = get_gPb(gpb, width, pt, tang);
	//cout << g << endl;*/
	//delete[] gpb;

	double * gpb = load_gPb("C:/Users/nyw/Desktop/BSR/BSDS500/data/gpb/246009.bin");
	cv::Mat img = cv::imread("C:/Users/nyw/Desktop/BSR/BSDS500/data/images/test/246009.jpg");
	int width = img.cols;
	cout << img.rows << "," << img.cols << endl;

	cv::Mat newimg(img.size(), CV_8U, cv::Scalar(0));
	
	for (int k = 0; k < 7; k++)
	{
		for (int i = 0; i < img.rows; i++)
		{
			for (int j = 0; j < img.cols; j++)
			{
				double g = get_gpb_at(gpb, width, i, j, k);
				newimg.at<uchar>(i, j) = uchar(g * 255);
			}
		}

		cv::imshow("hello", newimg);
		int key = cv::waitKey(0);
	}
	

	delete[] gpb;

	
}