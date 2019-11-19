#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>
#include "run.h"
using namespace std;


struct SC_info
{
	std::vector<std::vector<cv::Point2d>> contours;
	std::vector<std::vector<cv::Point2d>> UIs;
	std::vector<cv::Point2d> current_contour;
	std::vector<cv::Point2d> current_UI;
	
	int show_all_contours;
	int show_cur_contour;
	int show_UI_points;
};

void on_mouse_event(int event, int x, int y, int flags, void * param)
{
	
	SC_info * info = (SC_info*)(param);
	
	if (event == cv::EVENT_LBUTTONDOWN)
	{
		info->current_UI.push_back(cv::Point2d(x, y));
		info->show_UI_points = true;
	}
}

void UI_main(char * name)
{
	std::cout << "Help information:\n"
		<< "Left button: draw initial curve along the target contour (do not double click)\n"
		<<"\'z\': delete the last user edit point\n"
		<< "\'r\': run to extract contour\n"
		<<"\'a\': start another contour\n"
		<<"\'d\': delete the last contour\n"
		<<"\'1\': show all contours or not\n"
		<<"\'2\': show the last contour or not\n"
		<<"\'3\': show user edits or not\n"
		<< "\'l\': increase band width\n"
		<< "\'p\': decrease band width\n"
		<< "\'s\': save result\n"
		<< "ESC: exit\n"
		<< std::endl;
		
	double r = 10;
	SC_info info;
	info.contours.clear();
	info.UIs.clear();
	info.current_contour.clear();
	info.current_UI.clear();

	info.show_all_contours = false;
	info.show_cur_contour = false;
	info.show_UI_points = true;

	char filename[1024];

	/* Load gpb */
	sprintf_s(filename,1024, "./data/%s.bin", name);
	double * gpb;
	gpb = load_gPb(filename);
	
	/* Load source image */
	sprintf_s(filename,1024, "./data/%s.jpg", name);
	cv::Mat src = cv::imread(filename);
	
	int width = src.cols;
	int height = src.rows;

	cv::namedWindow("UI");
	cv::setMouseCallback("UI", on_mouse_event, &info);

	while (true)
	{
		cv::Mat img = src.clone();

		if (info.show_all_contours)
		{
			for (size_t i = 0; i < info.contours.size(); i++)
			{
				for (int j = 0; j < (int)(info.contours[i].size()) - 1; j++)
				{
					cv::line(img, info.contours[i][j], info.contours[i][j + 1], cv::Scalar(255, 0, 0));
				}
			}
		}
		if (info.show_cur_contour)
		{
			for (int i = 0; i < int(info.current_contour.size()) - 1; i++)
			{
				cv::line(img, info.current_contour[i], info.current_contour[i + 1], cv::Scalar(0, 0, 255));
			}
		}
		if (info.show_UI_points)
		{
			for (int i = 0; i < int(info.current_UI.size())-1; i++)
			{
				
				cv::line(img, info.current_UI[i], info.current_UI[i + 1], cv::Scalar(0, 255, 0));
			}
			for(size_t i = 0;i<info.current_UI.size();i++)
			{
				cv::circle(img, info.current_UI[i], 3, cv::Scalar(255, 0, 0), 2);
			}
		}

		cv::imshow("UI", img);
		int key = cv::waitKey(1);
		if (key == 27)
		{
			break;
		}
		else if (key == 'r')
		{
			run_UI(name, gpb, info.current_UI, width, height, info.current_contour, r);
			info.show_cur_contour = true;
		}
		else if (key == 'a')
		{
			if (info.current_UI.size() < 4)
			{
				continue;
			}
			run_UI(name, gpb, info.current_UI, width, height, info.current_contour, r);
			info.contours.push_back(info.current_contour);
			info.UIs.push_back(info.current_UI);
			info.current_contour.clear();
			info.current_UI.clear();
			info.show_all_contours = true;
		}
		else if (key == 'd')
		{
			if (!info.contours.empty())
			{
				info.contours.pop_back();
				info.UIs.pop_back();
				info.show_all_contours = true;
			}
		}
		else if (key == 'z')
		{
			if(!info.current_UI.empty())
				info.current_UI.pop_back();
		}
		else if (key == '1')
		{
			info.show_all_contours = !info.show_all_contours;
		}
		else if (key == '2')
		{
			info.show_cur_contour = !info.show_cur_contour;
		}
		else if (key == '3')
		{
			info.show_UI_points = !info.show_UI_points;
		}
		else if (key == 's')
		{
			cv::Mat save_img = src.clone();
			for (size_t i = 0; i < info.contours.size(); i++)
			{
				for (int j = 0; j < (int)(info.contours[i].size()) - 1; j++)
				{
					cv::line(save_img, info.contours[i][j], info.contours[i][j + 1], cv::Scalar(255, 255, 255));
				}
			}

			char filename1119[1024];
			sprintf_s(filename1119,1024, "./data/%s_result.png", name);
			cv::imwrite(filename1119, save_img);
			
			
		}
		else if (key == 'l')
		{
			r+=1;
			std::cout << r << endl;
		}
		else if (key == 'p')
		{
			r-=1;
			std::cout << r << endl;
		}
	}

	cv::destroyAllWindows();
	delete[] gpb;
}

