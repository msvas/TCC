#pragma once
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#define ATD at<double>
#define elif else if

typedef struct _unfilled_point {
	cv::Point2i pt;
	int fNeighbors;

} ufp;

typedef struct _Matches {
	cv::Point2i pt;
	double err;
} matches;

bool SortUFP(const ufp &v1, const ufp &v2);
bool allFilled(const cv::Mat &mat);
bool isinImage(int y, int x, const cv::Mat &img);
std::vector<cv::Point2i> GetUnfilledNeighbors(const cv::Mat &map);
cv::Mat getNeigborhoodWindow(const cv::Mat &img, cv::Point2i pt, int windowSize);
std::vector<matches> FindMatches(cv::Mat Template, cv::Mat SampleImage, cv::Mat img, cv::Point2i templateCenter, int windowSize, cv::Mat Map);
cv::Mat growImage(const cv::Mat &SampleImage, cv::Mat &Image, int windowSize, cv::Mat &map);
cv::Mat expandImage(const cv::Mat &img, int x_expand, int y_expand, int windowSize);

