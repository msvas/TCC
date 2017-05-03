#include "Main.h"
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <iostream>
#include <string>

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

class Main {
	public:
		Main();

};


Main::Main() {	
	Mat image1, outImg1, image2, outImg2;		//Create Matrix to store images
	vector<KeyPoint> keypoints1, keypoints2;	//Key points to be found on both images

	image1 = imread("stereotest1.jpg", 0);
	image2 = imread("stereotest2.jpg", 0);

	Ptr<SURF> extractor = SURF::create();
	extractor->detect(image1, keypoints1);
	extractor->detect(image2, keypoints2);
	drawKeypoints(image1, keypoints1, outImg1, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	drawKeypoints(image2, keypoints2, outImg2, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	namedWindow("SURF detector img1");
	imshow("SURF detector img1", outImg1);

	namedWindow("SURF detector img2");
	imshow("SURF detector img2", outImg2);

	waitKey();
}

int main() {
	Main base;

	return 0;
}