#include <stdio.h>
#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

void CalibrateCamera() {
	vector<Point2f> imageCorners;
	vector<Point3f> objectCorners;
	vector<vector<Point3f>> objectPoints;
	vector<vector<Point2f>> imagePoints;
	Size patternsize(8, 7);
	Size imageSize;
	Mat cameraMatrix;
	Mat distCoeffs;
	Mat viewGray;
	Mat frame;

	vector<string> filelist = { "calib_example/left01.jpg",
								"calib_example/left02.jpg",
								"calib_example/left03.jpg",
								"calib_example/left04.jpg",
								"calib_example/left05.jpg",
								"calib_example/left06.jpg",
								"calib_example/left07.jpg",
								"calib_example/left08.jpg",
								"calib_example/left09.jpg",
								"calib_example/left10.jpg"
							};

	for (int i = 0; i < patternsize.height; i++) {
		for (int j = 0; j < patternsize.width; j++) {
			objectCorners.push_back(cv::Point3f(i, j, 0.0f));
		}
	}

	bool found;

	for (int i = 0; i < filelist.size() - 1; i++) {
		frame = imread(filelist[i]);
		imageSize = Size(frame.rows, frame.cols);

		found = findChessboardCorners(frame, patternsize, imageCorners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);

		if (found) {
			cvtColor(frame, viewGray, CV_BGR2GRAY);
			cornerSubPix(viewGray, imageCorners, Size(5, 5), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

			if (imageCorners.size() == patternsize.area()) {
				// 2D image points from one view
				imagePoints.push_back(imageCorners);
				// corresponding 3D scene points
				objectPoints.push_back(objectCorners);
			}
		}

		drawChessboardCorners(frame, patternsize, Mat(imageCorners), found);
	}

		std::vector<cv::Mat> rvecs, tvecs;
		calibrateCamera(objectPoints, // the 3D points
						imagePoints,  // the image points
						imageSize,    // image size
						cameraMatrix, // output camera matrix
						distCoeffs,   // output distortion matrix
						rvecs, tvecs); // Rs, Ts 

		cout << cameraMatrix << endl;
}

int main() {
	CalibrateCamera();
}
