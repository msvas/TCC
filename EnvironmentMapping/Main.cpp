#include "Main.h"
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <GL/glew.h>
#include <GL/freeglut.h>

#include <iostream>
#include <string>

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

double tx = 0, ty = 0, tz = 0;

class Main {
	public:
		Main();
		int LoadImg(string imgName);
};


Main::Main() {	
	Mat image1, outImg1, image2, outImg2;		//Create Matrix to store images
	vector<KeyPoint> keypoints1, keypoints2;	//Key points to be found on both images
	Mat descriptors1, descriptors2;
	int minHessian = 400;
	//float Kdata[9] = { 1699, 0, 834, 0, 1696, 607, 0, 0, 1 };
	//Mat K(3, 3, CV_64F, &Kdata);

	image1 = imread("realtest3.png", 0);
	image2 = imread("realtest4.png", 0);

	Ptr<SURF> extractor = SURF::create(minHessian);
	extractor->detect(image1, keypoints1);
	extractor->detect(image2, keypoints2);
	
	//drawKeypoints(image1, keypoints1, outImg1, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	//drawKeypoints(image2, keypoints2, outImg2, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	extractor->compute(image1, keypoints1, descriptors1);
	extractor->compute(image2, keypoints2, descriptors2);

	cout << descriptors1.rows << " " << descriptors2.rows << endl;
	//cout << "Oi";

	//namedWindow("SURF detector img1");
	//imshow("SURF detector img1", outImg1);

	//namedWindow("SURF detector img2");
	//imshow("SURF detector img2", outImg2);

	BFMatcher matcher;
	vector<DMatch> matches, goodMatches;

	if (keypoints1.size() != 0 && keypoints2.size() != 0) {
		matcher.match(descriptors1, descriptors2, matches);
	}

	double max_dist = 0; 
	double min_dist = 100;

	cout << descriptors1.rows << "," << matches.size() << endl;

	//-- Quick calculation of max and min distances between keypoints
	for (int i = 0; i < descriptors1.rows; i++)	{
		double dist = matches[i].distance;

		if (dist < min_dist) 
			min_dist = dist;
		if (dist > max_dist) 
			max_dist = dist;
	}

	for (int i = 0; i < descriptors1.rows; i++) {
		if (matches[i].distance < 3 * min_dist) {
			goodMatches.push_back(matches[i]);
		}
	}

	//-- Localize the object
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;

	for (int i = 0; i < goodMatches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		obj.push_back(keypoints1[goodMatches[i].queryIdx].pt);
		scene.push_back(keypoints2[goodMatches[i].trainIdx].pt);
	}

	Mat H = findHomography(obj, scene, RANSAC);

	cout << "Homography: " << endl;
	cout << H << endl;

	//-- Step 4: calculate Fundamental Matrix
	vector<Point2f>imgpts1, imgpts2;
	for (unsigned int i = 0; i < matches.size(); i++)	{
		// queryIdx is the "left" image
		imgpts1.push_back(keypoints1[matches[i].queryIdx].pt);
		// trainIdx is the "right" image
		imgpts2.push_back(keypoints2[matches[i].trainIdx].pt);
	}
	Mat F = findFundamentalMat(imgpts1, imgpts2, FM_RANSAC, 0.1, 0.99);

	//cout << "F-Matrix size= " << F.rows << "," << F.cols << endl;
	
	Mat img_matches;
	//drawMatches(image1, keypoints1, image2, keypoints2, goodMatches, outImg1, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	//imshow("Good Matches & Object detection", outImg1);

	//-- Step 5: calculate Essential Matrix

	double data[] = {1189.46, 0.0, 805.49,
					0.0, 1191.78, 597.44,
					0.0, 0.0, 1.0 };
	Mat K(3, 3, CV_64F, data);
	Mat_<double> E = K.t() * F * K; //according to HZ (9.12)

	//vector<Mat> rots, trans, normals;
	//decomposeHomographyMat(H, K, rots, trans, normals);

	//cout << "Translations:" << endl;
	//cout << trans[0] << endl;

	//-- Step 6: calculate Rotation Matrix and Translation Vector
	Matx34d P;
	Matx34d P1;
	//decompose E to P' , HZ (9.19)
	SVD svd(E, SVD::MODIFY_A);
	Mat svd_u = svd.u;
	Mat svd_vt = svd.vt;
	Mat svd_w = svd.w;
	Matx33d W(0, -1, 0, 1, 0, 0, 0, 0, 1);//HZ 9.13
	Mat_<double> R = svd_u * Mat(W) * svd_vt; //HZ 9.19
	Mat_<double> t = svd_u.col(2); //u3
	
	tx = t.at<double>(0);
	ty = t.at<double>(1);
	tz = t.at<double>(2);
	
	/*
	tx = trans[0].at<double>(0);
	ty = trans[0].at<double>(1);
	tz = trans[0].at<double>(2);
	*/

	/*

	P1 = Matx34d(R(0, 0), R(0, 1), R(0, 2), t(0),
		R(1, 0), R(1, 1), R(1, 2), t(1),
		R(2, 0), R(2, 1), R(2, 2), t(2));

	//cout << "Translation: " << endl;
	//cout << t << endl;

	//-- Step 7: Reprojection Matrix and rectification data
	Mat R1, R2, P1_, P2_, Q;
	Rect validRoi[2];
	double dist[] = { -0.03432, 0.05332, -0.00347, 0.00106, 0.00000 };
	Mat D(1, 5, CV_64F, dist);

	stereoRectify(K, D, K, D, image1.size(), R, t, R1, R2, P1_, P2_, Q, CV_CALIB_ZERO_DISPARITY, 1, image1.size(), &validRoi[0], &validRoi[1]);

	// create the image in which we will save our disparities
	Mat imgDisparity16S = Mat(image1.rows, image1.cols, CV_16S);
	Mat imgDisparity8U = Mat(image1.rows, image1.cols, CV_8UC1);

	// Call the constructor for StereoBM
	int ndisparities = 16 * 5;  // < Range of disparity >
	int SADWindowSize = 5;      // < Size of the block window > Must be odd. Is the 
							    // size of averaging window used to match pixel  
							    // blocks(larger values mean better robustness to
							    // noise, but yield blurry disparity maps)

	Ptr<StereoBM> sbm = StereoBM::create(ndisparities, SADWindowSize);

	// Calculate the disparity image
	sbm->compute(image1, image2, imgDisparity16S);

	// Check its extreme values
	double minVal; 
	double maxVal;

	minMaxLoc(imgDisparity16S, &minVal, &maxVal);

	printf("Min disp: %f Max value: %f \n", minVal, maxVal);

	// Display it as a CV_8UC1 image
	imgDisparity16S.convertTo(imgDisparity8U, CV_8UC1, 255 / (maxVal - minVal));

	//namedWindow("windowDisparity", CV_WINDOW_NORMAL);
	//imshow("windowDisparity", imgDisparity8U);

	Mat image3d;
	reprojectImageTo3D(imgDisparity8U, image3d, Q, true);
	*/
		
	int key = (waitKey(0) & 0xFF);
	/*while (key != 'q') {
		key = (waitKey(0) & 0xFF);
	}*/
}

int Main::LoadImg(string imgName) {
	Mat image = imread(imgName);
	GLuint texid;

	if (image.empty()) {
		cout << "image empty" << endl;
	}
	else {
		glGenTextures(1, &texid);
		glBindTexture(GL_TEXTURE_2D, texid);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.cols, image.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, image.data);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		image.release();
	}

	return 1;
}

/* Initialize OpenGL Graphics */
void initGL(int w, int h)
{
	glViewport(0, 0, w, h); // use a screen size of WIDTH x HEIGHT
	glEnable(GL_TEXTURE_2D);     // Enable 2D texturing

	glMatrixMode(GL_PROJECTION);     // Make a simple 2D projection on the entire window
	glLoadIdentity();
	glOrtho(0.0, w, h, 0.0, 0.0, 100.0);

	glMatrixMode(GL_MODELVIEW);    // Set the matrix mode to object modeling

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClearDepth(0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Clear the window
}

void drawImage(int translatex, int translatey) {
	int min = 50;
	int max = 150;
	glBegin(GL_QUADS);
	glTexCoord2i(0, 0);
	glVertex2i(min + translatex, min + translatey);
	glTexCoord2i(0, 1);
	glVertex2i(min + translatex, max + translatey);
	glTexCoord2i(1, 1);
	glVertex2i(max + translatex, max + translatey);
	glTexCoord2i(1, 0);
	glVertex2i(max + translatex, min + translatey);
	glEnd();
}

void displayMe(void) {
	Main base;
	base.LoadImg("realtest3.png");
	// Clear color and depth buffers
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);     // Operate on model-view matrix

	/* Draw a quad */
	drawImage(0, 0);

	glutSwapBuffers();

	base.LoadImg("realtest4.png");
	// Clear color and depth buffers
	//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	//glMatrixMode(GL_MODELVIEW);     // Operate on model-view matrix

	/* Draw a quad */
	drawImage(tx*100, ty*100);

	glutSwapBuffers();
}

int main(int argc, char** argv) {
	Main base;

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_SINGLE);
	glutInitWindowSize(1366, 768);                    // window size
	glutInitWindowPosition(0, 0);                // distance from the top-left screen
	glutCreateWindow(argv[0]);    // message displayed on top bar window
	glutDisplayFunc(displayMe);

	initGL(300, 300);
	//base.LoadImg("test.png");

	glutMainLoop();

	return 0;
}