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

struct Vector3
{
	double x, y, z;
};

double tx = 0, ty = 0, tz = 0;
float deltaAngle = 0.0f;
float deltaMove = 0;
int xOrigin = -1;
float angle = 0;
float lx = 0.0f, lz = -1.0f;
float x = 0.0f, z = 5.0f;

int imagesTotal = 6;

class Main {
	public:
		vector<Vector3> pairDistances;
		float sizePerImg = 0;
		float sizeX, sizeY;
		std::list<GLuint> textures;

		Main();
		int LoadImg(string imgName);
		bool fileExists(string fileName);
		bool Compare(string img1, string img2);
		GLuint popTex();
};


Main::Main() {			
	int key = (waitKey(0) & 0xFF);
	/*while (key != 'q') {
		key = (waitKey(0) & 0xFF);
	}*/
}

bool Main::fileExists(string fileName)
{
	ifstream infile(fileName);
	return infile.good();
}

int Main::LoadImg(string imgName) {
	Mat image = imread(imgName);
	GLuint texid;

	textures.push_back(texid);

	if (image.empty()) {
		cout << "image empty" << endl;
		return 0;
	}
	else {
		cout << "loading image " << imgName << endl;
		glGenTextures(1, &texid);
		glBindTexture(GL_TEXTURE_2D, texid);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.cols, image.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, image.data);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		image.release();
	}

	return 1;
}

bool Main::Compare(string img1, string img2) {
	Mat image1, outImg1, image2, outImg2;		//Create Matrix to store images
	vector<KeyPoint> keypoints1, keypoints2;	//Key points to be found on both images
	Mat descriptors1, descriptors2;
	int minHessian = 400;

	cout << "Comparing " << img1 << " to " << img2 << endl;

	//pairDistances.clear();

	sizePerImg = (float) 2.0 / imagesTotal;

	if (fileExists(img1) && fileExists(img2)) {
		image1 = imread(img1, 0);
		image2 = imread(img2, 0);

		sizeX = image1.cols;
		sizeY = image1.rows;

		Ptr<SURF> extractor = SURF::create(minHessian);
		extractor->detect(image1, keypoints1);
		extractor->detect(image2, keypoints2);

		//drawKeypoints(image1, keypoints1, outImg1, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		//drawKeypoints(image2, keypoints2, outImg2, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

		extractor->compute(image1, keypoints1, descriptors1);
		extractor->compute(image2, keypoints2, descriptors2);

		cout << descriptors1.rows << " " << descriptors2.rows << endl;

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
		for (int i = 0; i < descriptors1.rows; i++) {
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
		for (unsigned int i = 0; i < matches.size(); i++) {
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

		double data[] = { 1189.46, 0.0, 805.49,
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

		Vector3 newPair;
		newPair.x = t.at<double>(0);
		newPair.y = t.at<double>(1);
		newPair.z = t.at<double>(2);
		pairDistances.push_back(newPair);

		tx = t.at<double>(0);
		ty = t.at<double>(1);
		tz = t.at<double>(2);

	}
	return 0;
}

GLuint Main::popTex() {
	GLuint tex;
	tex = textures.front();
	textures.pop_front();
	return tex;
}

/* Initialize OpenGL Graphics */
void initGL(int w, int h)
{
	glViewport(0, 0, w, h); // use a screen size of WIDTH x HEIGHT
	glDepthFunc(GL_NEVER);
	//glEnable(GL_LIGHTING);

	//glMatrixMode(GL_PROJECTION);     // Make a simple 2D projection on the entire window
	//glLoadIdentity();
	//glOrtho(0.0, w, h, 0.0, 0.0, 100.0);

}

void drawCube() {
	//Multi-colored side - FRONT
	glBegin(GL_POLYGON);

	glColor3f(1.0, 0.0, 0.0);     glVertex3f(0.5, -0.5, -0.5);      // P1 is red
	glColor3f(0.0, 1.0, 0.0);     glVertex3f(0.5, 0.5, -0.5);      // P2 is green
	glColor3f(0.0, 0.0, 1.0);     glVertex3f(-0.5, 0.5, -0.5);      // P3 is blue
	glColor3f(1.0, 0.0, 1.0);     glVertex3f(-0.5, -0.5, -0.5);      // P4 is purple

	glEnd();

	// White side - BACK
	glBegin(GL_POLYGON);
	glColor3f(1.0, 1.0, 1.0);
	glVertex3f(0.5, -0.5, 0.5);
	glVertex3f(0.5, 0.5, 0.5);
	glVertex3f(-0.5, 0.5, 0.5);
	glVertex3f(-0.5, -0.5, 0.5);
	glEnd();

	// Purple side - RIGHT
	glBegin(GL_POLYGON);
	glColor3f(1.0, 0.0, 1.0);
	glVertex3f(0.5, -0.5, -0.5);
	glVertex3f(0.5, 0.5, -0.5);
	glVertex3f(0.5, 0.5, 0.5);
	glVertex3f(0.5, -0.5, 0.5);
	glEnd();

	// Green side - LEFT
	glBegin(GL_POLYGON);
	glColor3f(0.0, 1.0, 0.0);
	glVertex3f(-0.5, -0.5, 0.5);
	glVertex3f(-0.5, 0.5, 0.5);
	glVertex3f(-0.5, 0.5, -0.5);
	glVertex3f(-0.5, -0.5, -0.5);
	glEnd();

	// Blue side - TOP
	glBegin(GL_POLYGON);
	glColor3f(0.0, 0.0, 1.0);
	glVertex3f(0.5, 0.5, 0.5);
	glVertex3f(0.5, 0.5, -0.5);
	glVertex3f(-0.5, 0.5, -0.5);
	glVertex3f(-0.5, 0.5, 0.5);
	glEnd();

	// Red side - BOTTOM
	glBegin(GL_POLYGON);
	glColor3f(1.0, 0.0, 0.0);
	glVertex3f(0.5, -0.5, -0.5);
	glVertex3f(0.5, -0.5, 0.5);
	glVertex3f(-0.5, -0.5, 0.5);
	glVertex3f(-0.5, -0.5, -0.5);
	glEnd();

}

void drawImage(GLuint texture, float translatex, float translatey, float min = -1, float max = 1, float order = 0) {
	cout << "Drawing image " << order << " with " << translatex << ", " << translatey << ", " << min << ", " << max << endl;

	glEnable(GL_TEXTURE_2D);
	glBegin(GL_QUADS);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexCoord2i(0, 0);
	glVertex2f(min + translatex, max + translatey);
	glTexCoord2i(0, 1);
	glVertex2f(min + translatex, min + translatey);
	glTexCoord2i(1, 1);
	glVertex2f(max + translatex, min + translatey);
	glTexCoord2i(1, 0);
	glVertex2f(max + translatex, max + translatey);
	glEnd();
	glDisable(GL_TEXTURE_2D);
}

void drawQuad(GLuint texture, float min = -0.5, float max = 0.5) {
	glEnable(GL_TEXTURE_2D);
	glBegin(GL_QUADS);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexCoord2i(0, 0);
	glVertex2f(min, min);
	glTexCoord2i(0, 1);
	glVertex2f(min, max);
	glTexCoord2i(1, 1);
	glVertex2f(max, max);
	glTexCoord2i(1, 0);
	glVertex2f(max, min);
	glEnd();
	glDisable(GL_TEXTURE_2D);
}

void computePos(float deltaMove) {

	x += deltaMove * lx * 0.1f;
	z += deltaMove * lz * 0.1f;
}

void displayMe(void) {
	Main base;
	string baseName = "coala";
	string format = ".jpg";
	int imgID = 1;
	int imgTotal;
	float min = -1, max = 1;
	float order = 0;
	float originX = 0, originY = 0;

	//computePos(deltaMove);

	while (base.fileExists(baseName + to_string(imgID + 1) + format)) {
		base.Compare(baseName + to_string(imgID) + format, baseName + to_string(imgID + 1) + format);
		imgID++;
	}

	imgTotal = imgID - 1;
	imgID = 1;

	min = -0.5;
	max = min + base.sizePerImg;

	base.LoadImg(baseName + to_string(imgID) + format);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();

	drawImage(base.popTex(), originX, originY, min, max, order);

	for (int i = 0; i < imgTotal; i++) {
		originX = base.pairDistances[i].x * base.sizePerImg;
		originY = base.pairDistances[i].y * base.sizePerImg;

		order += 1;

		base.LoadImg(baseName + to_string(i + 2) + format);
		drawImage(base.popTex(), originX, originY, min, max, order);
	}

	glutSwapBuffers();

	//drawCube();
	//glutSwapBuffers();

	/*
	drawQuad(base.popTex(), -0.5, -0.1);
	base.LoadImg(baseName + to_string(imgID + 1) + format);
	drawQuad(base.popTex(), -0.2, 0.2);
	base.LoadImg(baseName + to_string(imgID + 2) + format);
	drawQuad(base.popTex(), 0.1, 0.5);
	glutSwapBuffers();
	*/
	
}

void processNormalKeys(unsigned char key, int xx, int yy) {

	if (key == 27)
		exit(0);
}

int main(int argc, char** argv) {
	Main base;

	glutInit(&argc, argv);
	//glutInitDisplayMode(GLUT_SINGLE);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(1366, 768);                    // window size
	glutInitWindowPosition(0, 0);                // distance from the top-left screen
	glutCreateWindow(argv[0]);    // message displayed on top bar window
	glutDisplayFunc(displayMe);

	glutIgnoreKeyRepeat(1);
	glutKeyboardFunc(processNormalKeys);

	glewInit();

	initGL(300, 300);
	//base.LoadImg("test.png");

	glutMainLoop();

	return 0;
}