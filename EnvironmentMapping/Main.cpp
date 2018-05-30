#include "Main.h"
#include "ts.h"
#include "Inpainting.h"
#include "cObj.h"
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
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "glm-0.9.8.5/glm/glm.hpp"
#include "glm-0.9.8.5/glm/gtc/matrix_transform.hpp"
#include "glm-0.9.8.5/glm/gtc/type_ptr.hpp"

#include <iostream>
#include <string>

#define WIDTH 1366
#define HEIGHT 768

using namespace cv;
using namespace std;
using namespace glm;
using namespace cv::xfeatures2d;

struct Vector3
{
	double x, y, z;
};

bool renderBG = true;
bool skyspheremode = false;

// camera movement variables
float cam_angle = 0.0f;
double tx = 0, ty = 0, tz = 0;
float deltaAngle = 0.0f;
float deltaMove = 0;
int xOrigin = -1;
float lx = 0.0f, lz = -1.0f;
float x = 0.0f, z = 5.0f;
// end of camera movement variables

cObj sphere("models/liver.obj");
cObj skysphere("models/sphere3.obj");

glm::mat4 Projection;
glm::mat4 View;
glm::mat4 Model;
glm::mat4 M;

string baseName = "medical";
string imgformat = ".png";

int imagesTotal = 6;

GLuint glTransparency;
GLuint skyspheretex;

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
	Mat imageaux = imread(imgName, IMREAD_UNCHANGED);
	GLuint texid;

	Mat image = Mat(imageaux.rows - 10, imageaux.cols, imageaux.type(), imageaux.data);
	//Mat cropped = GetImgPiece(image, 0, 0, image.cols - 10, image.rows - 10);

	if (image.empty()) {
		cout << "image empty" << endl;
		return 0;
	}
	else {
		cout << "loading image " << imgName << endl;
		glGenTextures(1, &texid);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, texid);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.cols, image.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, image.data);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		image.release();

		textures.push_back(texid);
	}

	cout << "loaded texture: " << texid << " total size: " << textures.size() << endl;

	return 1;
}

bool Main::Compare(string img1, string img2) {
	Mat image1, outImg1, image2, outImg2;		//Create Matrix to store images
	vector<KeyPoint> keypoints1, keypoints2;	//Key points to be found on both images
	Mat descriptors1, descriptors2;
	int minHessian = 600;						// This threshold is used in points detection. A larger value inflicts in less but more salient results, a smaller one makes more results.

	cout << "Comparing " << img1 << " to " << img2 << endl;

	//pairDistances.clear();

	sizePerImg = (float) 6.0 / imagesTotal;

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

		double data[] = {	1189.46, 0.0, 805.49,
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
		Matx33d W(0, -1, 0, 1, 0, 0, 0, 0, 1); //HZ 9.13
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
	list<GLuint>::iterator it = textures.begin();
	advance(it, 0);
	tex = *it;
	textures.pop_front();
	cout << "popped. loaded texture: " << tex << " total size: " << textures.size() << endl;
	return tex;
}

class Skybox {
public:
	GLuint glProgram;
	GLint pvm;
	GLint vertex;
	GLuint vbo_cube_vertices;
	GLuint ibo_cube_indices;
	GLuint cubemap_texture;
	GLsizei cube_indices_size;
	GLuint skyboxVAO, skyboxVBO;

	Skybox();
	void Render();
	void SetBuffers();
};

Skybox::Skybox() {

}

void Skybox::Render() {
	glUseProgram(glProgram);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_CUBE_MAP, cubemap_texture);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_cube_vertices);
	glEnableVertexAttribArray(vertex);
	glVertexAttribPointer(vertex, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_cube_indices);
	glDrawElements(GL_QUADS, cube_indices_size / sizeof(GLushort), GL_UNSIGNED_SHORT, 0);
}

void Skybox::SetBuffers() {
	float vertices[] = {
		//              vertex1				vertex2				vertex3         vertex4
		/* xpos */		+1, -1, +1,			+1, +1, +1,			+1, +1, -1,		+1, -1, -1,
		/* xneg */		-1, -1, -1,			-1, +1, -1,			-1, +1, +1,		-1, -1, +1,
		/* ypos */		-1, +1, -1,			+1, +1, -1,			+1, +1, +1,		-1, +1, +1,
		/* yneg */		-1, -1, +1,			+1, -1, +1,			+1, -1, -1,		-1, -1, -1,
		/* zpos */		+1, -1, +1,			-1, -1, +1,			-1, +1, +1,		+1, +1, +1,
		/* zneg */		-1, -1, -1,			+1, -1, -1,			+1, +1, -1,		-1, +1, -1,
	};

	// vertexarray
	glGenVertexArrays(1, &(skyboxVAO));
	glBindVertexArray(skyboxVAO);

	glGenBuffers(1, &(skyboxVBO));
	glBindBuffer(GL_ARRAY_BUFFER, skyboxVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 3, (void*)(sizeof(float) * 0));
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);
}

Skybox skybox;
Main base;
int imgID = 1;
int imgTotal;

/* Initialize OpenGL Graphics */
void initGL(int w, int h)
{
	glViewport(0, 0, w, h); // use a screen size of WIDTH x HEIGHT
	glDepthFunc(GL_NEVER);
	glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_BLEND);
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
	cout << "Drawing image " << texture << " with " << translatex << ", " << translatey << ", " << min << ", " << max << endl;

	// Enable blending
	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

	float vertices[] = {
		min + translatex, max + translatey, 0, 0,
		min + translatex, min + translatey, 0, 1,
		max + translatex, min + translatey, 1, 1,
		max + translatex, max + translatey, 1, 0
	};

	GLuint vao, vbo;

	// vertexarray
	glGenVertexArrays(1, &(vao));
	glBindVertexArray(vao);

	glGenBuffers(1, &(vbo));
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(0 * sizeof(float)));
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
	glEnableVertexAttribArray(1);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	glEnable(GL_TEXTURE_2D);
	glUseProgram(glTransparency);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture);
	glBindVertexArray(vao);
	glUniform1i(glGetUniformLocation(glTransparency, "texture1"), 0);
	glDrawArrays(GL_QUADS, 0, 4);
	glBindVertexArray(0);
	glBindTexture(GL_TEXTURE_2D, 0);

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

Mat captureView(int width, int height, string printName) {
	Mat img(height, width, CV_8UC3);

	//use fast 4-byte alignment (default anyway) if possible
	glPixelStorei(GL_PACK_ALIGNMENT, (img.step & 3) ? 1 : 4);
	//set length of one complete row in destination data (doesn't need to equal img.cols)
	glPixelStorei(GL_PACK_ROW_LENGTH, img.step / img.elemSize());

	glReadPixels(0, 0, img.cols, img.rows, GL_BGR, GL_UNSIGNED_BYTE, img.data);

	Mat flipped(height, width, CV_8UC3);
	flip(img, flipped, 0);
	
	imwrite(printName, flipped);

	return flipped;
}

void computePos(float deltaMove) {

	x += deltaMove * lx * 0.1f;
	z += deltaMove * lz * 0.1f;
}

void processNormalKeys(unsigned char key, int xx, int yy) {

	if (key == 27)
		exit(0);
}

void pressKey(int key, int xx, int yy) {

	switch (key) {
	case GLUT_KEY_UP: deltaMove = 0.5f; break;
	case GLUT_KEY_DOWN: deltaMove = -0.5f; break;
	}
}

void releaseKey(int key, int x, int y) {

	switch (key) {
	case GLUT_KEY_UP:
	case GLUT_KEY_DOWN: deltaMove = 0; break;
	}
}

void mouseMove(int x, int y) {

	// this will only be true when the left button is down
	if (xOrigin >= 0) {

		// update deltaAngle
		deltaAngle = (x - xOrigin) * 0.001f;

		// update camera's direction
		lx = sin(cam_angle + deltaAngle);
		lz = -cos(cam_angle + deltaAngle);
	}
}

void mouseButton(int button, int state, int x, int y) {

	// only start motion if the left button is pressed
	if (button == GLUT_LEFT_BUTTON) {

		// when the button is released
		if (state == GLUT_UP) {
			cam_angle += deltaAngle;
			xOrigin = -1;
		}
		else {// state = GLUT_DOWN
			xOrigin = x;
		}
	}
}

unsigned int loadCubemap(vector<std::string> faces)
{
	unsigned int textureID;
	glGenTextures(1, &textureID);
	glBindTexture(GL_TEXTURE_CUBE_MAP, textureID);

	int width, height, nrChannels;
	for (unsigned int i = 0; i < faces.size(); i++)
	{
		unsigned char *data = stbi_load(faces[i].c_str(), &width, &height, &nrChannels, 0);
		if (data)
		{
			glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
			stbi_image_free(data);
		}
		else
		{
			std::cout << "Cubemap texture failed to load at path: " << faces[i] << std::endl;
			stbi_image_free(data);
		}
	}
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

	return textureID;
}

void setupCubeMap(GLuint& texture, Mat &xpos, Mat &xneg, Mat &ypos, Mat &yneg, Mat &zpos, Mat &zneg) {
	glActiveTexture(GL_TEXTURE0);
	glEnable(GL_TEXTURE_CUBE_MAP);
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_CUBE_MAP, texture);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X, 0, GL_RGB, xpos.cols, xpos.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, xpos.data);
	glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, 0, GL_RGB, xneg.cols, xneg.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, xneg.data);
	glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, 0, GL_RGB, ypos.cols, ypos.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, ypos.data);
	glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, 0, GL_RGB, yneg.cols, yneg.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, yneg.data);
	glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, 0, GL_RGB, zpos.cols, zpos.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, zpos.data);
	glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, 0, GL_RGB, zneg.cols, zneg.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, zneg.data);
}

GLuint LoadTexture(string filename) {
	GLuint texid;

	Mat image = imread(filename);

	glGenTextures(1, &texid);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texid);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.cols, image.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, image.data);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	return texid;
}

void DrawSkysphere(GLuint texture) {
	//Set up sky texture
	glEnable(GL_TEXTURE_2D);
	glEnable(GL_DEPTH_TEST);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture);
	cout << "t: " << texture << endl;
	GLUquadricObj *qobj = gluNewQuadric();
	gluQuadricOrientation(qobj, GLU_INSIDE);
	gluQuadricNormals(qobj, GLU_SMOOTH);
	gluQuadricTexture(qobj, GL_TRUE);

	glRotatef(90, 1.0f, 0.0f, 0.0f);  //Important!
	//glRotatef(rotation, 0.0, 0.0, 1.0);
	//rotation += 0.02;
	gluSphere(qobj, 5, 32, 32);
}

void displayMe(void) {
	float min = -1, max = 1;
	float order = 0;
	float originX = 0, originY = 0;

	if (deltaMove) {
		computePos(deltaMove);
		cout << "moved camera" << endl;
	}

	min = -0.5;
	max = min + base.sizePerImg;

	glMatrixMode(GL_MODELVIEW);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();

	if (renderBG) {

		imgID = 1;

		//base.LoadImg(baseName + to_string(imgID) + imgformat);

		drawImage(base.popTex(), originX, originY, min, max, order);

		for (int i = 0; i < imgTotal; i++) {
			originX = base.pairDistances[i].x * base.sizePerImg;
			originY = base.pairDistances[i].y * base.sizePerImg;

			order += 1;
			//base.LoadImg(baseName + to_string(i + 2) + imgformat);
			drawImage(base.popTex(), originX, originY, min, max, order);
		}

		Mat wholeTex;
		wholeTex = captureView(1366, 768, "out.jpg");

		// Extend  gray image.
		//Mat img = imread("out.jpg");
		
		Mat reduced, inpainted;
		if (!wholeTex.empty()) {
			reduced = reduceBlackPixels(wholeTex);
			inpainted = TeleaInpaint(reduced);
		}

		//imshow("inpainted", inpainted);

		Mat crop = GetImgPiece(inpainted, 0, 0, 512, 512);
		imwrite("cropped.jpg", crop);
		skyspheretex = LoadTexture("cropped.jpg");

		int xsize = floor(inpainted.cols / 4);
		int ysize = floor(inpainted.rows / 3);

		// finds the next power of 2
		int nextx = pow(2, ceil(log(xsize) / log(2)));
		int nexty = pow(2, ceil(log(ysize) / log(2)));

		// gets the smaller one
		int biggerpower = nextx < nexty ? nextx : nexty;

		/*
		Mat top = GetImgPiece(inpainted, xsize, 0, xsize, ysize);
		Mat left = GetImgPiece(inpainted, 0, ysize, xsize, ysize);
		Mat front = GetImgPiece(inpainted, xsize, ysize, xsize, ysize);
		Mat right = GetImgPiece(inpainted, 2 * xsize, ysize, xsize, ysize);
		Mat back = GetImgPiece(inpainted, 3 * xsize, ysize, xsize, ysize);
		Mat bottom = GetImgPiece(inpainted, xsize, 2 * ysize, xsize, ysize);
		*/

		Mat top = GetImgPiece(inpainted, xsize, 0, biggerpower, biggerpower);
		Mat left = GetImgPiece(inpainted, 0, biggerpower, biggerpower, biggerpower);
		Mat front = GetImgPiece(inpainted, xsize, biggerpower, biggerpower, biggerpower);
		Mat right = GetImgPiece(inpainted, 2 * xsize, biggerpower, biggerpower, biggerpower);
		Mat back = GetImgPiece(inpainted, 4 * xsize - biggerpower, ysize, biggerpower, biggerpower);
		Mat bottom = GetImgPiece(inpainted, xsize, 3 * ysize - biggerpower, biggerpower, biggerpower);

		imwrite("cubemap/top.jpg", top);
		imwrite("cubemap/left.jpg", left);
		imwrite("cubemap/front.jpg", front);
		imwrite("cubemap/right.jpg", right);
		imwrite("cubemap/back.jpg", back);
		imwrite("cubemap/bottom.jpg", bottom);

		//Mat resized;
		//resize(inpainted, resized, Size(inpainted.cols / 3, inpainted.rows / 3));
		//Mat result = expandImage(inpainted, 1, 1, 15);
		//imshow("img2", result);

		// cubemap
		skybox.cubemap_texture = loadCubemap({
			"cubemap/back.jpg",
			"cubemap/bottom.jpg",
			"cubemap/front.jpg",
			"cubemap/left.jpg",
			"cubemap/right.jpg",
			"cubemap/top.jpg"
		});
		
		//setupCubeMap(skybox.cubemap_texture, top, left, front, right, back, bottom);

		renderBG = false;
	} 
	else {
		View =  glm::lookAt(glm::vec3(x, 0.0f, z), glm::vec3(x + lx, 0.0f, z + lz), glm::vec3(0.0f, 1.0f, 0.0f));

		glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
		glClearDepth(1.0f);
		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LEQUAL);

		glPushMatrix();
		// seamless cubemap
		glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
		// skybox will be drawn first, so disallow writing into depth buffer
		glDepthMask(GL_FALSE);
		glUseProgram(skybox.glProgram);
		//View = glm::rotate(View, deltaAngle, glm::vec3(0.0f, 1.0f, 0.0f));
		Model = glm::mat4(1.0f);
		Model = glm::scale(Model, glm::vec3(600.0f, 600.0f, 600.0f));
		Model = glm::translate(Model, glm::vec3(0.0f, 0.0f, 0.0f));
		Projection = glm::perspective(45.0f, (float)WIDTH / (float)HEIGHT, 0.1f, 1000.0f);
		//View = glm::mat4(1.0f);
		glUniformMatrix4fv(glGetUniformLocation(skybox.glProgram, "Model"), 1, GL_FALSE, glm::value_ptr(Model));   // render cube around the view position
		glUniformMatrix4fv(glGetUniformLocation(skybox.glProgram, "View"), 1, GL_FALSE, glm::value_ptr(View));	    // use instead glm::lookAt(...)
		glUniformMatrix4fv(glGetUniformLocation(skybox.glProgram, "Projection"), 1, GL_FALSE, glm::value_ptr(Projection));
		glUniform1i(glGetUniformLocation(skybox.glProgram, "skybox"), 0);
		glActiveTexture(GL_TEXTURE0 + 0);
		glBindTexture(GL_TEXTURE_CUBE_MAP, skybox.cubemap_texture);
		glBindVertexArray(skybox.skyboxVAO);
		glDrawArrays(GL_QUADS, 0, 24);
		glBindVertexArray(0);
		glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
		glUseProgram(0);
		glDepthMask(GL_TRUE);
		//DrawSkysphere(skyspheretex);
		glPopMatrix();

		glPushMatrix();
		//View = glm::mat4(1.0f);
		//View = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 1));
		// render teapot
		Model = glm::mat4(0);
		//Model = glm::rotate(Model, .667f, glm::vec3(0.0f, 1.0f, 0.0f));
		//Model = glm::rotate(Model, .667f, glm::vec3(1.0f, 0.0f, 0.0f));
		//Model = glm::rotate(Model, .667f, glm::vec3(0.0f, 0.0f, 1.0f));
		Model = glm::translate(View, glm::vec3(0.0f, 0.0f, 0.0f));
		Model = glm::scale(Model, glm::vec3(0.2f, 0.2f, 0.2f));
		glm::vec3 light_position = glm::vec3(0.0f, 100.0f, 100.0f);
		glUseProgram(sphere.programID);
		glUniform3f(sphere.light_position1, light_position.x, light_position.y, light_position.z);
		glUniformMatrix4fv(sphere.Projection1, 1, GL_FALSE, glm::value_ptr(Projection));
		glUniformMatrix4fv(sphere.View1, 1, GL_FALSE, glm::value_ptr(View));
		glUniformMatrix4fv(sphere.Model1, 1, GL_FALSE, glm::value_ptr(Model));
		glUniform1i(glGetUniformLocation(sphere.programID, "cubemap"), 0);
		glActiveTexture(GL_TEXTURE0 + 0);
		glBindTexture(GL_TEXTURE_CUBE_MAP, skybox.cubemap_texture);

		sphere.render();
		glPopMatrix();

		if (skyspheremode) {
			// skysphere renderind
			glPushMatrix();
			Model = glm::mat4(1.0f);
			Model = glm::translate(View, glm::vec3(0.0f, 0.0f, 0.0f));
			Model = glm::scale(Model, glm::vec3(10.0f, 10.0f, 10.0f));
			glUseProgram(skysphere.programID);
			glUniformMatrix4fv(glGetUniformLocation(skysphere.programID, "Projection"), 1, GL_FALSE, glm::value_ptr(Projection));
			glUniformMatrix4fv(glGetUniformLocation(skysphere.programID, "View"), 1, GL_FALSE, glm::value_ptr(View));
			glUniformMatrix4fv(glGetUniformLocation(skysphere.programID, "Model"), 1, GL_FALSE, glm::value_ptr(Model));
			glUniform1i(glGetUniformLocation(skysphere.programID, "texture1"), 0);
			glActiveTexture(GL_TEXTURE0 + 0);
			glBindTexture(GL_TEXTURE_2D, skyspheretex);
			skysphere.render();
			glPopMatrix();
		}
	}
	//drawCube();
	
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

char* loadFile(const char *filename) {
	char* data;
	int len;
	std::ifstream ifs(filename, std::ifstream::in);

	ifs.seekg(0, std::ios::end);
	len = ifs.tellg();

	ifs.seekg(0, std::ios::beg);
	data = new char[len + 1];

	ifs.read(data, len);
	data[len] = 0;

	ifs.close();

	return data;
}

void createProgram(GLuint& glProgram, GLuint& glShaderV, GLuint& glShaderF, const char* vertex_shader, const char* fragment_shader) {
	glShaderV = glCreateShader(GL_VERTEX_SHADER);
	glShaderF = glCreateShader(GL_FRAGMENT_SHADER);
	const GLchar* vShaderSource = loadFile(vertex_shader);
	const GLchar* fShaderSource = loadFile(fragment_shader);
	glShaderSource(glShaderV, 1, &vShaderSource, NULL);
	glShaderSource(glShaderF, 1, &fShaderSource, NULL);
	//delete[] vShaderSource;
	//delete[] fShaderSource;
	glCompileShader(glShaderV);
	glCompileShader(glShaderF);
	glProgram = glCreateProgram();
	glAttachShader(glProgram, glShaderV);
	glAttachShader(glProgram, glShaderF);
	glLinkProgram(glProgram);
	//glUseProgram(glProgram);

	int  vlength, flength, plength;
	char vlog[2048], flog[2048], plog[2048];
	glGetShaderInfoLog(glShaderV, 2048, &vlength, vlog);
	glGetShaderInfoLog(glShaderF, 2048, &flength, flog);
	glGetProgramInfoLog(glProgram, 2048, &flength, plog);
	std::cout << vlog << std::endl << std::endl << flog << std::endl << std::endl << plog << std::endl << std::endl;
}

void releaseProgram(GLuint& glProgram, GLuint glShaderV, GLuint glShaderF) {
	glDetachShader(glProgram, glShaderF);
	glDetachShader(glProgram, glShaderV);
	glDeleteShader(glShaderF);
	glDeleteShader(glShaderV);
	glDeleteProgram(glProgram);
}

int main(int argc, char** argv) {
	while (base.fileExists(baseName + to_string(imgID + 1) + imgformat)) {
		base.Compare(baseName + to_string(imgID) + imgformat, baseName + to_string(imgID + 1) + imgformat);
		imgID++;
	}

	View = glm::mat4(1.0f);

	glutInit(&argc, argv);
	//glutInitDisplayMode(GLUT_SINGLE);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(WIDTH, HEIGHT);                    // window size
	glutInitWindowPosition(0, 0);                // distance from the top-left screen
	glutCreateWindow(argv[0]);    // message displayed on top bar window
	glutDisplayFunc(displayMe);

	glutIgnoreKeyRepeat(1);
	glutKeyboardFunc(processNormalKeys);
	glutSpecialFunc(pressKey);
	glutSpecialUpFunc(releaseKey);

	// here are the two new functions
	glutMouseFunc(mouseButton);
	glutMotionFunc(mouseMove);

	glewInit();

	imgTotal = imgID - 1;
	imgID = 1;

	base.LoadImg(baseName + to_string(imgID) + imgformat);

	for (int i = 0; i < imgTotal; i++) {
		base.LoadImg(baseName + to_string(i + 2) + imgformat);
	}

	initGL(WIDTH, HEIGHT);

	GLuint glShaderVT, glShaderFT;
	createProgram(glTransparency, glShaderVT, glShaderFT, "shaders/v_transparency.sh", "shaders/f_transparency.sh");
	
	skybox.SetBuffers();

	if (skyspheremode) {
		GLuint glSS, glShaderVS, glShaderFS;
		createProgram(glSS, glShaderVS, glShaderFS, "shaders/skysphere_v.sh", "shaders/skysphere_f.sh");
		skysphere.vertex1 = glGetAttribLocation(glSS, "vertex");
		skysphere.normal1 = glGetAttribLocation(glSS, "normal");
		skysphere.programID = glSS;
		skysphere.setupBufferObjects();
	}

	GLuint glShaderV, glShaderF;
	createProgram(skybox.glProgram, glShaderV, glShaderF, "shaders/vertex.sh", "shaders/fragment.sh");

	GLuint glProgram1, glShaderV1, glShaderF1;
	createProgram(glProgram1, glShaderV1, glShaderF1, "shaders/vertex1.sh", "shaders/fragment1.sh");
	sphere.vertex1 = glGetAttribLocation(glProgram1, "vertex");
	sphere.normal1 = glGetAttribLocation(glProgram1, "normal");
	sphere.light_position1 = glGetUniformLocation(glProgram1, "light_position");
	sphere.Projection1 = glGetUniformLocation(glProgram1, "Projection");
	sphere.View1 = glGetUniformLocation(glProgram1, "View");
	sphere.Model1 = glGetUniformLocation(glProgram1, "Model");

	sphere.programID = glProgram1;

	sphere.setupBufferObjects();

	glutMainLoop();

	return 0;
}