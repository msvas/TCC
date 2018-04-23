#include "Inpainting.h"

using namespace cv;
using namespace std;

int TeleaInpaint(const Mat &img) {
	Mat mask;

	threshold(img, mask, 250 / 255, 1, THRESH_BINARY_INV);
	mask.convertTo(mask, CV_16U, 255, 0);

	Mat dst, src;
	src = imread("out.jpg");
	inpaint(src, mask, dst, 3, INPAINT_TELEA);
	imshow("image", dst);
	waitKey(0);

	return 0;
}