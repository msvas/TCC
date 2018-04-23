#include "Inpainting.h"

using namespace cv;
using namespace std;

int TeleaInpaint(const Mat &img) {
	Mat mask;

	cvtColor(img, mask, COLOR_BGR2GRAY);
	mask.convertTo(mask, CV_64FC1, 1.0 / 255.0, 0);

	threshold(mask, mask, 220 /255, 1, THRESH_BINARY_INV);
	mask.convertTo(mask, CV_8U, 255, 0);
	imshow("o", mask);
	waitKey(0);

	Mat dst;
	cout << img.cols << " " << img.rows << " " << mask.cols << " " << mask.rows << endl;
	inpaint(img, mask, dst, 10, INPAINT_TELEA);
	imshow("image", dst);
	waitKey(0);

	return 0;
}