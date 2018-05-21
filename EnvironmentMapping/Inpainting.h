#pragma once
#include <stdio.h>
#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo/photo.hpp>

cv::Mat TeleaInpaint(const cv::Mat &img);
cv::Mat GetImgPiece(const cv::Mat img, int xinit, int yinit, int width, int height);
void WriteToFile(cv::Mat &tobewritten);