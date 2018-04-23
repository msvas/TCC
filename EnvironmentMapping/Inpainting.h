#pragma once
#include <stdio.h>
#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo/photo.hpp>

int TeleaInpaint(const cv::Mat &img);