/*
 * This code is provided as part of "A Practical Introduction to Computer Vision with OpenCV"
 * by Kenneth Dawson-Howe © Wiley & Sons Inc. 2014.  All rights reserved.
 */
#include "Utilities.h"
#include "opencv2\features2d.hpp"
#include <iostream>
#include <fstream>

#define first 0
#define second 1
#define third 2
#define fourth 3

static const char* videos[] = {
	"Media/Door1.avi",
	"Media/Door2.avi",
	"Media/Door3.avi",
	"Media/Door4.avi"
};

// Ground Truth
// Door location (TL, TR, BL, BR); First open-close; Second open-close
// (77,65), (225,67), (76,358), (205,359);	94-211; 373-490
// (77,65), (225,67), (75,358), (205,359);	71-184;	339-460
// (221,264), (573,265), (221,1035), (559,1040); 153-251; 380-492
// (221,264), (573,265), (221,1035), (559,1040); 134-230; 311-432

Rect door_1(Point(77, 65), Point(205, 359));
Rect door_2(Point(77, 65), Point(205, 359));
Rect door_3(Point(221, 264), Point(559, 1040));
Rect door_4(Point(221, 264), Point(559, 1040));

bool firstOpening = false, secondOpening = false;
int firstOpenFrame, firstCloseFrame;
int secondOpenFrame, secondCloseFrame;

void saveImage(VideoCapture video, int videoNum, int frameCount) {
	Mat frame;
	video.set(CV_CAP_PROP_POS_FRAMES, frameCount);
	video >> frame;
	imwrite((const char*)("Images/v" + std::to_string(videoNum) +
		"f" + std::to_string(frameCount) + ".png").c_str(), frame);
}

float getAngle(Vec4i line) {
	float x1 = line.val[0], y1 = line.val[1], x2 = line.val[2], y2 = line.val[3];
	return atan2(y2 - y1, x2 - x1) * 180.0 / CV_PI;
}

double calcAngle(Vec4i line) {
	double x1 = line.val[0], y1 = line.val[1], x2 = line.val[2], y2 = line.val[3];
	double x_dist = x1 - x2;
	double y_dist = y1 - y2;
	double x = pow(x_dist, 2), y = pow(y_dist, 2);
	double d = abs(sqrt(x + y));
	double radians = atan2(y, x);

	// return vlaue in degrees
	return radians * 180 / CV_PI;
}

bool isHorizontal(float angle) {
	return (angle > -5 && angle < 5) || (angle > 175 && angle < 185);
}

bool isVertical(float angle) {
	return (angle > 85 && angle < 95) || (angle > -95 && angle < -85);
}

// check if two points are within 25 pixels
bool tolerance(Point p1, Point p2) {
	double x_dist = pow(p1.x - p2.x, 2);
	double y_dist = pow(p1.y - p2.y, 2);
	return abs(sqrt(x_dist + y_dist)) < 25;
}

Point interpolate(Point p1, Point p2) {
	return Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2);
}

void detectDoor(VideoCapture& video) {
	Mat src, output;

	int frame_count = 0;
	while (waitKey(30)) {
		video.set(CV_CAP_PROP_POS_FRAMES, frame_count);
		video >> src;

		// equalise the colours during light changes
		Mat equalised;
		src.copyTo(equalised);
		cvtColor(equalised, equalised, CV_BGR2YCrCb);

		vector<Mat> channels;
		split(equalised, channels);
		equalizeHist(channels[0], channels[0]);
		merge(channels, equalised);
		cvtColor(equalised, equalised, CV_YCrCb2BGR);
		imshow("Equalised", equalised);

		// alternative adaptive threshold + Gaussian
		Mat input;
		src.copyTo(input);
		//cvtColor(input, input, CV_BGR2GRAY);
		//adaptiveThreshold(input, input, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 15, 5);
		//GaussianBlur(input, input, Size(5, 5), 5);

		// generate lines
		vector<Vec4i> lines, horizontalLines, verticalLines;

		Mat grey;
		input.copyTo(grey);
		cvtColor(grey, grey, CV_BGR2GRAY);
		Canny(grey, grey, 0, 255);
		HoughLinesP(grey, lines, 1, CV_PI / 180, 50, 50, 10);

		for (int i = 0; i < lines.size(); i++) {
			Vec4i l = lines[i];
			float ratio = l[1] / grey.rows;
			double angle = getAngle(l);

			if (isHorizontal(angle))
				horizontalLines.push_back(l);
			else if (isVertical(angle))
				verticalLines.push_back(l);
		}

		Mat edges;
		src.copyTo(edges);
		for (Vec4i line : horizontalLines) {
			Point point1(line[0], line[1]);
			Point point2(line[2], line[3]);
			cv::line(edges, point1, point2, Scalar(0, 0, 255), 2);
		}

		for (Vec4i line : verticalLines) {
			Point point1(line[0], line[1]);
			Point point2(line[2], line[3]);
			cv::line(edges, point1, point2, Scalar(255, 0, 0), 2);
		}

		vector<Point> doorPoints;

		for (int h = 0; h < horizontalLines.size(); h++) {
			for (int v = 0; v < verticalLines.size(); v++) {
				Vec4i h_line = horizontalLines[h];
				Vec4i v_line = verticalLines[v];

				Point h_p1 = Point(h_line[0], h_line[1]);
				Point h_p2 = Point(h_line[2], h_line[3]);

				Point v_p1 = Point(v_line[0], v_line[1]);
				Point v_p2 = Point(v_line[2], v_line[3]);

				// check if point within 25 pixels of one another
				if (tolerance(h_p1, v_p1)) {
					doorPoints.push_back(interpolate(h_p1, v_p1));
				}
				if (tolerance(h_p1, v_p2)) {
					doorPoints.push_back(interpolate(h_p1, v_p2));
				}
				if (tolerance(h_p2, v_p1)) {
					doorPoints.push_back(interpolate(h_p2, v_p1));
				}
				if (tolerance(h_p2, v_p2)) {
					doorPoints.push_back(interpolate(h_p2, v_p2));
				}
			}
		}

		for (int c = 0; c < doorPoints.size(); c++) {
			cv::circle(edges, doorPoints[c], 5, Scalar(0, 255, 0), 2);
		}

		cout << horizontalLines.size() << " horizontal lines" << endl;
		cout << verticalLines.size() << " vertical lines" << endl;
		cout << doorPoints.size() << " door points" << endl << endl;

		imshow("Edges", edges);

		if (frame_count == video.get(CAP_POS)
		frame_count += 1;
	}
}

void findDoor(int element) {
	std::string filename(videos[element]);

	VideoCapture video(filename);
	detectDoor(video);
	video.release();
}

int main(int argc, const char** argv) {
	findDoor(0);
	return 0;
}
