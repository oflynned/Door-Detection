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

const Rect door_1(Point(77, 65), Point(205, 359));
const Rect door_2(Point(77, 65), Point(205, 359));
const Rect door_3(Point(221, 264), Point(559, 1040));
const Rect door_4(Point(221, 264), Point(559, 1040));

int f_open = -1, f_close = -1, s_open = -1, s_close = -1;
bool firstEvent = false, secondEvent = false;

void saveImage(VideoCapture video, int videoNum, int frameCount) {
	Mat frame;
	video.set(CV_CAP_PROP_POS_FRAMES, frameCount);
	video >> frame;
	imwrite((const char*)("Images/v" + std::to_string(videoNum) +
		"f" + std::to_string(frameCount) + ".png").c_str(), frame);
}

void saveImage(Mat frame, int videoNum, int frameCount) {
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

	// return value in degrees
	return radians * 180 / CV_PI;
}

bool isHorizontal(float angle) {
	return (angle > -5 && angle < 5) || (angle > 175 && angle < 185);
}

bool isVertical(float angle) {
	return (angle > 85 && angle < 95) || (angle > -95 && angle < -85);
}

bool isSkewed(float angle) {
	return (angle > -45 && angle < 0);
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

void equaliseLighting(Mat image, bool showOutput = false) {
	// equalise the colours during light changes
	cvtColor(image, image, CV_BGR2YCrCb);

	vector<Mat> channels;
	split(image, channels);
	equalizeHist(channels[0], channels[0]);
	merge(channels, image);
	cvtColor(image, image, CV_YCrCb2BGR);

	if(showOutput)
		imshow("Equalised", image);
}

void detectDoor(VideoCapture& video, int element) {
	Mat src, output;

	int total_frames = video.get(CAP_PROP_FRAME_COUNT);
	int frame_count = 0;
	while (waitKey(30) && frame_count != total_frames) {
		video.set(CV_CAP_PROP_POS_FRAMES, frame_count);
		video >> src;
		
		// resize larger videos to half-size
		if(element == 2 || element == 3)
			resize(src, src, Size(), 0.5, 0.5);

		// generate lines
		vector<Vec4i> lines, horizontalLines, verticalLines, skewedLines;

		// Canny for getting outlines
		Mat grey;
		src.copyTo(grey);
		cvtColor(grey, grey, CV_BGR2GRAY);
		Canny(grey, grey, 50, 255);
		imshow("Canny", grey);

		// detect hough lines probabilistically
		HoughLinesP(grey, lines, 1, CV_PI / 180, 50, 100, 10);

		for (int i = 0; i < lines.size(); i++) {
			Vec4i l = lines[i];
			float ratio = l[1] / grey.rows;
			double angle = getAngle(l);

			if (isHorizontal(angle))
				horizontalLines.push_back(l);
			else if (isVertical(angle))
				verticalLines.push_back(l);
			else if (isSkewed(angle))
				skewedLines.push_back(l);
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

		for (Vec4i line : skewedLines) {
			Point point1(line[0], line[1]);
			Point point2(line[2], line[3]);
			cv::line(edges, point1, point2, Scalar(255, 255, 0), 2);

			const int DELAY = 50;

			if (skewedLines.size() > 0) {
				if (f_open == -1 && frame_count > DELAY) {
					f_open = frame_count;
				}
				else if (f_close == -1 && frame_count - (f_open + DELAY) > 0) {
					f_close = frame_count;
				}
				else if (s_open == -1 && frame_count - (f_close + DELAY) > 0) {
					s_open = frame_count;
				}
				else if (s_close == -1 && frame_count - (s_open + DELAY) > 0) {
					s_close = frame_count;
				}


				cout << "First Open " << f_open << endl;
				cout << "First Close " << f_close << endl;
				cout << "Second Open " << s_open << endl;
				cout << "Second Close " << s_close << endl;
			}
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

		/*cout << horizontalLines.size() << " horizontal lines" << endl;
		cout << verticalLines.size() << " vertical lines" << endl;
		cout << skewedLines.size() << " skewed lines" << endl;
		cout << doorPoints.size() << " door points" << endl << endl;*/

		// print frame count to mat
		string frame_text("Frame " + to_string(frame_count));
		putText(edges, frame_text, cvPoint(30, 30),
			FONT_HERSHEY_COMPLEX_SMALL, 0.5, cvScalar(0, 0, 255), 1, CV_AA);

		imshow("Edges", edges);

		/*if (frame_count == video.get(CV_CAP_PROP_FRAME_COUNT) - 1)
			frame_count = 0;
		else frame_count += 1;*/

		frame_count++;
	}
}

void findDoor(int element) {
	std::string filename(videos[element]);

	VideoCapture video(filename);
	detectDoor(video, element);
	video.release();

	cout << "First Open " << f_open << endl;
	cout << "First Close " << f_close << endl;
	cout << "Second Open " << s_open << endl;
	cout << "Second Close " << s_close << endl;

	// infinite loop broken on ESC
	while (cv::waitKey(1) != 27) {}
}

int main(int argc, const char** argv) {
	findDoor(1);
	return 0;
}
