/*
 * This code is provided as part of "A Practical Introduction to Computer Vision with OpenCV"
 * by Kenneth Dawson-Howe © Wiley & Sons Inc. 2014.  All rights reserved.
 */
#include "Utilities.h"

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

void findDoor(int element) {
	std::string filename(videos[element]);

	VideoCapture video(filename);
	medianBackgroundDemo(video, 120, door_2, true);
	video.release();
}

int main(int argc, const char** argv) {
	findDoor(1);
	return 0;
}
