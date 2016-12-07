/*
 * This code is provided as part of "A Practical Introduction to Computer Vision with OpenCV"
 * by Kenneth Dawson-Howe © Wiley & Sons Inc. 2014.  All rights reserved.
 */
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/video.hpp"
#include "opencv2/highgui.hpp"
#include <stdio.h>
#include <iostream>
#include "Utilities.h"

using namespace std;
using namespace cv;

void writeText(Mat image, char* text, int row, int column, Scalar passed_colour, double scale, int thickness)
{
	Scalar colour(0, 0, 255);
	Point location(column, row);
	putText(image, text, location, FONT_HERSHEY_SIMPLEX, scale, (passed_colour.val[0] == -1.0) ? colour : passed_colour, thickness);
}

Mat JoinImagesHorizontally(Mat& image1, char* name1, Mat& image2, char* name2, int spacing, Scalar passed_colour/*=-1.0*/)
{
	Mat result((image1.rows > image2.rows) ? image1.rows : image2.rows,
		image1.cols + image2.cols + spacing,
		image1.type());
	result.setTo(Scalar(255, 255, 255));
	Mat imageROI;
	imageROI = result(cv::Rect(0, 0, image1.cols, image1.rows));
	image1.copyTo(imageROI);
	if (spacing > 0)
	{
		imageROI = result(cv::Rect(image1.cols, 0, spacing, image1.rows));
		imageROI.setTo(Scalar(255, 255, 255));
	}
	imageROI = result(cv::Rect(image1.cols + spacing, 0, image2.cols, image2.rows));
	image2.copyTo(imageROI);
	writeText(result, name1, 13, 6, passed_colour);
	writeText(imageROI, name2, 13, 6, passed_colour);
	return result;
}

Mat JoinImagesVertically(Mat& image1, char* name1, Mat& image2, char* name2, int spacing, Scalar passed_colour/*=-1.0*/)
{
	Mat result(image1.rows + image2.rows + spacing,
		(image1.cols > image2.cols) ? image1.cols : image2.cols,
		image1.type());
	result.setTo(Scalar(255, 255, 255));
	Mat imageROI;
	imageROI = result(cv::Rect(0, 0, image1.cols, image1.rows));
	image1.copyTo(imageROI);
	if (spacing > 0)
	{
		imageROI = result(cv::Rect(0, image1.rows, image1.cols, spacing));
		imageROI.setTo(Scalar(255, 255, 255));
	}
	imageROI = result(cv::Rect(0, image1.rows + spacing, image2.cols, image2.rows));
	image2.copyTo(imageROI);
	writeText(result, name1, 13, 6, passed_colour);
	writeText(imageROI, name2, 13, 6, passed_colour);
	return result;
}

void addGaussianNoise(Mat &image, double average, double standard_deviation)
{
	// We need to work with signed images (as noise can be negative as well as positive).
	// We chose 16 bit signed images as if we converted an 8 bits unsigned image to a
	// signed version we would lose precision.
	int image_type = (image.channels() == 3) ? CV_16SC3 : CV_16SC1;
	Mat noise_image(image.size(), image_type);
	randn(noise_image, Scalar::all(average), Scalar::all(standard_deviation));
	Mat temp_image;
	image.convertTo(temp_image, image_type);
	addWeighted(temp_image, 1.0, noise_image, 1.0, 0.0, temp_image);
	temp_image.convertTo(image, image.type());
}

VideoWriter* OpenVideoFile(char* filename, VideoCapture& video_to_emulate, int horizontal_multiple, int vertical_multiple, int spacing)
{
	int codec = static_cast<int>(video_to_emulate.get(CV_CAP_PROP_FOURCC));
	Size image_size = Size((int)video_to_emulate.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)video_to_emulate.get(CV_CAP_PROP_FRAME_HEIGHT));
	double fps = video_to_emulate.get(CV_CAP_PROP_FPS);
	return OpenVideoFile(filename, codec, image_size, fps, horizontal_multiple, vertical_multiple, spacing);
}

VideoWriter* OpenVideoFile(char* filename, int codec, Size image_size, double fps, int horizontal_multiple, int vertical_multiple, int spacing)
{
	VideoWriter* output_video = new VideoWriter();
	Size video_size = Size((int)image_size.width*horizontal_multiple + spacing*(horizontal_multiple - 1),
		(int)image_size.height*vertical_multiple + spacing*(vertical_multiple - 1));
	output_video->open(filename, codec, fps, video_size, true);
	if (!output_video->isOpened())
	{
		cout << "Could not open the output video for write: " << filename << endl;
	}
	return output_video;
}

void WriteVideoFrame(VideoWriter* output_video, Mat& video_frame)
{
	*output_video << video_frame;
}

void CloseVideoFile(VideoWriter* video)
{
	delete video;
}



TimestampEvent::TimestampEvent()
{
	Reset("");
}
void TimestampEvent::Reset(String event_name)
{
	mEventName = event_name;
	mEventCount = 0;
	mAverageDuration = 0.0;
	mLastDuration = 0.0;
}
void TimestampEvent::RecordEvent(int duration)
{
	mLastDuration = duration;
	mAverageDuration = ((mAverageDuration*((double)mEventCount)) + duration) / ((double)(mEventCount + 1));
	mEventCount++;
}
double TimestampEvent::getLastTime()
{
	return mLastDuration;
}
double TimestampEvent::getAverageTime()
{
	return mAverageDuration;
}
String TimestampEvent::getEventName()
{
	return mEventName;
}
String TimestampEvent::getString(bool average, bool last)
{
	String event_string;
	std::ostringstream temp;
	temp << mEventName;
	temp.precision(1);
	if (average || last)
		temp << " ";
	if (last)
		temp << (int)(mLastDuration + 0.5) << "ms ";
	if ((average) && (mEventCount > 1))
		temp << fixed << "(av. " << mAverageDuration << "ms)";
	event_string = temp.str();
	return event_string;
}


Timestamper::Timestamper()
{
	mTickFrequency = getTickFrequency() / 1000.0;  // Tick frequency in ms
	reset();
}
void Timestamper::reset()
{
	ignoreTimeSinceLastRecorded();
	mEventCount = 0;
}
void Timestamper::ignoreTimeSinceLastRecorded()
{
	mLastTickCount = static_cast<double>(getTickCount());
}
void Timestamper::recordTime(String event)
{
	// Search for event
	int event_count = 0;
	while ((event_count < mEventCount) &&
		(event.compare(mEvents[event_count].getEventName()) != 0))
		event_count++;
	double tick_count = static_cast<double>(getTickCount());
	double processing_duration = (tick_count - mLastTickCount) / mTickFrequency;
	mLastTickCount = tick_count;
	if (event_count != MAX_EVENTS)
	{
		if (event_count == mEventCount)
		{
			// Add new event
			mEvents[event_count].Reset(event);
			mEventCount++;
		}
		mEvents[event_count].RecordEvent((int)processing_duration);
	}
}
void Timestamper::putTimes(Mat output_image)
{
	int line_step = 13;
	Scalar colour(0, 0, 255);
	Point location(7, 13);
	putText(output_image, "Execution times:", location, FONT_HERSHEY_SIMPLEX, 0.4, colour);

	for (int event_count = 0; event_count < mEventCount; event_count++)
	{
		String output = "";
		output += "-";
		output += mEvents[event_count].getString();
		location.y += line_step;
		putText(output_image, output, location, FONT_HERSHEY_SIMPLEX, 0.4, colour);
	}
}

Mat StretchImage(Mat& image)
{
	Mat& result = image.clone();
	// Find max value
	int image_rows = image.rows;
	int image_channels = image.channels();
	int values_on_each_row = image.cols * image_channels;
	uchar max = 0;
	for (int row = 0; row < image_rows; row++) {
		uchar* value = image.ptr<uchar>(row);
		for (int column = 0; column < values_on_each_row; column++)
		{
			if (*value > max)
				max = *value;
			value++;
		}
	}
	// Stretch values using a lookup-table
	int entries(256);
	Mat lut(1, entries, CV_8U);
	for (int i = 0; (i < 256); i++)
		lut.at<uchar>(i) = (255 * i) / max;
	LUT(image, lut, result);

	return result;
}

Mat convert_32bit_image_for_display(Mat& passed_image, double zero_maps_to/*=0.0*/, double passed_scale_factor/*=-1.0*/)
{
	Mat display_image;
	double scale_factor = passed_scale_factor;
	if (passed_scale_factor == -1.0)
	{
		double minimum, maximum;
		minMaxLoc(passed_image, &minimum, &maximum);
		scale_factor = (255.0 - zero_maps_to) / max(-minimum, maximum);
	}
	passed_image.convertTo(display_image, CV_8U, scale_factor, zero_maps_to);
	return display_image;
}

void show_32bit_image(char* window_name, Mat& passed_image, double zero_maps_to/*=0.0*/, double passed_scale_factor/*=-1.0*/)
{
	Mat display_image = convert_32bit_image_for_display(passed_image, zero_maps_to, passed_scale_factor);
	imshow(window_name, display_image);
}

Mat ComputeDefaultImage(Mat& passed_image)
{
	Mat five_by_five_element(5, 5, CV_8U, Scalar(1));
	Mat opened_image, image_gray, image_gray_edges, image_edges;
	morphologyEx(passed_image, opened_image, MORPH_OPEN, five_by_five_element);
	cvtColor(opened_image, image_gray, CV_BGR2GRAY);
	Canny(image_gray, image_gray_edges, 100, 150);
	cvtColor(image_gray_edges, image_edges, CV_GRAY2BGR);
	vector<Mat> input_planes(3);
	split(opened_image, input_planes);
	vector<Mat> output_planes;
	Mat temp_image = passed_image.clone();
	split(temp_image, output_planes);
	for (int plane = 0; plane < opened_image.channels(); plane++)
		Canny(input_planes[plane], output_planes[plane], 50, 120);
	Mat multispectral_edges, default_image;
	bitwise_or(output_planes[0], output_planes[2], output_planes[0]);
	bitwise_or(output_planes[0], output_planes[1], output_planes[0]);
	output_planes[2] = output_planes[0];
	output_planes[1] = output_planes[0];
	merge(output_planes, multispectral_edges);
	bitwise_or(passed_image, multispectral_edges, default_image);
	return default_image;
}

void DrawHistogram(MatND histograms[], int number_of_histograms, Mat& display_image)
{
	int number_of_bins = histograms[0].size[0];
	double max_value = 0, min_value = 0;
	double channel_max_value = 0, channel_min_value = 0;
	for (int channel = 0; (channel < number_of_histograms); channel++)
	{
		minMaxLoc(histograms[channel], &channel_min_value, &channel_max_value, 0, 0);
		max_value = ((max_value > channel_max_value) && (channel > 0)) ? max_value : channel_max_value;
		min_value = ((min_value < channel_min_value) && (channel > 0)) ? min_value : channel_min_value;
	}
	float scaling_factor = ((float)256.0) / ((float)number_of_bins);

	Mat histogram_image((int)(((float)number_of_bins)*scaling_factor), (int)(((float)number_of_bins)*scaling_factor), CV_8UC3, Scalar(255, 255, 255));
	display_image = histogram_image;
	int highest_point = static_cast<int>(0.9*((float)number_of_bins)*scaling_factor);
	for (int channel = 0; (channel < number_of_histograms); channel++)
	{
		int last_height;
		for (int h = 0; h < number_of_bins; h++)
		{
			float value = histograms[channel].at<float>(h);
			int height = static_cast<int>(value*highest_point / max_value);
			int where = (int)(((float)h)*scaling_factor);
			if (h > 0)
				line(histogram_image, Point((int)(((float)(h - 1))*scaling_factor), (int)(((float)number_of_bins)*scaling_factor) - last_height),
					Point((int)(((float)h)*scaling_factor), (int)(((float)number_of_bins)*scaling_factor) - height),
					Scalar(channel == 0 ? 255 : 0, channel == 1 ? 255 : 0, channel == 2 ? 255 : 0));
			last_height = height;
		}
	}
}

