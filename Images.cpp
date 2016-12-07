/*
 * This code is provided as part of "A Practical Introduction to Computer Vision with OpenCV"
 * by Kenneth Dawson-Howe © Wiley & Sons Inc. 2014.  All rights reserved.
 */
#include "Utilities.h"
//#include <cxcore.hpp>

// Example of simple grey scale image access.
void ChangeQuantisationGrey( Mat &image, int new_number_of_bits )
{
	CV_Assert( (image.type() == CV_8UC1) && (new_number_of_bits >= 1) && (new_number_of_bits <= 8) );
	uchar mask = 0xFF << (8-new_number_of_bits); // e.g. if new_number_of_bits=3, mask=0xE0
	for (int row=0; row < image.rows; row++)
		for (int col=0; col < image.cols; col++)
			image.at<uchar>(row,col) = image.at<uchar>(row,col) & mask;
}

// Example of simple colour image access.
void InvertColour( Mat& input_image, Mat& output_image )
{
	CV_Assert( input_image.type() == CV_8UC3 );
	output_image = input_image.clone();
	for (int row=0; row < input_image.rows; row++)
		for (int col=0; col < input_image.cols; col++)
			for (int channel=0; channel < input_image.channels(); channel++)
				output_image.at<Vec3b>(row,col)[channel] = 255 -
								input_image.at<Vec3b>(row,col)[channel];
}

// Example of simple skin pixel identification.
void SelectSkin( Mat& hls_image, Mat& skin_image )
{
	CV_Assert( hls_image.type() == CV_8UC3 );
	skin_image = hls_image.clone();
	for (int row=0; row < hls_image.rows; row++)
		for (int col=0; col < hls_image.cols; col++)
		{
			uchar hue = hls_image.at<Vec3b>(row,col)[0];
			uchar luminance = hls_image.at<Vec3b>(row,col)[1];
			uchar saturation = hls_image.at<Vec3b>(row,col)[2];
			double luminance_saturation_ratio = ((double) luminance) / ((double) saturation);
			bool skin_pixel = (saturation >= 50) && (luminance_saturation_ratio > 0.5) &&
				              (luminance_saturation_ratio < 3.0) && ((hue <= 14) || (hue >= 165));
			for (int channel=0; channel < hls_image.channels(); channel++)
				skin_image.at<Vec3b>(row,col)[channel] = skin_pixel ? hls_image.at<Vec3b>(row,col)[channel] : 0;
		}
}

// Example of simple red eye pixel identification.
void SelectRedEyePixels( Mat& bgr_image, Mat& redeye_image )
{
	CV_Assert( bgr_image.type() == CV_8UC3 );
	Mat hls_image;
	cvtColor(bgr_image,hls_image,CV_BGR2HLS);
	redeye_image = bgr_image.clone();
	for (int row=0; row < hls_image.rows; row++)
		for (int col=0; col < hls_image.cols; col++)
		{
			uchar hue = hls_image.at<Vec3b>(row,col)[0];
			uchar luminance = hls_image.at<Vec3b>(row,col)[1];
			uchar saturation = hls_image.at<Vec3b>(row,col)[2];
			double luminance_saturation_ratio = ((double) luminance) / ((double) saturation);
			bool red_eye_pixel = (luminance >= 64) && (saturation >= 100) &&
				                 (luminance_saturation_ratio > 0.5) && (luminance_saturation_ratio < 1.5) &&
								 ((hue <= 7) || (hue >= 162));
			for (int channel=0; channel < hls_image.channels(); channel++)
				redeye_image.at<Vec3b>(row,col)[channel] = red_eye_pixel ? bgr_image.at<Vec3b>(row,col)[channel] : luminance;
		}
}

// This routine is an example of efficient processing of an image.  To do this
// we have to avoid array indexing and instead use pointer arithmetic to work
// through the image values.  We also have to provide separate code for 1 and
// 3 channel images, and have to provide separate code for continuous and padded
// images.  (A padded image is one where there is some unused space at the end
// of each row, typically to align to a Word boundary).
void changeQuantisation(Mat &image, int new_number_of_bits)
{
	if ((new_number_of_bits >= 8) || (new_number_of_bits <= 0))
		return;
	int image_rows = image.rows;
	int image_columns = image.cols;
	int image_channels = image.channels();
	uchar mask = 0xFF << (8-new_number_of_bits); // e.g. if new_number_of_bits=3, mask=0xE0
	if (image.isContinuous()) // i.e. there is no padding at the end of rows
	{
		// Here we process each row of a 1 or 3 channel continuous image.  Hence we
		// can treat the image data values as a single contiguous array.
		uchar* value = image.ptr<uchar>(0);
		uchar* end_value = value + (image_columns*image_rows*image_channels);
		if (image_channels == 1)
			while (value < end_value)
				*value++ = *value & mask;
		else // if (image_channels == 3)
			while (value < end_value)
			{
				// The 3 channel version is more efficient as there are 3 times
				// less comparisons.
				*value++ = *value & mask;
				*value++ = *value & mask;
				*value++ = *value & mask;
			}
	}
	else if (image_channels == 1)
	{
		// Here we process each row of a 1 channel padded image.  We could use
		// this code for a continuous image but it would be a bit less efficient.
		for (int row=0; row < image_rows; row++) {
			uchar* value = image.ptr<uchar>(row);
			for (int column=0; column < image_columns; column++)
				*value++ = *value & mask;
		}
	}
	else // if (image_channels == 3)
	{
		// Here we process each row of a 3 channel padded image.   We could use
		// this code for a continuous image but it would be a bit less efficient.
		for (int row=0; row < image_rows; row++) {
			uchar* value = image.ptr<uchar>(row);
			for (int column=0; column < image_columns; column++)
			{
				*value++ = *value & mask;
				*value++ = *value & mask;
				*value++ = *value & mask;
			}
		}
	}
}

void addSaltAndPepperNoise(Mat &image, double noise_percentage)
{
	int image_rows = image.rows;
	int image_columns = image.cols;
	int image_channels = image.channels();
	int number_of_noise_points = (int) ((((double) image_rows*image_columns*image_channels)*noise_percentage)/100.0);
	for (int count = 0; count < number_of_noise_points; count++)
	{
		int row = rand() % image_rows;
		int column = rand() % image_columns;
		int channel = rand() % image_channels;
		uchar* pixel = image.ptr<uchar>(row) + (column*image_channels) + channel;
		*pixel = (rand()%2 == 1) ? 255 : 0;
	}
}


void invertImage(Mat &image, Mat &result_image)
{
	result_image.create( image.size(), image.type() );
	int image_rows = image.rows;
	int image_columns = image.cols;
	int image_channels = image.channels();
	if (image.isContinuous()) // i.e. there is no padding at the end of rows
	{
		// Here we process each row of a 1 or 3 channel continuous image.  Hence we
		// can treat the image data values as a single contiguous array.
		uchar* value = image.ptr<uchar>(0);
		uchar* result_value = result_image.ptr<uchar>(0);
		uchar* end_value = value + (image_columns*image_rows*image_channels);
		if (image_channels == 1)
			while (value < end_value)
				*result_value++ = *value++ ^ 0xFF;
		else // if (image_channels == 3)
			while (value < end_value)
			{
				// The 3 channel version is more efficient as there are 3 times
				// less comparisons.
				*result_value++ = *value++ ^ 0xFF;
				*result_value++ = *value++ ^ 0xFF;
				*result_value++ = *value++ ^ 0xFF;
			}
	}
	else if (image_channels == 1)
	{
		// Here we process each row of a 1 channel padded image.  We could use
		// this code for a continuous image but it would be a bit less efficient.
		for (int row=0; row < image_rows; row++) {
			uchar* value = image.ptr<uchar>(row);
			uchar* result_value = result_image.ptr<uchar>(row);
			for (int column=0; column < image_columns; column++)
				*result_value++ = *value++ ^ 0xFF;
		}
	}
	else // if (image_channels == 3)
	{
		// Here we process each row of a 3 channel padded image.   We could use
		// this code for a continuous image but it would be a bit less efficient.
		for (int row=0; row < image_rows; row++) {
			uchar* value = image.ptr<uchar>(row);
			uchar* result_value = result_image.ptr<uchar>(row);
			for (int column=0; column < image_columns; column++)
			{
				*result_value++ = *value++ ^ 0xFF;
				*result_value++ = *value++ ^ 0xFF;
				*result_value++ = *value++ ^ 0xFF;
			}
		}
	}
}


void ImagesDemos( Mat& image1, Mat& image2, Mat& logo_image, Mat& people_image )
{
	Timestamper* timer = new Timestamper();

	// Basic colour image access (demonstration using invert)
	Mat output_image;
	InvertColour( image1, output_image );
	Mat output1 = JoinImagesHorizontally(image1,"Original Image",output_image,"Inverted Image",4);
	imshow("Basic Image Processing", output1);
	char c = cvWaitKey();
    cvDestroyAllWindows();

	// Sampling & Quantisation (Grey scale)
	Mat image1_gray, smaller_image, resized_image, two_bit_image;
	cvtColor(image1, image1_gray, CV_BGR2GRAY);
	resize(image1_gray, smaller_image, Size( image1.cols/2, image1.rows/2 ));
	resize(smaller_image, resized_image, image1.size() );
	two_bit_image = image1_gray.clone();
	ChangeQuantisationGrey( two_bit_image, 2 );
	Mat image1_gray_display, smaller_image_display, resized_image_display, two_bit_image_display;
	cvtColor(image1_gray, image1_gray_display, CV_GRAY2BGR);
	cvtColor(smaller_image, smaller_image_display, CV_GRAY2BGR);
	cvtColor(resized_image, resized_image_display, CV_GRAY2BGR);
	cvtColor(two_bit_image, two_bit_image_display, CV_GRAY2BGR);
	output1 = JoinImagesHorizontally(two_bit_image_display,"Quantisation 8->2 bits",image1_gray_display,"Original Greyscale Image",4);
	Mat output2 = JoinImagesHorizontally(output1,"",smaller_image_display,"Half sized image",4);
	Mat output3 = JoinImagesHorizontally(output2,"",resized_image_display,"Resized image",4);
	// Sampling & Quantisation
	Mat quantised_frame;
	quantised_frame = image1.clone();
	resize(image1, smaller_image, Size( image1.cols/2, image1.rows/2 ));
	resize(smaller_image, resized_image, image1.size(), 0.0, 0.0, INTER_NEAREST );
	changeQuantisation(quantised_frame, 2);
	output1 = JoinImagesHorizontally(quantised_frame,"Quantisation 8->2 bits",image1,"Original Colour Image",4);
	output2 = JoinImagesHorizontally(output1,"",smaller_image,"Half sized image",4);
	Mat output4 = JoinImagesHorizontally(output2,"",resized_image,"Resized image",4);
	Mat output5 = JoinImagesVertically(output3,"",output4,"",4);
	imshow("Sampling & Quantisation", output5);
	c = cvWaitKey();
    cvDestroyAllWindows();

	// Colour channels.
	resize(image2, smaller_image, Size( image2.cols/2, image2.rows/2 ));
	vector<Mat> input_planes(3);
	split(smaller_image,input_planes);
	Mat channel1_display, channel2_display, channel3_display;
	cvtColor(input_planes[2], channel1_display, CV_GRAY2BGR);
	cvtColor(input_planes[1], channel2_display, CV_GRAY2BGR);
	cvtColor(input_planes[0], channel3_display, CV_GRAY2BGR);
	output1 = JoinImagesHorizontally(channel1_display,"Red",channel2_display,"Green",4);
	output2 = JoinImagesHorizontally(output1,"",channel3_display,"Blue",4);

	Mat yuv_image;
	cvtColor(smaller_image, yuv_image, CV_BGR2YUV);
	split(yuv_image,input_planes);
	cvtColor(input_planes[0], channel1_display, CV_GRAY2BGR);
	cvtColor(input_planes[1], channel2_display, CV_GRAY2BGR);
	cvtColor(input_planes[2], channel3_display, CV_GRAY2BGR);
	output1 = JoinImagesHorizontally(channel1_display,"Y",channel2_display,"U",4);
	output3 = JoinImagesHorizontally(output1,"",channel3_display,"V",4);
	output4 = JoinImagesVertically(output2,"",output3,"",4);

	Mat hls_image;
	cvtColor(smaller_image, hls_image, CV_BGR2HLS);
	vector<Mat> hls_planes(3);
	split(hls_image,hls_planes);
	Mat& hue_image = hls_planes[0];
	cvtColor(hls_planes[0], channel1_display, CV_GRAY2BGR);
	cvtColor(hls_planes[1], channel2_display, CV_GRAY2BGR);
	cvtColor(hls_planes[2], channel3_display, CV_GRAY2BGR);
	output1 = JoinImagesHorizontally(channel1_display,"Hue",channel2_display,"Luminance",4);
	output2 = JoinImagesHorizontally(output1,"",channel3_display,"Saturation",4);
	output3 = JoinImagesVertically(output4,"",output2,"",4);
	Mat lab_image;
	cvtColor(smaller_image, lab_image, CV_BGR2Lab);
	vector<Mat> lab_planes(3);
	split(lab_image,lab_planes);
	cvtColor(lab_planes[0], channel1_display, CV_GRAY2BGR);
	cvtColor(lab_planes[1], channel2_display, CV_GRAY2BGR);
	cvtColor(lab_planes[2], channel3_display, CV_GRAY2BGR);
	output1 = JoinImagesHorizontally(channel1_display,"Luminance",channel2_display,"A",4);
	output2 = JoinImagesHorizontally(output1,"",channel3_display,"B",4);
	output4 = JoinImagesVertically(output3,"",output2,"",4);
	output3 = JoinImagesHorizontally(smaller_image,"",output4,"",4);
	imshow("Colour Models - RGB, YUV, HLS, Lab", output3);
	c = cvWaitKey();
    cvDestroyAllWindows();

	Mat hls_people_image, hls_skin_image, skin_image, redeye_image;
	cvtColor(people_image, hls_people_image, CV_BGR2HLS);
	SelectSkin( hls_people_image, hls_skin_image );
	SelectRedEyePixels( people_image, redeye_image );
	cvtColor(hls_skin_image, skin_image, CV_HLS2BGR);
	output1 = JoinImagesHorizontally(people_image,"Original Image",skin_image,"Possible skin pixels",4);
	output2 = JoinImagesHorizontally(output1,"",redeye_image,"Possible Red-Eye pixels",4);
	imshow("Skin & Redeye detection", output2);
	c = cvWaitKey();
    cvDestroyAllWindows();

	// Noise & Smoothing
	resize(image1, smaller_image, Size( image1.cols*3/4, image1.rows*3/4 ));
	Mat noise_test = smaller_image.clone();
	addGaussianNoise(noise_test, 0.0, 20.0);
	Mat noise_test1 = noise_test.clone();
	Mat noise_test2 = noise_test.clone();
	Mat noise_test3 = noise_test.clone();
	blur(noise_test1,noise_test1,Size(5,5));
	GaussianBlur(noise_test2,noise_test2,Size(5,5),1.5);
	medianBlur(noise_test3,noise_test3,5);
	output1 = JoinImagesHorizontally(noise_test,"Gaussian Noise (0, 20)",noise_test1,"Local Average",4);
	output2 = JoinImagesHorizontally(output1,"",noise_test2,"Gaussian filtered",4);
	output3 = JoinImagesHorizontally(output2,"",noise_test3,"Median filtered",4);
	noise_test = smaller_image.clone();
	addSaltAndPepperNoise(noise_test, 5.0);
	noise_test1 = noise_test.clone();
	noise_test2 = noise_test.clone();
	noise_test3 = noise_test.clone();
	blur(noise_test1,noise_test1,Size(5,5));
	GaussianBlur(noise_test2,noise_test2,Size(5,5),1.5);
	medianBlur(noise_test3,noise_test3,5);
	output1 = JoinImagesHorizontally(noise_test,"Salt and Pepper Noise (5%)",noise_test1,"Local Average",4);
	output2 = JoinImagesHorizontally(output1,"",noise_test2,"Gaussian filtered",4);
	output4 = JoinImagesHorizontally(output2,"",noise_test3,"Median filtered",4);
	output5 = JoinImagesVertically(output3,"",output4,"",4);
	output1 = JoinImagesHorizontally(smaller_image,"Original Image",output5,"",4);
	imshow("Noise and Smoothing", output1);
	c = cvWaitKey();
    cvDestroyAllWindows();

	// Regions of Interest and weighted image addition.
	Mat watermarked_image = image1.clone();
	double scale = (((double)logo_image.cols)/((double)image1.cols)) > (((double)logo_image.rows)/((double)image1.rows)) ?
		             0.5/(((double)logo_image.cols)/((double)image1.cols)) : 0.5/(((double)logo_image.rows)/((double)image1.rows));
	int new_logo_size = image1.cols < image1.rows ? image1.cols/8 : image1.rows/8;
	resize(logo_image,logo_image,Size(((int) (((double) logo_image.cols)*scale)),((int) (((double) logo_image.rows)*scale))));
	Mat imageROI;
	imageROI = watermarked_image(cv::Rect((image1.cols-logo_image.cols)/2,(image1.rows-logo_image.rows)/2,logo_image.cols,logo_image.rows));
	addWeighted(imageROI,1.0,logo_image,0.1,0.0,imageROI);
	output1 = JoinImagesHorizontally(image1,"Original Image",logo_image,"Watermark",4);
	output2 = JoinImagesHorizontally(output1,"",watermarked_image,"Watermarked Image",4);
    imshow("Watermarking (Demo of Image ROIs & weighted addition)", output2);
	c = cvWaitKey();
    cvDestroyAllWindows();
}
