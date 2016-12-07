/*
 * This code is provided as part of "A Practical Introduction to Computer Vision with OpenCV"
 * by Kenneth Dawson-Howe © Wiley & Sons Inc. 2014.  All rights reserved.
 */
#include "Utilities.h"

Mat original_image, binary_image;
int current_threshold, max_threshold;

void onBinaryThresholdSlider(int pos, void*)
{
	threshold(original_image,binary_image,current_threshold,max_threshold,THRESH_BINARY);
	setTrackbarPos("Threshold","Binary",current_threshold);
	imshow("Binary", binary_image);
}

void BinaryDemos( Mat& pcb_image, Mat& stationery_image )
{
	Mat gray_pcb_image, gray_stationery_image;
	cvtColor(pcb_image, gray_pcb_image, CV_BGR2GRAY);
	cvtColor(stationery_image, gray_stationery_image, CV_BGR2GRAY);

	// Binary thresholding (manual)
	current_threshold = 128;
	max_threshold = 255;
    namedWindow("Binary", CV_WINDOW_AUTOSIZE );
	createTrackbar("Threshold","Binary",&current_threshold,max_threshold,onBinaryThresholdSlider);
	original_image = gray_pcb_image;

	char c;
	do {
		threshold(gray_pcb_image,binary_image,current_threshold,max_threshold,THRESH_BINARY);
		imshow("Original", pcb_image);
		imshow("Binary", binary_image);
		c = cvWaitKey(100);
    } while (c == -1);
    cvDestroyAllWindows();

	// Otsu thresholding
	Mat otsu_binary_image;
	threshold(gray_stationery_image,otsu_binary_image,current_threshold,max_threshold,
		THRESH_BINARY | THRESH_OTSU);
	Mat otsu_binary_image_display;
	cvtColor(otsu_binary_image, otsu_binary_image_display, CV_GRAY2BGR);

	// Band thresholding
	Mat binary_image1,binary_image2,band_thresholded_image;
	int threshold1 = 55, threshold2 = 73;
	threshold(gray_stationery_image,binary_image1,threshold1,max_threshold,THRESH_BINARY);
	threshold(gray_stationery_image,binary_image2,threshold2,max_threshold,THRESH_BINARY_INV);
	bitwise_and( binary_image1, binary_image2, band_thresholded_image );

	// Semi thresholding
	Mat semi_thresholded_image;
	threshold(gray_stationery_image,otsu_binary_image,current_threshold,max_threshold,
		THRESH_BINARY_INV | THRESH_OTSU);
	bitwise_and( gray_stationery_image, otsu_binary_image, semi_thresholded_image );

	// Adaptive thresholding
	threshold(gray_stationery_image,otsu_binary_image,current_threshold,max_threshold,
		THRESH_BINARY | THRESH_OTSU);
	Mat adaptive_thresholded_binary_image1,adaptive_thresholded_binary_image2;
	adaptiveThreshold(gray_stationery_image,adaptive_thresholded_binary_image1,255.0,ADAPTIVE_THRESH_MEAN_C,
		THRESH_BINARY, 21, 20 );
	adaptiveThreshold(gray_stationery_image,adaptive_thresholded_binary_image2,255.0,ADAPTIVE_THRESH_MEAN_C,
		THRESH_BINARY, 101, 20 );
	Mat band_thresholded_image_display, adaptive_thresholded_binary_image1_display, adaptive_thresholded_binary_image2_display, semi_thresholded_image_display;
	cvtColor(semi_thresholded_image, semi_thresholded_image_display, CV_GRAY2BGR);
	cvtColor(adaptive_thresholded_binary_image1, adaptive_thresholded_binary_image1_display, CV_GRAY2BGR);
	cvtColor(adaptive_thresholded_binary_image2, adaptive_thresholded_binary_image2_display, CV_GRAY2BGR);
	cvtColor(band_thresholded_image, band_thresholded_image_display, CV_GRAY2BGR);
	Mat output1 = JoinImagesHorizontally( stationery_image, "Original Image", otsu_binary_image_display, "Otsu thresholded Image", 4 );
	Mat output2 = JoinImagesHorizontally( output1, "", band_thresholded_image_display, "Band thresholded Image (55-73)", 4 );
	Mat output3 = JoinImagesHorizontally( semi_thresholded_image_display, "Semi-Thresholded Image", adaptive_thresholded_binary_image1_display, "Adaptive Thresholded Image (21x21)", 4 );
	Mat output4 = JoinImagesHorizontally( output3, "", adaptive_thresholded_binary_image2_display, "Adaptive Thresholded Image (101x101)", 4 );
	Mat thresholding_output = JoinImagesVertically( output2, "", output4, "", 4 );
	imshow("Binary Thresholding", thresholding_output);
	c = cvWaitKey();
    cvDestroyAllWindows();

	// Erosion & Dilation
	Mat eroded_image, dilated_image, eroded5_image, dilated5_image;
	threshold(gray_pcb_image,binary_image,current_threshold,max_threshold,
		THRESH_BINARY | THRESH_OTSU);
	erode(binary_image,eroded_image,Mat());
	dilate(binary_image,dilated_image,Mat());
	Mat five_by_five_element(5,5,CV_8U,Scalar(1));
	erode(binary_image,eroded5_image,five_by_five_element);
	dilate(binary_image,dilated5_image,five_by_five_element);
	Mat original_display, eroded3_display, otsu_display, eroded5_display, dilated3_display, dilated5_display;
	cvtColor(gray_pcb_image, original_display, CV_GRAY2BGR);
	cvtColor(binary_image, otsu_display, CV_GRAY2BGR);
	cvtColor(eroded_image, eroded3_display, CV_GRAY2BGR);
	cvtColor(eroded5_image, eroded5_display, CV_GRAY2BGR);
	cvtColor(dilated_image, dilated3_display, CV_GRAY2BGR);
	cvtColor(dilated5_image, dilated5_display, CV_GRAY2BGR);
	output1 = JoinImagesHorizontally( original_display, "Original Image", eroded3_display, "Eroded Image (3x3)", 4 );
	output2 = JoinImagesHorizontally( output1, "", eroded5_display, "Eroded Image (5x5)", 4 );
	output3 = JoinImagesHorizontally( otsu_display, "Otsu Thresholded Image", dilated3_display, "Dilated Image (3x3)", 4 );
	output4 = JoinImagesHorizontally( output3, "", dilated5_display, "Dilated Image (5x5)", 4 );
	Mat morphology_output = JoinImagesVertically( output2, "", output4, "", 4 );
	imshow("Mathematical Morphology - Erosion & Dilation", morphology_output);
	c = cvWaitKey();
    cvDestroyAllWindows();

	// Opening and closing
	Mat opened_image, closed_image, opened5_image, closed5_image;
	morphologyEx(binary_image,opened_image,MORPH_OPEN,Mat());
	morphologyEx(binary_image,closed_image,MORPH_CLOSE,Mat());
	morphologyEx(binary_image,opened5_image,MORPH_OPEN,five_by_five_element);
	morphologyEx(binary_image,closed5_image,MORPH_CLOSE,five_by_five_element);
	Mat opened3_display, closed3_display, opened5_display, closed5_display;
	cvtColor(opened_image, opened3_display, CV_GRAY2BGR);
	cvtColor(closed_image, closed3_display, CV_GRAY2BGR);
	cvtColor(opened5_image, opened5_display, CV_GRAY2BGR);
	cvtColor(closed5_image, closed5_display, CV_GRAY2BGR);
	output1 = JoinImagesHorizontally( original_display, "Original Image", opened3_display, "Opened Image (3x3)", 4 );
	output2 = JoinImagesHorizontally( output1, "", opened5_display, "Opened Image (5x5)", 4 );
	output3 = JoinImagesHorizontally( otsu_display, "Otsu Thresholded Image", closed3_display, "Closed Image (3x3)", 4 );
	output4 = JoinImagesHorizontally( output3, "", closed5_display, "Closed Image (5x5)", 4 );
	morphology_output = JoinImagesVertically( output2, "", output4, "", 4 );
	imshow("Mathematical Morphology - Opening & Closing", morphology_output);
	c = cvWaitKey();
    cvDestroyAllWindows();


	// Grey scale morphology + Connected Components Analysis
	morphologyEx(gray_pcb_image,opened_image,MORPH_OPEN,five_by_five_element);
	threshold(opened_image,binary_image,135,max_threshold,
		THRESH_BINARY );
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	Mat opened_image_copy = binary_image.clone();
	findContours(opened_image_copy,contours,hierarchy,CV_RETR_TREE,CV_CHAIN_APPROX_NONE);
	Mat contours_image = Mat::zeros(opened_image.size(), CV_8UC3);
	for (int contour_number=0; (contour_number<(int)contours.size()); contour_number++)
	{
        Scalar colour( rand()&0xFF, rand()&0xFF, rand()&0xFF );
        drawContours( contours_image, contours, contour_number, colour, CV_FILLED, 8, hierarchy );
	}
	Mat opened_display,binary_display;
	cvtColor(opened_image, opened_display, CV_GRAY2BGR);
	cvtColor(binary_image, binary_display, CV_GRAY2BGR);
	output1 = JoinImagesHorizontally( original_display, "Original Image", opened_display, "Opened Image (5x5)", 4 );
	output2 = JoinImagesHorizontally( binary_display, "Thresholded Image", contours_image, "Connected Components", 4 );
	morphology_output = JoinImagesVertically( output1, "", output2, "", 4 );
	imshow("Grey Scale Morphology & Connected Components", morphology_output);
	c = cvWaitKey();
    cvDestroyAllWindows();

	// Colour morphology
	Mat nine_by_nine_element(9,9,CV_8U,Scalar(1));
	morphologyEx(pcb_image,opened_image,MORPH_OPEN,five_by_five_element);
	morphologyEx(pcb_image,closed_image,MORPH_CLOSE,nine_by_nine_element);
	output1 = JoinImagesHorizontally( closed_image, "Closed Image (9x9)", pcb_image, "Original Image", 4 );
	morphology_output = JoinImagesHorizontally( output1, "", opened_image, "Opened Image (5x5)", 4 );
	imshow("Colour Morphology", morphology_output);
	c = cvWaitKey();
    cvDestroyAllWindows();
}
