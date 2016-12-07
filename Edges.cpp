/*
 * This code is provided as part of "A Practical Introduction to Computer Vision with OpenCV"
 * by Kenneth Dawson-Howe © Wiley & Sons Inc. 2014.  All rights reserved.
 */
#include "Utilities.h"
#include "opencv2/video.hpp"
//#include <cxcore.hpp>

// Draw a passed line using a random colour if one is not provided
void DrawLine(Mat result_image, Point point1, Point point2, Scalar passed_colour=-1.0)
{
    Scalar colour( rand()&0xFF, rand()&0xFF, rand()&0xFF );
	line( result_image, point1, point2, (passed_colour.val[0] == -1.0) ? colour : passed_colour );
}

// Draw line segments delineated by end points
void DrawLines(Mat result_image, vector<Vec4i> lines, Scalar passed_colour=-1.0)
{
	for (vector<cv::Vec4i>::const_iterator current_line = lines.begin();
		    (current_line != lines.end()); current_line++)
	{
		Point point1((*current_line)[0],(*current_line)[1]);
		Point point2((*current_line)[2],(*current_line)[3]);
		DrawLine(result_image, point1, point2, passed_colour);
	}
}

// Draw lines defined by rho and theta parameters
void DrawLines(Mat result_image, vector<Vec2f> lines, Scalar passed_colour=-1.0)
{
	for (vector<cv::Vec2f>::const_iterator current_line = lines.begin();
		    (current_line != lines.end()); current_line++)
	{
		float rho = (*current_line)[0];
		float theta = (*current_line)[1];
		// To avoid divide by zero errors we offset slightly from 0.0
		float cos_theta = (cos(theta) == 0.0) ? (float) 0.000000001 : (float) cos(theta);
		float sin_theta = (sin(theta) == 0.0) ? (float) 0.000000001 : (float) sin(theta);
		Point left((int) (rho/cos(theta)),0);
		Point right((int) ((rho-(result_image.rows-1)*sin(theta))/cos(theta)),(int) ((result_image.rows-1)));
		Point top(0,(int) (rho/sin(theta)));
		Point bottom((int)(result_image.cols-1),(int) ((rho-(result_image.cols-1)*cos(theta))/sin(theta)));
		Point* point1 = NULL;
		Point* point2 = NULL;
		if ((left.y >= 0.0) && (left.y <= (result_image.rows-1)))
			point1 = &left;
		if ((right.y >= 0.0) && (right.y <= (result_image.rows-1)))
			if (point1 == NULL)
				point1 = &right;
			else point2 = &right;
		if ((point2 == NULL) && (top.x >= 0.0) && (top.x <= (result_image.cols-1)))
			if (point1 == NULL)
				point1 = &top;
			else if ((point1->x != top.x) || (point1->y != top.y))
				point2 = &top;
		if (point2 == NULL)
			point2 = &bottom;
		DrawLine(result_image, *point1, *point2, passed_colour);
	}
}

void DrawCircles(Mat result_image, vector<Vec3f> circles, Scalar passed_colour=-1.0)
{
	for (vector<cv::Vec3f>::const_iterator current_circle = circles.begin();
		    (current_circle != circles.end()); current_circle++)
	{
		Scalar colour( rand()&0xFF, rand()&0xFF, rand()&0xFF );
		circle(result_image,Point((int) ((*current_circle)[0]),int((*current_circle)[1])),(int) ((*current_circle)[2]),
			(passed_colour.val[0] == -1.0) ? colour : passed_colour,
			2);
	}
}

void NonMaximaEdgeGradientSuppression( Mat& gradients, Mat& orientations, Mat& nms_result, float min_gradient = 50.0 )
{
/*
	// An inefficient (but easier to understand) implementation:
	nms_result = gradients.clone();
	for (int row=1; row < gradients.rows-1; row++)
		for (int column=1; column < gradients.cols-1; column++)
		{
			float curr_gradient = gradients.at<float>(row,column);
			float curr_orientation = orientations.at<float>(row,column);
			// Determine which neighbours to check
			int direction = (((int) (16.0*(curr_orientation)/(2.0*PI))+15)%8)/2;
			float gradient1 = 0.0, gradient2 = 0.0;
			switch(direction)
			{
			case 0:
				gradient1 = gradients.at<float>(row-1,column-1);
				gradient2 = gradients.at<float>(row+1,column+1);
				break;
			case 1:
				gradient1 = gradients.at<float>(row-1,column);
				gradient2 = gradients.at<float>(row+1,column);
				break;
			case 2:
				gradient1 = gradients.at<float>(row-1,column+1);
				gradient2 = gradients.at<float>(row+1,column-1);
				break;
			case 3:
				gradient1 = gradients.at<float>(row,column+1);
				gradient2 = gradients.at<float>(row,column-1);
				break;
			}
			if ((gradient1 > curr_gradient) || (gradient2 > curr_gradient))
				nms_result.at<float>(row,column) = 0.0;
		}
*/
	// An efficient implementation:
	nms_result = gradients.clone();
	int image_channels = gradients.channels();
	int image_rows = gradients.rows;
	int values_on_each_row = (gradients.cols-2) * image_channels;
	int max_row = image_rows-1;
	for (int row=1; row < max_row-1; row++)
	{
		float* curr_gradient = gradients.ptr<float>(row) + image_channels;
		float* curr_orientation = orientations.ptr<float>(row) + image_channels;
		float* output_point = nms_result.ptr<float>(row);
		*output_point = 0.0;
		output_point += image_channels;
		for (int column=0; column < values_on_each_row; column++)
		{
			if (*curr_gradient < min_gradient)
				*output_point = 0.0;
			else
			{
				// Determine which neighbours to check
				int direction = (((int) (16.0*(*curr_orientation)/(2.0*PI))+15)%8)/2;
				float gradient1 = 0.0, gradient2 = 0.0;
				switch(direction)
				{
				case 0:
					gradient1 = *(gradients.ptr<float>(row-1) + (column)*image_channels);
					gradient2 = *(gradients.ptr<float>(row+1) + (column+2)*image_channels);
					break;
				case 1:
					gradient1 = *(gradients.ptr<float>(row-1) + (column+1)*image_channels);
					gradient2 = *(gradients.ptr<float>(row+1) + (column+1)*image_channels);
					break;
				case 2:
					gradient1 = *(gradients.ptr<float>(row-1) + (column+2)*image_channels);
					gradient2 = *(gradients.ptr<float>(row+1) + (column)*image_channels);
					break;
				case 3:
					gradient1 = *(curr_gradient - image_channels);
					gradient2 = *(curr_gradient + image_channels);
					break;
				}
				if ((gradient1 > *curr_gradient) || (gradient2 > *curr_gradient))
					*output_point = 0.0;
			}
			curr_gradient += image_channels;
			curr_orientation += image_channels;
			output_point += image_channels;
		}
		*output_point = 0.0;
	}
}

void FindZeroCrossings( Mat& laplacian, Mat& zero_crossings )
{
	Mat* result = new Mat( laplacian.size(), CV_8U, Scalar(0) );
	zero_crossings = *result;
	int image_rows = laplacian.rows;
	int image_channels = laplacian.channels();
	int values_on_each_row = laplacian.cols * image_channels;
	float laplacian_threshold = 0.0;
	// Find Zero Crossings
	for (int row=1; row < image_rows; row++) {
		float* prev_row_pixel = laplacian.ptr<float>(row-1) +1;
		float* curr_row_pixel = laplacian.ptr<float>(row);
		uchar* output_pixel = zero_crossings.ptr<uchar>(row) +1;
		for (int column=1; column < values_on_each_row; column++)
		{
			float prev_value_on_row = *curr_row_pixel;
			curr_row_pixel++;
			float curr_value = *curr_row_pixel;
			float prev_value_on_column = *prev_row_pixel;
			float difference = 0.0;
			if (((curr_value > 0) && (prev_value_on_row < 0)) ||
				((curr_value < 0) && (prev_value_on_row > 0)))
				difference = abs(curr_value - prev_value_on_row);
			if ((((curr_value > 0) && (prev_value_on_column < 0)) ||
				 ((curr_value < 0) && (prev_value_on_column > 0))) &&
				(abs(curr_value - prev_value_on_column) > difference))
				difference = abs(curr_value - prev_value_on_column);
 			*output_pixel = (difference > laplacian_threshold) ? 255 : 0;// (int) ((100 * difference) / laplacian_threshold);
			prev_row_pixel++;
			output_pixel++;
		}
	}
}

void EdgeDemos( Mat& passed_image1, Mat& image2 )
{
	Mat image1 = passed_image1.clone();
	bool half_size = true;
	if (half_size)
		resize(passed_image1, image1, Size( image1.cols/2, image1.rows/2 ));
	Mat image1_gray, image2_gray;
	cvtColor(image1, image1_gray, CV_BGR2GRAY);
	cvtColor(image2, image2_gray, CV_BGR2GRAY);

	// First derivative (Roberts) Edge detection
	int kernel_size = 2;
	Mat kernel = Mat::eye( 2, 2, CV_32FC1 );
	Point anchor = Point( 0,0 );
	*(((float*) (kernel.data))) = 0.0;  // [0][1] 
	*(((float*) (kernel.data))+1) = 1.0;  // [0][1] 
	*(((float*) (kernel.data))+2) = -1.0;  // [1][0] 
	*(((float*) (kernel.data))+3) = 0.0;  // [1][0] 
	Mat roberts1;
	filter2D(image1_gray, roberts1, CV_32F, kernel, anchor);
	Mat roberts1_gray = convert_32bit_image_for_display( roberts1, 128.0, 0.4 );
	*(((float*) (kernel.data))) = 1.0;  // [0][1] 
	*(((float*) (kernel.data))+1) = 0.0;  // [0][1] 
	*(((float*) (kernel.data))+2) = 0.0;  // [1][0] 
	*(((float*) (kernel.data))+3) = -1.0;  // [1][0] 
	Mat roberts2;
	filter2D(image1_gray, roberts2, CV_32F, kernel, anchor);
	Mat roberts2_gray = convert_32bit_image_for_display( roberts2, 128.0, 0.4 );
	//cartToPolar(roberts1,roberts2,l2norm_gradient,orientation);
	Mat abs_gradient = abs(roberts1) + abs(roberts2);
	Mat roberts_gradient = convert_32bit_image_for_display(abs_gradient);
	Mat image1_gray_display, roberts1_gray_display, roberts2_gray_display, roberts_gradient_display;
	cvtColor(image1_gray, image1_gray_display, CV_GRAY2BGR);
	cvtColor(roberts1_gray, roberts1_gray_display, CV_GRAY2BGR);
	cvtColor(roberts2_gray, roberts2_gray_display, CV_GRAY2BGR);
	cvtColor(roberts_gradient, roberts_gradient_display, CV_GRAY2BGR);
	Mat row1_output = JoinImagesHorizontally( image1_gray_display, "Original Image", roberts_gradient_display, "Roberts (L1 norm) Gradient Image", 4 );
	Mat row2_output = JoinImagesHorizontally( roberts1_gray_display, "Roberts Partial Derivative (1)", roberts2_gray_display, "Roberts Partial Derivative (2)", 4 );
	Mat roberts_output = JoinImagesVertically(row1_output,"",row2_output,"", 4);
	imshow("Roberts Edge Detection", roberts_output );
	char c = cvWaitKey();
    cvDestroyAllWindows();

	// First derivative (Sobel) Edge detection
	Mat horizontal_partial_derivative, vertical_partial_derivative;
	Mat l2norm_gradient, orientation;
	Sobel(image1_gray,horizontal_partial_derivative,CV_32F,1,0);
	Sobel(image1_gray,vertical_partial_derivative,CV_32F,0,1);
	abs_gradient = abs(horizontal_partial_derivative) + abs(vertical_partial_derivative);
	cartToPolar(horizontal_partial_derivative,vertical_partial_derivative,l2norm_gradient,orientation);
	Mat horizontal_partial_derivative_gray = convert_32bit_image_for_display( horizontal_partial_derivative );
	Mat vertical_partial_derivative_gray = convert_32bit_image_for_display( vertical_partial_derivative );
	Mat abs_gradient_gray = convert_32bit_image_for_display( abs_gradient );
	Mat l2norm_gradient_gray = convert_32bit_image_for_display( l2norm_gradient );
	Mat l2norm_gradient_mask,display_orientation,the_gradient;
	l2norm_gradient.convertTo(l2norm_gradient_mask,CV_8U);
	threshold(l2norm_gradient_mask,l2norm_gradient_mask,50,255,THRESH_BINARY);
	orientation.copyTo(display_orientation, l2norm_gradient_mask);
	Mat orientation_gray = convert_32bit_image_for_display( orientation, 0.0, 255.0/(2.0*PI) );
	Mat display_orientation_gray = convert_32bit_image_for_display( display_orientation, 0.0, 255.0/(2.0*PI) );
	Mat nms_result;
	NonMaximaEdgeGradientSuppression( l2norm_gradient, orientation, nms_result );
	Mat nms_result_gray = convert_32bit_image_for_display( nms_result );
	Mat horizontal_partial_derivative_gray_display, vertical_partial_derivative_gray_display, abs_gradient_gray_display, l2norm_gradient_gray_display, orientation_gray_display, display_orientation_gray_display, nms_result_gray_display;
	cvtColor(horizontal_partial_derivative_gray, horizontal_partial_derivative_gray_display, CV_GRAY2BGR);
	cvtColor(vertical_partial_derivative_gray, vertical_partial_derivative_gray_display, CV_GRAY2BGR);
	cvtColor(abs_gradient_gray, abs_gradient_gray_display, CV_GRAY2BGR);
	cvtColor(l2norm_gradient_gray, l2norm_gradient_gray_display, CV_GRAY2BGR);
	cvtColor(orientation_gray, orientation_gray_display, CV_GRAY2BGR);
	cvtColor(display_orientation_gray, display_orientation_gray_display, CV_GRAY2BGR);
	cvtColor(nms_result_gray, nms_result_gray_display, CV_GRAY2BGR);
	Mat output1 = JoinImagesHorizontally( image1_gray_display, "Original Image", abs_gradient_gray_display, "Sobel (L1 norm) Gradient Image", 4 );
	Mat output2 = JoinImagesHorizontally( output1, "", l2norm_gradient_gray_display, "Sobel (L2 norm) Gradient Image", 4 );
	row1_output = JoinImagesHorizontally( output2, "", orientation_gray_display, "Sobel Orientation Image", 4 );
	output1 = JoinImagesHorizontally( horizontal_partial_derivative_gray_display, "Horizontal partial derivative", vertical_partial_derivative_gray_display, "Vertical partial derivative", 4 );
	output2 = JoinImagesHorizontally( output1, "", display_orientation_gray_display, "Selected Orientations", 4 );
	row2_output = JoinImagesHorizontally( output2, "", nms_result_gray_display, "Non-Maxima Suppressed gradients", 4 );
	Mat sobel_output = JoinImagesVertically(row1_output,"",row2_output,"", 4);
	imshow("Sobel Edge Detection", sobel_output );
	c = cvWaitKey();
    cvDestroyAllWindows();

	// Second derivative (Laplacian of Gaussian)
	Mat laplacian;
	Mat blurred_image1_gray;
	GaussianBlur(image1_gray,blurred_image1_gray,Size(5,5),0.5);
	Laplacian(blurred_image1_gray,laplacian,CV_32F,3);
	Mat zero_crossings;
	FindZeroCrossings(laplacian,zero_crossings);
	Sobel(blurred_image1_gray,horizontal_partial_derivative,CV_32F,1,0);
	Sobel(blurred_image1_gray,vertical_partial_derivative,CV_32F,0,1);
	abs_gradient = abs(horizontal_partial_derivative) + abs(vertical_partial_derivative);
	abs_gradient.convertTo(the_gradient,CV_8U);
	bitwise_and( the_gradient, zero_crossings, zero_crossings );
	// Use a different size Gaussian filter
	Mat laplacian2;
	Mat blurred_image1_gray2;
	GaussianBlur(image1_gray,blurred_image1_gray2,Size(41,41),2.0);
	Laplacian(blurred_image1_gray2,laplacian2,CV_32F,3);
	Mat zero_crossings2;
	FindZeroCrossings(laplacian2,zero_crossings2);
	Sobel(blurred_image1_gray2,horizontal_partial_derivative,CV_32F,1,0);
	Sobel(blurred_image1_gray2,vertical_partial_derivative,CV_32F,0,1);
	abs_gradient = abs(horizontal_partial_derivative) + abs(vertical_partial_derivative);
	abs_gradient.convertTo(the_gradient,CV_8U);
	bitwise_and( the_gradient, zero_crossings2, zero_crossings2 );
	// Create composite display image
	Mat laplacian_gray1 = convert_32bit_image_for_display( laplacian, 128.0 );
	Mat laplacian_gray2 = convert_32bit_image_for_display( laplacian2, 128.0 );
	Mat blurred_image1_gray_display, blurred_image1_gray2_display, zero_crossings_display, zero_crossings2_display, laplacian_gray1_display, laplacian_gray2_display;
	cvtColor(blurred_image1_gray, blurred_image1_gray_display, CV_GRAY2BGR);
	cvtColor(blurred_image1_gray2, blurred_image1_gray2_display, CV_GRAY2BGR);
	cvtColor(zero_crossings, zero_crossings_display, CV_GRAY2BGR);
	cvtColor(zero_crossings2, zero_crossings2_display, CV_GRAY2BGR);
	cvtColor(laplacian_gray1, laplacian_gray1_display, CV_GRAY2BGR);
	cvtColor(laplacian_gray2, laplacian_gray2_display, CV_GRAY2BGR);
	output1 = JoinImagesHorizontally( blurred_image1_gray_display, "Gaussian Smoothed image (sigma=0.5)", laplacian_gray1_display, "Laplacian of Gaussian", 4 );
	row1_output = JoinImagesHorizontally( output1, "", zero_crossings_display, "Zero crossings", 4 );
	output1 = JoinImagesHorizontally( blurred_image1_gray2_display, "Gaussian Smoothed image (sigma=2.0)", laplacian_gray2_display, "Laplacian of Gaussian", 4 );
	row2_output = JoinImagesHorizontally( output1, "", zero_crossings2_display, "Zero crossings", 4 );
    Mat log_output = JoinImagesVertically( row1_output, "", row2_output, "", 4 );
	imshow("Second Derivative (Laplacian of Gaussian)",log_output);
	c = cvWaitKey();
    cvDestroyAllWindows();

	// Multi spectral edge detection (using Canny)
	vector<Mat> input_planes(3);
	Mat processed_image = image2.clone();
	vector<Mat> output_planes;
	split(processed_image,output_planes);
	split(image2,input_planes);
	for (int plane=0; plane < image2.channels(); plane++)
		Canny(input_planes[plane],output_planes[plane],100,200);
	Mat multispectral_edges;
	merge(output_planes, multispectral_edges);
	Mat image2_gray_edges = image2_gray.clone();
	Canny(image2_gray,image2_gray_edges,100,200);
	Mat image2_gray_display, image2_gray_edges_display;
	cvtColor(image2_gray, image2_gray_display, CV_GRAY2BGR);
	cvtColor(image2_gray_edges, image2_gray_edges_display, CV_GRAY2BGR);
	row1_output = JoinImagesHorizontally( image2, "Colour Image", multispectral_edges, "Multispectral (3 channel Canny) edges", 4 );
	row2_output = JoinImagesHorizontally( image2_gray_display, "Greyscale Image", image2_gray_edges_display, "Greyscale (Canny) edges", 4 );
    Mat multispectral_output = JoinImagesVertically( row1_output, "", row2_output, "", 4 );
	imshow( "Multispectral edge detection using Canny", multispectral_output );
	c = cvWaitKey();
    cvDestroyAllWindows();

	// Sharpen Image
	Mat image_32bit, sharpened_image;
	image1.convertTo( image_32bit, CV_32F );
	Laplacian(image1,laplacian,CV_32F,3);
	Mat sharpened_image_32bit = image_32bit - 0.3*laplacian;
	sharpened_image_32bit.convertTo( sharpened_image, CV_8U );
	Mat laplacian_display = convert_32bit_image_for_display( laplacian, 128.0 );
	output1 = JoinImagesHorizontally( image1, "Original Image", laplacian_display, "Laplacian of Gaussian", 4 );
	row1_output = JoinImagesHorizontally( output1, "", sharpened_image, "Sharpened Image", 4 );
	imshow( "Image Sharpening", row1_output );
	c = cvWaitKey();
    cvDestroyAllWindows();

	// Contours and straight line segments
	Mat canny_edge_image;
	Canny(image2,canny_edge_image,80,150);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	Mat canny_edge_image_copy = canny_edge_image.clone();
	findContours(canny_edge_image_copy,contours,hierarchy,CV_RETR_TREE,CV_CHAIN_APPROX_NONE);

	vector<Vec4i> line_segments;
	vector<vector<Point>> approx_contours( contours.size() );
	for (int contour_number=0; (contour_number<(int)contours.size()); contour_number++)
	{	// Approximate each contour as a series of line segments.
		approxPolyDP( Mat(contours[contour_number]), approx_contours[contour_number], 6, true );
	}
	// Extract line segments from the contours.
	for (int contour_number=0; (contour_number<(int)contours.size()); contour_number++)
	{
		for (int line_segment_number=0; (line_segment_number<(int)approx_contours[contour_number].size()-1); line_segment_number++)
		{
			line_segments.push_back(Vec4i(approx_contours[contour_number][line_segment_number].x,approx_contours[contour_number][line_segment_number].y,
				approx_contours[contour_number][line_segment_number+1].x,approx_contours[contour_number][line_segment_number+1].y));
		}
	}
	// Draw the contours and then the segments
	Mat contours_image = Mat::zeros(canny_edge_image.size(), CV_8UC3);
	Mat line_segments_image = Mat::zeros(canny_edge_image.size(), CV_8UC3);
	for (int contour_number=0; (contour_number<(int)contours.size()); contour_number++)
	{
	    Scalar colour( rand()&0xFF, rand()&0xFF, rand()&0xFF );
        drawContours( contours_image, contours, contour_number, colour, 1, 8, hierarchy );
	}
	DrawLines(line_segments_image,line_segments);
	Mat canny_edge_image_display;
	cvtColor(canny_edge_image, canny_edge_image_display, CV_GRAY2BGR);
	row1_output = JoinImagesHorizontally( image2, "Original Image", canny_edge_image_display, "Canny Edge Image", 4 );
	row2_output = JoinImagesHorizontally( contours_image, "Boundary Chain Codes", line_segments_image, "Line Segments", 4 );
	Mat line_segments_display = JoinImagesVertically( row1_output, "", row2_output, "", 4 );
	imshow( "Line segment extraction", line_segments_display );
	c = cvWaitKey();
    cvDestroyAllWindows();

	// Hough tranform for (full) line detection
	vector<Vec2f> hough_lines;
	HoughLines(canny_edge_image, hough_lines, 1, PI/200.0, 100);
	Mat hough_lines_image = image2.clone();
	DrawLines(hough_lines_image, hough_lines);
	// Probabilistic Hough transform for line segments
	vector<Vec4i> hough_line_segments;
	HoughLinesP(canny_edge_image, hough_line_segments, 1.0, PI/200.0, 20, 20, 5);
	Mat hough_line_segments_image = Mat::zeros(canny_edge_image.size(), CV_8UC3);
	DrawLines(hough_line_segments_image, hough_line_segments);
	output1 = JoinImagesHorizontally( image2, "Original Image", hough_lines_image, "Hough for Lines", 4 );
	row1_output = JoinImagesHorizontally( output1, "", hough_line_segments_image, "Probabilistic Hough (for line segments)", 4 );
	
	// Hough for circles
	vector<Vec3f> circles;
	HoughCircles(image1_gray, circles, CV_HOUGH_GRADIENT,0.5,8,200,20,10,25);//2,20,100,20,5,30);
	Mat hough_circles_image = image1.clone();
	DrawCircles(hough_circles_image, circles);
	row2_output = JoinImagesHorizontally( image1, "Original Image", hough_circles_image, "Hough for Circles", 4 );
	Mat hough_output = JoinImagesVertically( row1_output, "", row2_output, "", 4 );
	imshow( "Hough transformation", hough_output );
	c = cvWaitKey();
    cvDestroyAllWindows();
}
