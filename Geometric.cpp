/*
 * This code is provided as part of "A Practical Introduction to Computer Vision with OpenCV"
 * by Kenneth Dawson-Howe © Wiley & Sons Inc. 2014.  All rights reserved.
 */
#include "Utilities.h"

void GeometricDemos( Mat& image1, Mat& image2, Mat& image3 )
{
	// Rotation
	Mat rotation_matrix( 2, 3, CV_32FC1 ), rotated_image;
	Point center = Point( image1.cols/2, image1.rows/2 );
	double angle = -15.0;
	double scale = 1.0;
	rotation_matrix = getRotationMatrix2D( center, angle, scale );
	warpAffine( image1, rotated_image, rotation_matrix, Size( image1.cols*5/4, image1.rows*5/4 ) );

	// Skewing
	Mat skewing_matrix = Mat::eye( 2, 3, CV_32FC1 ), skewed_image;
	*(((float*) (skewing_matrix.data))+1) = (float) 0.35;  // [0][1] Skewing
	*(((float*) (skewing_matrix.data))+2) = -30.0; // [0][2] Translate to the left
	warpAffine( rotated_image, skewed_image, skewing_matrix, rotated_image.size() );

	// Panoramic distortion
	Mat panoramic_matrix = Mat::eye( 2, 3, CV_32FC1), panoramic_image;
	*((float*) (panoramic_matrix.data)) = 1.25;      // [0][0] Panoramic distortion
	*(((float*) (panoramic_matrix.data))+2) = 0.0; // [0][2] Translate to the left
	warpAffine( skewed_image, panoramic_image, panoramic_matrix, skewed_image.size() );

	// Affine transformation
	Mat affine_matrix( 2, 3, CV_32FC1 ), affine_warped_image;
	affine_warped_image = Mat::zeros( 100, 400, image1.type() );
	Point2f source_points[4], destination_points[4];
	source_points[0] = Point2f( 29.0, 66.0 );
	source_points[1] = Point2f( 25.0, 135.0 );
	source_points[2] = Point2f( 256.0, 9.0 );
	destination_points[0] = Point2f( 0.0, 0.0 );
	destination_points[1] = Point2f( 0.0, 99.0 );
	destination_points[2] = Point2f( 399.0, 0.0 );
	
	affine_matrix = getAffineTransform( source_points, destination_points );
	warpAffine( image1, affine_warped_image, affine_matrix, affine_warped_image.size() );

	// Perspective transformation
	Mat perspective_matrix( 3, 3, CV_32FC1 ), perspective_warped_image;
	perspective_warped_image = Mat::zeros( 100, 400, image1.type() );
	source_points[3] = Point2f( 252.0, 71.0 );
	destination_points[3] = Point2f( 399.0, 99.0 );

	perspective_matrix = getPerspectiveTransform( source_points, destination_points );
	warpPerspective( image1, perspective_warped_image, perspective_matrix, perspective_warped_image.size() );
	Mat output1 = JoinImagesVertically( image1, "Original Image", skewed_image, "Skewed Image", 4, Scalar( 255, 0, 0 ) );
	Mat col1_output = JoinImagesVertically( output1, "", affine_warped_image, "Affine transformation", 4, Scalar( 255, 0, 0 ) );
	output1 = JoinImagesVertically( rotated_image, "Rotated Image (-15 degrees)", panoramic_image, "Panoramic Distortion", 4, Scalar( 255, 0, 0 ) );
	Mat col2_output = JoinImagesVertically( output1, "", perspective_warped_image, "Perspective transformation", 4, Scalar( 255, 0, 0 ) );
	Mat geometric_output = JoinImagesHorizontally( col1_output, "", col2_output, "", 4, Scalar( 255, 0, 0 ) );
	imshow("Geometric transformations", geometric_output );

	char c = cvWaitKey();

	// Image expansion (showing different interpolation methods)
	Mat nearest_expanded_image, bilinear_expanded_image, bicubic_expanded_image;
	int expansion_factor=2;
	for (int point=0; point<4; point++)
	{
		destination_points[point].x *= expansion_factor;
		destination_points[point].y *= expansion_factor;
	}
	perspective_matrix = getPerspectiveTransform( source_points, destination_points );
	Size result_size(perspective_warped_image.cols*expansion_factor, perspective_warped_image.rows*expansion_factor);
	warpPerspective( image1, nearest_expanded_image, perspective_matrix, result_size, INTER_NEAREST );
	warpPerspective( image1, bilinear_expanded_image, perspective_matrix, result_size, INTER_LINEAR );
	warpPerspective( image1, bicubic_expanded_image, perspective_matrix, result_size, INTER_CUBIC  );
	output1 = JoinImagesVertically( nearest_expanded_image, "Nearest neighbour interpolation", bilinear_expanded_image, "Bilinear interpolation", 4 );
	Mat interpolation_output = JoinImagesVertically( output1, "", bicubic_expanded_image, "Bicubic interpolation", 4 );
	imshow("Interpolation schemes", interpolation_output );
	c = cvWaitKey();
	cvDestroyAllWindows();

}
