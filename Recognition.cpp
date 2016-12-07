/*
 * This code is provided as part of "A Practical Introduction to Computer Vision with OpenCV"
 * by Kenneth Dawson-Howe © Wiley & Sons Inc. 2014.  All rights reserved.
 */
#include "Utilities.h"
#include "opencv2/objdetect.hpp"
#include "opencv2/imgcodecs.hpp"
#include <opencv2/ml.hpp>
#include <fstream>

using namespace cv;
using namespace cv::ml;

void CompareRecognitionResults( Mat& locations_found, Mat& ground_truth, double& precision, double& recall, double& accuracy, double& specificity, double& f1 )
{
	CV_Assert( locations_found.type() == CV_8UC1 );
	CV_Assert( ground_truth.type() == CV_8UC1 );
	int false_positives = 0;
	int false_negatives = 0;
	int true_positives = 0;
	int true_negatives = 0;
	for (int row=0; row < ground_truth.rows; row++)
		for (int col=0; col < ground_truth.cols; col++)
		{
			uchar result = locations_found.at<uchar>(row,col);
			uchar gt = ground_truth.at<uchar>(row,col);
			if ( gt > 0 )
				if ( result > 0 )
					true_positives++;
				else false_negatives++;
			else if ( result > 0 )
				false_positives++;
			else true_negatives++;
		}
	precision = ((double) true_positives) / ((double) (true_positives+false_positives));
	recall = ((double) true_positives) / ((double) (true_positives+false_negatives));
	accuracy = ((double) (true_positives+true_negatives)) / ((double) (true_positives+false_positives+true_negatives+false_negatives));
	specificity = ((double) true_negatives) / ((double) (false_positives+true_negatives));
	f1 = 2.0*precision*recall / (precision + recall);
}

void FindLocalMaxima( Mat& input_image, Mat& local_maxima, double threshold_value )
{
	Mat dilated_input_image,thresholded_input_image,thresholded_input_8bit;
	dilate(input_image,dilated_input_image,Mat());
	compare(input_image,dilated_input_image,local_maxima,CMP_EQ);
	threshold( input_image, thresholded_input_image, threshold_value, 255, THRESH_BINARY );
	thresholded_input_image.convertTo( thresholded_input_8bit, CV_8U );
	bitwise_and( local_maxima, thresholded_input_8bit, local_maxima );
}

void FindLocalMinima( Mat& input_image, Mat& local_minima, double threshold_value )
{
	Mat eroded_input_image,thresholded_input_image,thresholded_input_8bit;
	erode(input_image,eroded_input_image,Mat());
	compare(input_image,eroded_input_image,local_minima,CMP_EQ);
	threshold( input_image, thresholded_input_image, threshold_value, 255, THRESH_BINARY_INV );
	thresholded_input_image.convertTo( thresholded_input_8bit, CV_8U );
	bitwise_and( local_minima, thresholded_input_8bit, local_minima );
}

void DrawMatchingTemplateRectangles( Mat& display_image, Mat& matched_template_map, Mat& template_image, Scalar passed_colour=-1.0 )
{
	int image_channels = matched_template_map.channels();
	int values_on_each_row = matched_template_map.cols;
	for (int row=0; row < matched_template_map.rows; row++) {
		uchar* curr_point = matched_template_map.ptr<uchar>(row);
		for (int column=0; column < values_on_each_row; column++)
		{
			if (*curr_point > 0)
			{
				Scalar colour( rand()&0xFF, rand()&0xFF, rand()&0xFF );
				Point location( column, row );
				rectangle( display_image, location, Point( column + template_image.cols , row + template_image.rows ), (passed_colour.val[0] == -1.0) ? colour : passed_colour, 1, 8, 0 );
			}
			curr_point += image_channels;
		}
	}
}


void ChamferMatching( Mat& chamfer_image, Mat& model, Mat& matching_image )
{
	// Extract the model points (as they are sparse).
	vector<Point> model_points;
	int image_channels = model.channels();
	for (int model_row=0; (model_row < model.rows); model_row++)
	{
		uchar *curr_point = model.ptr<uchar>(model_row);
		for (int model_column=0; (model_column < model.cols); model_column++)
		{
			if (*curr_point > 0)
			{
				Point& new_point = Point(model_column,model_row);
				model_points.push_back(new_point);
			}
			curr_point += image_channels;
		}
	}
	int num_model_points = model_points.size();
	image_channels = chamfer_image.channels();
	// Try the model in every possible position
	matching_image = Mat(chamfer_image.rows-model.rows+1, chamfer_image.cols-model.cols+1, CV_32FC1);
	for (int search_row=0; (search_row <= chamfer_image.rows-model.rows); search_row++)
	{
		float *output_point = matching_image.ptr<float>(search_row);
		for (int search_column=0; (search_column <= chamfer_image.cols-model.cols); search_column++)
		{
			float matching_score = 0.0;
			for (int point_count=0; (point_count < num_model_points); point_count++)
				matching_score += (float) *(chamfer_image.ptr<float>(model_points[point_count].y+search_row) + search_column + model_points[point_count].x*image_channels);
			*output_point = matching_score;
			output_point++;
		}
	}
}

#define MAX_SAMPLES 100
#define MAX_FEATURE_VALUE 511
#define UNKNOWN_CLASS 3

void myConvexityDefects( vector<Point> contour, vector<int> hull_indices, vector<Vec4i>& convexity_defects, int minimum_distance_required=1 )
{
	if (minimum_distance_required < 1)
		minimum_distance_required = 1;
	int previous_hull_index = hull_indices.size()-1;
	for (unsigned int hull_index=0; (hull_index < hull_indices.size()); previous_hull_index=hull_index, hull_index++)
	{
		if (hull_indices[hull_index] != ((hull_indices[previous_hull_index]+1)%(contour.size()))) // Not the next point on the contour
		{
			int x1 = contour[hull_indices[previous_hull_index]].x;
			int y1 = contour[hull_indices[previous_hull_index]].y;
			int x2 = contour[hull_indices[hull_index]].x;
			int y2 = contour[hull_indices[hull_index]].y;
			int max_distance = 0;
			int max_distance_index = -1;
			// Look for the further point away
			for (int between_index=(hull_indices[previous_hull_index]+1)%(contour.size()); (between_index != hull_indices[hull_index]); 
				                                               between_index=(between_index+1)%(contour.size()))
			{
				// Compute the distance to the internal point
				int x0 = contour[between_index].x;
				int y0 = contour[between_index].y;
				//int area = ((x1-x0)*(y2-y0) - (x2-x0)*(y1-y0))/2;
				int distance = (int) (((double) abs((y2-y1)*x0-(x2-x1)*y0+x2*y1-y2*x1))/sqrt((y2-y1)*(y2-y1)+(x2-x1)*(x2-x1)));
				if (distance > max_distance)
				{
					max_distance = distance;
					max_distance_index = between_index;
				}
			}
			if (max_distance > minimum_distance_required)
			{
				// Store new convexity
				Vec4i* new_convexity = new Vec4i(hull_indices[previous_hull_index],hull_indices[hull_index],max_distance_index,max_distance);
				convexity_defects.push_back(*new_convexity);
			}
		}
	}
}

void SupportVectorMachineDemo(Mat& class1_samples, char* class1_name, Mat& class2_samples, char* class2_name, Mat& unknown_samples)
{
    float labels[MAX_SAMPLES];
    float training_data[MAX_SAMPLES][2];
	Ptr<SVM> svm;

    // Image for visual representation of (2-D) feature space
    int width = MAX_FEATURE_VALUE+1, height = MAX_FEATURE_VALUE+1;
    Mat feature_space = Mat::zeros(height, width, CV_8UC3);

	int number_of_samples = 0;
	// Loops three times:
	//  1st time - extracts feature values for class 1
	//  2nd time - extracts feature values for class 2 AND trains SVM
	//  3rd time - extracts feature values for unknowns AND predicts their classes using SVM
	for (int current_class = 1; current_class<=UNKNOWN_CLASS; current_class++)
	{
		Mat gray_image,binary_image;
		if (current_class == 1)
			cvtColor(class1_samples, gray_image, CV_BGR2GRAY);
		else if (current_class == 2)
			cvtColor(class2_samples, gray_image, CV_BGR2GRAY);
		else cvtColor(unknown_samples, gray_image, CV_BGR2GRAY);
		threshold(gray_image,binary_image,128,255,THRESH_BINARY_INV);
	    vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		findContours(binary_image,contours,hierarchy,CV_RETR_TREE,CV_CHAIN_APPROX_NONE);
		Mat contours_image = Mat::zeros(binary_image.size(), CV_8UC3);
		contours_image = Scalar(255,255,255);
		// Do some processing on all contours (objects and holes!)
		vector<vector<Point>> hulls(contours.size());
		vector<vector<int>> hull_indices(contours.size());
		vector<vector<Vec4i>> convexity_defects(contours.size());
		vector<Moments> contour_moments(contours.size());
		for (int contour_number=0; (contour_number>=0); contour_number=hierarchy[contour_number][0])
		{
			if (contours[contour_number].size() > 10)
			{
				convexHull(contours[contour_number], hulls[contour_number]);
				convexHull(contours[contour_number], hull_indices[contour_number], true);
				//convexityDefects( contours[contour_number], hull_indices[contour_number], convexity_defects[contour_number]);
				myConvexityDefects( contours[contour_number], hull_indices[contour_number], convexity_defects[contour_number], 2);
				contour_moments[contour_number] = moments( contours[contour_number] );
				// Draw the shape and features
				Scalar colour( rand()&0x7F, rand()&0x7F, rand()&0x7F );
				drawContours( contours_image, contours, contour_number, colour, CV_FILLED, 8, hierarchy );
				char output[500];
				double area = contourArea(contours[contour_number])+contours[contour_number].size()/2+1;
				// Draw the convex hull
				drawContours( contours_image, hulls, contour_number, Scalar(0, 255, 0) );
				// Highlight any convexities
				int largest_convexity_depth=0;
				for (int convexity_index=0; convexity_index < (int)convexity_defects[contour_number].size(); convexity_index++)
				{
					if (convexity_defects[contour_number][convexity_index][3] > largest_convexity_depth) 
						largest_convexity_depth = convexity_defects[contour_number][convexity_index][3];
					if (convexity_defects[contour_number][convexity_index][3] > 2)//256*2) 
					{
						line( contours_image, contours[contour_number][convexity_defects[contour_number][convexity_index][0]], contours[contour_number][convexity_defects[contour_number][convexity_index][2]], Scalar(0,0, 255));
						line( contours_image, contours[contour_number][convexity_defects[contour_number][convexity_index][1]], contours[contour_number][convexity_defects[contour_number][convexity_index][2]], Scalar(0,0, 255));
					}
				}
				// Compute moments and a measure of the deepest convexity
				double hu_moments[7];
				HuMoments( contour_moments[contour_number], hu_moments );
				double diameter = ((double) contours[contour_number].size())/PI;
				//double convexity_depth = ((double) largest_convexity_depth)/256.0;
				double convex_measure = largest_convexity_depth/diameter;
				int class_id = current_class;
				float feature[2] = { (float) convex_measure*((float) MAX_FEATURE_VALUE), (float) hu_moments[0]*((float) MAX_FEATURE_VALUE) };
				if (feature[0] > ((float) MAX_FEATURE_VALUE)) feature[0] = ((float) MAX_FEATURE_VALUE);
				if (feature[1] > ((float) MAX_FEATURE_VALUE)) feature[1] = ((float) MAX_FEATURE_VALUE);
				if (current_class == UNKNOWN_CLASS)
				{
					// Try to predict the class
					Mat sampleMat = (Mat_<float>(1,2) << feature[0], feature[1]);
					float prediction = svm->predict(sampleMat);
					class_id = (prediction > 0.0) ? 1 : (prediction < 0.0) ? 2 : 0;
				}
				char* current_class_name = (class_id==1) ? class1_name : (class_id==2) ? class2_name : "Unknown";

				sprintf(output,"Class=%s, Features %.2f, %.2f", current_class_name, feature[0]/((float) MAX_FEATURE_VALUE), feature[1]/((float) MAX_FEATURE_VALUE));
				Point location( contours[contour_number][0].x-40, contours[contour_number][0].y-3 );
				putText( contours_image, output, location, FONT_HERSHEY_SIMPLEX, 0.4, colour );
				if (current_class == UNKNOWN_CLASS)
				{
				}
				else if (number_of_samples < MAX_SAMPLES)
				{
					labels[number_of_samples] = (float) ((current_class == 1) ? 1.0 : -1.0);
					training_data[number_of_samples][0] = feature[0];
					training_data[number_of_samples][1] = feature[1];
					number_of_samples++;
				}
			}
		}
		if (current_class == 1)
		{
			Mat temp_output = contours_image.clone();
			imshow(class1_name, temp_output );
		}
		else if (current_class == 2)
		{
			Mat temp_output2 = contours_image.clone();
			imshow(class2_name, temp_output2 );

			// Now that features for both classes have been determined, train the SVM
			svm = ml::SVM::create();
			svm->setType(SVM::C_SVC);
		    svm->setKernel(SVM::LINEAR);
			Mat labelsMat(number_of_samples, 1, CV_32SC1, labels);
			Mat trainingDataMat(number_of_samples, 2, CV_32FC1, training_data);
			Ptr<ml::TrainData> tData = ml::TrainData::create(trainingDataMat, ml::SampleTypes::ROW_SAMPLE, labelsMat);
			svm->train(tData);

			// Show the SVM classifier for all possible feature values
			Vec3b green(192,255,192), blue (255,192,192);
			// Show the decision regions given by the SVM
			for (int i = 0; i < feature_space.rows; ++i)
				for (int j = 0; j < feature_space.cols; ++j)
				{
					Mat sampleMat = (Mat_<float>(1,2) << j,i);
					float prediction = svm->predict(sampleMat);
					if (prediction > 0.0)
						feature_space.at<Vec3b>(i,j) = green;
					else if (prediction < 0.0)
					    feature_space.at<Vec3b>(i,j)  = blue;
				}
			// Show the training data (as dark circles)
			for(int sample=0; sample < number_of_samples; sample++)
				if (labels[sample] == 1.0)
					circle( feature_space, Point((int) training_data[sample][0], (int) training_data[sample][1]), 5, Scalar( 255, 255, 0 ), -1, 8);
				else circle( feature_space, Point((int) training_data[sample][0], (int) training_data[sample][1]), 5, Scalar( 255, 0, 0 ), -1, 8);
			// Highlight the support vectors (in red)
			Mat support_vectors = svm->getSupportVectors();
			for (int support_vector_index = 0; support_vector_index < support_vectors.rows; ++support_vector_index)
			{
				const float* v = support_vectors.ptr<float>(support_vector_index);
				circle( feature_space,  Point( (int) v[0], (int) v[1]),   5,  Scalar(0, 0, 255));
			}
			imshow("SVM feature space", feature_space);
		}
		else if (current_class == 3)
		{
			imshow("Classification of unknowns", contours_image );
		}
	}
}

void PCAFaceRecognition() {
	/*
// The following code is not supported in the main release of OpenCV 3.0.0.  A special version of the
// software will be released containing updated code using opencv_contrib will be released shortly.
#define NUMBER_OF_FACES 10
#define NUMBER_OF_IMAGES_PER_FACE 3
    vector<Mat> known_face_images;
    vector<int> known_labels;
    vector<Mat> unknown_face_images;
    vector<int> unknown_labels;
    // Load greyscale face images (which are from http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html)
	char file_name[40];
	Mat original_images,row_so_far,image_so_far,temp_image1;
	int face_number = 1;
	for (; face_number<=NUMBER_OF_FACES; face_number++)
	{
		for (int image_number = 1; image_number<=NUMBER_OF_IMAGES_PER_FACE; image_number++)
		{
			sprintf(file_name,"Media/att_faces/s%d/%d.pgm",face_number,image_number);
			Mat current_image = imread(file_name,0);
			if (image_number>1)
			{
				known_face_images.push_back(current_image);
				known_labels.push_back(face_number);
			}
			else
			{
				// Keep the last image of each face as a test case.
				unknown_face_images.push_back(current_image);
				unknown_labels.push_back(face_number);
			}
			cvtColor(current_image, current_image, CV_GRAY2BGR);
			if (image_number == 2)
			{
				if (face_number%10 == 1)
				{
					if (face_number > 1)
						if (face_number == 11)
							original_images = row_so_far.clone();
						else original_images = JoinImagesVertically( original_images, "", row_so_far, "", 1 );
					row_so_far = current_image.clone();
				}
				else
				{
					char image_number_string[10],previous_image_number_string[10];
					sprintf(previous_image_number_string,"%d",face_number-1);
					sprintf(image_number_string,"%d",face_number);
					row_so_far = JoinImagesHorizontally( row_so_far, (face_number%10==2)?previous_image_number_string:"", current_image, image_number_string, 1 );
				}
			}
		}
	}
	if (face_number <= 11)
		original_images = row_so_far.clone();
	else original_images = JoinImagesVertically( original_images, "", row_so_far, "", 1 );
	imshow("Known face images", original_images);
	imwrite("pca_unknown_faces.bmp",original_images);
    Ptr<FaceRecognizer> face_recogniser = createEigenFaceRecognizer();
    face_recogniser->train(known_face_images, known_labels);
	char previous_face_number_string[100]="";
	char face_number_string[100]="";
	int correct_count = 0;
	for (face_number = 0; face_number < (int)unknown_face_images.size(); face_number++)
	{
		int predicted_face_number = 0;
		double recognition_confidence = 0.0;
		face_recogniser->predict(unknown_face_images[face_number],predicted_face_number,recognition_confidence);
		if (unknown_labels[face_number]==predicted_face_number)
			correct_count++;
		strcpy(previous_face_number_string,face_number_string);
		cvtColor(unknown_face_images[face_number], temp_image1, CV_GRAY2BGR);
		sprintf(face_number_string,"%d (%.0f)",predicted_face_number,recognition_confidence);
		Point location(2,15);
		putText( temp_image1, face_number_string, location, FONT_HERSHEY_SIMPLEX, 0.4, unknown_labels[face_number]==predicted_face_number?Scalar( 0,255,0 ):Scalar( 0,0,255 ) );
		if (face_number%10 == 0)
		{
			if (face_number > 10)
				image_so_far = JoinImagesVertically( image_so_far, "", row_so_far, "", 1 );
			else image_so_far = row_so_far.clone();
			row_so_far = temp_image1.clone();
		}
		else 
		{
			row_so_far = JoinImagesHorizontally( row_so_far, "", temp_image1, "", 1 );
		}
	}
	if (face_number > 10)
		image_so_far = JoinImagesVertically( image_so_far, "", row_so_far, "", 1 );
	else image_so_far = row_so_far.clone();
	char output[300];
	sprintf(output,"OVERALL Recognised %d/%d (with %d training image%s of %d subjects)",correct_count,unknown_face_images.size(),NUMBER_OF_IMAGES_PER_FACE-1,(NUMBER_OF_IMAGES_PER_FACE-1==1)?"":"s",NUMBER_OF_FACES);
	Point location(10,image_so_far.rows-10);
	putText( image_so_far, output, location, FONT_HERSHEY_SIMPLEX, 0.4, Scalar( 255,0,0 ) );
	imshow("Recognised faces using PCA (Eigenfaces)", image_so_far);
	*/
}

void PCASimpleExample()
{
	// Different possible samples with varying variances along the secondary axis
	//float samples[][2] = { {30,30}, {50,50}, {101,101}, {151,151}, {201,201}, {251,251}, {311,311}, {351,351}, {411,411}, {391,391} };
	float samples[][2] = { {30,30}, {70,50}, {101,81}, {151,171}, {181,201}, {251,271}, {291,311}, {351,371}, {421,411}, {391,391} };
	//float samples[][2] = { {30,30}, {120,50}, {41,81}, {221,171}, {151,201}, {211,271}, {391,311}, {301,371}, {501,411}, {391,391} };
	int number_of_samples = sizeof(samples)/sizeof(int[2]);
	Mat samples_matrix(number_of_samples, 2, CV_32FC1, samples);
	int width = 800;  int height = 512;
	char output[500];
    Mat feature_space = Mat::zeros(height, width, CV_8UC3);
	feature_space.setTo(Scalar(255,255,255));
	// Show the samples (as circles)
	for(int sample=0; sample < number_of_samples; sample++)
		circle( feature_space, Point((int) samples[sample][0], (int) samples[sample][1]), 3, Scalar( 0,128,0 ), -1, 8);

	PCA pca(samples_matrix, Mat(), 0, 2 );
	Mat eigenvalues = pca.eigenvalues;
	Mat eigenvectors = pca.eigenvectors;
	Mat mean = pca.mean;
	sprintf(output,"Mean (%.1f, %.1f)",((float*) mean.data)[0],((float*) mean.data)[1]);
	Point location( (int) ((float*) mean.data)[0]+3, (int) ((float*) mean.data)[1] );
	putText( feature_space, output, location, FONT_HERSHEY_SIMPLEX, 0.4, Scalar( 0,0,255 ) );
	for (int row=0; row<eigenvalues.rows; row++)
	{
		float eigenvalue = ((float*) eigenvalues.data)[row*eigenvalues.cols];
		float length = ((float) height)/(((float) 1.8)*(((float) 1.0)+((float) row)));
		sprintf(output,"Eigenvalue %.1f  Eigenvector (%.4f, %.4f) ",eigenvalue,((float*) eigenvectors.data)[0+row*eigenvectors.cols],((float*) eigenvectors.data)[1+row*eigenvectors.cols]);
		Point location((int)(((float*) mean.data)[0]+length*((float*) eigenvectors.data)[0+row*eigenvectors.cols])+3,(int)(((float*) mean.data)[0]+length*((float*) eigenvectors.data)[1+row*eigenvectors.cols])+3);
		putText( feature_space, output, location, FONT_HERSHEY_SIMPLEX, 0.4, Scalar( 0,0,255 ) );
		arrowedLine(feature_space,Point((int)(((float*) mean.data)[0]-length*((float*) eigenvectors.data)[0+row*eigenvectors.cols]),(int)(((float*) mean.data)[0]-length*((float*) eigenvectors.data)[1+row*eigenvectors.cols])),
								Point((int)(((float*) mean.data)[0]+length*((float*) eigenvectors.data)[0+row*eigenvectors.cols]),(int)(((float*) mean.data)[0]+length*((float*) eigenvectors.data)[1+row*eigenvectors.cols])),
								Scalar(0,0,255));
	}
	Mat transformed_samples_matrix = samples_matrix.clone();
	for (int sample_number = 0; sample_number<number_of_samples; sample_number++)
	{
		Mat sample = samples_matrix.row(sample_number), transformed_sample;
		pca.project(sample,transformed_sample);
		transformed_sample.row(0).copyTo(transformed_samples_matrix.row(sample_number));
		sprintf(output,"(%.0f, %.0f) -> (%.0f, %.0f)",((float*) sample.data)[0],((float*) sample.data)[1],((float*) transformed_sample.data)[0],((float*) transformed_sample.data)[1]);
		Point location( (int) ((float*) sample.data)[0]+3, (int) ((float*) sample.data)[1] );
		putText( feature_space, output, location, FONT_HERSHEY_SIMPLEX, 0.4, Scalar( 0,0,0 ) );
	}
	{  // OPTIONAL: Write covariance matrices on images
		Mat covariance,average,new_covariance,new_average;
		calcCovarMatrix(samples_matrix,covariance,average,CV_COVAR_NORMAL | CV_COVAR_ROWS);
		calcCovarMatrix(transformed_samples_matrix,new_covariance,new_average,CV_COVAR_NORMAL | CV_COVAR_ROWS);
		covariance = covariance/number_of_samples;
		new_covariance = new_covariance/number_of_samples;
		location.y = height-100;
		location.x = 10;
		sprintf(output,"Original covariance matrix:",((float*) covariance.data)[0+0*covariance.cols],((float*) covariance.data)[1+0*covariance.cols]);
		putText( feature_space, output, location, FONT_HERSHEY_SIMPLEX, 0.4, Scalar( 0,0,0 ) );
		sprintf(output,"%5.1f, %5.1f",((double*) covariance.data)[0+0*covariance.cols],((double*) covariance.data)[1+0*covariance.cols]);
		location.x = 20;
		location.y += 15;
		putText( feature_space, output, location, FONT_HERSHEY_SIMPLEX, 0.4, Scalar( 0,0,0 ) );
		sprintf(output,"%5.1f, %5.1f",((double*) covariance.data)[0+1*covariance.cols],((double*) covariance.data)[1+1*covariance.cols]);
		location.y += 15;
		putText( feature_space, output, location, FONT_HERSHEY_SIMPLEX, 0.4, Scalar( 0,0,0 ) );
		location.y += 30;
		location.x = 10;
		sprintf(output,"New covariance matrix:");
		putText( feature_space, output, location, FONT_HERSHEY_SIMPLEX, 0.4, Scalar( 0,0,0 ) );
		sprintf(output,"%5.1f, %5.1f",((double*) new_covariance.data)[0+0*new_covariance.cols],((double*) new_covariance.data)[1+0*new_covariance.cols]);
		location.x = 20;
		location.y += 15;
		putText( feature_space, output, location, FONT_HERSHEY_SIMPLEX, 0.4, Scalar( 0,0,0 ) );
		sprintf(output,"%5.1f, %5.1f",((double*) new_covariance.data)[0+1*new_covariance.cols],((double*) new_covariance.data)[1+1*new_covariance.cols]);
		location.y += 15;
		putText( feature_space, output, location, FONT_HERSHEY_SIMPLEX, 0.4, Scalar( 0,0,0 ) );
	}
	imshow("PCA transformation",feature_space);
}



void RecognitionDemos( Mat& full_image, Mat& template1, Mat& template2, Mat& template1locations, Mat& template2locations, VideoCapture& bicycle_video, Mat& bicycle_background, Mat& bicycle_model, VideoCapture& people_video, CascadeClassifier& cascade, Mat& numbers, Mat& good_orings, Mat& bad_orings, Mat& unknown_orings )
{
	Timestamper* timer = new Timestamper();

	// Principal Components Analysis
	PCASimpleExample();
    char ch = cvWaitKey();
	cvDestroyAllWindows();

// The following code is not supported in the main release of OpenCV 3.0.0.  A special version of the
// software will be released containing updated code using opencv_contrib will be released shortly.
//	PCAFaceRecognition();
//    ch = cvWaitKey();
//	cvDestroyAllWindows();

	// Statistical Pattern Recognition
	Mat gray_numbers,binary_numbers;
	cvtColor(numbers, gray_numbers, CV_BGR2GRAY);
	threshold(gray_numbers,binary_numbers,128,255,THRESH_BINARY_INV);
    vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(binary_numbers,contours,hierarchy,CV_RETR_TREE,CV_CHAIN_APPROX_NONE);
	Mat contours_image = Mat::zeros(binary_numbers.size(), CV_8UC3);
	contours_image = Scalar(255,255,255);
	// Do some processing on all contours (objects and holes!)
	vector<RotatedRect> min_bounding_rectangle(contours.size());
	vector<vector<Point>> hulls(contours.size());
	vector<vector<int>> hull_indices(contours.size());
	vector<vector<Vec4i>> convexity_defects(contours.size());
	vector<Moments> contour_moments(contours.size());
	for (int contour_number=0; (contour_number<(int)contours.size()); contour_number++)
	{
		if (contours[contour_number].size() > 10)
		{
			min_bounding_rectangle[contour_number] = minAreaRect(contours[contour_number]);
			convexHull(contours[contour_number], hulls[contour_number]);
			convexHull(contours[contour_number], hull_indices[contour_number]);
			convexityDefects( contours[contour_number], hull_indices[contour_number], convexity_defects[contour_number]);
			contour_moments[contour_number] = moments( contours[contour_number] );
		}
	}
	for (int contour_number=0; (contour_number>=0); contour_number=hierarchy[contour_number][0])
	{
		if (contours[contour_number].size() > 10)
		{
        Scalar colour( rand()&0x7F, rand()&0x7F, rand()&0x7F );
        drawContours( contours_image, contours, contour_number, colour, CV_FILLED, 8, hierarchy );
		char output[500];
		double area = contourArea(contours[contour_number])+contours[contour_number].size()/2+1;
		// Process any holes (removing the area from the are of the enclosing contour)
		for (int hole_number=hierarchy[contour_number][2]; (hole_number>=0); hole_number=hierarchy[hole_number][0])
		{
			area -= (contourArea(contours[hole_number])-contours[hole_number].size()/2+1);
			Scalar colour( rand()&0x7F, rand()&0x7F, rand()&0x7F );
 			drawContours( contours_image, contours, hole_number, colour, CV_FILLED, 8, hierarchy );
			sprintf(output,"Area=%.0f", contourArea(contours[hole_number])-contours[hole_number].size()/2+1);
			Point location( contours[hole_number][0].x +20, contours[hole_number][0].y +5 );
			putText( contours_image, output, location, FONT_HERSHEY_SIMPLEX, 0.4, colour );
		}
		// Draw the minimum bounding rectangle
		Point2f bounding_rect_points[4];
		min_bounding_rectangle[contour_number].points(bounding_rect_points);
		line( contours_image, bounding_rect_points[0], bounding_rect_points[1], Scalar(0, 0, 127));
		line( contours_image, bounding_rect_points[1], bounding_rect_points[2], Scalar(0, 0, 127));
		line( contours_image, bounding_rect_points[2], bounding_rect_points[3], Scalar(0, 0, 127));
		line( contours_image, bounding_rect_points[3], bounding_rect_points[0], Scalar(0, 0, 127));
		float bounding_rectangle_area = min_bounding_rectangle[contour_number].size.area();
		// Draw the convex hull
        drawContours( contours_image, hulls, contour_number, Scalar(127,0,127) );
		// Highlight any convexities
		int largest_convexity_depth=0;
		for (int convexity_index=0; convexity_index < (int)convexity_defects[contour_number].size(); convexity_index++)
		{
			if (convexity_defects[contour_number][convexity_index][3] > largest_convexity_depth)
				largest_convexity_depth = convexity_defects[contour_number][convexity_index][3];
			if (convexity_defects[contour_number][convexity_index][3] > 256*2)
			{
				line( contours_image, contours[contour_number][convexity_defects[contour_number][convexity_index][0]], contours[contour_number][convexity_defects[contour_number][convexity_index][2]], Scalar(0,0, 255));
				line( contours_image, contours[contour_number][convexity_defects[contour_number][convexity_index][1]], contours[contour_number][convexity_defects[contour_number][convexity_index][2]], Scalar(0,0, 255));
			}
		}
		double hu_moments[7];
		HuMoments( contour_moments[contour_number], hu_moments );
		sprintf(output,"Perimeter=%d, Area=%.0f, BArea=%.0f, CArea=%.0f", contours[contour_number].size(),area,min_bounding_rectangle[contour_number].size.area(),contourArea(hulls[contour_number]));
		Point location( contours[contour_number][0].x, contours[contour_number][0].y-3 );
		putText( contours_image, output, location, FONT_HERSHEY_SIMPLEX, 0.4, colour );
		sprintf(output,"HuMoments = %.2f, %.2f, %.2f", hu_moments[0],hu_moments[1],hu_moments[2]);
		Point location2( contours[contour_number][0].x+100, contours[contour_number][0].y-3+15 );
		putText( contours_image, output, location2, FONT_HERSHEY_SIMPLEX, 0.4, colour );
		}
	}
	imshow("Shape Statistics", contours_image );
	char c = cvWaitKey();
	cvDestroyAllWindows();

	// Support Vector Machine
	imshow("Good - original",good_orings);
	imshow("Defective - original",bad_orings);
	imshow("Unknown - original",unknown_orings);
	SupportVectorMachineDemo(good_orings,"Good",bad_orings,"Defective",unknown_orings);
	c = cvWaitKey();
	cvDestroyAllWindows();

	// Template Matching
	Mat display_image, correlation_image;
	full_image.copyTo( display_image );
	double min_correlation, max_correlation;
	Mat matched_template_map;
	int result_columns =  full_image.cols - template1.cols + 1;
	int result_rows = full_image.rows - template1.rows + 1;
	correlation_image.create( result_columns, result_rows, CV_32FC1 );
	timer->reset();
	double before_tick_count = static_cast<double>(getTickCount());
	matchTemplate( full_image, template1, correlation_image, CV_TM_CCORR_NORMED );
	double after_tick_count = static_cast<double>(getTickCount());
	double duration_in_ms = 1000.0*(after_tick_count-before_tick_count)/getTickFrequency();
	minMaxLoc( correlation_image, &min_correlation, &max_correlation );
	FindLocalMaxima( correlation_image, matched_template_map, max_correlation*0.99 );
	timer->recordTime("Template Matching (1)");
	Mat matched_template_display1;
	cvtColor(matched_template_map, matched_template_display1, CV_GRAY2BGR);
	Mat correlation_window1 = convert_32bit_image_for_display( correlation_image, 0.0 );
	DrawMatchingTemplateRectangles( display_image, matched_template_map, template1, Scalar(0,0,255) );
	double precision, recall, accuracy, specificity, f1;
	Mat template1locations_gray;
	cvtColor(template1locations, template1locations_gray, CV_BGR2GRAY);
	CompareRecognitionResults( matched_template_map, template1locations_gray, precision, recall, accuracy, specificity, f1 );
	char results[400];
	Scalar colour( 255, 255, 255);
	sprintf( results, "precision=%.2f", precision);
	Point location( 7, 213 );
	putText( display_image, "Results (1)", location, FONT_HERSHEY_SIMPLEX, 0.4, colour );
	location.y += 13;
	putText( display_image, results, location, FONT_HERSHEY_SIMPLEX, 0.4, colour );
	sprintf( results, "recall=%.2f", recall);
	location.y += 13;
	putText( display_image, results, location, FONT_HERSHEY_SIMPLEX, 0.4, colour );
	sprintf( results, "accuracy=%.2f", accuracy);
	location.y += 13;
	putText( display_image, results, location, FONT_HERSHEY_SIMPLEX, 0.4, colour );
	sprintf( results, "specificity=%.2f", specificity);
	location.y += 13;
	putText( display_image, results, location, FONT_HERSHEY_SIMPLEX, 0.4, colour );
	sprintf( results, "f1=%.2f", f1);
	location.y += 13;
	putText( display_image, results, location, FONT_HERSHEY_SIMPLEX, 0.4, colour );
  
	result_columns =  full_image.cols - template2.cols + 1;
	result_rows = full_image.rows - template2.rows + 1;
	correlation_image.create( result_columns, result_rows, CV_32FC1 );
	timer->ignoreTimeSinceLastRecorded();
	matchTemplate( full_image, template2, correlation_image, CV_TM_CCORR_NORMED );
	minMaxLoc( correlation_image, &min_correlation, &max_correlation );
	FindLocalMaxima( correlation_image, matched_template_map, max_correlation*0.99 );
	timer->recordTime("Template Matching (2)");
	Mat matched_template_display2;
	cvtColor(matched_template_map, matched_template_display2, CV_GRAY2BGR);
	Mat correlation_window2 = convert_32bit_image_for_display( correlation_image, 0.0 );
	DrawMatchingTemplateRectangles( display_image, matched_template_map, template2, Scalar(0,0,255) );
	timer->putTimes(display_image);
	Mat template2locations_gray;
	cvtColor(template2locations, template2locations_gray, CV_BGR2GRAY);
	CompareRecognitionResults( matched_template_map, template2locations_gray, precision, recall, accuracy, specificity, f1 );
	sprintf( results, "precision=%.2f", precision);
	location.x = 123;
	location.y = 213;
	putText( display_image, "Results (2)", location, FONT_HERSHEY_SIMPLEX, 0.4, colour );
	location.y += 13;
	putText( display_image, results, location, FONT_HERSHEY_SIMPLEX, 0.4, colour );
	sprintf( results, "recall=%.2f", recall);
	location.y += 13;
	putText( display_image, results, location, FONT_HERSHEY_SIMPLEX, 0.4, colour );
	sprintf( results, "accuracy=%.2f", accuracy);
	location.y += 13;
	putText( display_image, results, location, FONT_HERSHEY_SIMPLEX, 0.4, colour );
	sprintf( results, "specificity=%.2f", specificity);
	location.y += 13;
	putText( display_image, results, location, FONT_HERSHEY_SIMPLEX, 0.4, colour );
	sprintf( results, "f1=%.2f", f1);
	location.y += 13;
	putText( display_image, results, location, FONT_HERSHEY_SIMPLEX, 0.4, colour );
	Mat correlation_display1, correlation_display2;
	cvtColor(correlation_window1, correlation_display1, CV_GRAY2BGR);
	cvtColor(correlation_window2, correlation_display2, CV_GRAY2BGR);

	Mat output1 = JoinImagesVertically(template1,"Template (1)",correlation_display1,"Correlation (1)",4);
	Mat output2 = JoinImagesVertically(output1,"",matched_template_display1,"Local maxima (1)",4);
	Mat output3 = JoinImagesVertically(template2,"Template (2)",correlation_display2,"Correlation (2)",4);
	Mat output4 = JoinImagesVertically(output3,"",matched_template_display2,"Local maxima (2)",4);
	Mat output5 = JoinImagesHorizontally( full_image, "Original Image", output2, "", 4 );
	Mat output6 = JoinImagesHorizontally( output5, "", output4, "", 4 );
	Mat output7 = JoinImagesHorizontally( output6, "", display_image, "", 4 );
	imshow( "Template matching result", output7 );
	c = cvWaitKey();
	cvDestroyAllWindows();

	// Chamfer Matching
    Mat model_gray,model_edges,model_edges2;
	cvtColor(bicycle_model, model_gray, CV_BGR2GRAY);
	threshold(model_gray,model_edges,127,255,THRESH_BINARY);
	Mat current_frame;
	bicycle_video.set(CV_CAP_PROP_POS_FRAMES,400);  // Just in case the video has already been used.
	bicycle_video >> current_frame;
	bicycle_background = current_frame.clone();
	bicycle_video.set(CV_CAP_PROP_POS_FRAMES,500); 
	timer->reset();
	int count = 0;
	while (!current_frame.empty() && (count < 8))
    {
		Mat result_image = current_frame.clone();
		count++;
		Mat difference_frame, difference_gray, current_edges;
		absdiff(current_frame,bicycle_background,difference_frame);
		cvtColor(difference_frame, difference_gray, CV_BGR2GRAY);
		Canny(difference_frame, current_edges, 100, 200, 3);

		vector<vector<Point> > results;
		vector<float> costs;
		threshold(model_gray,model_edges,127,255,THRESH_BINARY);
		Mat matching_image, chamfer_image, local_minima;
		timer->ignoreTimeSinceLastRecorded();
		threshold(current_edges,current_edges,127,255,THRESH_BINARY_INV);
		distanceTransform( current_edges, chamfer_image, CV_DIST_L2 , 3);
		timer->recordTime("Chamfer Image");
		ChamferMatching( chamfer_image, model_edges, matching_image );
		timer->recordTime("Matching");
		FindLocalMinima( matching_image, local_minima, 500.0 );
		timer->recordTime("Find Minima");
		DrawMatchingTemplateRectangles( result_image, local_minima, model_edges, Scalar( 255, 0, 0 ) );
		Mat chamfer_display_image = convert_32bit_image_for_display( chamfer_image );
		Mat matching_display_image = convert_32bit_image_for_display( matching_image );
		//timer->putTimes(result_image);
		Mat current_edges_display, local_minima_display, model_edges_display, colour_matching_display_image, colour_chamfer_display_image;
		cvtColor(current_edges, current_edges_display, CV_GRAY2BGR);
		cvtColor(local_minima, local_minima_display, CV_GRAY2BGR);
		cvtColor(model_edges, model_edges_display, CV_GRAY2BGR);
		cvtColor(matching_display_image, colour_matching_display_image, CV_GRAY2BGR);
		cvtColor(chamfer_display_image, colour_chamfer_display_image, CV_GRAY2BGR);

		Mat output1 = JoinImagesVertically(current_frame,"Video Input",current_edges_display,"Edges from difference", 4);
		Mat output2 = JoinImagesVertically(output1,"",model_edges_display,"Model", 4);
		Mat output3 = JoinImagesVertically(bicycle_background,"Static Background",colour_chamfer_display_image,"Chamfer image", 4);
		Mat output4 = JoinImagesVertically(output3,"",colour_matching_display_image,"Degree of fit", 4);
		Mat output5 = JoinImagesVertically(difference_frame,"Difference",result_image,"Result", 4);
		Mat output6 = JoinImagesVertically(output5,"",local_minima_display,"Local minima", 4);
		Mat output7 = JoinImagesHorizontally( output2, "", output4, "", 4 );
		Mat output8 = JoinImagesHorizontally( output7, "", output6, "", 4 );
		imshow("Chamfer matching", output8);
		c = waitKey(1000);  // This makes the image appear on screen
		bicycle_video >> current_frame;
	}
	c = cvWaitKey();
	cvDestroyAllWindows();

	// Cascade of Haar classifiers (most often shown for face detection).
    VideoCapture camera;
	camera.open(1);
	camera.set(CV_CAP_PROP_FRAME_WIDTH, 320);
	camera.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
    if( camera.isOpened() )
	{
		timer->reset();
		Mat current_frame;
		do {
			camera >> current_frame;
			if( current_frame.empty() )
				break;
			vector<Rect> faces;
			timer->ignoreTimeSinceLastRecorded();
			Mat gray;
			cvtColor( current_frame, gray, CV_BGR2GRAY );
			equalizeHist( gray, gray );
			cascade.detectMultiScale( gray, faces, 1.1, 2, CV_HAAR_SCALE_IMAGE, Size(30, 30) );
			timer->recordTime("Haar Classifier");
			for( int count = 0; count < (int)faces.size(); count++ )
				rectangle(current_frame, faces[count], cv::Scalar(255,0,0), 2);
			//timer->putTimes(current_frame);
			imshow( "Cascade of Haar Classifiers", current_frame );
			c = waitKey(10);  // This makes the image appear on screen
        } while (c == -1);
	}
	cvDestroyAllWindows();

	// Histogram of Oriented Gradients for people detection.
	HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
	int frame_number = 990;
	people_video.set(CV_CAP_PROP_POS_FRAMES,frame_number);
	people_video >> current_frame;
	Mat bigger_frame;
	timer->reset();
	while ((!current_frame.empty()) && (frame_number++ < 1010))
    {
		Mat imageROI;
		imageROI= current_frame(cv::Rect(current_frame.cols/5,current_frame.rows/5,current_frame.cols*2/3,current_frame.rows*2/3));
		resize(imageROI,bigger_frame,Size( imageROI.cols*4, imageROI.rows*4 ));
	    vector<Rect> people;
		timer->ignoreTimeSinceLastRecorded();
	    hog.detectMultiScale(bigger_frame, people);
		timer->recordTime("Hisogram of Oriented Gradients");
	    for( int count = 0; count < (int)people.size(); count++ )
	    {
		    Rect current_person = people[count];
		    // HoG returns overly large rectangles so we shrink them for display.
		    current_person.x += current_person.width/5;
		    current_person.y += current_person.height/8;
		    current_person.height = current_person.height*4/5;
		    current_person.width = current_person.width/2;
		    rectangle(bigger_frame, current_person, cv::Scalar(255,0,0), 2);
	    }
		//timer->putTimes(bigger_frame);
	    imshow("Histogram of Oriented Gradients (for People Detection)", bigger_frame);
		c = waitKey(10);  // This makes the image appear on screen
		people_video >> current_frame;
    }
	c = cvWaitKey();
	cvDestroyAllWindows();
}
