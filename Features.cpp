/*
 * This code is provided as part of "A Practical Introduction to Computer Vision with OpenCV"
 * by Kenneth Dawson-Howe © Wiley & Sons Inc. 2014.  All rights reserved.
 */
#include "Utilities.h"
#include "opencv2/video.hpp"
#include "opencv2/features2d.hpp"
//#include "opencv2/xfeatures2d.hpp"

void FeaturesDemos( Mat& image1, Mat& image2, Mat& image3 )
{
	Timestamper* timer = new Timestamper();
	Mat image2_ROI, image3_ROI;
	image2_ROI = image2(cv::Rect(155,120,195,90));
	image3_ROI = image3(cv::Rect(155,120,195,90));
	Mat image1_gray, image2_gray, image3_gray;
	cvtColor(image1, image1_gray, CV_BGR2GRAY);
	cvtColor(image2_ROI, image2_gray, CV_BGR2GRAY);
	cvtColor(image3_ROI, image3_gray, CV_BGR2GRAY);


	Mat harris_cornerness,possible_harris_corners,harris_corners;
	cornerHarris(image1_gray,harris_cornerness,3,3,0.02);
	// V3.0.0 change
	Ptr<FeatureDetector> harris_feature_detector = GFTTDetector::create( 1000, 0.01, 10, 3, true );
	//GoodFeaturesToTrackDetector harris_detector( 1000, 0.01, 10, 3, true );
	vector<KeyPoint> keypoints;
	timer->reset();
	// V3.0.0 change
	harris_feature_detector->detect( image1_gray, keypoints );
	//harris_detector.detect( image1_gray,keypoints );
	timer->recordTime("Harris");
	Mat harris_corners_image;
	drawKeypoints( image1, keypoints, harris_corners_image, Scalar( 0, 0, 255 ) );
	Mat harris_cornerness_display_image = convert_32bit_image_for_display(harris_cornerness);
	Mat original_image, harris_cornerness_image;
	cvtColor(image1_gray, original_image, CV_GRAY2BGR);
	cvtColor(harris_cornerness_display_image, harris_cornerness_image, CV_GRAY2BGR);

	vector<KeyPoint> keypoints2;
	// V3.0.0 change
	Ptr<FeatureDetector> min_eigen_feature_detector = GFTTDetector::create( 1000, 0.01, 10, 3, false );
	//GoodFeaturesToTrackDetector min_eigen_detector( 1000, 0.01, 10, 3, false );
	timer->ignoreTimeSinceLastRecorded();
	// V3.0.0 change
	min_eigen_feature_detector->detect( image1_gray, keypoints2 );
	//min_eigen_detector.detect(image1_gray,keypoints2);
	timer->recordTime("Min Eigen values");
	Mat good_features_image;
	drawKeypoints( image1, keypoints2, good_features_image, Scalar( 0, 0, 255 ) );

	// V3.0.0 change
	Ptr<FeatureDetector> feature_detector = ORB::create();
	Mat fast_corners_image;
	//Ptr<FeatureDetector> feature_detector = FeatureDetector::create("FAST");
/*
// The following code has been dropped as FAST does not seem to work with VS2015 & OpenCV3.1.0.  Not sure why 8-9-16.
	vector<KeyPoint> FAST_keypoints;
	timer->ignoreTimeSinceLastRecorded();
	feature_detector->detect( image1_gray, FAST_keypoints );
	timer->recordTime("FAST");
	drawKeypoints( image1, FAST_keypoints, fast_corners_image, Scalar( 0, 0, 255 ) );
*/	
	/*
	// The following code is not supported in the main release of OpenCV 3.0.0.  A special version of the
	// software is needed containing updated code using opencv_contrib.
	vector<KeyPoint> SIFT_keypoints;
	// V3.0.0 change
	Ptr<Feature2D> sift_feature_detector = xfeatures2d::SIFT::create();
	//feature_detector = FeatureDetector::create("SIFT");
	timer->ignoreTimeSinceLastRecorded();
	sift_feature_detector->detect( image1_gray, SIFT_keypoints );
	timer->recordTime("SIFT");
	Mat sift_features_image;
	drawKeypoints( image1, SIFT_keypoints, sift_features_image, Scalar( 0, 0, 255 ) );
	
	vector<KeyPoint> SURF_keypoints;
	feature_detector = FeatureDetector::create("SURF");
	timer->ignoreTimeSinceLastRecorded();
	feature_detector->detect( image1_gray, SURF_keypoints );
	timer->recordTime("SURF");
	Mat surf_features_image;
	drawKeypoints( image1, SURF_keypoints, surf_features_image, Scalar( 0, 0, 255 ) );
*/
	Mat output1 = JoinImagesHorizontally( original_image, "Original (for processing)", harris_cornerness_image, "Harris Cornerness", 4 );
	Mat output2 = JoinImagesHorizontally( output1, "", harris_corners_image, "Harris Corners", 4 );
	output1 = JoinImagesHorizontally( output2, "", good_features_image, "Min Eigen Values Features", 4 );
/*
// The following code has been dropped as FAST does not seem to work with VS2015 & OpenCV3.1.0.  Not sure why 8-9-16.
	Mat output3 = JoinImagesHorizontally( fast_corners_image, "FAST Corners", sift_features_image, "SIFT Features", 4 );
	Mat output4 = JoinImagesHorizontally( output3, "", surf_features_image, "SURF Features", 4 );
	Mat times_image = image1.clone();
	timer->putTimes(times_image);
	output3 = JoinImagesHorizontally( output4, "", times_image, "", 4 );
	Mat features_output = JoinImagesVertically(output1,"",output3,"", 4);
*/
	Mat features_output = JoinImagesVertically(output1,"",fast_corners_image,"FAST Corners", 4);
	imshow("Features", features_output );
	char c = cvWaitKey();
/*
// The following code is not supported in the main release of OpenCV 3.0.0.  A special version of the
// software is needed containing updated code using opencv_contrib.
	// Find SURF features
	SurfFeatureDetector detector( 400 );
	vector<KeyPoint> keypoints_2, keypoints_3;
	detector.detect( image2_gray, keypoints_2 );
	detector.detect( image3_gray, keypoints_3 );
	// Extract feature descriptors
	SurfDescriptorExtractor extractor;
	Mat descriptors_2, descriptors_3;
	extractor.compute( image2_gray, keypoints_2, descriptors_2 );
	extractor.compute( image3_gray, keypoints_3, descriptors_3 );
	// Match descriptors.
	BFMatcher matcher(NORM_L2);
	vector< DMatch > matches;
	matcher.match( descriptors_2, descriptors_3, matches );
	// Display SURF matches
	Mat img_matches;
	drawMatches( image2_gray, keypoints_2, image3_gray, keypoints_3, matches, img_matches );
	
	// Find SIFT features
	SiftFeatureDetector sift_detector;
	sift_detector.detect( image2_gray, keypoints_2 );
	sift_detector.detect( image3_gray, keypoints_3 );
	// Extract feature descriptors
	SiftDescriptorExtractor sift_extractor;
	sift_extractor.compute( image2_gray, keypoints_2, descriptors_2 );
	sift_extractor.compute( image3_gray, keypoints_3, descriptors_3 );
	// Match descriptors.
	BFMatcher sift_matcher(NORM_L2);
	vector< DMatch > matches2;
	sift_matcher.match( descriptors_2, descriptors_3, matches2 );
	// Display SIFT matches
	Mat sift_matches;
	drawMatches( image2_gray, keypoints_2, image3_gray, keypoints_3, matches2, sift_matches );
	output1 = JoinImagesHorizontally( image2_ROI, "First image", image3_ROI, "Second image" );
	output2 = JoinImagesVertically(output1,"",sift_matches,"SIFT Matches", 4);
	Mat feature_matching_output = JoinImagesVertically(output2,"",img_matches,"SURF Matches", 4);
	imshow("Feature matching", feature_matching_output );
	*/
	c = cvWaitKey();
    cvDestroyAllWindows();
}

bool isFeatureMatchedOnlyOnce( vector< DMatch > matches, int query_index, int train_index )
{
	int query_count = 0, train_count = 0;
	for (vector<DMatch>::const_iterator current_match = matches.begin();
		    (current_match != matches.end()); current_match++)
	{
		if ((*current_match).queryIdx == query_index)
			query_count++;
		if ((*current_match).trainIdx == train_index)
			train_count++;
	}
	return ((train_count == 1) && (query_count == 1));
}

void drawMatchesInConsistentColours( vector< DMatch > matches, vector<KeyPoint> current_keypoints, vector<Scalar> &current_colours, vector<int> &current_counts, vector<KeyPoint> previous_keypoints, vector<Scalar> previous_colours, vector<int> previous_counts, Mat& result_image )
{
	// Assign random colours to all features and initialise the count of frames to 1.
	current_colours.clear();
	current_counts.clear();
	for (int index=0; index < (int) current_keypoints.size(); index++)
	{
		current_colours.push_back(Scalar(rand()&0xFF, rand()&0xFF, rand()&0xFF));
		current_counts.push_back(1);
	}
	for (vector<DMatch>::const_iterator current_match = matches.begin();
		    (current_match != matches.end()); current_match++)
		if (isFeatureMatchedOnlyOnce( matches, (*current_match).queryIdx, (*current_match).trainIdx ))
		{   // Only consider features which are matched once between the frames.
			// Update the colour and the count of frames for the matched feature.
			current_colours[(*current_match).queryIdx] = previous_colours[(*current_match).trainIdx];
			current_counts[(*current_match).queryIdx] = previous_counts[(*current_match).trainIdx]+1;
			Point current_location;
			current_location.x = (int) (current_keypoints.at( (*current_match).queryIdx ).pt.x);
			current_location.y = (int) (current_keypoints.at( (*current_match).queryIdx ).pt.y);
			// Draw a coloured circle for each feature where the thickness of the line is indicative of
			// how many frames the feature has been matched.
			circle( result_image, current_location, 5, current_colours.at( (*current_match).queryIdx ),current_counts[(*current_match).queryIdx]/2);
		}
}

void TrackFeaturesDemo( VideoCapture& video, int starting_frame_number, int ending_frame_number )
{
	/*
// The following code is not supported in the main release of OpenCV 3.0.0.  A special version of the
// software will be released containing updated code using opencv_contrib will be released shortly.
	video.set(CV_CAP_PROP_POS_FRAMES,starting_frame_number);
	Mat current_frame, current_frame_gray;
	vector<KeyPoint> current_keypoints, previous_keypoints;
	vector<Scalar> current_colours, previous_colours;
	vector<int> current_counts, previous_counts;
	Mat current_descriptors, previous_descriptors;
	video >> current_frame;
	int frame_number = starting_frame_number;
	while ((!current_frame.empty()) && (frame_number++ < ending_frame_number))
	{
		vector< DMatch > matches;
		cvtColor(current_frame, current_frame_gray, CV_BGR2GRAY);
		// Find SIFT features
		SiftFeatureDetector sift_detector;
		sift_detector.detect( current_frame_gray, current_keypoints );
		SiftDescriptorExtractor sift_extractor;
		sift_extractor.compute( current_frame_gray, current_keypoints, current_descriptors );
		// Match descriptors.
		BFMatcher sift_matcher(NORM_L2);
		if (!previous_descriptors.empty())
			sift_matcher.match( current_descriptors, previous_descriptors, matches );
		// Display SIFT matches
		Mat sift_matches_image = current_frame.clone();
		drawMatchesInConsistentColours( matches, current_keypoints, current_colours, current_counts, previous_keypoints, previous_colours, previous_counts, sift_matches_image );
		imshow( "Tracking SIFT features", sift_matches_image );
		char c = cvWaitKey(10);
		previous_keypoints = current_keypoints;
		current_keypoints.clear();
		previous_colours = current_colours;
		current_colours.clear();
		previous_counts = current_counts;
		current_counts.clear();
		previous_descriptors = current_descriptors.clone();
		video >> current_frame;
	}
	char c = cvWaitKey();
    cvDestroyAllWindows();
	*/
}
