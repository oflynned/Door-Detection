/*
 * This code is provided as part of "A Practical Introduction to Computer Vision with OpenCV"
 * by Kenneth Dawson-Howe © Wiley & Sons Inc. 2014.  All rights reserved.
 */
#include "Utilities.h"
#include "opencv2/video.hpp"


void MeanShiftDemo( VideoCapture& video, Rect& starting_position, int starting_frame_number, int end_frame)
{
	bool half_size = true;
	video.set(CV_CAP_PROP_POS_FRAMES,starting_frame_number);
	Mat current_frame, hls_image;
	std::vector<cv::Mat> hls_planes(3);
	video >> current_frame;
	Rect current_position(starting_position);
	if (half_size)
	{
		resize(current_frame, current_frame, Size( current_frame.cols/2, current_frame.rows/2 ));
		current_position.height /= 2;
		current_position.width /= 2;
		current_position.x /= 2;
		current_position.y /= 2;
	}
	cvtColor(current_frame, hls_image, CV_BGR2HLS);
	split(hls_image,hls_planes);
    int chosen_channel = 0;  // Hue channel
	Mat image1ROI = hls_planes[chosen_channel](current_position);

	float channel_range[2] = { 0.0, 255.0 };
    int channel_numbers[1] = { 0 };
	int number_bins[1] = { 32 };
	MatND histogram[1];
    const float* channel_ranges = channel_range;
	calcHist(&(image1ROI), 1, channel_numbers, Mat(), histogram[0], 1 , number_bins, &channel_ranges);
    normalize(histogram[0],histogram[0],1.0);
	rectangle(current_frame,current_position,Scalar(0,255,0),2);
	Mat starting_frame = current_frame.clone();
	int frame_number = starting_frame_number;
	while (!current_frame.empty() && (frame_number < end_frame))
    {
		// Calculate back projection
		Mat back_projection_probabilities;
        calcBackProject(&(hls_planes[chosen_channel]),1,channel_numbers,*histogram,back_projection_probabilities,&channel_ranges,255.0);
		// Remove low saturation points from consideration
		Mat saturation_mask;
        inRange( hls_image, Scalar(0,10,50,0),Scalar(180,256,256,0), saturation_mask );
		bitwise_and( back_projection_probabilities, back_projection_probabilities,back_projection_probabilities, saturation_mask );
		// Mean shift
		TermCriteria criteria(cv::TermCriteria::MAX_ITER,5,0.01);
		meanShift(back_projection_probabilities,current_position,criteria);
		// Output to screen
		rectangle(current_frame,current_position,Scalar(0,255,0),2);
		Mat chosen_channel_image, back_projection_image;
		cvtColor(hls_planes[chosen_channel], chosen_channel_image, CV_GRAY2BGR);
		cvtColor(back_projection_probabilities, back_projection_image, CV_GRAY2BGR);
		Mat row1_output = JoinImagesHorizontally( starting_frame, "Starting position", chosen_channel_image, "Chosen channel (Hue)", 4 );
		Mat row2_output = JoinImagesHorizontally( back_projection_image, "Back projection", current_frame, "Current position", 4 );
		Mat mean_shift_output = JoinImagesVertically(row1_output,"",row2_output,"", 4);
        imshow("Mean Shift Tracking", mean_shift_output );
		// Advance to next frame
		video >> current_frame;
		if (half_size)
			resize(current_frame, current_frame, Size( current_frame.cols/2, current_frame.rows/2 ));
		cvtColor(current_frame, hls_image, CV_BGR2HLS);
		split(hls_image,hls_planes);
		frame_number++;
	    cvWaitKey(1000);
	}
	char c = cvWaitKey();
    cvDestroyAllWindows();
}

void drawOpticalFlow(Mat& optical_flow, Mat& display, int spacing, Scalar passed_line_colour=-1.0, Scalar passed_point_colour=-1.0)
{
	Scalar colour( rand()&0xFF, rand()&0xFF, rand()&0xFF );
    for (int row = spacing/2; row < display.rows; row += spacing)
        for(int column = spacing/2; column < display.cols; column += spacing)
        {
            const Point2f& fxy = optical_flow.at<Point2f>(row,column);
            circle(display, Point(column,row), 1, (passed_point_colour.val[0] == -1.0) ? colour : passed_point_colour, -1);
            line(display, Point(column,row), Point(cvRound(column+fxy.x), cvRound(row+fxy.y)),
                 (passed_line_colour.val[0] == -1.0) ? colour : passed_line_colour);
        }
}

#define MAX_FEATURES 400
void LucasKanadeOpticalFlow(Mat& previous_gray_frame, Mat& gray_frame, Mat& display_image)
{
	Size img_sz = previous_gray_frame.size();
	int win_size = 10;
	cvtColor(previous_gray_frame, display_image, CV_GRAY2BGR);
	vector<Point2f> previous_features, current_features;
	const int MAX_CORNERS = 500;
	goodFeaturesToTrack(previous_gray_frame, previous_features, MAX_CORNERS, 0.05, 5, noArray(), 3, false, 0.04);
	cornerSubPix(previous_gray_frame, previous_features, Size(win_size, win_size), Size(-1,-1),
                 TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.03));
    vector<uchar> features_found;
	calcOpticalFlowPyrLK(previous_gray_frame, gray_frame, previous_features, current_features, features_found, noArray(),
                         Size(win_size*4+1,win_size*4+1), 5,
                         TermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, .3 ));
    for( int i = 0; i < (int)previous_features.size(); i++ )
	{
		if( !features_found[i] )
			continue;
        circle(display_image, previous_features[i], 1, Scalar(0,0,255));
		line(display_image, previous_features[i], current_features[i], Scalar(0,255,0));   
	}
}

class MedianBackground
{
private:
	Mat mMedianBackground;
	float**** mHistogram;
	float*** mLessThanMedian;
	float mAgingRate;
	float mCurrentAge;
	float mTotalAges;
	int mValuesPerBin;
	int mNumberOfBins;
public:
	MedianBackground( Mat initial_image, float aging_rate, int values_per_bin );
	Mat GetBackgroundImage();
	void UpdateBackground( Mat current_frame );
	float getAgingRate()
	{
		return mAgingRate;
	}
};

MedianBackground::MedianBackground( Mat initial_image, float aging_rate, int values_per_bin )
{
	mCurrentAge = 1.0;
	mAgingRate = aging_rate;
	mTotalAges = 0.0;
	mValuesPerBin = values_per_bin;
	mNumberOfBins = 256/mValuesPerBin;
	mMedianBackground = Mat::zeros(initial_image.size(), initial_image.type());
	mLessThanMedian = (float***) new float**[mMedianBackground.rows];
	mHistogram = (float****) new float***[mMedianBackground.rows];
	for (int row=0; (row<mMedianBackground.rows); row++)
	{
		mHistogram[row] = (float***) new float**[mMedianBackground.cols];
		mLessThanMedian[row] = (float**) new float*[mMedianBackground.cols];
		for (int col=0; (col<mMedianBackground.cols); col++)
		{
			mHistogram[row][col] = (float**) new float*[mMedianBackground.channels()];
			mLessThanMedian[row][col] = new float[mMedianBackground.channels()];
			for (int ch=0; (ch<mMedianBackground.channels()); ch++)
			{
				mHistogram[row][col][ch] = new float[mNumberOfBins];
				mLessThanMedian[row][col][ch] = 0.0;
				for (int bin=0; (bin<mNumberOfBins); bin++)
				{
					mHistogram[row][col][ch][bin] = (float) 0.0;
				}
			}
		}
	}
}

Mat MedianBackground::GetBackgroundImage()
{
	return mMedianBackground;
}

void MedianBackground::UpdateBackground( Mat current_frame )
{
	mTotalAges += mCurrentAge;
	float total_divided_by_2 = mTotalAges/((float) 2.0);
	for (int row=0; (row<mMedianBackground.rows); row++)
	{
		for (int col=0; (col<mMedianBackground.cols); col++)
		{
			for (int ch=0; (ch<mMedianBackground.channels()); ch++)
			{
				int new_value = (mMedianBackground.channels()==3) ? current_frame.at<Vec3b>(row,col)[ch] : current_frame.at<uchar>(row,col);
				int median = (mMedianBackground.channels()==3) ? mMedianBackground.at<Vec3b>(row,col)[ch] : mMedianBackground.at<uchar>(row,col);
				int bin = new_value/mValuesPerBin;
				mHistogram[row][col][ch][bin] += mCurrentAge;
				if (new_value < median)
					mLessThanMedian[row][col][ch] += mCurrentAge;
				int median_bin = median/mValuesPerBin;
				while ((mLessThanMedian[row][col][ch] + mHistogram[row][col][ch][median_bin] < total_divided_by_2) && (median_bin < 255))
				{
					mLessThanMedian[row][col][ch] += mHistogram[row][col][ch][median_bin];
					median_bin++;
				}
				while ((mLessThanMedian[row][col][ch] > total_divided_by_2) && (median_bin > 0))
				{
					median_bin--;
					mLessThanMedian[row][col][ch] -= mHistogram[row][col][ch][median_bin];
				}
				if (mMedianBackground.channels()==3)
					mMedianBackground.at<Vec3b>(row,col)[ch] = median_bin*mValuesPerBin;
				else mMedianBackground.at<uchar>(row,col) = median_bin*mValuesPerBin;
			}
		}
	}
	mCurrentAge *= mAgingRate;
}

void medianBackgroundDemo(VideoCapture& video, int starting_frame, Rect groundTruth, bool showGT) {
	Mat current_frame, first_frame;
	double learning_rate = 0.01;

	video.set(CV_CAP_PROP_POS_FRAMES, starting_frame);
	video >> current_frame;

	first_frame = current_frame.clone();

	MedianBackground median_background(current_frame, (float) 1.002, 1);
	MedianBackground median_background_2(current_frame, (float) 1.001, 1);

	Mat median_bg_img, median_fg_img;
	Mat median_bg_img_2, median_fg_img_2;
	Mat median_fg_difference;

	double frame_rate = video.get(CV_CAP_PROP_FPS);
	double time_between_frames = 1000.0 / frame_rate;
	Timestamper* timer = new Timestamper();
	int frame_count = 0;

	bool isFirstOpening = false, isSecondOpening = false;
	int frameFirstOpening, frameSecondOpening;
	int totalChange = 0, avgChange = 0;

	//train
	for (int i = 0; i < 100; i++) {
		median_background.UpdateBackground(current_frame);
		median_background_2.UpdateBackground(current_frame);
	}

	while ((!current_frame.empty()) && (frame_rate++ < video.get(CV_CAP_PROP_FRAME_COUNT))) {
		// fast median
		double duration = static_cast<double>(getTickCount());

		// slow median
		timer->ignoreTimeSinceLastRecorded();
		median_background_2.UpdateBackground(current_frame);
		timer->recordTime("median2");
		median_bg_img_2 = median_background.GetBackgroundImage();

		Mat median_difference_2;
		absdiff(median_bg_img_2, current_frame, median_difference_2);
		cvtColor(median_difference_2, median_difference_2, CV_BGR2GRAY);
		threshold(median_difference_2, median_difference_2, 30, 255, THRESH_BINARY);
		median_fg_img_2.setTo(Scalar(0, 0, 0));
		current_frame.copyTo(median_fg_img_2, median_difference_2);

		// inter-frame times
		duration = static_cast<double>(getTickCount()) - duration;
		duration /= getTickFrequency() / 1000.0;
		int delay = (time_between_frames > duration) ? ((int)(time_between_frames - duration)) : 1;
		char c = cvWaitKey(delay);

		// output to screen
		char frame_str[100];
		sprintf(frame_str, "Frame %d", frame_count);

		if(showGT)
			cv::rectangle(current_frame, groundTruth, Scalar(0, 0, 255));

		Mat temp_median_output_2 = JoinImagesHorizontally(current_frame, frame_str, median_bg_img_2, "Median Background 2", 4);
		Mat median_output_2 = JoinImagesHorizontally(temp_median_output_2, "", median_fg_img_2, "Foreground 2", 4);
		imshow("Median Background Model 2", median_output_2);

		threshold(median_difference_2, median_difference_2, 20, 255, THRESH_BINARY);
		imshow("Thresholded", median_difference_2);

		totalChange += countNonZero(median_difference_2);
		avgChange = totalChange / (frame_count / 10);

		cout << totalChange << endl;
		cout << avgChange << endl;

		video >> current_frame;
		frame_count += 10;
	}

	cvDestroyAllWindows();
	delete timer;
}

void VideoDemos( VideoCapture& surveillance_video, int starting_frame, bool clean_binary_images )
{
	Mat previous_gray_frame, optical_flow, optical_flow_display;
	Mat current_frame, thresholded_image, closed_image, first_frame;
	Mat current_frame_gray, running_average_background;
	Mat temp_running_average_background, running_average_difference;
	Mat running_average_foreground_mask, running_average_foreground_image;
	Mat selective_running_average_background;
	Mat temp_selective_running_average_background, selective_running_average_difference;
	Mat selective_running_average_foreground_mask, selective_running_average_background_mask, selective_running_average_foreground_image;
	double running_average_learning_rate = 0.01;
	surveillance_video.set(CV_CAP_PROP_POS_FRAMES,starting_frame);
	surveillance_video >> current_frame;
	first_frame = current_frame.clone();
	cvtColor(current_frame, current_frame_gray, CV_BGR2GRAY);
	current_frame.convertTo(running_average_background, CV_32F);
	selective_running_average_background = running_average_background.clone();
	int rad = running_average_background.depth();
	MedianBackground median_background( current_frame, (float) 1.005, 1 );
	Mat median_background_image, median_foreground_image;

	int codec = static_cast<int>(surveillance_video.get(CV_CAP_PROP_FOURCC));
	// V3.0.0 update on next line.  OLD CODE was    BackgroundSubtractorMOG2 gmm; //(50,16,true);
    Ptr<BackgroundSubtractorMOG2> gmm = createBackgroundSubtractorMOG2();
	Mat foreground_mask, foreground_image = Mat::zeros(current_frame.size(), CV_8UC3);

	double frame_rate = surveillance_video.get(CV_CAP_PROP_FPS);
	double time_between_frames = 1000.0/frame_rate;
	Timestamper* timer = new Timestamper();
	int frame_count = 0;
	while ((!current_frame.empty()) && (frame_count++ < 1000))//1800))
    {
 		double duration = static_cast<double>(getTickCount());
		vector<Mat> input_planes(3);
		split(current_frame,input_planes);
		cvtColor(current_frame, current_frame_gray, CV_BGR2GRAY);
/*
// The following code has been dropped as Lucas Kanade Optical Flow does not seem to work with VS2015 & OpenCV3.1.0.  Not sure why 8-9-16.
		if (frame_count%2 == 0)  // Skip every second frame so the flow is greater.
		{
			if ( previous_gray_frame.data )
			{
				Mat lucas_kanade_flow;
				timer->ignoreTimeSinceLastRecorded();
				LucasKanadeOpticalFlow(previous_gray_frame, current_frame_gray, lucas_kanade_flow);
				timer->recordTime("Lucas Kanade Optical Flow");
				calcOpticalFlowFarneback(previous_gray_frame, current_frame_gray, optical_flow, 0.5, 3, 15, 3, 5, 1.2, 0);
				cvtColor(previous_gray_frame, optical_flow_display, CV_GRAY2BGR);
				drawOpticalFlow(optical_flow, optical_flow_display, 8, Scalar(0, 255, 0), Scalar(0, 0, 255));
				timer->recordTime("Farneback Optical Flow");
				char frame_str[100];
				sprintf( frame_str, "Frame = %d", frame_count);
 				Mat temp_output = JoinImagesHorizontally( current_frame, frame_str, optical_flow_display, "Farneback Optical Flow", 4 );
				Mat optical_flow_output = JoinImagesHorizontally( temp_output, "", lucas_kanade_flow, "Lucas Kanade Optical Flow", 4 );
				imshow("Optical Flow", optical_flow_output );
			}
			std::swap(previous_gray_frame, current_frame_gray);
		}
	*/
		// Static background image
		Mat difference_frame, binary_difference;
		Mat structuring_element(3,3,CV_8U,Scalar(1));
		timer->ignoreTimeSinceLastRecorded();
		absdiff(current_frame,first_frame,difference_frame);
		cvtColor(difference_frame, thresholded_image, CV_BGR2GRAY);
		threshold(thresholded_image,thresholded_image,30,255,THRESH_BINARY);
		if (clean_binary_images)
		{
			morphologyEx(thresholded_image,closed_image,MORPH_CLOSE,structuring_element);
			morphologyEx(closed_image,binary_difference,MORPH_OPEN,structuring_element);
			current_frame.copyTo(binary_difference, thresholded_image);
		}
		else
		{
			binary_difference.setTo(Scalar(0,0,0));
		    current_frame.copyTo(binary_difference, thresholded_image);
		}
		timer->recordTime("Static difference");

		// Running Average (three channel version)
		vector<Mat> running_average_planes(3);
		split(running_average_background,running_average_planes);
		accumulateWeighted(input_planes[0], running_average_planes[0], running_average_learning_rate);
		accumulateWeighted(input_planes[1], running_average_planes[1], running_average_learning_rate);
		accumulateWeighted(input_planes[2], running_average_planes[2], running_average_learning_rate);
		merge(running_average_planes,running_average_background);
		running_average_background.convertTo(temp_running_average_background,CV_8U);
		absdiff(temp_running_average_background,current_frame,running_average_difference);
		split(running_average_difference,running_average_planes);
		// Determine foreground points as any point with a difference of more than 30 on any one channel:
		threshold(running_average_difference,running_average_foreground_mask,30,255,THRESH_BINARY);
		split(running_average_foreground_mask,running_average_planes);
		bitwise_or( running_average_planes[0], running_average_planes[1], running_average_foreground_mask );
		bitwise_or( running_average_planes[2], running_average_foreground_mask, running_average_foreground_mask );
		if (clean_binary_images)
		{
			morphologyEx(running_average_foreground_mask,closed_image,MORPH_CLOSE,structuring_element);
			morphologyEx(closed_image,running_average_foreground_mask,MORPH_OPEN,structuring_element);
		}
		running_average_foreground_image.setTo(Scalar(0,0,0));
	    current_frame.copyTo(running_average_foreground_image, running_average_foreground_mask);
		timer->recordTime("Running Average");

		// Running Average with selective update
		vector<Mat> selective_running_average_planes(3);
		// Find Foreground mask
		selective_running_average_background.convertTo(temp_selective_running_average_background,CV_8U);
		absdiff(temp_selective_running_average_background,current_frame,selective_running_average_difference);
		split(selective_running_average_difference,selective_running_average_planes);
		// Determine foreground points as any point with an average difference of more than 30 over all channels:
		Mat temp_sum = (selective_running_average_planes[0]/3 + selective_running_average_planes[1]/3 + selective_running_average_planes[2]/3);
		threshold(temp_sum,selective_running_average_foreground_mask,30,255,THRESH_BINARY_INV);
		// Update background
		split(selective_running_average_background,selective_running_average_planes);
		accumulateWeighted(input_planes[0], selective_running_average_planes[0], running_average_learning_rate,selective_running_average_foreground_mask);
		accumulateWeighted(input_planes[1], selective_running_average_planes[1], running_average_learning_rate,selective_running_average_foreground_mask);
		accumulateWeighted(input_planes[2], selective_running_average_planes[2], running_average_learning_rate,selective_running_average_foreground_mask);
    	invertImage(selective_running_average_foreground_mask,selective_running_average_foreground_mask);
		accumulateWeighted(input_planes[0], selective_running_average_planes[0], running_average_learning_rate/3.0,selective_running_average_foreground_mask);
		accumulateWeighted(input_planes[1], selective_running_average_planes[1], running_average_learning_rate/3.0,selective_running_average_foreground_mask);
		accumulateWeighted(input_planes[2], selective_running_average_planes[2], running_average_learning_rate/3.0,selective_running_average_foreground_mask);
		merge(selective_running_average_planes,selective_running_average_background);
		if (clean_binary_images)
		{
			morphologyEx(selective_running_average_foreground_mask,closed_image,MORPH_CLOSE,structuring_element);
			morphologyEx(closed_image,selective_running_average_foreground_mask,MORPH_OPEN,structuring_element);
		}
 		selective_running_average_foreground_image.setTo(Scalar(0,0,0));
	    current_frame.copyTo(selective_running_average_foreground_image, selective_running_average_foreground_mask);
		timer->recordTime("Selective Running Average");

		// Median background
		timer->ignoreTimeSinceLastRecorded();
		median_background.UpdateBackground( current_frame );
		timer->recordTime("Median");
		median_background_image = median_background.GetBackgroundImage();
		Mat median_difference;
		absdiff(median_background_image,current_frame,median_difference);
		cvtColor(median_difference, median_difference, CV_BGR2GRAY);
		threshold(median_difference,median_difference,30,255,THRESH_BINARY);
		median_foreground_image.setTo(Scalar(0,0,0));
	    current_frame.copyTo(median_foreground_image, median_difference);

		// Update the Gaussian Mixture Model
 		// V3.0.0 update on next line.  OLD CODE was  gmm(current_frame, foreground_mask);
        gmm->apply(current_frame, foreground_mask);
		// Clean the resultant binary (moving pixel) mask using an opening.
		threshold(foreground_mask,thresholded_image,150,255,THRESH_BINARY);
		Mat moving_incl_shadows, shadow_points;
		threshold(foreground_mask,moving_incl_shadows,50,255,THRESH_BINARY);
		absdiff( thresholded_image, moving_incl_shadows, shadow_points );
		Mat cleaned_foreground_mask;
		if (clean_binary_images)
		{
			morphologyEx(thresholded_image,closed_image,MORPH_CLOSE,structuring_element);
			morphologyEx(closed_image,cleaned_foreground_mask,MORPH_OPEN,structuring_element);
		}
		else cleaned_foreground_mask = thresholded_image.clone();
 		foreground_image.setTo(Scalar(0,0,0));
        current_frame.copyTo(foreground_image, cleaned_foreground_mask);
		timer->recordTime("Gaussian Mixture Model");
		// Create an average background image (just for information)
        Mat mean_background_image;
		timer->ignoreTimeSinceLastRecorded();
		// V3.0.0 update on next line.  OLD CODE was   gmm.getBackgroundImage(mean_background_image);
        gmm->getBackgroundImage(mean_background_image);

		duration = static_cast<double>(getTickCount())-duration;
		duration /= getTickFrequency()/1000.0;
		int delay = (time_between_frames>duration) ? ((int) (time_between_frames-duration)) : 1;
		char c = cvWaitKey(delay);
		
		char frame_str[100];
		sprintf( frame_str, "Frame = %d", frame_count);
		Mat temp_static_output = JoinImagesHorizontally( current_frame, frame_str, first_frame, "Static Background", 4 );
		Mat static_output = JoinImagesHorizontally( temp_static_output, "", binary_difference, "Foreground", 4 );
        imshow("Static Background Model", static_output );
 		/*
		Mat temp_running_output = JoinImagesHorizontally( current_frame, frame_str, temp_running_average_background, "Running Average Background", 4 );
		Mat running_output = JoinImagesHorizontally( temp_running_output, "", running_average_foreground_image, "Foreground", 4 );
		imshow("Running Average Background Model", running_output );
 		Mat temp_selective_output = JoinImagesHorizontally( current_frame, frame_str, temp_selective_running_average_background, "Selective Running Average Background", 4 );
		Mat selective_output = JoinImagesHorizontally( temp_selective_output, "", selective_running_average_foreground_image, "Foreground", 4 );
        imshow("Selective Running Average Background Model", selective_output );
 		Mat temp_median_output = JoinImagesHorizontally( current_frame, frame_str, median_background_image, "Median Background", 4 );
		Mat median_output = JoinImagesHorizontally( temp_median_output, "", median_foreground_image, "Foreground", 4 );
        imshow("Median Background Model", median_output );
  		Mat temp_gaussian_output = JoinImagesHorizontally( current_frame, frame_str, mean_background_image, "GMM Background", 4 );
		Mat gaussian_output = JoinImagesHorizontally( temp_gaussian_output, "", foreground_image, "Foreground", 4 );
        imshow("Gaussian Mixture Model", gaussian_output );
		*/
		timer->putTimes( current_frame );
		imshow( "Computation Times", current_frame );
	 	surveillance_video >> current_frame;
	}
	cvDestroyAllWindows();
}
