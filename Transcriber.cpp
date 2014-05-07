//	Transcriber.cpp

//	Transcriber class at the top.
//	main() at the bottom.

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <stdio.h>
#include <cmath>
#include <sstream>
#include <string>
#include <utility>

#include <iostream>

#define NUM_DOWNSAMPLES 5
#define UPMAP(point) (point * (32))
#define NUM_BINS 512

#define CANNY_LOW_THRESH 50
#define CANNY_HIGH_THRESH 150
#define HOUGH_THRESH 20
#define HOUGH_MIN_LEN 10
#define HOUGH_MAX_GAP 20
#define RAD2DEG(a) (a*180.0/CV_PI)

#define SKIN_HUE_LOW 0
#define SKIN_SAT_LOW 0
#define SKIN_VAL_LOW 128
#define SKIN_HUE_HIGH 20
#define SKIN_SAT_HIGH 120
#define SKIN_VAL_HIGH 255

#define CHOP_SLACK 0.075

struct LineUpComparer {
	bool operator() (cv::Vec4i a, cv::Vec4i b) {
		int y1a = a[1] < a[3] ? a[1] : a[3];
		int y1b = b[1] < b[3] ? b[1] : b[3];
		return y1a < y1b;
	}
} pt_higher;

struct LineDownComparer {
	bool operator() (cv::Vec4i a, cv::Vec4i b) {
		int y1a = a[1] < a[3] ? a[3] : a[1];
		int y1b = b[1] < b[3] ? b[3] : b[1];
		return y1a > y1b;
	}
} pt_lower;

struct ContourAreaComparer {
	bool operator() (cv::vector<cv::Point> contour1,
					 cv::vector<cv::Point> contour2) {
		return cv::contourArea(contour1) > cv::contourArea(contour2);
	}
} cont_comp;

class Transcriber {
	private:
	const std::string video_filename;
	const unsigned num_strings;
	const std::string window_name;
	const std::string alt_window_name;

	//	A vector of (presumably) 4-6 lines/strings, that is, their fretting.
	std::vector<std::string> output;

	cv::Mat first_frame;
	cv::Mat guitar_template;

	bool drawing_box;
	cv::Rect guitar_bbox;


	void get_guitar_box();
	static void guitar_roi_callback(int event, int x, int y, int d,
							 		void *param);

	void downsample(const cv::Mat& frame, cv::Mat& dst, int num_times) const;
	cv::Point get_match_point(const cv::Mat& frame) const;

	void draw_histogram(const std::vector<cv::Vec2f> lines) const;
	void find_two_max_bins(std::vector< std::vector<cv::Vec4i> > bins,
						   int& max_size1, int& max_size2, int& max_bin1,
						   int& max_bin2) const;

	float get_r(cv::Vec4i points) const;
	float get_theta(cv::Vec4i points) const;
	void draw_rtheta_line(cv::Vec2f line, cv::Mat& frame) const;
	void draw_twopt_line(cv::Vec4i points, cv::Mat& frame) const;

	void find_four_closest(std::vector<cv::Point>& pts) const;

	public:
	Transcriber(const char *filename, const unsigned num_strings);
	~Transcriber();

	void generate_tabs();
	void print_output() const;
};

Transcriber::Transcriber(const char *filename, const unsigned strings)
						:	video_filename(filename),
							num_strings(strings),
							window_name("Frame"),
							alt_window_name("Frame2") {
	int i;
	for (i = 0; i < num_strings; ++i) {
		output.push_back("");
	}
	cv::namedWindow(window_name);
}

Transcriber::~Transcriber() {
	cv::destroyAllWindows();
}

void Transcriber::print_output() const {
	for (int i = 0; i < num_strings; ++i) {
		std::cout << output[i] << '\n';
	}
}

void Transcriber::guitar_roi_callback(int event, int x, int y, int d,
									  void *param) {
	
	Transcriber *t = (Transcriber *) param;

	switch (event) {
		case cv::EVENT_LBUTTONDOWN:
			t -> drawing_box = true;
			t -> guitar_bbox = cv::Rect(x, y, 0, 0);
			break;
		case cv::EVENT_MOUSEMOVE:
			if (t -> drawing_box) {
				t -> guitar_bbox.width = x - t -> guitar_bbox.x;
				t -> guitar_bbox.height = y - t -> guitar_bbox.y;
			}
			break;
		case cv::EVENT_LBUTTONUP:
			t -> drawing_box = false;
			if( t -> guitar_bbox.width < 0 ){
				t -> guitar_bbox.x += t -> guitar_bbox.width;
				t -> guitar_bbox.width *= -1;
			}
			if( t -> guitar_bbox.height < 0 ){
				t -> guitar_bbox.y += t -> guitar_bbox.height;
				t -> guitar_bbox.height *= -1;
			}
			std::cout << "Drawing rectangle\n";
			cv::rectangle(t -> first_frame,
						  cv::Point(t->guitar_bbox.x, t->guitar_bbox.y),
						  cv::Point(t->guitar_bbox.x + t->guitar_bbox.width,
									t->guitar_bbox.y + t->guitar_bbox.height),
						  cv::Scalar(255, 0, 0));
			std::cout << "Showing selected region\n";
			cv::imshow(t -> window_name, t -> first_frame);
			cv::waitKey(5000);
			break;
		default: break;
	}
}

void Transcriber::get_guitar_box() {
	std::cout << "Inside get_guitar_box\n";
	cv::setMouseCallback(window_name,
						 Transcriber::guitar_roi_callback,
						 this);
	cv::imshow(window_name, first_frame);
	cv::waitKey(-1);
	guitar_template = first_frame(guitar_bbox);
}

void Transcriber::downsample(const cv::Mat& frame,
							 cv::Mat& dest,
							 int num_times) const {
	frame.copyTo(dest);
	while (num_times--) {
		pyrDown(dest, dest,
				cv::Size(dest.cols / 2, dest.rows / 2));
	}
}

cv::Point Transcriber::get_match_point (const cv::Mat& frame) const {

	//	Downsample the image to help performance further.
	cv::Mat down_frame;
	downsample(frame, down_frame, NUM_DOWNSAMPLES);

	//	Downsample the template as well.
	cv::Mat down_template;
   	downsample(guitar_template, down_template, NUM_DOWNSAMPLES);

	cv::Mat result;

	/// Create the result matrix
	int result_cols =  down_frame.cols - down_template.cols + 1;
	int result_rows = down_frame.rows - down_template.rows + 1;
	result.create( result_cols, result_rows, CV_32FC1 );

	/// Do the Matching and Normalize
	cv::matchTemplate( down_frame, down_template, result, CV_TM_CCORR_NORMED);
	cv::normalize( result, result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );

	/// Localizing the best match with minMaxLoc
	double minVal; double maxVal; cv::Point minLoc; cv::Point maxLoc;
	cv::Point matchLoc;

	cv::minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat() );
	matchLoc = maxLoc;

	matchLoc.x = UPMAP(matchLoc.x);
	matchLoc.y = UPMAP(matchLoc.y);
	
	return matchLoc;
}

void
Transcriber::draw_histogram(const std::vector<cv::Vec2f> lines) const {
	//	Histogram of thetas.
	std::vector<cv::Mat> rtplanes;
	cv::split(lines, rtplanes);

	int hist_size = 512;
	float range[] = { 0, 3.14 };
	const float *hist_range = { range };
	cv::Mat r_hist, t_hist;
	cv::calcHist( &rtplanes[0], 1, 0, cv::Mat(), r_hist, 1,
				  &hist_size, &hist_range, true, false);
	cv::calcHist( &rtplanes[1], 1, 0, cv::Mat(), t_hist, 1,
				  &hist_size, &hist_range, true, false);

	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound( (double) hist_w/hist_size );

	cv::Mat histImage( hist_h, hist_w, CV_8UC3, cv::Scalar( 0,0,0) );

	/// Normalize the result to [ 0, histImage.rows ]
	normalize(t_hist, t_hist, 0, histImage.rows,
			  cv::NORM_MINMAX, -1, cv::Mat() );

	/// Draw for each channel
	for( int i = 1; i < hist_size; i++ ) {
		cv::line(histImage,
				 cv::Point( bin_w*(i-1),
				 			hist_h - cvRound(t_hist.at<float>(i-1)) ),
				 cv::Point( bin_w*(i),
					  		hist_h - cvRound(t_hist.at<float>(i)) ),
				 cv::Scalar( 0, 255, 0), 2, 8, 0);
	}
	cv::imshow(alt_window_name, histImage);
}

void Transcriber::draw_rtheta_line(cv::Vec2f line, cv::Mat& frame) const {
	float rho = line[0], theta = line[1];
	cv::Point pt1, pt2;
	double a = cv::cos(theta), b = cv::sin(theta);
	double x0 = a*rho, y0 = b*rho;
	pt1.x = cvRound(x0 + 1000*(-b));
	pt1.y = cvRound(y0 + 1000*(a));
	pt2.x = cvRound(x0 - 1000*(-b));
	pt2.y = cvRound(y0 - 1000*(a));
	cv::line( frame, pt1, pt2, cv::Scalar(255,255,255), 5, CV_AA);
}

float Transcriber::get_r(cv::Vec4i points) const {
	float x1 = points[0];
	float y1 = points[1];
	float x2 = points[2];
	float y2 = points[3];
	return std::fabs((x1 * y2 - x2 * y1))
		/ std::sqrt( (x1 - x2) * (x1 - x2)
					+(y1 - y2) * (y1 - y2));
}

float Transcriber::get_theta(cv::Vec4i points) const {
	if (points[2] != points[0]) {
		float t = 
			std::atan( (float) (points[3] - points[1]) 
						/ (float) (points[2] - points[0]) );
		return (CV_PI / 2) + t;
	} else {
		return 0.0;
	}
}

void Transcriber::draw_twopt_line(cv::Vec4i points, cv::Mat& frame) const {
	cv::line(frame, cv::Point(points[0], points[1]),
					cv::Point(points[2], points[3]), cv::Scalar(255, 0, 0),
					2, CV_AA);
}

void Transcriber::find_two_max_bins(std::vector< std::vector<cv::Vec4i> > bins,
								    int& max_size1, int& max_size2,
									int& max_bin1, int& max_bin2) const {
	max_bin1 = max_bin2 = -1;
	max_size1 = max_size2 = 0;
	for( size_t i = 0; i < bins.size(); i++ ) {
		if (bins[i].size() > max_size1) {
			max_size2 = max_size1;
			max_bin2 = max_bin1;
			max_size1 = bins[i].size();
			max_bin1 = i;
		} else if (bins[i].size() > max_size2) {
			max_size2 = bins[i].size();
			max_bin2  = i;
		}
	}
}

float get_sum_distance(const std::vector<cv::Point>& pts,
					   int i, int j, int k, int l) {
	float dist = 0.0;
	dist += (cv::norm(pts[i] - pts[j])
			+ cv::norm(pts[i] - pts[k])
			+ cv::norm(pts[i] - pts[l])
			+ cv::norm(pts[j] - pts[k])
			+ cv::norm(pts[j] - pts[l])
			+ cv::norm(pts[k] - pts[l]));
	return dist;
}

void Transcriber::find_four_closest(std::vector<cv::Point>& pts) const {
	//	shuffle the pts vector here. First 4 have the smallest combined
	//	distance between them.
	if (pts.size() <= 4) return;

	float min_sum_distance = std::numeric_limits<float>::max();
	int min1, min2, min3, min4;
	float sum_distance;
	for (int i = 0; i < pts.size(); ++i) {
		for (int j = i + 1; j < pts.size(); ++j) {
			for (int k = j + 1; k < pts.size(); ++k) {
				for (int l = k + 1; l < pts.size(); ++l) {
					sum_distance = get_sum_distance(pts, i, j, k, l);
					if (sum_distance < min_sum_distance) {
						min_sum_distance = sum_distance;
						min1 = i;
						min2 = j;
						min3 = k;
						min4 = l;
					}
				}
			}
		}
	}
	std::swap(pts[0], pts[min1]);
	std::swap(pts[1], pts[min2]);
	std::swap(pts[2], pts[min3]);
	std::swap(pts[3], pts[min4]);
}


void Transcriber::generate_tabs() {
	cv::Mat frame;	//	current frame
	int frame_num = 0;

	cv::VideoCapture capture(video_filename);
	if (!capture.isOpened()) {
		std::cerr << "Unable to open video file : " << video_filename << "\n";
		exit(1);
	}

	//	Read the first frame in and ask the user to input a bounding box for
	//	the guitar.
	if (!capture.read(first_frame)) {
		std::cerr << "Unable to read the first frame.\nExiting...\n";
		exit(1);
	}

	get_guitar_box();
	
	char key_input = 0;
	while (key_input != 27) {	//	27 = ESC !!!
		
		//capture.set(CV_CAP_PROP_POS_FRAMES, fno); // !!!set postion
		//	read the current frame
		if (!capture.read(frame)) {
			std::cerr << "Unable to read the next frame.\nExiting...\n";
			exit(1);
		}
		frame_num++;
		
		//	Get the point where the template matches.
		cv::Point matchLoc = get_match_point(frame);
		cv::Rect guitar_part(matchLoc.x, matchLoc.y,
							 guitar_template.cols,
							 guitar_template.rows);
		frame = frame(guitar_part);
		cv::Mat frame_orig(frame);

		//	Now that we have a more ... relevant part of the image, we can
		//	get to work on it.
		//	First up, get all lines in this video. Some of those will be
		//	string, some will be frets, and some others will be unnecessary
		//	noise...
		//
		//	Find all lines in this frame.
		//
		//	Blur it first.
		// cv::GaussianBlur(frame, frame, cv::Size(3,3), 0);
		cv::Canny(frame, frame, CANNY_LOW_THRESH, CANNY_HIGH_THRESH, 3);

		//	Use the Hough Transform Line Detector to find lines of the
		//	equation form:
		//		x cos T + y sin T = p
		//		or
		//		the 2-point form, as the case may be.
		std::vector<cv::Vec4i> lines;
		cv::HoughLinesP(frame, lines, 1, CV_PI / 180,
						HOUGH_THRESH, HOUGH_MIN_LEN, HOUGH_MAX_GAP);

		frame = frame_orig;

		//	Partition the lines into bins.
		std::vector< std::vector<cv::Vec4i> > bins(NUM_BINS);
		for( size_t i = 0; i < lines.size(); i++ ) {
			int bin_no = (int) ((get_theta(lines[i]) * NUM_BINS) / CV_PI);
			bins[bin_no].push_back(lines[i]);
		}

		//	Find the 2 most frequent bins.
		//	Number 1 should be the strings, and 2 the frets.
		//	(Hopefully...)
		int max_size1, max_size2;
		int max_bin1, max_bin2;
		find_two_max_bins(bins, max_size1, max_size2, max_bin1, max_bin2);

		//	Rotate the image to make the strings horizontal.
		cv::Mat rotMatrix = 
			cv::getRotationMatrix2D(cv::Point(frame.cols/2, frame.rows/2), 
						RAD2DEG((max_bin1 * CV_PI)/NUM_BINS) - 90, 1);
		cv::warpAffine(frame, frame, rotMatrix, frame.size());
		cv::flip(frame, frame, 1);
		cv::Mat flipped_orig(frame);
		cv::Mat grad_x, abs_grad_x;
		cv::Sobel(frame, grad_x, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT );
  		cv::convertScaleAbs( grad_x, abs_grad_x );

		//	Threshold to get only the strongest x-gradients.
		//	Will mostly be frets.
		cv::threshold(abs_grad_x, abs_grad_x, 128, 255, 0);

		//	Hough transform to get lines again.
		cv::cvtColor( abs_grad_x,abs_grad_x, CV_BGR2GRAY );
		
		std::vector<cv::Vec4i> fretboard_lines;
		cv::HoughLinesP(abs_grad_x, fretboard_lines, 1, CV_PI / 180,
						HOUGH_THRESH, HOUGH_MIN_LEN, HOUGH_MAX_GAP);

		float epsilon = 0.2;
		std::vector<cv::Vec4i> fret_lines;
		
		for (size_t i = 0; i < fretboard_lines.size(); ++i) {
			float t = get_theta(fretboard_lines[i]); 
			if(!std::fabs(t) < epsilon) {
				//	parallel to a fret.
				fret_lines.push_back(fretboard_lines[i]);
			}else{
				draw_twopt_line(fretboard_lines[i], frame);
			}
		}

		//	Chop off the parts that don't contain the guitar.
		//	Find the median vertical line.
		//	It will likely be a fret.
		//	Take a factor of safety around this line, and chop off a
		//	rectangular block.
		//	This is the area we wanted to work in from the start.
		std::sort(fret_lines.begin(), fret_lines.end(), pt_higher);
		cv::Vec4i median_up = fret_lines[fret_lines.size() / 2];
		int y1 = median_up[1] < median_up[3] ? median_up[1] : median_up[3];
		std::sort(fret_lines.begin(), fret_lines.end(), pt_lower);
		cv::Vec4i median_down = fret_lines[fret_lines.size() / 2];
		int y2 = median_down[1] < median_down[3]
				? median_down[1] : median_down[3];
		cv::Rect fret_box(0, y1 - (int) (CHOP_SLACK * frame.rows),
						  frame.cols,
						  std::min<int>((y2 - y1 
								  		+ (int)( 2 * CHOP_SLACK * frame.rows)),
							  			frame.rows - y1 - 1));
		frame = frame(fret_box);

		//	Convert to HSV for skin detection.
		frame.copyTo(frame_orig);
		cv::cvtColor(frame, frame, CV_BGR2HSV);
		cv::inRange(frame,
					cv::Scalar(SKIN_HUE_LOW, SKIN_SAT_LOW, SKIN_VAL_LOW),
					cv::Scalar(SKIN_HUE_HIGH, SKIN_SAT_HIGH, SKIN_VAL_HIGH),
					frame);

		std::vector< std::vector<cv::Point> > contours;
		std::vector<cv::Vec4i> hierarchy;
		
		cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
											cv::Size(3,3));
		cv::erode(frame, frame, element);
		findContours(frame, contours, hierarchy,
					 CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE,
					 cv::Point(0, 0));
		frame = frame_orig;
		std::sort(contours.begin(), contours.end(), cont_comp);
		
		std::vector <cv::Point> max_y_pts_vec;
		for( int i = 0; i < contours.size(); i++ ) {
			if (max_y_pts_vec.size() == 6) break;
			
			 cv::drawContours(frame, contours, i,
							 cv::Scalar(255, 0, 255), 2, 8,
							 hierarchy, 0, cv::Point() );
			

			cv::Point max_y_pt = contours[i][0];
			for(int j = 0; j < contours[i].size(); j++){
				if(contours[i][j].y < max_y_pt.y)
					max_y_pt = contours[i][j];
			}

			max_y_pts_vec.push_back(max_y_pt);
		}

		find_four_closest(max_y_pts_vec);
		for( int i = 0; i < 4; i++ ) {
			cv::circle(frame, max_y_pts_vec[i], 4, cv::Scalar(0, 255, 255), -1);
		}
		
		//	show the current frame.
		cv::imshow(window_name, frame);
		cv::imwrite(std::string("screenshots/screenshot_")
				+ std::to_string(frame_num)
				+ std::string(".png"), frame);

		key_input = cv::waitKey(1);
	}

	capture.release();
}

int main(int argc, char **argv) {
	if (argc != 3) {
		std::cerr << "usage : <binary> <filename> <num_strings>\n.";
		std::cerr << "Exiting...\n";
		return -1;
	}
	Transcriber t(argv[1], atoi(argv[2]));
	t.generate_tabs();
}
