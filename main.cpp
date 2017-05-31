#include <jni.h>
#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>
#include <android/log.h>

#define  LOG_TAG    "JNI_PART"
#define LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG, __VA_ARGS__)
#define LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG,LOG_TAG, __VA_ARGS__)
#define LOGW(...)  __android_log_print(ANDROID_LOG_WARN,LOG_TAG, __VA_ARGS__)
#define LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,LOG_TAG, __VA_ARGS__)
#define LOGF(...)  __android_log_print(ANDROID_LOG_FATAL,LOG_TAG, __VA_ARGS__)
using namespace cv;
using namespace std;
extern "C" {
jstring Java_com_martin_ads_testopencv_MainActivity_stringFromJNI(
        JNIEnv *env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}

int thresh = 120;
double max_size_ratio = 0.6;
double min_size_ratio = 0.1;
static double angle(Point pt1, Point pt2, Point pt0) {
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1 * dx2 + dy1 * dy2) /
           sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
}


void
selectFeatureBounds(Mat &frame, CvPoint topLeftPoint, CvPoint oppssiteTotopLeft,
                    MatSize imageSize) {
    cv::Mat overlay;
    double alpha = 0.3;
    frame.copyTo(overlay);
    cv::rectangle(overlay, topLeftPoint, oppssiteTotopLeft, cvScalar(0, 0, 255, 0), CV_FILLED, 4);
//    cv::rectangle(frame, topLeftPoint, oppssiteTotopLeft, cvScalar(0, 0, 255 * 0.7), CV_FILLED, 4);
    cv::addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame);
}


static void findSquares(const Mat &image, vector<vector<Point> > &squares, double resize_scale) {
    squares.clear();
    Mat pyr, timg, gray0(image.size(), CV_8U), gray;
    image.copyTo(timg);
    int max_size = image.cols * image.rows;
    vector<vector<Point> > contours;
    int N = 4;
    for (int c = 0; c < 3; c++) {
        int ch[] = {c, 0};
        mixChannels(&timg, 1, &gray0, 1, ch, 1);
        for (int l = 0; l < N; l++) {
            if (l == 0) {
                Canny(gray0, gray, 0, thresh, 3);
                dilate(gray, gray, Mat(), Point(-1, -1));
            } else {
                gray = (gray0 >= (l + 1) * 255 / N);
            }
            findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
            vector<Point> approx;
            for (size_t i = 0; i < contours.size(); i++) {
                double area0 = contourArea(contours[i]);
                if ((area0 > max_size_ratio * max_size) || (area0 < min_size_ratio * max_size))
                    continue;
                approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true) * 0.015,
                             true);
                if (approx.size() == 4 && isContourConvex(Mat(approx))) {
                    double maxCosine = 0;
                    for (int j = 2; j < 5; j++) {
                        double cosine = fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
                        maxCosine = MAX(maxCosine, cosine);
                    }
                    //if( maxCosine < 0.5 ) //angle must be larger than 60
                    if (maxCosine < 0.25) //angle must be larger than 75
                        squares.push_back(approx);
                }
            }
        }
    }
    //pick the center one
    if (squares.size()) {
        double min_dis = 100000;
        vector<Point> pts;
        for (size_t i = 0; i < squares.size(); i++) {
            double new_dis =
                    fabs(squares[i][0].x + squares[i][1].x + squares[i][2].x + squares[i][3].x) /
                    4.0 +
                    (squares[i][0].y + squares[i][1].y + squares[i][2].y + squares[i][3].y) / 4.0 -
                    image.cols / 2.0 - image.rows / 2.0;
            if (new_dis < min_dis) {
                min_dis = new_dis;
                pts = squares[i];
            }
        }
        squares.clear();
        for (size_t i = 0; i < pts.size(); i++) {
            pts[i].x = int(pts[i].x * (1.0 / resize_scale));
            pts[i].y = int(pts[i].y * (1.0 / resize_scale));
        }
        double max_x = -1, min_x = 10000, max_y = -1, min_y = 100000;
        for (size_t i = 0; i < pts.size(); i++) {
            max_x = max_x > pts[i].x ? max_x : pts[i].x;
            min_x = min_x < pts[i].x ? min_x : pts[i].x;
            max_y = max_y > pts[i].y ? max_y : pts[i].y;
            min_y = min_y < pts[i].y ? min_y : pts[i].y;
        }
        pts.clear();
        pts.push_back(Point(min_x, min_y));
        pts.push_back(Point(max_x, min_y));
        pts.push_back(Point(max_x, max_y));
        pts.push_back(Point(min_x, max_y));
        squares.push_back(pts);
    }
}


static void
drawSquares(Mat &image, const vector<vector<Point> > &squares
) {
    Size boxSize = image.size();
    int height = boxSize.height;
    int width = boxSize.width;
    int horizontallineLength = boxSize.width / 4;
    int verticallineLength = height / 5;
    int lineLength = 30;
    for (size_t i = 0; i < squares.size(); i++) {
        const Point *p = &squares[i][0];
        int n = (int) squares[i].size();


        cv::line(image,
                 cvPoint(squares[i][0].x, squares[i][0].y),
                 cvPoint(squares[i][0].x + lineLength,
                         squares[i][0].y), cvScalar(0, 0, 255, 0),
                 8, 4);
        cv::line(image,
                 cvPoint(squares[i][0].x, squares[i][0].y),
                 cvPoint(squares[i][0].x,
                         squares[i][0].y + lineLength),
                 cvScalar(0, 0, 255, 0), 8, 4);
        /**
       *    right bottom
       */
        cv::line(image, cvPoint(squares[i][1].x,
                                squares[i][1].y),
                 cvPoint(squares[i][1].x - lineLength,
                         squares[i][1].y), cvScalar(0, 0, 255, 0),
                 8, 4);
        cv::line(image, cvPoint(squares[i][1].x,
                                squares[i][1].y),
                 cvPoint(squares[i][1].x,
                         squares[i][1].y + lineLength),
                 cvScalar(0, 0, 255, 0), 8, 4);
        /**
       *    left bottom
       */
        cv::line(image, cvPoint(squares[i][2].x,
                                squares[i][2].y),
                 cvPoint(squares[i][2].x,
                         squares[i][2].y - lineLength),
                 cvScalar(255, 255, 255, 0), 8, 4);
        cv::line(image, cvPoint(squares[i][2].x,
                                squares[i][2].y),
                 cvPoint(squares[i][2].x - lineLength,
                         squares[i][2].y),
                 cvScalar(255, 255, 255, 0), 8, 4);
        /**
     *    left top
     */
        cv::line(image, cvPoint(squares[i][3].x,
                                squares[i][3].y),
                 cvPoint(squares[i][3].x + lineLength,
                         squares[i][3].y),
                 cvScalar(255, 255, 255, 0), 8, 4);
        cv::line(image, cvPoint(squares[i][3].x,
                                squares[i][3].y),
                 cvPoint(squares[i][3].x,
                         squares[i][3].y - lineLength),
                 cvScalar(255, 255, 255, 0), 8, 4);


//        selectFeatureBounds(image, cvPoint(squares[i][3].x,
//                                           squares[i][3].y),
//                            cvPoint(squares[i][1].x,
//                                    squares[i][1].y), image.size);


    }
}
void main(JNIEnv *, jobject, jlong addrGray,jlong addrRgba) {

    Mat &image = *(Mat *) addrRgba;
    vector<vector<Point> > squares;
    Mat newImage = image.clone();

    float scale = 0.;
    resize(image, newImage, Size(), scale, scale);
    findSquares(newImage, squares, scale);
    drawSquares(image, squares);
    newImage.release();

}

}
