#include <stdio.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "include/accelerator.hpp"

using namespace cv;
using namespace std;

int main(int argc, const char **argv) {
    Mat frame;
    Accelerator dev;

    vector<vector<float>> data;
    data.push_back(vector<float>());
    data.push_back(vector<float>());
    data.push_back(vector<float>());

    data.at(0).push_back(-1.0f);
    data.at(0).push_back(-1.0f);
    data.at(0).push_back(-1.0f);

    data.at(1).push_back(-1.0f);
    data.at(1).push_back(8.0f);
    data.at(1).push_back(-1.0f);

    data.at(2).push_back(-1.0f);
    data.at(2).push_back(-1.0f);
    data.at(2).push_back(-1.0f);

    VideoCapture cap;
    cap.open(0);

    while (1) {
        cap >> frame;
        Mat result = dev.conv(frame, data);
        imshow("test", result);
        waitKey(20);
    }

    return 0;
}