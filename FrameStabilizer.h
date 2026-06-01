#pragma once
#include <opencv2/opencv.hpp>

class FrameStabilizer
{
public:
    FrameStabilizer();

    cv::Mat  Stabilize(
        const cv::Mat& Frame);

    cv::Rect TransformBox(
        const cv::Rect& Box,
        int W, int H);

    void Reset();

    bool IsReady() const
    {
        return bReady;
    }

private:
    cv::Mat PrevGray;
    cv::Mat Transform;
    bool    bReady = false;

    std::vector<cv::Point2f> PrevPoints;
    std::vector<cv::Point2f> CurrPoints;
    std::vector<cv::Mat>     History;
    int HistorySize = 5;

    cv::Mat GetSmoothed();
};
