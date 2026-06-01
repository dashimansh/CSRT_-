#pragma once

#include <opencv2/opencv.hpp>

class KalmanTracker
{
public:
    KalmanTracker();
    void     Init(const cv::Rect& InitBox);
    cv::Rect Update(const cv::Rect& Measured);
    cv::Rect Predict();
    void     Reset();
    cv::Rect GetPredictedBox() const
    {
        return PredictedBox;
    }
    bool IsInitialized() const
    {
        return bInitialized;
    }

private:
    cv::KalmanFilter KF;
    cv::Mat          Measurement;
    cv::Rect         PredictedBox;
    bool             bInitialized = false;

    cv::Rect MatToRect(
        const cv::Mat& State);
};
