#include "KalmanTracker.h"

KalmanTracker::KalmanTracker()
{
    KF = cv::KalmanFilter(6, 4, 0);
    Measurement =
        cv::Mat::zeros(4, 1, CV_32F);
}

void KalmanTracker::Init(
    const cv::Rect& InitBox)
{
    float td[] = {
        1,0,0,0,1,0,
        0,1,0,0,0,1,
        0,0,1,0,0,0,
        0,0,0,1,0,0,
        0,0,0,0,1,0,
        0,0,0,0,0,1 };

    KF.transitionMatrix =
        cv::Mat(6, 6, CV_32F, td).clone();

    cv::setIdentity(KF.measurementMatrix);
    cv::setIdentity(KF.processNoiseCov,
        cv::Scalar::all(1e-4));
    cv::setIdentity(KF.measurementNoiseCov,
        cv::Scalar::all(1e-2));
    cv::setIdentity(KF.errorCovPost,
        cv::Scalar::all(1));

    KF.statePost.at<float>(0) =
        (float)InitBox.x;
    KF.statePost.at<float>(1) =
        (float)InitBox.y;
    KF.statePost.at<float>(2) =
        (float)InitBox.width;
    KF.statePost.at<float>(3) =
        (float)InitBox.height;
    KF.statePost.at<float>(4) = 0;
    KF.statePost.at<float>(5) = 0;

    PredictedBox = InitBox;
    bInitialized = true;
}

cv::Rect KalmanTracker::Update(
    const cv::Rect& Measured)
{
    if (!bInitialized)
    {
        Init(Measured);
        return Measured;
    }

    KF.predict();

    Measurement.at<float>(0) =
        (float)Measured.x;
    Measurement.at<float>(1) =
        (float)Measured.y;
    Measurement.at<float>(2) =
        (float)Measured.width;
    Measurement.at<float>(3) =
        (float)Measured.height;

    cv::Mat Corrected =
        KF.correct(Measurement);

    PredictedBox = MatToRect(Corrected);
    return PredictedBox;
}

cv::Rect KalmanTracker::Predict()
{
    if (!bInitialized) return cv::Rect();
    cv::Mat Pred = KF.predict();
    PredictedBox = MatToRect(Pred);
    return PredictedBox;
}

void KalmanTracker::Reset()
{
    bInitialized = false;
    PredictedBox = cv::Rect();
}

cv::Rect KalmanTracker::MatToRect(
    const cv::Mat& S)
{
    int X = std::max(0,
        (int)S.at<float>(0));
    int Y = std::max(0,
        (int)S.at<float>(1));
    int W = std::max(1,
        (int)S.at<float>(2));
    int H = std::max(1,
        (int)S.at<float>(3));
    return cv::Rect(X, Y, W, H);
}
