#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include "KalmanTracker.h"

enum class ETrackerType
{
    CSRT,
    KCF,
    CAMSHIFT
};

enum class ETrackState
{
    Idle,
    Tracking,
    Occluded,
    Lost
};

class ObjectTracker
{
public:
    ObjectTracker();

    bool Init(
        const cv::Mat& Frame,
        const cv::Rect& BBox,
        ETrackerType Type =
            ETrackerType::CSRT);

    cv::Rect Update(
        const cv::Mat& Frame);

    bool Reinit(
        const cv::Mat& Frame,
        const cv::Rect& NewBBox);

    void Reset();
    void Draw(cv::Mat& Frame);

    ETrackState GetState() const
    {
        return State;
    }
    cv::Rect GetCurrentBox() const
    {
        return CurrentBox;
    }
    cv::Rect GetPredictedBox() const
    {
        return KFilter.GetPredictedBox();
    }
    int GetLostCount() const
    {
        return LostFrameCount;
    }
    bool IsTracking() const
    {
        return State == ETrackState::Tracking;
    }
    bool IsOccluded() const
    {
        return State == ETrackState::Occluded;
    }
    bool IsLost() const
    {
        return State == ETrackState::Lost;
    }

private:
    KalmanTracker KFilter;
    ETrackState   State       = ETrackState::Idle;
    ETrackerType  TrackerType = ETrackerType::CSRT;
    cv::Rect      CurrentBox;
    int           LostFrameCount = 0;

    // CSRT/KCF tracker
    cv::Ptr<cv::Tracker> CVTracker;

    // CamShift
    cv::Mat  RoiHist;
    cv::Rect TrackWindow;

    // Original template — never updated!
    cv::Mat ObjTemplate;

    // Color histogram of object
    cv::Mat ObjHistogram;

    // Background subtractor
    cv::Ptr<cv::BackgroundSubtractorMOG2>
        BGSub;

    bool UpdateCVTracker(
        const cv::Mat& Frame,
        cv::Rect& OutBox);

    bool UpdateCamShift(
        const cv::Mat& Frame,
        cv::Rect& OutBox);

    // Compare color histogram
    double CompareHistogram(
        const cv::Mat& Frame,
        const cv::Rect& Box);

    // Find by template + color combined
    bool FindByTemplateAndColor(
        const cv::Mat& Frame,
        cv::Rect& OutBox);

    // Find by motion + color
    bool FindByMotion(
        const cv::Mat& Frame,
        cv::Rect& OutBox);

    // Try all methods to re-find object
    bool TryReacquire(
        const cv::Mat& Frame);
};
