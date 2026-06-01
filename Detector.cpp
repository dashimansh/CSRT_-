#include "Detector.h"
#include "Config.h"
#include <iostream>

Detector::Detector()
{
    ColorLower1 = cv::Scalar(
        RED_LOWER1_H,
        RED_LOWER1_S,
        RED_LOWER1_V);
    ColorUpper1 = cv::Scalar(
        RED_UPPER1_H,
        RED_UPPER1_S,
        RED_UPPER1_V);
    ColorLower2 = cv::Scalar(
        RED_LOWER2_H,
        RED_LOWER2_S,
        RED_LOWER2_V);
    ColorUpper2 = cv::Scalar(
        RED_UPPER2_H,
        RED_UPPER2_S,
        RED_UPPER2_V);
    bHasTwoRanges = true;
}

void Detector::SetColorRange(
    const cv::Scalar& Lower1,
    const cv::Scalar& Upper1,
    const cv::Scalar& Lower2,
    const cv::Scalar& Upper2)
{
    ColorLower1 = Lower1;
    ColorUpper1 = Upper1;
    if (Lower2[0] >= 0)
    {
        ColorLower2 = Lower2;
        ColorUpper2 = Upper2;
        bHasTwoRanges = true;
    }
    else bHasTwoRanges = false;
}

void Detector::BuildMask(
    const cv::Mat& Frame)
{
    cv::cvtColor(Frame, HSVFrame,
        cv::COLOR_BGR2HSV);
    cv::inRange(HSVFrame,
        ColorLower1, ColorUpper1, Mask);
    if (bHasTwoRanges)
    {
        cv::inRange(HSVFrame,
            ColorLower2, ColorUpper2,
            Mask2);
        cv::bitwise_or(Mask, Mask2, Mask);
    }
    cv::erode(Mask, Mask,
        cv::Mat(), cv::Point(-1, -1), 2);
    cv::dilate(Mask, Mask,
        cv::Mat(), cv::Point(-1, -1), 3);
}

DetectionResult Detector::FindLargestBlob()
{
    DetectionResult Result;
    std::vector<std::vector<cv::Point>>
        Contours;
    cv::findContours(Mask, Contours,
        cv::RETR_EXTERNAL,
        cv::CHAIN_APPROX_SIMPLE);

    if (Contours.empty()) return Result;

    double LargestArea = 0;
    int    LargestIdx  = -1;
    for (int i = 0;
        i < (int)Contours.size(); i++)
    {
        double Area =
            cv::contourArea(Contours[i]);
        if (Area > LargestArea)
        {
            LargestArea = Area;
            LargestIdx  = i;
        }
    }

    if (LargestArea < MIN_OBJECT_AREA)
        return Result;

    Result.bFound      = true;
    Result.Area        = LargestArea;
    Result.BoundingBox =
        cv::boundingRect(
            Contours[LargestIdx]);
    Result.Center = cv::Point(
        Result.BoundingBox.x +
        Result.BoundingBox.width  / 2,
        Result.BoundingBox.y +
        Result.BoundingBox.height / 2);

    float BoxArea = (float)(
        Result.BoundingBox.width *
        Result.BoundingBox.height);
    Result.Confidence = std::min(1.f,
        (float)(LargestArea / BoxArea));
    return Result;
}

DetectionResult Detector::Detect(
    const cv::Mat& Frame)
{
    if (Frame.empty())
        return DetectionResult();
    BuildMask(Frame);
    return FindLargestBlob();
}

void Detector::AdaptColorRange(
    const cv::Mat& Frame,
    const cv::Rect& ObjectBox)
{
    cv::Rect SafeBox = ObjectBox &
        cv::Rect(0, 0,
            Frame.cols, Frame.rows);
    if (SafeBox.area() <= 0) return;

    cv::Mat ObjectHSV;
    cv::cvtColor(Frame(SafeBox),
        ObjectHSV, cv::COLOR_BGR2HSV);

    cv::Scalar Mean, StdDev;
    cv::meanStdDev(ObjectHSV,
        Mean, StdDev);

    float Margin = 15.f;
    cv::Scalar NewLower(
        std::max(0.0, Mean[0] - Margin),
        std::max(0.0, Mean[1] - 40.0),
        std::max(0.0, Mean[2] - 40.0));
    cv::Scalar NewUpper(
        std::min(180.0, Mean[0] + Margin),
        255.0, 255.0);

    for (int i = 0; i < 3; i++)
    {
        ColorLower1[i] =
            ColorLower1[i] *
            (1 - AdaptRate) +
            NewLower[i] * AdaptRate;
        ColorUpper1[i] =
            ColorUpper1[i] *
            (1 - AdaptRate) +
            NewUpper[i] * AdaptRate;
    }
    bHasTwoRanges = false;
}

cv::Rect Detector::SelectROI(
    cv::Mat& Frame)
{
    cv::Rect ROI = cv::selectROI(
        "Select Object - press Enter",
        Frame, false, false);
    cv::destroyWindow(
        "Select Object - press Enter");
    return ROI;
}

void Detector::Draw(
    cv::Mat& Frame,
    const DetectionResult& Result)
{
    if (!Result.bFound) return;
    cv::rectangle(Frame,
        Result.BoundingBox,
        cv::Scalar(0, 165, 255), 2);
    cv::putText(Frame,
        "DETECTED " +
        std::to_string(
            (int)(Result.Confidence * 100))
        + "%",
        cv::Point(
            Result.BoundingBox.x,
            Result.BoundingBox.y - 8),
        cv::FONT_HERSHEY_SIMPLEX,
        0.55,
        cv::Scalar(0, 165, 255), 2);
}
