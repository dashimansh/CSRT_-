#include "FrameStabilizer.h"

FrameStabilizer::FrameStabilizer()
{
    History.reserve(HistorySize);
}

cv::Mat FrameStabilizer::Stabilize(
    const cv::Mat& Frame)
{
    cv::Mat Gray;
    cv::cvtColor(Frame, Gray,
        cv::COLOR_BGR2GRAY);

    if (!bReady)
    {
        PrevGray  = Gray.clone();
        bReady    = true;
        Transform =
            cv::Mat::eye(2, 3, CV_64F);
        return Frame.clone();
    }

    cv::goodFeaturesToTrack(
        PrevGray, PrevPoints,
        200, 0.01, 30);

    if (PrevPoints.empty())
    {
        PrevGray = Gray.clone();
        return Frame.clone();
    }

    std::vector<uchar> Status;
    std::vector<float> Error;
    cv::calcOpticalFlowPyrLK(
        PrevGray, Gray,
        PrevPoints, CurrPoints,
        Status, Error);

    std::vector<cv::Point2f> GP, GC;
    for (size_t i = 0;
        i < Status.size(); i++)
    {
        if (Status[i])
        {
            GP.push_back(PrevPoints[i]);
            GC.push_back(CurrPoints[i]);
        }
    }

    if (GP.size() < 4)
    {
        PrevGray = Gray.clone();
        return Frame.clone();
    }

    cv::Mat T =
        cv::estimateAffinePartial2D(
            GP, GC);

    if (T.empty())
    {
        PrevGray = Gray.clone();
        return Frame.clone();
    }

    History.push_back(T.clone());
    if ((int)History.size() > HistorySize)
        History.erase(History.begin());

    Transform = GetSmoothed();

    cv::Mat Inv = Transform.clone();
    Inv.at<double>(0, 2) =
        -Inv.at<double>(0, 2);
    Inv.at<double>(1, 2) =
        -Inv.at<double>(1, 2);

    cv::Mat Stabilized;
    cv::warpAffine(Frame, Stabilized,
        Inv, Frame.size());

    PrevGray = Gray.clone();
    return Stabilized;
}

cv::Mat FrameStabilizer::GetSmoothed()
{
    if (History.empty())
        return cv::Mat::eye(2, 3, CV_64F);

    cv::Mat Sum =
        cv::Mat::zeros(2, 3, CV_64F);
    for (auto& T : History) Sum += T;
    return Sum / (double)History.size();
}

cv::Rect FrameStabilizer::TransformBox(
    const cv::Rect& Box, int W, int H)
{
    if (Transform.empty()) return Box;

    std::vector<cv::Point2f> Corners = {
        { (float)Box.x,
          (float)Box.y },
        { (float)(Box.x + Box.width),
          (float)Box.y },
        { (float)Box.x,
          (float)(Box.y + Box.height) },
        { (float)(Box.x + Box.width),
          (float)(Box.y + Box.height) }
    };

    std::vector<cv::Point2f> Trans;
    cv::transform(Corners, Trans,
        Transform);

    float MinX = Trans[0].x;
    float MinY = Trans[0].y;
    float MaxX = Trans[0].x;
    float MaxY = Trans[0].y;

    for (auto& P : Trans)
    {
        MinX = std::min(MinX, P.x);
        MinY = std::min(MinY, P.y);
        MaxX = std::max(MaxX, P.x);
        MaxY = std::max(MaxY, P.y);
    }

    cv::Rect R(
        (int)MinX, (int)MinY,
        (int)(MaxX - MinX),
        (int)(MaxY - MinY));
    R &= cv::Rect(0, 0, W, H);
    return R;
}

void FrameStabilizer::Reset()
{
    bReady = false;
    PrevGray.release();
    History.clear();
    Transform =
        cv::Mat::eye(2, 3, CV_64F);
}
