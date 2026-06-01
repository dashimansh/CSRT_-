#include "Tracker.h"
#include "Config.h"
#include <iostream>

ObjectTracker::ObjectTracker()
{
    State          = ETrackState::Idle;
    LostFrameCount = 0;

    BGSub =
        cv::createBackgroundSubtractorMOG2(
            500, 16, false);
}

bool ObjectTracker::Init(
    const cv::Mat& Frame,
    const cv::Rect& BBox,
    ETrackerType Type)
{
    TrackerType = Type;

    cv::Rect SafeBox = BBox &
        cv::Rect(0, 0,
            Frame.cols, Frame.rows);
    if (SafeBox.area() <= 0)
        return false;

    // Save original template
    ObjTemplate = Frame(SafeBox).clone();

    // Save color histogram
    // (HSV — all 3 channels)
    cv::Mat HSV;
    cv::cvtColor(Frame, HSV,
        cv::COLOR_BGR2HSV);

    cv::Mat ROI = HSV(SafeBox);
    int HistSize[] = { 8, 8, 8 };
    float HRange[] = { 0, 180 };
    float SRange[] = { 0, 256 };
    float VRange[] = { 0, 256 };
    const float* Ranges[] = {
        HRange, SRange, VRange };
    int Channels[] = { 0, 1, 2 };

    cv::calcHist(&ROI, 1, Channels,
        cv::Mat(), ObjHistogram,
        3, HistSize, Ranges);
    cv::normalize(ObjHistogram,
        ObjHistogram, 0, 1,
        cv::NORM_MINMAX);

    if (Type == ETrackerType::CSRT)
    {
        CVTracker =
            cv::TrackerCSRT::create();
        CVTracker->init(Frame, SafeBox);
        std::cout << "CSRT initialized\n";
    }
    else if (Type == ETrackerType::KCF)
    {
        CVTracker =
            cv::TrackerKCF::create();
        CVTracker->init(Frame, SafeBox);
        std::cout << "KCF initialized\n";
    }
    else if (Type == ETrackerType::CAMSHIFT)
    {
        cv::Mat HSV2;
        cv::cvtColor(Frame, HSV2,
            cv::COLOR_BGR2HSV);

        cv::Mat ROI2   = HSV2(SafeBox);
        int   HistSize2 = 16;
        float Range2[]  = { 0, 180 };
        const float* Ranges2 = Range2;
        int Channels2 = 0;

        cv::calcHist(&ROI2, 1, &Channels2,
            cv::Mat(), RoiHist,
            1, &HistSize2, &Ranges2);
        cv::normalize(RoiHist, RoiHist,
            0, 255, cv::NORM_MINMAX);

        TrackWindow = SafeBox;
        std::cout << "CamShift initialized\n";
    }

    CurrentBox     = SafeBox;
    State          = ETrackState::Tracking;
    LostFrameCount = 0;
    KFilter.Init(SafeBox);
    return true;
}

bool ObjectTracker::UpdateCVTracker(
    const cv::Mat& Frame,
    cv::Rect& OutBox)
{
    if (!CVTracker) return false;

    bool bOK = CVTracker->update(
        Frame, OutBox);

    if (!bOK) return false;

    OutBox &= cv::Rect(0, 0,
        Frame.cols, Frame.rows);

    if (OutBox.area() < 100)
        return false;

    return true;
}

bool ObjectTracker::UpdateCamShift(
    const cv::Mat& Frame,
    cv::Rect& OutBox)
{
    if (RoiHist.empty()) return false;

    cv::Mat HSV, BackProj;
    cv::cvtColor(Frame, HSV,
        cv::COLOR_BGR2HSV);

    float Range[] = { 0, 180 };
    const float* Ranges = Range;
    int Channels = 0;

    cv::calcBackProject(&HSV, 1,
        &Channels, RoiHist,
        BackProj, &Ranges);

    cv::TermCriteria TC(
        cv::TermCriteria::EPS |
        cv::TermCriteria::COUNT,
        10, 1);

    cv::Rect SearchWin  = TrackWindow;
    SearchWin.x        -= 20;
    SearchWin.y        -= 20;
    SearchWin.width    += 40;
    SearchWin.height   += 40;
    SearchWin &= cv::Rect(0, 0,
        Frame.cols, Frame.rows);

    if (SearchWin.area() <= 0)
        return false;

    cv::Mat SearchBP = BackProj(SearchWin);
    cv::Rect LocalWin(
        0, 0,
        SearchWin.width,
        SearchWin.height);

    cv::RotatedRect RR =
        cv::CamShift(SearchBP,
            LocalWin, TC);

    OutBox    = RR.boundingRect();
    OutBox.x += SearchWin.x;
    OutBox.y += SearchWin.y;
    OutBox   &= cv::Rect(0, 0,
        Frame.cols, Frame.rows);

    if (OutBox.area() < 100)
        return false;

    TrackWindow = OutBox;

    cv::Mat Region = BackProj(
        OutBox & cv::Rect(0, 0,
            Frame.cols, Frame.rows));
    cv::Scalar Mean = cv::mean(Region);
    return Mean[0] > 30.0;
}

// Compare color histogram of
// a region with object histogram
double ObjectTracker::CompareHistogram(
    const cv::Mat& Frame,
    const cv::Rect& Box)
{
    if (ObjHistogram.empty())
        return 0.0;

    cv::Rect SafeBox = Box &
        cv::Rect(0, 0,
            Frame.cols, Frame.rows);
    if (SafeBox.area() <= 0)
        return 0.0;

    cv::Mat HSV;
    cv::cvtColor(Frame, HSV,
        cv::COLOR_BGR2HSV);

    cv::Mat ROI = HSV(SafeBox);
    int HistSize[] = { 8, 8, 8 };
    float HRange[] = { 0, 180 };
    float SRange[] = { 0, 256 };
    float VRange[] = { 0, 256 };
    const float* Ranges[] = {
        HRange, SRange, VRange };
    int Channels[] = { 0, 1, 2 };

    cv::Mat Hist;
    cv::calcHist(&ROI, 1, Channels,
        cv::Mat(), Hist,
        3, HistSize, Ranges);
    cv::normalize(Hist, Hist,
        0, 1, cv::NORM_MINMAX);

    return cv::compareHist(
        ObjHistogram, Hist,
        cv::HISTCMP_CORREL);
}

bool ObjectTracker::FindByTemplateAndColor(
    const cv::Mat& Frame,
    cv::Rect& OutBox)
{
    if (ObjTemplate.empty())
        return false;

    if (Frame.cols < ObjTemplate.cols ||
        Frame.rows < ObjTemplate.rows)
        return false;

    cv::Mat Result;
    cv::matchTemplate(Frame,
        ObjTemplate, Result,
        cv::TM_CCOEFF_NORMED);

    // Find top 5 matches
    double BestScore = 0;
    cv::Rect BestBox;

    for (int i = 0; i < 5; i++)
    {
        double MinVal, MaxVal;
        cv::Point MinLoc, MaxLoc;
        cv::minMaxLoc(Result,
            &MinVal, &MaxVal,
            &MinLoc, &MaxLoc);

        if (MaxVal < 0.5) break;

        cv::Rect CandBox(
            MaxLoc.x, MaxLoc.y,
            ObjTemplate.cols,
            ObjTemplate.rows);
        CandBox &= cv::Rect(0, 0,
            Frame.cols, Frame.rows);

        // Check color match too
        double ColorScore =
            CompareHistogram(
                Frame, CandBox);

        // Combined score
        double CombinedScore =
            MaxVal * 0.5 +
            ColorScore * 0.5;

        if (CombinedScore > BestScore)
        {
            BestScore = CombinedScore;
            BestBox   = CandBox;
        }

        // Suppress this location
        cv::rectangle(Result,
            cv::Point(
                MaxLoc.x - ObjTemplate.cols/2,
                MaxLoc.y - ObjTemplate.rows/2),
            cv::Point(
                MaxLoc.x + ObjTemplate.cols/2,
                MaxLoc.y + ObjTemplate.rows/2),
            cv::Scalar(0), -1);
    }

    // Need both template AND
    // color to match well
    if (BestScore < 0.55)
        return false;

    OutBox = BestBox;
    std::cout
        << "Re-acquired! Score: "
        << (int)(BestScore * 100)
        << "%\n";

    return true;
}

bool ObjectTracker::FindByMotion(
    const cv::Mat& Frame,
    cv::Rect& OutBox)
{
    if (!BGSub) return false;

    cv::Mat FGMask;
    BGSub->apply(Frame, FGMask);

    cv::erode(FGMask, FGMask,
        cv::Mat(), cv::Point(-1,-1), 2);
    cv::dilate(FGMask, FGMask,
        cv::Mat(), cv::Point(-1,-1), 3);

    std::vector<std::vector<cv::Point>>
        Contours;
    cv::findContours(FGMask, Contours,
        cv::RETR_EXTERNAL,
        cv::CHAIN_APPROX_SIMPLE);

    if (Contours.empty()) return false;

    // Find best motion region
    // that matches object color
    double BestScore = 0;
    cv::Rect BestBox;

    for (auto& C : Contours)
    {
        double Area = cv::contourArea(C);
        if (Area < 500) continue;

        cv::Rect MotionBox =
            cv::boundingRect(C);

        // Check size similarity
        if (!ObjTemplate.empty())
        {
            float WR = (float)MotionBox.width
                / ObjTemplate.cols;
            float HR = (float)MotionBox.height
                / ObjTemplate.rows;

            if (WR < 0.4f || WR > 2.5f ||
                HR < 0.4f || HR > 2.5f)
                continue;
        }

        // Check color match
        double ColorScore =
            CompareHistogram(
                Frame, MotionBox);

        if (ColorScore > BestScore)
        {
            BestScore = ColorScore;
            BestBox   = MotionBox;
        }
    }

    // Need good color match
    if (BestScore < 0.5)
        return false;

    OutBox = BestBox;
    std::cout
        << "Motion re-acquired! "
        << "Color match: "
        << (int)(BestScore * 100)
        << "%\n";
    return true;
}

bool ObjectTracker::TryReacquire(
    const cv::Mat& Frame)
{
    cv::Rect NewBox;

    // Try template + color first
    if (FindByTemplateAndColor(
        Frame, NewBox))
    {
        if (CVTracker &&
            (TrackerType ==
                ETrackerType::CSRT ||
             TrackerType ==
                ETrackerType::KCF))
        {
            CVTracker.release();
            if (TrackerType ==
                ETrackerType::CSRT)
                CVTracker =
                    cv::TrackerCSRT::create();
            else
                CVTracker =
                    cv::TrackerKCF::create();
            CVTracker->init(Frame, NewBox);
        }

        CurrentBox     = NewBox;
        State          = ETrackState::Tracking;
        LostFrameCount = 0;
        KFilter.Init(NewBox);
        std::cout << "AUTO RECOVERED!\n";
        return true;
    }

    // Then try motion + color
    if (FindByMotion(Frame, NewBox))
    {
        if (CVTracker &&
            (TrackerType ==
                ETrackerType::CSRT ||
             TrackerType ==
                ETrackerType::KCF))
        {
            CVTracker.release();
            if (TrackerType ==
                ETrackerType::CSRT)
                CVTracker =
                    cv::TrackerCSRT::create();
            else
                CVTracker =
                    cv::TrackerKCF::create();
            CVTracker->init(Frame, NewBox);
        }

        CurrentBox     = NewBox;
        State          = ETrackState::Tracking;
        LostFrameCount = 0;
        KFilter.Init(NewBox);
        std::cout << "AUTO RECOVERED!\n";
        return true;
    }

    return false;
}

cv::Rect ObjectTracker::Update(
    const cv::Mat& Frame)
{
    if (State == ETrackState::Idle)
        return cv::Rect();

    if (BGSub &&
        State != ETrackState::Tracking)
    {
        cv::Mat FGMask;
        BGSub->apply(Frame, FGMask);
    }

    cv::Rect TrackerBox;
    bool bOK = false;

    if (TrackerType == ETrackerType::CSRT ||
        TrackerType == ETrackerType::KCF)
        bOK = UpdateCVTracker(
            Frame, TrackerBox);
    else if (TrackerType ==
        ETrackerType::CAMSHIFT)
        bOK = UpdateCamShift(
            Frame, TrackerBox);

    if (bOK)
    {
        cv::Point LastC(
            CurrentBox.x +
            CurrentBox.width  / 2,
            CurrentBox.y +
            CurrentBox.height / 2);
        cv::Point NewC(
            TrackerBox.x +
            TrackerBox.width  / 2,
            TrackerBox.y +
            TrackerBox.height / 2);
        float Dist =
            (float)cv::norm(LastC - NewC);
        if (Dist > MAX_JUMP_DISTANCE)
            bOK = false;
    }

    if (bOK)
    {
        CurrentBox =
            KFilter.Update(TrackerBox);
        State          = ETrackState::Tracking;
        LostFrameCount = 0;
    }
    else
    {
        LostFrameCount++;

        if (LostFrameCount <= MAX_LOST_FRAMES)
        {
            CurrentBox = KFilter.Predict();
            State      = ETrackState::Occluded;
            std::cout << "Occluded ["
                << LostFrameCount << "]\n";
        }
        else
        {
            State = ETrackState::Lost;
            TryReacquire(Frame);
        }
    }
    return CurrentBox;
}

bool ObjectTracker::Reinit(
    const cv::Mat& Frame,
    const cv::Rect& NewBBox)
{
    return Init(Frame, NewBBox, TrackerType);
}

void ObjectTracker::Reset()
{
    KFilter.Reset();
    CVTracker.release();
    RoiHist.release();
    ObjTemplate.release();
    ObjHistogram.release();
    State          = ETrackState::Idle;
    LostFrameCount = 0;
    CurrentBox     = cv::Rect();
}

void ObjectTracker::Draw(cv::Mat& Frame)
{
    if (State == ETrackState::Idle)
        return;

    cv::Scalar  Color;
    std::string Text;

    switch (State)
    {
    case ETrackState::Tracking:
        Color = cv::Scalar(0, 255, 0);
        Text  = "TRACKING";
        break;
    case ETrackState::Occluded:
        Color = cv::Scalar(0, 165, 255);
        Text  = "OCCLUDED";
        break;
    case ETrackState::Lost:
        Color = cv::Scalar(0, 0, 255);
        Text  = "SEARCHING...";
        break;
    default: return;
    }

    cv::rectangle(Frame,
        CurrentBox, Color, 2);

    if (SHOW_KALMAN_PRED &&
        State == ETrackState::Occluded)
    {
        cv::Rect P =
            KFilter.GetPredictedBox();
        cv::rectangle(Frame, P,
            cv::Scalar(255, 255, 0), 1);
        cv::putText(Frame, "KALMAN",
            cv::Point(P.x, P.y - 5),
            cv::FONT_HERSHEY_SIMPLEX,
            0.4,
            cv::Scalar(255, 255, 0), 1);
    }

    cv::Point C(
        CurrentBox.x + CurrentBox.width  / 2,
        CurrentBox.y + CurrentBox.height / 2);

    cv::line(Frame,
        cv::Point(C.x - 15, C.y),
        cv::Point(C.x + 15, C.y),
        Color, 2);
    cv::line(Frame,
        cv::Point(C.x, C.y - 15),
        cv::Point(C.x, C.y + 15),
        Color, 2);

    std::string TypeText;
    switch (TrackerType)
    {
    case ETrackerType::CSRT:
        TypeText = "[CSRT]";
        break;
    case ETrackerType::KCF:
        TypeText = "[KCF]";
        break;
    case ETrackerType::CAMSHIFT:
        TypeText = "[CAMSHIFT]";
        break;
    }

    cv::putText(Frame,
        Text + " " + TypeText,
        cv::Point(CurrentBox.x,
            CurrentBox.y - 10),
        cv::FONT_HERSHEY_SIMPLEX,
        0.6, Color, 2);
}
