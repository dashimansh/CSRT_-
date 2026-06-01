#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include "Config.h"
#include "Detector.h"
#include "Tracker.h"
#include "FrameStabilizer.h"
#include "ThreadedCamera.h"

class FPSCounter
{
public:
    void Tick()
    {
        auto N =
            std::chrono::steady_clock::now();
        double E =
            std::chrono::duration<double>(
                N - Last).count();
        Last = N;
        FPS  = 1.0 / E;
    }
    double Get() const { return FPS; }

private:
    std::chrono::steady_clock::time_point
        Last =
        std::chrono::steady_clock::now();
    double FPS = 0.0;
};

void DrawHUD(
    cv::Mat& Frame,
    double FPS,
    ETrackState State,
    int Lost)
{
    cv::rectangle(
        Frame,
        cv::Rect(0, 0, Frame.cols, 75),
        cv::Scalar(0, 0, 0), -1);

    cv::putText(Frame,
        "FPS: " + std::to_string((int)FPS),
        cv::Point(10, 24),
        cv::FONT_HERSHEY_SIMPLEX,
        0.7, cv::Scalar(0, 255, 255), 2);

    std::string S;
    cv::Scalar  SC;

    switch (State)
    {
    case ETrackState::Idle:
        S  = "IDLE";
        SC = cv::Scalar(200, 200, 200);
        break;
    case ETrackState::Tracking:
        S  = "TRACKING";
        SC = cv::Scalar(0, 255, 0);
        break;
    case ETrackState::Occluded:
        S  = "OCCLUDED [" +
            std::to_string(Lost) + "/" +
            std::to_string(MAX_LOST_FRAMES)
            + "]";
        SC = cv::Scalar(0, 165, 255);
        break;
    case ETrackState::Lost:
        S  = "SEARCHING...";
        SC = cv::Scalar(0, 0, 255);
        break;
    }

    cv::putText(Frame,
        "STATE: " + S,
        cv::Point(160, 24),
        cv::FONT_HERSHEY_SIMPLEX,
        0.65, SC, 2);

    cv::putText(Frame,
        "S=Select R=Reset "
        "1=CSRT 2=KCF "
        "3=CamShift Q=Quit",
        cv::Point(10, 55),
        cv::FONT_HERSHEY_SIMPLEX,
        0.45,
        cv::Scalar(180, 180, 180), 1);
}

int main()
{
    std::cout
        << "Object Tracker\n"
        << "==============\n"
        << "S = Select object\n"
        << "R = Reset\n"
        << "1 = CSRT (most accurate)\n"
        << "2 = KCF  (fastest)\n"
        << "3 = CamShift (color)\n"
        << "Q = Quit\n\n";

    // ================================================
    // SELECT YOUR SOURCE
    // MODE 1 = Laptop webcam
    // MODE 2 = Video file
    // MODE 3 = IP camera  <-- for office
    // ================================================
    const int MODE = 3;

    const int CAMERA_INDEX = 0;

    const std::string VIDEO_PATH =
        "C:/Users/Victus/Downloads/video.mp4";

    // !! CHANGE IP, USERNAME, PASSWORD !!
    const std::string IP_URL =
        "rtsp://admin:admin123@192.168.1.64:554/stream";
    // ================================================

    ThreadedCamera  Camera;
    Detector        Det;
    ObjectTracker   Tracker;
    FrameStabilizer Stab;
    FPSCounter      FPS;

    if (MODE == 1)
    {
        std::cout << "Opening webcam\n";
        if (!Camera.Open(CAMERA_INDEX))
        {
            std::cerr
                << "Failed to open webcam!\n";
            return -1;
        }
        std::cout << "Webcam opened!\n";
    }
    else if (MODE == 2)
    {
        std::cout
            << "Opening video: "
            << VIDEO_PATH << "\n";
        if (!Camera.OpenFile(VIDEO_PATH))
        {
            std::cerr
                << "Failed! Check path.\n";
            return -1;
        }
        std::cout << "Video opened!\n";
    }
    else if (MODE == 3)
    {
        std::cout
            << "Connecting to: "
            << IP_URL << "\n";
        if (!Camera.OpenFile(IP_URL))
        {
            std::cerr
                << "Failed! Check URL.\n"
                << IP_URL << "\n";
            return -1;
        }
        std::cout << "IP Camera connected!\n";
    }

    cv::Mat Frame;
    int  FrameCount = 0;
    bool bStabilize = false;

    cv::namedWindow(
        WINDOW_NAME,
        cv::WINDOW_NORMAL);
    cv::resizeWindow(
        WINDOW_NAME, 960, 540);

    while (true)
    {
        if (!Camera.GetLatestFrame(Frame))
            continue;

        FrameCount++;
        FPS.Tick();

        cv::Mat PF = Frame.clone();

        if (bStabilize)
            PF = Stab.Stabilize(Frame);

        if (Tracker.GetState() !=
            ETrackState::Idle)
            Tracker.Update(PF);

        Tracker.Draw(PF);
        DrawHUD(PF, FPS.Get(),
            Tracker.GetState(),
            Tracker.GetLostCount());

        cv::imshow(WINDOW_NAME, PF);

        char Key = (char)cv::waitKey(1);

        if (Key == 's' || Key == 'S')
        {
            cv::Rect ROI =
                Det.SelectROI(PF);
            if (ROI.width > 0 &&
                ROI.height > 0)
                Tracker.Init(PF, ROI,
                    ETrackerType::CSRT);
        }
        else if (Key == 'r' || Key == 'R')
        {
            Tracker.Reset();
            Stab.Reset();
            std::cout << "Reset\n";
        }
        else if (Key == 't' || Key == 'T')
        {
            bStabilize = !bStabilize;
            std::cout << "Stabilize: "
                << (bStabilize ?
                    "ON" : "OFF") << "\n";
        }
        else if (Key == '1')
        {
            if (Tracker.GetState() !=
                ETrackState::Idle)
            {
                Tracker.Init(PF,
                    Tracker.GetCurrentBox(),
                    ETrackerType::CSRT);
                std::cout << "CSRT\n";
            }
        }
        else if (Key == '2')
        {
            if (Tracker.GetState() !=
                ETrackState::Idle)
            {
                Tracker.Init(PF,
                    Tracker.GetCurrentBox(),
                    ETrackerType::KCF);
                std::cout << "KCF\n";
            }
        }
        else if (Key == '3')
        {
            if (Tracker.GetState() !=
                ETrackState::Idle)
            {
                Tracker.Init(PF,
                    Tracker.GetCurrentBox(),
                    ETrackerType::CAMSHIFT);
                std::cout << "CamShift\n";
            }
        }
        else if (Key == 'q' ||
            Key == 'Q' || Key == 27)
            break;
    }

    Camera.Stop();
    cv::destroyAllWindows();
    return 0;
}
