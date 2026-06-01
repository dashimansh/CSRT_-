#include "ThreadedCamera.h"
#include "Config.h"
#include <iostream>

ThreadedCamera::ThreadedCamera() {}

ThreadedCamera::~ThreadedCamera()
{
    Stop();
}

bool ThreadedCamera::Open(int ID)
{
    Cap.open(ID);
    if (!Cap.isOpened())
    {
        std::cerr
            << "Cannot open camera\n";
        return false;
    }
    SetupCamera();
    return true;
}

bool ThreadedCamera::OpenFile(
    const std::string& Path)
{
    Cap.open(Path);
    if (!Cap.isOpened()) return false;

    Width  = (int)Cap.get(
        cv::CAP_PROP_FRAME_WIDTH);
    Height = (int)Cap.get(
        cv::CAP_PROP_FRAME_HEIGHT);
    FPS    = (int)Cap.get(
        cv::CAP_PROP_FPS);

    std::cout << "Opened: "
        << Width << "x"
        << Height << " @ "
        << FPS << "fps\n";

    bRunning  = true;
    CamThread = std::thread(
        &ThreadedCamera::CaptureLoop,
        this);
    return true;
}

void ThreadedCamera::SetupCamera()
{
    Cap.set(cv::CAP_PROP_BUFFERSIZE,
        CAMERA_BUFFER_SIZE);
    Cap.set(cv::CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH);
    Cap.set(cv::CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT);
    Cap.set(cv::CAP_PROP_FPS,
        CAMERA_FPS);

    Width  = (int)Cap.get(
        cv::CAP_PROP_FRAME_WIDTH);
    Height = (int)Cap.get(
        cv::CAP_PROP_FRAME_HEIGHT);
    FPS    = (int)Cap.get(
        cv::CAP_PROP_FPS);

    std::cout << "Camera: "
        << Width << "x"
        << Height << " @ "
        << FPS << "fps\n";

    bRunning  = true;
    CamThread = std::thread(
        &ThreadedCamera::CaptureLoop,
        this);
}

void ThreadedCamera::CaptureLoop()
{
    while (bRunning)
    {
        cv::Mat Frame;
        Cap >> Frame;
        if (Frame.empty())
        {
            Cap.set(
                cv::CAP_PROP_POS_FRAMES,
                0);
            continue;
        }
        std::lock_guard<std::mutex>
            Lock(FrameMutex);
        LatestFrame = Frame.clone();
        bNewFrame   = true;
    }
}

bool ThreadedCamera::GetLatestFrame(
    cv::Mat& Out)
{
    if (!bNewFrame) return false;
    std::lock_guard<std::mutex>
        Lock(FrameMutex);
    Out       = LatestFrame.clone();
    bNewFrame = false;
    return true;
}

void ThreadedCamera::Stop()
{
    bRunning = false;
    if (CamThread.joinable())
        CamThread.join();
    Cap.release();
}
